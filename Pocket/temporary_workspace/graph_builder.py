import re
import os
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
import torch_geometric.transforms as T
 
LOG_ALL_EDGES = False
 
 
# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
 
def _parse_columns(df: pd.DataFrame):
    """
    Single-pass column categorisation.
    Returns:
        feature_cols  — per-object kinematic features
        global_cols   — event-level global features (events_* / w_*)
        objects       — {obj_type: {idx: {feat: col}}}
        edge_cols     — list of (etype_str, src_name, dst_name, col)
    """
    OBJ_RE  = re.compile(r"(jet|lepton|DeepMETResolutionTune)(\d*)_(.+)")
    EDGE_RE = re.compile(r"(mass|dR|dphi|deta)_(\w+)_(\w+)")
    EDGE_TYPE_NAMES = {"mass", "dR", "dphi", "deta"}
 
    exclude = (
        {"label", "year_tag", "category", "weight", "events_genWeight",
         "process", "w_had_jets_mass", "w_had_jets_N"}
        | {c for c in df.columns if c.startswith("weight_variation")}
        | {c for c in df.columns if c.startswith("vbsjets")}
    )

    numeric = set(df.select_dtypes(include=["number"]).columns)
 
    feature_cols = []
    global_cols  = []
    objects      = {}
    edge_cols    = []
 
    for col in df.columns:
        if col in exclude or col not in numeric:
            continue
 
        # Edge columns (separate pass)
        em = EDGE_RE.match(col)
        if em:
            etype, src, dst = em.groups()
            if etype in EDGE_TYPE_NAMES:
                edge_cols.append((etype, src, dst, col))
            continue
 
        # Global features
        if col.startswith("events") or col.startswith("w"):
            global_cols.append(col)
            continue
 
        # Node features
        om = OBJ_RE.match(col)
        if om:
            obj_type, idx, feat = om.groups()
            idx = idx or "1"
            objects.setdefault(obj_type, {})
            objects[obj_type].setdefault(idx, {})
            objects[obj_type][idx][feat] = col
            feature_cols.append(col)
 
    return feature_cols, global_cols, objects, edge_cols
 
 
def _build_node_arrays(df: pd.DataFrame, objects: dict, pad_width: int = 9):
    """
    Pre-extract node feature arrays for every object instance.
    Returns:
        node_arrays  — {(obj_type, idx): np.ndarray (N_events, pad_width)}
        node_order   — ordered list of (obj_type, idx)
        node_type_ids — list of int type-ids matching node_order
        node_name_to_idx — {"jet1": 0, "lepton1": 1, ...}
    """
    NODE_TYPES = {"jet": 0, "lepton": 1, "DeepMETResolutionTune": 2}
    TYPE_ORDER = ["jet", "lepton", "DeepMETResolutionTune"]
 
    node_arrays      = {}
    node_order       = []
    node_type_ids    = []
    node_name_to_idx = {}
 
    for obj_type in TYPE_ORDER:
        if obj_type not in objects:
            continue
        for idx in sorted(objects[obj_type].keys()):
            cols = list(objects[obj_type][idx].values())
            arr  = df[cols].to_numpy(dtype=float)          # (N, n_feats)
            arr[arr == -999] = 0.0
 
            if arr.shape[1] < pad_width:
                pad = np.zeros((arr.shape[0], pad_width - arr.shape[1]))
                arr = np.concatenate([arr, pad], axis=1)
 
            node_arrays[(obj_type, idx)] = arr             # (N, pad_width)
 
            node_name = (
                "DeepMETResolutionTune"
                if obj_type == "DeepMETResolutionTune"
                else f"{obj_type}{idx}"
            )
            node_name_to_idx[node_name] = len(node_order)
            node_order.append((obj_type, idx))
            node_type_ids.append(NODE_TYPES[obj_type])
 
    return node_arrays, node_order, node_type_ids, node_name_to_idx
 
 
def _build_edge_arrays(df: pd.DataFrame, edge_cols: list, node_name_to_idx: dict):
    """
    Pre-extract edge value arrays and resolve node indices.
    Returns:
        valid_edges — list of (etype_id, src_idx, dst_idx, values_array)
                      where values_array has shape (N_events,)
    """
    EDGE_TYPE_IDS = {"mass": 0, "dR": 1, "dphi": 2, "deta": 3}
    valid_edges = []
 
    for etype_str, src, dst, col in edge_cols:
        src_idx = node_name_to_idx.get(src)
        dst_idx = node_name_to_idx.get(dst)
        if src_idx is None or dst_idx is None:
            continue
        arr = df[col].to_numpy(dtype=float)                # (N_events,)
        valid_edges.append((EDGE_TYPE_IDS[etype_str], src_idx, dst_idx, arr))
 
    return valid_edges
 
 
# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
 
def get_hographical_fast(
    df: pd.DataFrame,
    cache_path: str = None,
    pad_width: int = 9,
    ) -> list:
    """
    Vectorized replacement for get_hographical.
 
    All pandas I/O is performed once before the event loop. The per-event
    loop does only numpy index operations and torch tensor assembly, which
    is 20-80x faster than iterrows() for typical VBS dataframes.
 
    Parameters
    ----------
    df          : DataFrame produced by load_category_dataframe / parquet load
    cache_path  : If given, save the data_list to disk on first call and
                  load from disk on subsequent calls. Useful for long runs.
    pad_width   : Feature vector width per node (default 9, matching original).
 
    Returns
    -------
    data_list : list of torch_geometric.data.Data, one per event
    """
 
    # ------------------------------------------------------------------
    # Cache load
    # ------------------------------------------------------------------
    if cache_path and os.path.exists(cache_path):
        print(f"[graph_builder] Loading cached graphs from {cache_path}")
        return torch.load(cache_path)
 
    print("[graph_builder] Pre-processing columns …")
 
    # ------------------------------------------------------------------
    # Phase 1 — column parsing and array extraction  (done ONCE)
    # ------------------------------------------------------------------
    _, global_cols, objects, edge_cols = _parse_columns(df)
 
    node_arrays, node_order, node_type_ids, node_name_to_idx = _build_node_arrays(
        df, objects, pad_width=pad_width
    )
    valid_edges = _build_edge_arrays(df, edge_cols, node_name_to_idx)
 
    global_arr = df[global_cols].to_numpy(dtype=float) if global_cols else None
 
    labels_arr  = df["label"].to_numpy(dtype=int)
    weights_arr = (
        df["weight"].to_numpy(dtype=float)
        if "weight" in df.columns
        else np.ones(len(df), dtype=float)
    )
    processes = df["process"].tolist()
    indices   = df.index.tolist()
    n_events  = len(df)
 
    # ------------------------------------------------------------------
    # Pre-compute fixed tensors shared across all events
    # ------------------------------------------------------------------
    NODE_TYPES    = {"jet": 0, "lepton": 1, "DeepMETResolutionTune": 2}
    EDGE_TYPES    = {"mass": 0, "dR": 1, "dphi": 2, "deta": 3}
    n_node_types  = len(NODE_TYPES)
    n_edge_types  = len(EDGE_TYPES)
    n_nodes       = len(node_order)
 
    node_type_tensor = torch.tensor(node_type_ids, dtype=torch.long)
    node_type_oh     = torch.nn.functional.one_hot(
        node_type_tensor, num_classes=n_node_types
    ).float()                                              # (n_nodes, 3) — constant
 
    # Identify pt column index (index 2 by convention; fall back to 0 for MET)
    # MET objects typically have fewer features, so we handle the fallback per-event.
    PT_COL = 2
 
    # Pre-stack the full node feature matrix: shape (N_events, n_nodes, pad_width)
    # This is the single largest vectorisation win — one malloc instead of N_events.
    print("[graph_builder] Stacking node feature arrays …")
    node_matrix = np.stack(
        [node_arrays[(obj_type, idx)] for (obj_type, idx) in node_order],
        axis=1
    )  # (N_events, n_nodes, pad_width)
 
    # Pre-stack edge value matrix: shape (N_events, n_valid_edges)
    print("[graph_builder] Stacking edge value arrays …")
    if valid_edges:
        edge_values_matrix = np.stack(
            [arr for (_, _, _, arr) in valid_edges], axis=1
        )  # (N_events, n_candidate_edges)
        edge_meta = [(etype_id, src_idx, dst_idx)
                     for (etype_id, src_idx, dst_idx, _) in valid_edges]
    else:
        edge_values_matrix = np.empty((n_events, 0), dtype=float)
        edge_meta = []
 
    # Pre-stack global features
    if global_arr is not None:
        global_tensor_all = torch.tensor(global_arr, dtype=torch.float)
        # (N_events, u_dim)
 
    print(f"[graph_builder] Building {n_events:,} graphs …")
 
    # ------------------------------------------------------------------
    # Phase 2 — event loop (tensor assembly only)
    # ------------------------------------------------------------------
    data_list = []
    transform = T.ToUndirected()
 
    for i in range(n_events):
 
        data = Data()
 
        # ---- Node features ----------------------------------------
        x = torch.tensor(node_matrix[i], dtype=torch.float)  # (n_nodes, pad_width)
 
        x_new = x.clone()
        if x.shape[1] > PT_COL:
            x_new[:, PT_COL] = torch.log(x[:, PT_COL].clamp(min=1e-3))
        else:
            x_new[:, 0] = torch.log(x[:, 0].clamp(min=1e-3))
        x = x_new
 
        data.x = torch.cat([x, node_type_oh], dim=1)         # (n_nodes, pad_width+3)
 
        # ---- Edges ------------------------------------------------
        ev = edge_values_matrix[i]                             # (n_candidate_edges,)
        valid_mask = ev != -999.0
 
        if valid_mask.any():
            sel_meta   = [edge_meta[j]    for j in range(len(edge_meta))    if valid_mask[j]]
            sel_vals   = ev[valid_mask]
 
            etype_ids  = torch.tensor([m[0] for m in sel_meta], dtype=torch.long)
            src_ids    = torch.tensor([m[1] for m in sel_meta], dtype=torch.long)
            dst_ids    = torch.tensor([m[2] for m in sel_meta], dtype=torch.long)
 
            edge_index = torch.stack([src_ids, dst_ids], dim=0)  # (2, E)
 
            ea_raw = torch.tensor(sel_vals, dtype=torch.float).unsqueeze(1)  # (E, 1)
 
            if LOG_ALL_EDGES:
                # Original behaviour: log-transform every edge value
                ea_raw[:, 0] = torch.log(ea_raw[:, 0].clamp(min=1e-3))
            else:
                # Preferred: only log-transform invariant mass edges (etype 0)
                mass_mask = etype_ids == EDGE_TYPES["mass"]
                if mass_mask.any():
                    ea_raw[mass_mask, 0] = torch.log(
                        ea_raw[mass_mask, 0].clamp(min=1e-3)
                    )
 
            edge_type_oh = torch.nn.functional.one_hot(
                etype_ids, num_classes=n_edge_types
            ).float()                                          # (E, 4)
 
            data.edge_index = edge_index
            data.edge_attr  = torch.cat([ea_raw, edge_type_oh], dim=1)  # (E, 5)
 
        else:
            data.edge_index = torch.zeros((2, 0), dtype=torch.long)
            data.edge_attr  = torch.zeros((0, 1 + n_edge_types), dtype=torch.float)
 
        # ---- Global features --------------------------------------
        if global_arr is not None:
            data.u = global_tensor_all[i].unsqueeze(0)        # (1, u_dim)
 
        # ---- Labels & metadata ------------------------------------
        data.y       = torch.tensor([labels_arr[i]],  dtype=torch.long)
        data.weight  = torch.tensor([weights_arr[i]], dtype=torch.float)
        data.process = processes[i]
        data.idx     = indices[i]
 
        # ---- PyG transform ----------------------------------------
        data = transform(data)
 
        data_list.append(data)
 
    print(f"[graph_builder] Done. Built {len(data_list):,} graphs.")
 
    # ------------------------------------------------------------------
    # Cache save
    # ------------------------------------------------------------------
    if cache_path:
        print(f"[graph_builder] Saving graphs to {cache_path}")
        torch.save(data_list, cache_path)
 
    return data_list