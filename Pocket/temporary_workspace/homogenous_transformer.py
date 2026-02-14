### Optimus Prime 
import re
from itertools import product
import sys
import os
import types
import pandas as pd
import coffea.util
import coffea
print(coffea.__file__)
import awkward as ak
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import Any, Dict, Optional
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
# import wandb
import networkx as nx
# from networkx.algorithms import community
# %run visualize.py #IDK IF I CAN DO THIS
# from visualize import GraphVisualization 

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.storage import GlobalStorage
from torch_geometric.utils import to_networkx
from torch.nn import (
    BatchNorm1d,
    Embedding,
    Linear,
    ModuleList,
    ReLU,
    Sequential,
    Sigmoid,
    LayerNorm,
    GELU
)
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch_geometric.transforms as T

from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, GPSConv, global_add_pool, global_max_pool, global_mean_pool, HeteroConv, TransformerConv, GlobalAttention
from torch_geometric.nn.attention import PerformerAttention

import mplhep as hep
hep.style.use("CMS")


def load_category_from_coffea(coffea_file, category_name, out_prefix):
    #Load and combine all processes for a given category from a .coffea file
    ####### DEBUGGING?

    # if 'coffea.processor.accumulator' not in sys.modules:
    #     sys.modules['coffea.processor.accumulator'] = types.ModuleType('accumulator')
    
    # # You can optionally map classes if needed:
    # try:
    #     from coffea.processor import processor
    #     sys.modules['coffea.processor.accumulator'].ProcessorAccumulator = getattr(processor, 'ProcessorAccumulator', None)
    # except ImportError:
    #     pass  # if ProcessorAccumulator no longer exists, keep as None
    

    ####### DEBUGGING?

    merged_file = coffea.util.load(coffea_file)
    columns = merged_file["columns"]
    gen_weight_normalization = merged_file["sum_genweights"]
    all_dfs = []
    count=False

    for process_name, process_dict in columns.items():
        for subkey, year_dict in process_dict.items():
            if not isinstance(year_dict, dict):
                continue
            if category_name not in year_dict:
                continue  # Skip missing categories
            category_dict = year_dict[category_name]
            #print(category_dict)
            print(f"Processing {process_name} ({subkey}) for category {category_name}")

            data_dict = {}

            for var, arr in category_dict['nominal'].items():
                # print(f"--- {var} ---")
                val = getattr(arr, "value", arr)
                # print("type:", type(val))
                # print("shape:", getattr(val, "shape", None))

                try:
                    arr_np = np.asarray(val)

                    # Ensure column vector (N,) not (N,1)
                    if arr_np.ndim == 2 and arr_np.shape[1] == 1:
                        arr_np = arr_np[:, 0]

                    # Skip empty arrays
                    if arr_np.size == 0:
                        print(f"Skipping empty array for {var}")
                        continue

                    # Store flattened 1D array
                    data_dict[var] = arr_np

                except Exception as e:
                    print(f"Could not process {var}: {e}")
                    continue

            if not data_dict:
                continue

            # print("DEBUG: Checking data_dict for scalar entries...")
            # for key, value in data_dict.items():
            #     if not isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            #         print(f"  --> SCALAR FOUND: key='{key}', value='{value}' (type={type(value)})")
            #     else:
            #         # Also catch 0-length arrays/lists
            #         try:
            #             if len(value) == 0:
            #                 print(f"  --> EMPTY sequence: key='{key}'")
            #         except TypeError:
            #             print(f"  --> NON-ITERABLE but not scalar? key='{key}', type={type(value)}")

            df = pd.DataFrame(data_dict)
            df["process"] = process_name
            df["year_tag"] = subkey
            df["category"] = category_name
            all_dfs.append(df)
            count=True
            if count:
                continue

    if not all_dfs:
        raise ValueError(f"No data found for category '{category_name}' in {coffea_file}")

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all["label"] = df_all["process"].apply(lambda x: 1 if "EWK" in str(x) else 0) 

    labels = df_all["label"].to_numpy()
    # # 3. Event weights
    if "weight" in df_all.columns:
        #print(df["year_tag"], " year and scaled correction ", norm_table)
        df_all["weight"] = df_all.apply( lambda row: row["weight"] / gen_weight_normalization.get(row["year_tag"], 1.0), axis=1 )
        weights = df_all["weight"].to_numpy()
    else:
        weights = None


    return df_all,gen_weight_normalization



def load_category_dataframe(parquet_path):
    df = pd.read_parquet(parquet_path)
    return df


def get_hographical(df):
    data_list = []  # 1 GRAPH PER EVENT

    # -----------------------------
    # Column handling
    # -----------------------------
    weights = df["weight"].to_numpy() if "weight" in df.columns else None

    exclude_cols = ["label", "year_tag", "category", "weight",
                    "events_genWeight", "process", "w_had_jets_mass"] + \
                   [c for c in df.columns if c.startswith("weight_variation")] + \
                   [c for c in df.columns if c.startswith("vbsjets")]

    feature_cols = (
        df.drop(columns=[c for c in exclude_cols if c in df.columns])
          .select_dtypes(include=["number"])
          .columns.tolist()
    )

    global_cols = [c for c in feature_cols if c.startswith("events") or c.startswith("w")]
    feature_cols = [c for c in feature_cols if c not in global_cols]

    # -----------------------------
    # Parse object structure
    # -----------------------------
    OBJ_RE = re.compile(r"(jet|lepton|PuppiMET)(\d*)_(.+)")
    objects = {}

    for col in feature_cols:
        m = OBJ_RE.match(col)
        if not m:
            continue
        obj_type, idx, feat = m.groups()
        idx = idx or "1"
        objects.setdefault(obj_type, {})
        objects[obj_type].setdefault(idx, {})
        objects[obj_type][idx][feat] = col

    NODE_TYPES = {"jet": 0, "lepton": 1, "PuppiMET": 2}
    EDGE_TYPES = {"mass": 0, "dR": 1, "dphi": 2, "deta": 3}

    # -----------------------------
    # Build one graph per event
    # -----------------------------
    for evt_idx, row in df.iterrows():
        data = Data()

        node_features = []
        node_types = []
        node_name_to_idx = {}

        # ---------- JETS ----------
        for idx, feats in sorted(objects.get("jet", {}).items()):
            cols = list(feats.values())
            vals = row[cols].values.astype(float)
            vals[vals == -999] = 0.0
            # if (vals == -999).any():
            #     continue
            if vals.shape[0] < 9:
                vals = np.pad(vals, (0, 9 - vals.shape[0]), constant_values=0.0)
            
            node_name_to_idx[f"jet{idx}"] = len(node_features)
            node_features.append(vals)
            node_types.append(NODE_TYPES["jet"])

        # ---------- LEPTONS ----------
        for idx, feats in sorted(objects.get("lepton", {}).items()):
            cols = list(feats.values())
            vals = row[cols].values.astype(float)
            vals[vals == -999] = 0.0
            if vals.shape[0] < 9:
                vals = np.pad(vals, (0, 9 - vals.shape[0]), constant_values=0.0)
            
            node_name_to_idx[f"lepton{idx}"] = len(node_features)
            node_features.append(vals)
            node_types.append(NODE_TYPES["lepton"])

        # ---------- MET ----------
        met_feats = objects.get("PuppiMET", {}).get("1")
        if met_feats:
            cols = list(met_feats.values())
            vals = row[cols].values.astype(float)

            if vals.shape[0] < 9:
                vals = np.pad(vals, (0, 9 - vals.shape[0]), constant_values=0.0)
            
            node_name_to_idx["PuppiMET"] = len(node_features)
            node_features.append(vals)
            node_types.append(NODE_TYPES["PuppiMET"])

        # ---------- Node tensors ----------
        x = torch.tensor(node_features, dtype=torch.float)

        # log-pt (same logic as before)
        x_new = x.clone()
        try:
            x_new[:, 2] = torch.log(x[:, 2].clamp(min=1e-3))
        except Exception:
            x_new[:, 0] = torch.log(x[:, 0].clamp(min=1e-3))
        x = x_new

        node_type = torch.tensor(node_types, dtype=torch.long)
        node_type_oh = torch.nn.functional.one_hot(
            node_type, num_classes=len(NODE_TYPES)
        ).float()

        # concatenate type flags to features
        data.x = torch.cat([x, node_type_oh], dim=1)

        # -----------------------------
        # Build homogeneous edges
        # -----------------------------
        edge_index = []
        edge_attr = []
        edge_type = []

        EDGE_RE = re.compile(r"(mass|dR|dphi|deta)_(\w+)_(\w+)")

        for col in df.columns:
            m = EDGE_RE.match(col)
            if not m:
                continue

            etype, src, dst = m.groups()
            val = float(row[col])
            if val == -999:
                continue

            src_idx = node_name_to_idx.get(src)
            dst_idx = node_name_to_idx.get(dst)
            if src_idx is None or dst_idx is None:
                continue

            edge_index.append([src_idx, dst_idx])
            edge_attr.append([val])
            edge_type.append(EDGE_TYPES[etype])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        # log-mass
        edge_attr[:, 0] = torch.log(edge_attr[:, 0].clamp(min=1e-3))

        edge_type = torch.tensor(edge_type, dtype=torch.long)
        edge_type_oh = torch.nn.functional.one_hot(
            edge_type, num_classes=len(EDGE_TYPES)
        ).float()

        data.edge_index = edge_index
        # data.edge_attr = torch.cat([edge_attr, edge_type_oh], dim=1)

        # num_edges = data.edge_index.size(1)

        # num_edges = data.edge_index.size(1)
        # raw_edge_feat = edge_attr  # shape [num_edges, feat_dim]
        # edge_type_oh = torch.zeros((num_edges, len(EDGE_TYPES)), device=raw_edge_feat.device)

        # # Then fill edge_type_oh according to your edge types:
        # for i, etype in enumerate(edge_type_list_per_edge):  # list of length num_edges
        #     edge_type_oh[i, etype] = 1.0

        # Concatenate along feature dimension
        data.edge_attr = torch.cat([edge_attr, edge_type_oh], dim=1)
        num_edges = data.edge_index.size(1)
        # if data.edge_attr.size(0) != num_edges:
        #     raise ValueError(f"Edge features size {data.edge_attr.size(0)} does not match number of edges {num_edges}")
        if data.edge_attr is None:
            # No edge features → initialize zeros
            data.edge_attr = torch.zeros((num_edges, 1), device=data.x.device)
        elif data.edge_attr.size(0) != num_edges:
            raise ValueError(f"Edge features size {data.edge_attr.size(0)} does not match number of edges {num_edges}")

        # -----------------------------
        # Global features
        # -----------------------------
        if global_cols:
            data.u = torch.tensor(
                [row[global_cols].values.astype(float)],
                dtype=torch.float
            )

        # -----------------------------
        # Labels & metadata
        # -----------------------------
        data.y = torch.tensor([int(row["label"])], dtype=torch.long)
        data.weight = torch.tensor([float(row["weight"])], dtype=torch.float)
        data.process = row["process"]
        data.idx = evt_idx

        # -----------------------------
        # PyG transforms
        # -----------------------------
        data = T.ToUndirected()(data)
        # data = T.AddSelfLoops()(data)

        data_list.append(data)

    return data_list



def get_graphical(df):
    data_list = [] # GOAL IS 1 GRAPH PER EVENT

    # Event weights
    weights = df["weight"].to_numpy() if "weight" in df.columns else None # MAKE SURE WE PROPAGATE WEIGHTS INTO DA SYSTEM

    exclude_cols = ["label", "year_tag", "category", "weight", "events_genWeight", "process", "w_had_jets_mass"] + \
                   [c for c in df.columns if c.startswith("weight_variation")] + \
                   [c for c in df.columns if c.startswith("vbsjets")]
                   # EXCLUDE COLUMNS THAT DON'T CONTAIN FEATURES

    feature_cols = ( # DROP THE ABOVE COLUMNS FROM GRAPH
        df.drop(columns=[c for c in exclude_cols if c in df.columns])
        .select_dtypes(include=["number"])
        .columns.tolist()
    )

    # Separate global vs node features
    global_cols = [c for c in feature_cols if c.startswith("events") or c.startswith("w")] # set global feature list
    feature_cols = [c for c in feature_cols if c not in global_cols] # set rest of features excluding the above

    # type_prefixes = ["lepton", "jet", "PuppiMET"] # 3 types of nodes supposdely
    OBJ_RE = re.compile(r"(jet|lepton|PuppiMET)(\d*)_(.+)")
    objects = {}
    for col in feature_cols:
        m = OBJ_RE.match(col)
        if not m:
            continue

        obj_type, idx, feat = m.groups()
        idx = idx or "1"  

        objects.setdefault(obj_type, {})
        objects[obj_type].setdefault(idx, {})
        objects[obj_type][idx][feat] = col

    # -----------------------------
    # Build one graph per event
    # -----------------------------
    for evt_idx, row in df.iterrows():
        data = HeteroData()

        # ---------- JETS (many nodes, one type) ----------
        jet_features = []
        jet_index_map = {}  # jet1 → 0, jet2 → 1, ...

        for idx, feats in sorted(objects.get("jet", {}).items()):
            cols = list(feats.values())
            vals = row[cols].values.astype(float)

            if (vals == -999).any():
                continue

            jet_index_map[f"jet{idx}"] = len(jet_features)
            jet_features.append(vals)

        if jet_features:
            data["jet"].x = torch.tensor(jet_features, dtype=torch.float)
            data["jet"].num_nodes = data["jet"].x.size(0)
        # ---------- LEPTONS (one or more nodes) ----------
        lep_features = []
        lep_index_map = {}  # lepton1 -> 0, lepton2 -> 1, etc.

        for idx, feats in sorted(objects.get("lepton", {}).items()):
            cols = list(feats.values())
            vals = row[cols].values.astype(float)
            vals[vals == -999] = 0.0
            lep_index_map[f"lepton{idx}"] = len(lep_features)
            lep_features.append(vals)

        if lep_features:
            data["lepton"].x = torch.tensor(lep_features, dtype=torch.float)
            data["lepton"].num_nodes = data["lepton"].x.size(0)

        # ---------- MET (single node) ----------
        met_feats = objects.get("PuppiMET", {}).get("1")
        if met_feats:
            cols = list(met_feats.values())
            vals = row[cols].values.astype(float)

            data["PuppiMET"].x = torch.tensor([vals], dtype=torch.float)
            data["PuppiMET"].num_nodes = 1

        # ---------- DEBUG: node sanity ----------
        # print(f"\n[EVENT {evt_idx}] NODE SUMMARY")
        for nt in data.node_types:
            x = data[nt].x
            if x is None or x.numel() == 0:
                # No node features → fill with zeros
                data[nt].x = torch.zeros((data[nt].num_nodes, self.channels), device=x.device)
                continue

            x_new=x.clone()
            try:
                x_new[:, 2] = torch.log(x[:, 2].clamp(min=1e-3)) #set pt to log scale
            except:
                x_new[:, 0] = torch.log(x[:, 0].clamp(min=1e-3))#set pt to log scale (MET)
            data[nt].x = x_new
            # print(f"Node type: {nt}, shape: {x_new.shape}")
            # print(f"First 5 nodes:\n{x_new[:5, :]}")
            # print(f"{nt} nodes: mean={x_new.mean():.3f}, std={x_new.std():.3f}, " f"min={x_new.min():.1f}, max={x_new.max():.1f}" )
        # -----------------------------
        # Build edges from existing columns
        # -----------------------------

        edges = {etype: [] for etype in ["mass","dR","dphi","deta"]}
        edge_attrs = {etype: [] for etype in ["mass","dR","dphi","deta"]}


        EDGE_RE = re.compile(r"(mass|dR|dphi|deta)_(\w+)_(\w+)")
        edge_dict = {}
        for col in df.columns:
            m = EDGE_RE.match(col)
            if not m:
                continue

            etype, src, dst = m.groups()
            val = float(row[col])

            if val == -999:
                continue

            def resolve(obj): # this function makes sure the jet index is appropriately handled
                if obj.startswith("jet"):
                    return "jet", jet_index_map.get(obj)
                if obj.startswith("lepton"):
                    return "lepton", 0
                if obj.startswith("PuppiMET"):
                    return "PuppiMET", 0
                return None, None

            src_type, src_idx = resolve(src)
            dst_type, dst_idx = resolve(dst)

            if src_idx is None or dst_idx is None:
                continue
            key = (src_type, etype, dst_type)
            if key not in edge_dict:
                edge_dict[key] = {'edge_index': [], 'edge_attr': []}

            # Append the edge and its feature
            edge_dict[key]['edge_index'].append([src_idx, dst_idx])
            edge_dict[key]['edge_attr'].append([val])

        # for etype in edges:
        #     if not edges[etype]:
        #         continue
        for key, d in edge_dict.items():
            edge_index = torch.tensor(d['edge_index'], dtype=torch.long).t().contiguous()
            # edge_attr = torch.tensor(d['edge_attr'], dtype=torch.float)
            if len(d['edge_attr']) == 0:
                edge_attr = torch.zeros((edge_index.shape[1], 1), dtype=torch.float)
            else:
                edge_attr = torch.tensor(d['edge_attr'], dtype=torch.float)
            # if edge_index.shape[1] != edge_attr.shape[0]:
            #     min_len = min(edge_index.shape[1], edge_attr.shape[0])
            #     edge_index = edge_index[:, :min_len]
            #     edge_attr = edge_attr[:min_len]
            #     print(f"[WARN] Truncated edges for {key} to {min_len} to match edge_attr")

            # edge_attr = torch.full((len(edges), node_channels), val, dtype=torch.float)
            assert edge_index.shape[1] == edge_attr.shape[0], f"{key} mismatch!"
            # print(key, len(d['edge_index']), len(d['edge_attr']))

            data[key].edge_index = edge_index
            data[key].edge_attr = edge_attr

        # ---------- DEBUG: edge sanity ----------
        for (src_type, etype, dst_type), edge_data in data.items():
            ei = edge_data.edge_index
            ea = edge_data.edge_attr
            x = data[src_type].x
            y = data[dst_type].x

            # Assert edge_index vs edge_attr lengths
            assert ei.shape[1] == ea.shape[0], f"{src_type}-{etype}-{dst_type} mismatch"

            # Assert node indices are valid
            assert ei.max() < x.shape[0], f"{src_type}-{etype}-{dst_type} src index out of bounds"
            assert ei.min() >= 0, f"{src_type}-{etype}-{dst_type} src index negative"
            assert ei.max(1)[1] < y.shape[0], f"{src_type}-{etype}-{dst_type} dst index out of bounds"

            # Optionally: check feature dimension of edge_attr
            assert ea.shape[1] == 1, f"{src_type}-{etype}-{dst_type} edge_attr dim mismatch"

        # -----------------------------
        # Global features
        # -----------------------------
        if global_cols:
            data.u = torch.tensor([row[global_cols].values.astype(float)], dtype=torch.float)

        for etype in data.edge_types:
            ea = data[etype].edge_attr
            if ea is None or ea.numel() == 0:
                # No edge features → fill with zeros
                num_edges = data[etype].edge_index.shape[1]
                data[etype].edge_attr = torch.zeros((num_edges, self.channels), device=data[etype].edge_index.device)
                continue

            ea_new = ea.clone()
            # assume mass is the first column for edges where it exists
            if ea_new.shape[1] > 0:
                ea_new[:, 0] = torch.log(ea[:, 0].clamp(min=1e-3))#set di-object mass to log scale
            data[etype].edge_attr = ea_new

            # print(f"{etype} edges: min mass={ea_new[:,0].min():.3f}, max mass={ea_new[:,0].max():.3f}")
        # Labels & metadata
        # -----------------------------
        data.y = torch.tensor([int(row["label"])], dtype=torch.long)
        data.weight = torch.tensor([float(row["weight"])], dtype=torch.float)
        data.process = row["process"]
        data.idx = evt_idx

        # PyG transforms
        data = T.ToUndirected()(data)
        data = T.AddSelfLoops()(data)

        data_list.append(data)

    return data_list

# def get_graphical(df):
#     data_list = []

#     # df["label"] = df["process"].apply(lambda x: 1 if "EWK" in str(x) else 0) 
#     node_feature_dims = {
#         "jet1": 3,
#         "jet2": 3,
#         "jet3": 3,
#         "jet4": 3,
#         "jet5": 3,
#         "jet6": 3,
#         "lepton1": 3,
#         "PuppiMET": 2,
#     }
#     labels = df["label"].to_numpy()
#     # 3. Event weights
#     if "weight" in df.columns:
#         #print(df["year_tag"], " year and scaled correction ", norm_table)
#         # df["weight"] = df.apply( lambda row: row["weight"] / norm_table.get(row["year_tag"], 1.0), axis=1 )
#         weights = df["weight"].to_numpy()
#     else:
#         weights = None

    
#     exclude_cols = ["label", "year_tag", "category", "weight", "events_genWeight", "process",  'weight_variation_PileupWeightUp', 'weight_variation_PileupWeightDown', 'weight_variation_sf_mu_idUp', 'weight_variation_sf_mu_idDown', 'weight_variation_sf_mu_isoUp', 'weight_variation_sf_mu_isoDown', 'weight_variation_sf_ele_idUp', 'weight_variation_sf_ele_idDown', 'weight_variation_sf_ele_recoUp', 'weight_variation_sf_ele_recoDown', 'weight_variation_sf_L1prefiringUp', 'weight_variation_sf_L1prefiringDown', 'weight_variation_sf_mu_triggerUp', 'weight_variation_sf_mu_triggerDown', 'weight_variation_sf_jet_puIdUp', 'weight_variation_sf_jet_puIdDown', 'weight_variation_sf_partonshower_isrUp', 'weight_variation_sf_partonshower_isrDown', 'weight_variation_sf_partonshower_fsrUp', 'weight_variation_sf_partonshower_fsrDown']#, 'PuppiMET_pt','PuppiMET_phi', 'w_had_jets_centrality_resolved','w_had_jets_N']


#     feature_cols = (
#         df.drop(columns=[c for c in exclude_cols if c in df.columns])
#         .select_dtypes(include=["number"])
#         .columns.tolist()
#     )
#     # print("Features:", feature_cols)
#     # print(df[feature_cols].dtypes)

#     EDGE_PREFIXES = ("mass", "dR", "dphi", "deta")
#     feature_cols = [
#         col for col in feature_cols
#         if not any(col.startswith(p + "_") for p in EDGE_PREFIXES)
#     ]

#     global_cols = [c for c in feature_cols if c.startswith("events") or c.startswith("w")]
#     feature_cols = [c for c in feature_cols if c not in global_cols]

#     type_prefixes = ["lepton", "jet", "PuppiMET"]
#     feature_groups = {t: [] for t in type_prefixes}

#     # feature_groups = {}
    
#     for col in feature_cols:
#         for t in type_prefixes:
#             if col.startswith(t):
#                 feature_groups[t].append(col)
#                 break

#     enumerated_groups = {}
#     for obj in type_prefixes:
#         # find all columns starting with obj
#         obj_cols = [c for c in feature_cols if c.startswith(obj)]
#         # group by object index
#         obj_dict = {}
#         for col in obj_cols:
#             m = re.match(rf"{obj}(\d*)_(.*)", col)
#             if m:
#                 idx = m.group(1) or ""  # default to blank if no index
#                 obj_idx = f"{obj}{idx}"
#                 obj_dict.setdefault(obj_idx, []).append(col)
#         enumerated_groups.update(obj_dict)


#     print("\nFeature groups:")
#     for k, v in feature_groups.items():
#         print(f"  {k}: {v}")

#     row_num=0
#     data_list = []
#     for _, row in df.iterrows():

#         data = HeteroData()
#         # --------------------------------------------------------------
#         # Create node types with their own feature vectors
#         # One node per feature group
#         # --------------------------------------------------------------
#         for node_type, cols in enumerated_groups.items():
#             vals = row[cols].values.astype(float)
#             # data[node_type].num_nodes = 0
#             if np.all(vals == -999):
#                 vals = np.zeros(node_feature_dims[node_type], dtype=float)
#             vals[vals == -999] = 0.0
#             # shape => (1, num_features_for_this_group)
            
#             data[node_type].x = torch.tensor([vals], dtype=torch.float)
#             data[node_type].num_nodes = data[node_type].x.size(0)
#         # --------------------------------------------------------------
#         # Build fully connected edges between all node types
#         # With exactly 1 node in each type, the index is always [[0],[0]]
#         # --------------------------------------------------------------
#         node_types = list(enumerated_groups.keys())

#         EDGE_QUANTITIES = auto_generate_mappings(df, node_types)

#         EDGE_TYPES = ["mass", "dR", "dphi","deta"]

#         for src, dst in product(node_types, repeat=2):
#             if src not in data.node_types or dst not in data.node_types:
#                 # skip edges if either node type doesn't exist
#                 continue
#             for etype in EDGE_TYPES:
#                     # get correct event-based column
#                     col = EDGE_QUANTITIES.get((src, etype, dst))
#                     if col is None:
#                         # data[src, etype, dst].edge_index = torch.zeros((2, 0), dtype=torch.long)
#                         # data[src, etype, dst].edge_attr = torch.zeros((0, 1), dtype=torch.float)
#                         continue

#                     # 1-edge graph (0 → 0)
#                     edge_index = torch.tensor([[0], [0]], dtype=torch.long)

#                     # lookup precomputed scalar
#                     val = float(row[col])
#                     edge_attr = torch.tensor([[val]], dtype=torch.float)

#                     data[src, etype, dst].edge_index = edge_index
#                     data[src, etype, dst].edge_attr  = edge_attr
                
#         if global_cols:
#             global_vals = row[global_cols].values.astype(float)
#             data.u = torch.tensor([global_vals], dtype=torch.float)

#         # --------------------------------------------------------------
#         # Attach labels and metadata
#         # --------------------------------------------------------------
#         data.y = torch.tensor([int(row["label"])], dtype=torch.long)
#         data.weight = torch.tensor([float(row["weight"])], dtype=torch.float)
#         data.process = row["process"]

#         data.node_types = node_types 

#         data = T.ToUndirected()(data)
#         data = T.AddSelfLoops()(data)
#         # data = T.AddRandomWalkPE(walk_length=20, attr_name='pe')(data)
#         data_list.append(data)
#         # if row_num > 3:
#         #     continue
#         # row_num+=1

#     node_set = set()
#     for data in data_list:
#         node_set.update(data.node_types)
#     print(node_set)   
    
#     return data_list

def visualize_graph(graph, index):
    """
    Visualize a HomogenousData graph event-by-event including:
      - Nodes
      - Edge features
      - Global features in the title
    """
    g = nx.Graph()

    # --- Add nodes per node type ---
    node_offset = 0
    node_map = {}  # map (node_type, local_idx) -> global idx
    for node_type in graph.node_types:
        x = graph[node_type].x
        for i in range(x.shape[0]):
            global_idx = node_offset + i
            g.add_node(global_idx, label=node_type)
            node_map[(node_type, i)] = global_idx
        node_offset += x.shape[0]

    # --- Add edges per edge type ---
    for src, etype, dst in graph.edge_types:
        # print("etype", etype)
        edge_index = graph[src, etype, dst].edge_index
        edge_attr = graph[src, etype, dst].edge_attr
        for i in range(edge_index.shape[1]):
            u = node_map[(src, edge_index[0, i].item())]
            v = node_map[(dst, edge_index[1, i].item())]
            g.add_edge(u, v, label=f"{etype}: {edge_attr[i].item():.2f}" if edge_attr is not None else etype)

    # --- Draw graph ---
    pos = nx.spring_layout(g, seed=42)
    plt.figure(figsize=(8, 8))
    labels = nx.get_node_attributes(g, 'label')
    edge_labels = nx.get_edge_attributes(g, 'label')
    nx.draw(g, pos, with_labels=True, labels=labels, node_size=800, node_color='lightblue')
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=8)

    # --- Global features ---
    if hasattr(graph, 'u') and graph.u is not None:
        u_vals = graph.u.squeeze().tolist()
        if not isinstance(u_vals, list):
            u_vals = [u_vals]
        plt.title("Global: " + ", ".join([f"{v:.2f}" for v in u_vals]), fontsize=10)

    # plt.tight_layout()
    plt.savefig(f"event_graph_{index}.png")
    plt.close()

# fig = create_graph(dataset[0])

def auto_generate_mappings(df, node_types, edge_types=("mass", "dR", "dphi", "deta")):
    """
    Automatically map (src, etype, dst) -> df column name,
    with direction independence:
      mass_jet_lep maps to both (jet, mass, lep) and (lep, mass, jet).
    """
    EDGE_QUANTITIES = {}

    # regex: mass_jet_lep  → (etype, src, dst)
    pattern = re.compile(rf"^({'|'.join(edge_types)})_(\w+)_(\w+)$")

    for col in df.columns:
        m = pattern.match(col)
        if not m:
            continue

        etype, src, dst = m.groups()

        # Only accept if src/dst are actual node types
        if src not in node_types or dst not in node_types:
            continue

        # Store mapping for both directions
        EDGE_QUANTITIES[(src, etype, dst)] = col
        EDGE_QUANTITIES[(dst, etype, src)] = col

    return EDGE_QUANTITIES

# class RedrawProjection:
#     def __init__(self, model: torch.nn.Module,
#                 redraw_interval: Optional[int] = None):
#         self.model = model
#         self.redraw_interval = redraw_interval
#         self.num_last_redraw = 0

#     def redraw_projections(self):
#         if not self.model.training or self.redraw_interval is None:
#             return
#         if self.num_last_redraw >= self.redraw_interval:
#             fast_attentions = [
#                 module for module in self.model.modules()
#                 if isinstance(module, PerformerAttention)
#             ]
#             for fast_attention in fast_attentions:
#                 fast_attention.redraw_projection_matrix()
#             self.num_last_redraw = 0
#             return
#         self.num_last_redraw += 1

class HomoGraph(torch.nn.Module):
    def __init__(
        self,
        channels,
        num_layers,
        u_dim,
        heads=3,
    ):
        super().__init__()

        self.channels = channels
        self.heads = heads

        # ---------------------------------
        # Node encoder
        # ---------------------------------
        # Full node feature vector INCLUDING type flags
        self.node_emb = None  # lazy init

        # ---------------------------------
        # Edge encoder
        # ---------------------------------
        self.edge_emb = None

        # self.edge_emb = Sequential(
        #     Linear(None, channels),  # lazy init
        #     ReLU(),
        #     Linear(channels, channels),
        # )

        self.edge_emb_rd = torch.nn.ModuleList([
            Sequential(
                Linear(channels * (heads ** i), channels * (heads ** (i + 1))),
                ReLU(),
            )
            for i in range(num_layers - 1)
        ])

        # ---------------------------------
        # Message passing
        # ---------------------------------
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        for layer_idx in range(num_layers):
            in_ch = channels if layer_idx == 0 else channels * heads

            self.convs.append(
                TransformerConv(
                    in_channels=in_ch,
                    out_channels=channels,
                    heads=heads,
                    dropout=0.1,
                    edge_dim=channels * (heads ** layer_idx),
                )
            )

            self.norms.append(LayerNorm(channels * heads))

        self.res_proj = Linear(channels, channels * heads)

        # ---------------------------------
        # Global feature encoder
        # ---------------------------------
        self.u_mlp = Sequential(
            Linear(u_dim, channels),
            LayerNorm(channels),
            ReLU(),
        )

        # ---------------------------------
        # Output MLP
        # ---------------------------------
        self.mlp = Sequential(
            Linear(channels * heads + channels, channels // 2),
            ReLU(),
            Linear(channels // 2, channels // 4),
            ReLU(),
            Linear(channels // 4, 1),
            Sigmoid(),
        )

        self.ffn = torch.nn.ModuleList([
            Sequential(
                Linear(channels * heads, 4 * channels * heads),
                GELU(),
                Linear(4 * channels * heads, channels * heads),
            ) for _ in range(num_layers)
        ])

        self.norms2 = torch.nn.ModuleList([
            LayerNorm(channels * heads) for _ in range(num_layers)
        ])

    # --------------------------------------------------
    def forward(self, data: Data):
        # device = next(model.parameters()).device
        # data.x = torch.tensor(node_features, dtype=torch.float, device=device)

        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch
        device = x.device

        # if self.edge_emb is None:
        #     self.edge_emb = Sequential()
        #         Linear(edge_attr.size(1), self.channels),
        #         ReLU(),
        #         Linear(self.channels, self.channels),
        #     ).to(edge_attr.device)
        #     # ---------------------------------
        #     # Lazy init encoders
        #     # ---------------------------------
        if self.node_emb is None:
            self.node_emb = Linear(
                x.size(1),
                self.channels,
                device=x.device
            ).to(device)

        if self.edge_emb is None:
            self.edge_emb = Sequential(
                Linear(edge_attr.size(1), self.channels),
                ReLU(),
                Linear(self.channels, self.channels),
            ).to(device)

        # ---------------------------------
        # Encode nodes / edges
        # ---------------------------------
        h = self.node_emb(x)
        ea = self.edge_emb(edge_attr)

        # ---------------------------------
        # Message passing
        # ---------------------------------
        for layer_idx, conv in enumerate(self.convs):
            if layer_idx == 0:
                h_new = conv(h, edge_index, ea)
                h = h_new + self.res_proj(h)
            else:
                ea = self.edge_emb_rd[layer_idx - 1](ea)
                h_new = conv(h, edge_index, ea)
                h = h_new + h

            h = self.norms[layer_idx](h)

            h_ffn = self.ffn[layer_idx](h)
            h = self.norms2[layer_idx](h + h_ffn)

        # ---------------------------------
        # Global pooling (ALL nodes)
        # ---------------------------------
        pooled = global_add_pool(h, batch)
            # global_max_pool(h, batch),
            # global_mean_pool(h, batch)],
            # dim=-1)

        # ---------------------------------
        # Global features
        # ---------------------------------
        u = self.u_mlp(data.u)

        out = torch.cat([pooled, u], dim=-1)
        return self.mlp(out)



class HeteroGraph(torch.nn.Module):
    def __init__(self, channels,node_types, edge_types, num_layers,num_node_categories, num_edge_categories, u_dim, attn_type='multihead'):
        super().__init__()

        self.channels = channels 
        # input_dim = channels * len(node_types) + channels


        node_feature_dims = {
            "jet": 3,
            "lepton": 3,
            "PuppiMET": 2,
            # "w": 2,
        }

        self.edge_type_to_key = {
            etype: f"{etype[0]}__{etype[1]}__{etype[2]}" for etype in edge_types
        }
        self.key_to_edge_type = {
            v: k for k, v in self.edge_type_to_key.items()
        }


        self.node_emb = torch.nn.ModuleDict({
            nt: Linear(node_feature_dims[nt], channels) 
            for nt in node_types
        })

        self.edge_emb = torch.nn.ModuleDict({
            "__".join(etype): Sequential(
                Linear(1, channels),
                ReLU(),
                Linear(channels, channels)
            )
            for etype in edge_types
        })

        self.edge_emb_rd2 = torch.nn.ModuleList([
            torch.nn.ModuleDict({
                "__".join(etype): Sequential(
                    Linear(channels * (3 ** i), channels * (3 ** (i + 1))),
                    ReLU(),
                )
                for etype in edge_types
            })
            for i in range(num_layers - 1)
        ])

        self.u_mlp = Sequential(
            Linear(u_dim, channels),
            LayerNorm(channels),
            ReLU(),
        )


        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList([
            LayerNorm(channels * 3)
            for _ in range(num_layers)
            ])
        for layer_idx in range(num_layers):
            if layer_idx == 0:
                conv = HeteroConv({
                    (src, rel, dst): TransformerConv(
                        in_channels=channels,  # match node_emb output
                        out_channels=channels,
                        heads=3,
                        dropout=0.1,
                        edge_dim=channels,
                    )
                    for (src, rel, dst) in edge_types
                }, aggr='mean')
            else: # layer_idx % 2 == 1:
                conv = HeteroConv({
                    (src, rel, dst): TransformerConv(
                        in_channels=channels*3,  # previous out_channels * heads
                        out_channels=channels,
                        heads=3,
                        dropout=0.1,
                        edge_dim=channels*(3**layer_idx),
                    )
                    for (src, rel, dst) in edge_types
                }, aggr='mean')
            # elif layer_idx % 2 == 0:
            #     conv = HeteroConv({
            #         (src, rel, dst): TransformerConv(
            #             in_channels=channels,  # previous out_channels * heads
            #             out_channels=channels,
            #             heads=3,
            #             dropout=0.1,
            #             edge_dim=channels,
            #         )
            #         for (src, rel, dst) in edge_types
            #     }, aggr='mean')
            # self.u_mlp = Sequential( Linear(u_dim, channels), ReLU(),)

            self.convs.append(conv)
        
        self.mlp = None
        self.node_types = node_types
        self.res_proj = torch.nn.ModuleDict({
            nt: Linear(channels, channels *3)  # heads = 3
            for nt in node_types
        })

        self.edge_res_scale = torch.nn.Parameter(torch.tensor(1.0))


        pooled_nodes_dim = len(node_types) * channels *3
        input_dim = pooled_nodes_dim + channels
        # print("input dim: ", input_dim)
        # self.mlp = None
        self.mlp = Sequential(
                Linear(input_dim, channels // 2),
                ReLU(),
                Linear(channels // 2, channels // 4),
                ReLU(),
                Linear(channels // 4, 1),
                Sigmoid()
            )

    def forward(self, data: HeteroData):

        # ----- Node inputs -----
        x_dict = { nt: data[nt].x for nt in data.node_types }
        # ----- Encode nodes -----
        h = {}
        out_device = data['u'].device
        batch_size = data['u'].size(0) 
        for nt in x_dict:
            # print(
            #     "Node type:", nt,
            #     "min:", x_dict[nt].min().item(),
            #     "max:", x_dict[nt].max().item(),
            #     # "num_categories:", num_node_categories[nt]
            # )
            h_nt = torch.cat([ self.node_emb[nt](x_dict[nt]), ], dim=-1)
            h[nt] = h_nt
        edge_index_dict = {
            etype: data[etype].edge_index
            for etype in data.edge_types
        }

        for etype in data.edge_types:
            ei = data[etype].edge_index
            ea = data[etype].edge_attr
            num_edges = ei.shape[1]
            if ea is None or ea.shape[0] == 0:
                # No edge features at all → fill with zeros
                data[etype].edge_attr = torch.zeros(
                    (num_edges, self.channels),
                    device=ei.device
                )

            elif ea.shape[0] < num_edges:
                # Missing some edge features → pad with zeros
                pad = torch.zeros(
                    (num_edges - ea.shape[0], ea.shape[1]),
                    device=ea.device
                )
                data[etype].edge_attr = torch.cat([ea, pad], dim=0)

            elif ea.shape[0] > num_edges:
                # Extra edge features (rare but possible)
                data[etype].edge_attr = ea[:num_edges]

        
        # layers
        h0 = h.copy()

        edge_attr_dict_prev={}
        for layer_idx, conv in enumerate(self.convs):
            edge_attr_dict={}
            
            for etype in data.edge_types:    # etype = (src, rel, dst)
                if 'edge_attr' not in data[etype]:
                    continue
                key = "__".join(etype)
                
                if layer_idx == 0:
                    ea = data[etype].edge_attr
                    edge_attr_dict[etype] = self.edge_emb[key](ea)
                else:
                    ea = edge_attr_dict_prev[etype]
                    edge_attr_dict[etype] = self.edge_emb_rd2[layer_idx - 1][key](ea)
            edge_attr_dict_prev = edge_attr_dict.copy()

                # edge_attr_dict[etype] = self.edge_emb[key](data[etype].edge_attr)

            h_new = conv(
                h,
                edge_index_dict=edge_index_dict,
                edge_attr_dict=edge_attr_dict
            )
            h_out = {}
            for nt in h:
                if nt in h_new:
                    if layer_idx == 0:
                        h_res = h_new[nt] + self.res_proj[nt](h[nt])
                    else:
                        h_res = h_new[nt] + h[nt]
                else:
                    if layer_idx == 0:
                        h_res = self.res_proj[nt](h[nt])
                    else:
                        h_res = h[nt]
                # shared LayerNorm
                h_out[nt] = self.norms[layer_idx](h_res)
            h = h_out
        num_graphs = data.num_graphs 
        pooled = []

        for nt in self.node_types:  
            if nt in data.node_types:
                h_nt = h[nt]                      # [num_nodes, C]
                batch = data[nt].batch            # [num_nodes]

                # 1) pool over nodes - [num_graphs, C]
                pooled_nt = global_add_pool(
                    h_nt, batch, size=num_graphs
                )
                # pooled_nt = self.pool[nt](h_nt, batch)
                # # 2) reduce over features - [num_graphs, 1]
                # pooled_nt = pooled_nt.sum(dim=1, keepdim=True)

            else:
                # Node type missing entirely - zero scalar per graph
                pooled_nt = torch.zeros(
                    num_graphs, self.channels *3, device=out_device
                )

            pooled.append(pooled_nt)

        pooled_nodes = torch.cat(pooled, dim=-1)


        u = data['u']          # shape [batch, u_dim]
        u_emb = self.u_mlp(u)  # map to channel size

        out = torch.cat([pooled_nodes, u_emb], dim=-1)

        return self.mlp(out)


def neural_net_initialization(full_data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        print("Using CPU")

    node_types, edge_types, num_node_categories, num_edge_categories, u_dim = collect_graph_metadata(full_data)

    # 6. Hyperparameters
    channels = 32
    num_layers = 3
    attn_type = 'multihead'
    attn_kwargs = {'dropout': 0.1}

    model = HomoGraph(
        channels=channels,
        num_layers=num_layers,
        u_dim=u_dim,    
    ).to(device)

    # model = HeteroGraph(
    #     node_types=node_types,
    #     edge_types=edge_types,
    #     channels=channels,
    #     num_layers=num_layers,
    #     num_node_categories=num_node_categories,
    #     num_edge_categories=num_edge_categories,
    #     u_dim=u_dim).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)
    return model, optimizer, scheduler, device

def plot_training_curves(train_losses, val_errors):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(7,5))
    plt.plot(epochs, train_losses, label="Training")
    plt.plot(epochs, val_errors, label="Validation")
    plt.xlabel("Epoch")
    # plt.yscale('log')
    plt.ylabel("Loss")
    plt.title("Training Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("transform_training_curve_2018.png")
    plt.close()

def plot_auc_curves(train_aucs, val_aucs):
    epochs = range(1, len(train_aucs) + 1)

    plt.figure(figsize=(7,5))
    plt.plot(epochs, train_aucs, label="Train AUC")
    plt.plot(epochs, val_aucs, label="Validation AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title("AUC vs Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("transform_auc_vs_epoch.png")
    plt.close()


def plot_roc_curves(datasets, title="ROC Curve Transformer", filename="roc_comparison.png"):
    """
    Plot multiple ROC curves on the same figure.
    
    datasets: list of dicts, each with keys:
        - 'labels': tensor or numpy array of true labels
        - 'probs' : tensor or numpy array of predicted probabilities
        - 'weights': optional tensor or numpy array
        - 'name'  : string label for the curve
    """
    plt.figure(figsize=(6,6))
    
    for data in datasets:
        labels = data['labels']
        probs  = data['probs']
        weights = data.get('weights', None)
        name = data.get('name', 'ROC')

        # Convert to numpy if needed
        if hasattr(labels, 'numpy'):
            labels = labels.numpy()
        if hasattr(probs, 'numpy'):
            probs = probs.numpy()
        if weights is not None and hasattr(weights, 'numpy'):
            weights = weights.numpy()

        fpr, tpr, _ = roc_curve(labels, probs, sample_weight=weights)
        fpr, idx = np.unique(fpr, return_index=True)
        tpr = tpr[idx]

        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
        # roc_auc = auc(fpr, tpr)
        # plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

    plt.plot([0,1], [0,1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title,fontsize=15)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_score_distributions(labels, probs, weights=None,outname="transform_score_distrib_2018.png"):
    plt.figure(figsize=(7,5))

    sig = labels == 1
    bkg = labels == 0

    if weights is None:
        plt.hist(probs[bkg], bins=40, alpha=0.6, label="Background")
        plt.hist(probs[sig], bins=40, alpha=0.6, label="Signal")
    else:
        plt.hist(probs[bkg], bins=40, weights=weights[bkg],
                 alpha=0.6, label="Background")
        plt.hist(probs[sig], bins=40, weights=weights[sig],
                 alpha=0.6, label="Signal")

    plt.xlabel("Transformer discriminator output",fontsize=14)
    plt.ylabel("Yields",fontsize=14)
    # plt.title("Classifier Output Distribution")
    # plt.yscale('log')
    hep.cms.label(
        "Preliminary",
        data=False,             
        loc=0, 
        fontsize=16,   # reduce from the very large default
        lumi=41.5,       
        com=13                  
    )
    plt.legend(fontsize=8)
    # plt.grid(True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outname)

def plot_score_distributions_stacked(
    df,
    scores,
    bins=40,
    max_events=None,
    outname="transform_score_distrib_stacked_2018.png",
):
    """
    Plot stacked classifier output distribution grouped by process groups.
    Uses df indexing to stay aligned with shuffled data.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.cm import get_cmap

    if max_events is not None:
        df = df.iloc[:max_events]
        scores = scores[:max_events]

    # --- extract aligned arrays ---
    processes = df["process"].values
    if "weight" in df.columns:
        weights = df['weight'].values #SCALING TO FULL 2017
    else:
        weights = np.ones(len(df))

    if hasattr(scores, "cpu"):  # torch tensor
        scores = scores.cpu().numpy()

    # --- same process grouping ---
    process_groups = {
        "VBS_EWK": [
            "WminusTo2JZTo2LJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WminusToLNuWminusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WminusToLNuZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WplusTo2JWminusToLNuJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WplusTo2JZTo2LJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WplusToLNuWminusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WplusToLNuWplusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WplusToLNuZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "ZTo2LZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
        ],
        "Top/ttbar": [
            "ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8",
            "ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8",
            "ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8",
            "ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8",
            "ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8",
            "ttWJets_TuneCP5_13TeV_madgraphMLM_pythia8",
            "ttZJets_TuneCP5_13TeV_madgraphMLM_pythia8",
            "TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8",
            "TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8",
        ],
        "W+jets": [
            "WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8",
            "WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8",
            "WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8",
            "WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8",
            "WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8",
            "WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8",
            "WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8",
            "WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8",
            "WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8",
        ],
        "DY": [
            "DYJetsToLL_M-10to50_TuneCP5_13TeV-amcatnloFXFX-pythia8",
            "DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8",
        ],
        "QCD-VV": [ "WminusTo2JZTo2LJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WminusToLNuWminusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WminusToLNuZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WplusTo2JZTo2LJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WplusTo2JWminusToLNuJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WplusToLNuWminusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WplusToLNuWplusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WplusToLNuZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "ZTo2LZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8"]
    }

    def map_to_group(proc):
        for g, members in process_groups.items():
            if proc in members:
                return g
        return "Other"

    groups = np.array([map_to_group(p) for p in processes])
    unique_groups = sorted(np.unique(groups))

    # --- plotting ---
    plt.figure(figsize=(7, 5))

    cmap = get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(unique_groups))]

    stacked_scores = []
    stacked_weights = []

    for g in unique_groups:
        mask = groups == g
        stacked_scores.append(scores[mask])
        stacked_weights.append(weights[mask])

    plt.hist(
        stacked_scores,
        bins=bins,
        weights=stacked_weights,
        stacked=True,
        label=unique_groups,
        color=colors,
        edgecolor=None,
        alpha=0.85,
    )

    plt.xlabel("Transformer discriminator output",fontsize=14)
    plt.ylabel("Yields",fontsize=14)
    # plt.title("Classifier Output Distribution")
    # plt.yscale('log')
    hep.cms.label(
        "Preliminary",
        data=False,             
        loc=0, 
        fontsize=16,   # reduce from the very large default
        lumi=41.5,       
        com=13                  
    )
    plt.legend(fontsize=8)

    # plt.xlabel("Model output (sigmoid)")
    # plt.ylabel("Events")
    # plt.title("Classifier Output Distribution")
    plt.legend(fontsize=8)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outname, dpi=300)
    plt.close()


def collect_graph_metadata(dataset):
    node_types = set()
    edge_types = set()
    num_node_categories = {}
    num_edge_categories = {}
    u_dim = None

    # for g in dataset:
    #     # Node types
    #     for nt in g.node_types:
    #         node_types.add(nt)
    #         # Track max feature size per node type
    #         # print("node type before error", nt)
    #         x_size = g[nt].x.size(1)
    #         if nt not in num_node_categories or x_size > num_node_categories[nt]:
    #             num_node_categories[nt] = x_size
    #     # Edge types
    #     for et in g.edge_types:
    #         edge_types.add(et)
    #         if hasattr(g[et], 'edge_attr'):
    #             e_size = g[et].edge_attr.size(1)
    #             if et not in num_edge_categories or e_size > num_edge_categories[et]:
    #                 num_edge_categories[et] = e_size
    #         else:
    #             if et not in num_edge_categories:
    #                 num_edge_categories[et] = 0

        # Global feature u
    # if u_dim is None:
    u_dim = dataset[0].u.size(1)

    return list(node_types), list(edge_types), num_node_categories, num_edge_categories, u_dim



def balance_signal_background_weights_graphs(data_list, balance_to="signal", force_pos=True):
    """
    Rescale signal or background weights so that
    sum(signal_weights) == sum(background_weights).

    """

    # Extract labels and weights
    labels = np.array([int(d.y.item()) for d in data_list])

    weights = np.array([
        float(d.weight.item()) if hasattr(d, "weight") else 1.0
        for d in data_list
    ])

    if force_pos:
        weights=abs(weights)
    sig_mask = labels == 1
    bkg_mask = labels == 0

    sum_sig = weights[sig_mask].sum()
    sum_bkg = weights[bkg_mask].sum()

    if sum_sig == 0 or sum_bkg == 0:
        raise ValueError("Error: One of the classes has zero total weight!")

    if balance_to == "background":
        scale = sum_bkg / sum_sig
        weights[sig_mask] *= scale
        print(f"Scaled signal weights by {scale:.3f} to match background total.")

    elif balance_to == "signal":
        scale = sum_sig / sum_bkg
        weights[bkg_mask] *= scale
        print(f"Scaled background weights by {scale:.3f} to match signal total.")
    
    elif balance_to == "unity":
        scale1 = 1 / sum_bkg
        scale2 = 1 / sum_sig
        weights[bkg_mask] *= scale1
        weights[sig_mask] *= scale2
        print(f"Scaled sig and bkg weights by {scale1:.3f} and {scale2:.3f} to match unity.")

    else:
        raise ValueError("balance_to must be either 'signal' or 'background'.")

    # Write weights back into the graphs
    for d, w in zip(data_list, weights):
        d.weight = torch.tensor([w], dtype=torch.float)

    print(
        f"After reweighting: "
        f"sum(signal)={weights[sig_mask].sum():.3f}, "
        f"sum(background)={weights[bkg_mask].sum():.3f}"
    )

    return data_list

def plot_raw_logit_output(first_batch_logits, last_batch_logits,epoch):
   
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.hist(first_batch_logits.numpy(), bins=30, color='skyblue')
    plt.title("Logits - First Batch")
    plt.xlabel("Logit value")
    plt.ylabel("Count")

    plt.subplot(1,2,2)
    plt.hist(last_batch_logits.numpy(), bins=30, color='salmon')
    plt.title("Logits - Last Batch")
    plt.xlabel("Logit value")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.savefig(f"transform_logits_{epoch}.png")


def get_kin_range(df, kin, cfg):
    if kin not in cfg:
        return None  

    spec = cfg[kin]

    if "range" in spec:
        return spec["range"]

    if "pct" in spec:
        lo, hi = spec["pct"]
        return np.percentile(df[kin].values, [lo, hi])

    raise ValueError(f"Invalid axis spec for {kin}")


def plot_kinematic_vs_score_2d(df, scores, kinematics, bins=(20,20), max_events=None):
    if max_events is not None:
        df = df.iloc[:max_events]
        scores = scores[:max_events]

    labels = df['label'].values
    if 'weight' in df.columns:
        event_weights = df['weight'].values
    else:
        event_weights = np.ones(len(df))
    KIN_AXES = {
        "jet1_pt":     dict(range=(0, 500)),
        "jet2_pt":     dict(range=(0, 500)),
        "mass_jet1_jet2": dict(range=(0, 1500)),
        "mass_jet3_jet4": dict(range=(0, 200)),
    }

    for kin in kinematics:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

        x_range = get_kin_range(df, kin, KIN_AXES)
        if x_range is None:
            x_range = (df[kin].min(), df[kin].max())
        x_bins = bins[0]

        for ax, lbl, title in zip(axes, [0, 1], ["Background", "Signal"]):
            mask = labels == lbl
            x = df[kin].values[mask]
            w = event_weights[mask]
            y = scores[mask]
            if isinstance(y, torch.Tensor):
                y = y.cpu().numpy()   
            # --- 2D histogram (normalized) ---
            h = ax.hist2d(
                x,
                y,
                bins=bins,
                range=[x_range, (0, 1.0)],
                cmap='viridis',
                weights=w,
                density=True    # ← normalize Z
            )
            fig.colorbar(h[3], ax=ax, label='Density')

            # --- Profile: mean score per x-bin ---
            x_edges = np.linspace(x_range[0], x_range[1], x_bins + 1)
            x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])

            mean_scores = np.full(x_bins, np.nan)
            for i in range(x_bins):
                in_bin = (x >= x_edges[i]) & (x < x_edges[i+1])
                if np.any(in_bin):
                    mean_scores[i] = np.average(y[in_bin], weights=w[in_bin])

            ax.plot(
                x_centers,
                mean_scores,
                color='red',
                marker='o',
                markersize=4,
                linewidth=2,
                #label=r'$\langle \mathrm{score} \rangle$'
            )

            ax.set_xlabel(kin)
            ax.set_ylabel("Probability")
            ax.set_title(title)
            ax.set_ylim(0, 1.0)
            ax.legend()

        plt.tight_layout()
        plt.savefig(f"score_comp_{kin}_2018.png")
        plt.close()

process_groups = {
    "VBS_EWK": [
        "WminusTo2JZTo2LJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
        "WminusToLNuWminusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
        "WminusToLNuZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
        "WplusTo2JWminusToLNuJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
        "WplusTo2JZTo2LJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
        "WplusToLNuWminusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
        "WplusToLNuWplusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
        "WplusToLNuZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
        "ZTo2LZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
    ],
    "Top/ttbar": [
        "ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8",
        "ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8",
        "ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8",
        "ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8",
        "ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8",
        "ttWJets_TuneCP5_13TeV_madgraphMLM_pythia8",
        "ttZJets_TuneCP5_13TeV_madgraphMLM_pythia8",
        "TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8",
        "TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8",
    ],
    "W+jets": [
        "WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8",
        "WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8",
        "WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8",
        "WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8",
        "WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8",
        "WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8",
        "WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8",
        "WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8",
        "WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8",
    ],
    "DY": [
        "DYJetsToLL_M-10to50_TuneCP5_13TeV-amcatnloFXFX-pythia8",
        "DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8",
    ],
    "QCD-VV": [ "WminusTo2JZTo2LJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WminusToLNuWminusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WminusToLNuZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WplusTo2JZTo2LJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WplusTo2JWminusToLNuJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WplusToLNuWminusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WplusToLNuWplusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WplusToLNuZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "ZTo2LZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8"]
}

def build_roc_dataset(df, scores, signal_group, background_groups):
    """
    Returns labels, probs, weights for ROC construction.
    signal_group: string (e.g. "VBS_EWK")
    background_groups: list of group names OR "all"
    """

    processes = df["process"].values
    weights = df["weight"].values if "weight" in df.columns else np.ones(len(df))

    if hasattr(scores, "cpu"):
        scores = scores.cpu().numpy()

    # map process → group
    proc_to_group = {}
    for g, members in process_groups.items():
        for m in members:
            proc_to_group[m] = g

    groups = np.array([proc_to_group.get(p, "Other") for p in processes])

    if background_groups == "all":
        mask = (groups == signal_group) | (groups != signal_group)
    else:
        mask = (groups == signal_group) | np.isin(groups, background_groups)

    labels = (groups[mask] == signal_group).astype(int)
    probs  = scores[mask]
    wts    = weights[mask]

    return labels, probs, wts

# def plot_roc_per_background(df, scores, outdir="roc_groups"):
#     import os
#     os.makedirs(outdir, exist_ok=True)

#     for bkg in ["W+jets", "Top/ttbar", "DY", "QCD-VV"]:
#         labels, probs, wts = build_roc_dataset(
#             df, scores, "VBS_EWK", background_groups=[bkg]
#         )

#         plot_roc_curves(
#             datasets=[{
#                 "labels": labels,
#                 "probs": probs,
#                 "weights": wts,
#                 "name": f"VBS_EWK vs {bkg}"
#             }],
#             title=f"Transformer: VBS_EWK vs {bkg}",
#             filename=f"{outdir}/roc_vbs_vs_{bkg.replace('+','').replace('/','_')}.png"
#         )

def plot_roc_per_background(df, scores, outdir="roc_groups"):
    os.makedirs(outdir, exist_ok=True)

    # --- align scores ---
    if hasattr(scores, "cpu"):
        scores = scores.cpu().numpy()

    processes = df["process"].values
    weights = df["weight"].values if "weight" in df.columns else np.ones(len(df))
    weights = abs(weights)
    # --- SAME process grouping as reference ---
    process_groups = {
        "VBS_EWK": [
            "WminusTo2JZTo2LJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WminusToLNuWminusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WminusToLNuZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WplusTo2JWminusToLNuJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WplusTo2JZTo2LJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WplusToLNuWminusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WplusToLNuWplusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WplusToLNuZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "ZTo2LZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
        ],
        "Top/ttbar": [
            "ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8",
            "ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8",
            "ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8",
            "ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8",
            "ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8",
            "ttWJets_TuneCP5_13TeV_madgraphMLM_pythia8",
            "ttZJets_TuneCP5_13TeV_madgraphMLM_pythia8",
            "TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8",
            "TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8",
        ],
        "W+jets": [
            "WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8",
            "WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8",
            "WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8",
            "WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8",
            "WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8",
            "WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8",
            "WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8",
            "WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8",
            "WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8",
        ],
        "DY": [
            "DYJetsToLL_M-10to50_TuneCP5_13TeV-amcatnloFXFX-pythia8",
            "DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8",
        ],
        "QCD-VV": [
            "WminusTo2JZTo2LJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WminusToLNuWminusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WminusToLNuZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WplusTo2JZTo2LJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WplusTo2JWminusToLNuJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WplusToLNuWminusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WplusToLNuWplusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WplusToLNuZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "ZTo2LZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
        ],
    }

    def map_to_group(proc):
        for g, members in process_groups.items():
            if proc in members:
                return g
        return "Other"

    groups = np.array([map_to_group(p) for p in processes])

    # --- loop over backgrounds ---
    for bkg in ["W+jets", "DY", "QCD-VV", "Top/ttbar", "Other"]:

        mask = (groups == "VBS_EWK") | (groups == bkg)

        if not np.any(groups == bkg):
            print(f"[SKIP] {bkg}: no events found")
            continue

        labels = (groups[mask] == "VBS_EWK").astype(int)
        probs = scores[mask]
        wts = weights[mask]

        if len(np.unique(labels)) < 2:
            print(f"[SKIP] {bkg}: only one class after selection")
            continue

        fpr, tpr, _ = roc_curve(labels, probs, sample_weight=wts)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], "--", color="gray")

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Transformer: VBS_EWK vs {bkg}", fontsize=15)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        outname = f"{outdir}/roc_vbs_vs_{bkg.replace('+','').replace('/','_')}_2018.png"
        plt.savefig(outname, dpi=300)
        plt.close()

        print(f"Saved {outname}")


def main():

    parser = argparse.ArgumentParser(description="Convert specific category from .coffea to pandas DataFrame")
    parser.add_argument("coffea_file", help="Path to input .coffea file")
    parser.add_argument("category", help="Event category key (e.g. whad_withbveto_e)")
    parser.add_argument("--out", help="Optional output parquet/csv file", default=None)
    parser.add_argument(    '--attn_type', default='multihead',    help="Global attention type such as 'multihead' or 'performer'.")

    args = parser.parse_args()
    ########## SAMPLE USAGE ###############
    ##### python homogenous_transformer.py /eos/user/v/vslokenb/vbs_semilep/outputs/v9_rerun_btag/output_merged_v9_rerun_btag.coffea whad_withbveto_mu --out '/eos/user/v/vslokenb/vbs_semilep/outputs/with_btag.parquet'
    #######################################
    # df_init, norm = load_category_from_coffea(args.coffea_file, args.category)

    # df_init.to_parquet(args.out, index=False)

    df = load_category_dataframe(args.out)
    data_list=get_hographical(df)
    # data_list = balance_signal_background_weights_graphs( data_list,    balance_to="unity")
  
    from torch_geometric.loader import DataLoader
    train_val_list, test_list = train_test_split(data_list, test_size=0.2, random_state=42)
    train_list, val_list = train_test_split(train_val_list, test_size=0.25, random_state=42 )
    # 0.25 x 0.8 = 0.2, so final split is 60% train / 20% val / 20% test
    #  0.5 * 0.9 = 0.45, so split is 45, 45, 10 (need stats for debugging)

    train_loader = DataLoader(balance_signal_background_weights_graphs( train_list, balance_to="unity"), batch_size=32, shuffle=True)
    val_loader   = DataLoader(balance_signal_background_weights_graphs( val_list, balance_to="unity"), batch_size=64, shuffle=False)
    test_loader  = DataLoader(balance_signal_background_weights_graphs( test_list, balance_to="unity"), batch_size=64, shuffle=False)

    print(f"Train size: {len(train_list)}, Val size: {len(val_list)}, Test size: {len(test_list)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model, optimizer, scheduler, device = neural_net_initialization(data_list)
    criterion = torch.nn.BCELoss(reduction="none")

    @torch.enable_grad()
    def train(model, optimizer, train_loader, device):
        model.train()

        total_loss = 0.0
        total_weight = 0.0

        all_logits = []
        all_labels = []
        all_weights = []

        for data_idx, data in enumerate(train_loader):
            data = data.to(device)

            optimizer.zero_grad()
            # model.redraw_projection.redraw_projections()

            out = model(data).squeeze()      # logits
            y   = data.y.float()

            # out = torch.sigmoid(out)
            loss_per_event = criterion(out, y)

            if hasattr(data, 'weight'):
                w = data.weight.float()

                loss = (loss_per_event * w).sum() / w.sum()

                total_loss   += (loss_per_event * w).sum().item()
                total_weight += w.sum().item()

                all_weights.append(w.detach())
            else:
                loss = loss_per_event.mean()

                total_loss   += loss_per_event.sum().item()
                total_weight += loss_per_event.numel()

                all_weights.append(torch.ones_like(y))

            loss.backward()
            optimizer.step()

            if data_idx == 0:
                first_batch_logits = out.detach()
            last_batch_logits = out.detach()
            all_logits.append(out.detach())
            all_labels.append(y.detach())

        all_logits = torch.cat(all_logits).cpu()
        all_labels = torch.cat(all_labels).cpu()
        all_weights = torch.cat(all_weights).cpu()
        first_batch_logits = first_batch_logits.cpu()
        last_batch_logits = last_batch_logits.cpu()


        avg_loss = total_loss / total_weight

        return avg_loss, all_logits, all_labels, all_weights, first_batch_logits, last_batch_logits





    @torch.no_grad()
    def test(model, loader, device):
        model.eval()

        total_loss = 0.0
        total_weight = 0.0

        all_logits = []
        all_labels = []
        all_weights = []

        for data in loader:
            data = data.to(device)

            out = model(data).squeeze()     # logits
            y   = data.y.float()
            # out = torch.sigmoid(out)
            loss_per_event = criterion(out, y)

            if hasattr(data, 'weight'):
                w = data.weight.float()

                total_loss   += (loss_per_event * w).sum().item()
                total_weight += w.sum().item()

                all_weights.append(w.detach())
            else:
                total_loss   += loss_per_event.sum().item()
                total_weight += loss_per_event.numel()

                all_weights.append(torch.ones_like(y))

            all_logits.append(out.detach())
            all_labels.append(y.detach())

        all_logits  = torch.cat(all_logits).cpu()
        all_labels  = torch.cat(all_labels).cpu()
        all_weights = torch.cat(all_weights).cpu()

        avg_loss = total_loss / total_weight

        return avg_loss, all_logits, all_labels, all_weights



    num_epochs = 500
    train_losses = []
    val_errors   = []
    val_aucs     = []
    train_aucs   = []

    patience = 15  # how many epochs to wait for improvement
    best_val_error = float('inf')
    epochs_no_improve = 0
    best_model_path = "best_model_2018.pt"  # path to save the best model

    for epoch in range(1, num_epochs + 1):
        # -------- TRAIN --------
        train_loss, train_logits, train_labels, train_weights,first_batch_logits, last_batch_logits = train(
            model, optimizer, train_loader, device
        )
        if hasattr(model, 'edge_res_scale'):
            print(f"Epoch {epoch}: edge_res_scale = {model.edge_res_scale.item():.3f}")

        print("Logits range: ", train_logits.min().item(), train_logits.max().item())
        train_probs = (train_logits)
        print("probs range: ", train_probs.min().item(), train_probs.max().item())
        # train_auc = roc_auc_score(
        #     train_labels.numpy(),
        #     train_probs.numpy(),
        #     sample_weight=train_weights.numpy()
        # )

        if epoch == 1 or epoch == num_epochs:
            plot_raw_logit_output(first_batch_logits, last_batch_logits, epoch)
        # -------- VALIDATION --------
        val_error, val_logits, val_labels, val_weights = test(
            model, val_loader, device
        )

        val_probs = (val_logits)
        # val_auc = roc_auc_score(
        #     val_labels.numpy(),
        #     val_probs.numpy(),
        #     sample_weight=val_weights.numpy()
        # )

        # -------- SCHEDULER --------
        scheduler.step(val_error)

        # -------- LOGGING --------
        train_losses.append(train_loss)
        val_errors.append(val_error)
        # train_aucs.append(train_auc)
        # val_aucs.append(val_auc)

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            # f"Train AUC: {train_auc:.4f} | "
            f"Val Error: {val_error:.4f} | "
            # f"Val AUC: {val_auc:.4f}"
        )

        # -------- EARLY STOPPING --------
        if val_error < best_val_error:
            best_val_error = val_error
            epochs_no_improve = 0
            # Save the best model
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved at epoch {epoch}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    # After loop, you can load the best model if needed
    # model.load_state_dict(torch.load(best_model_path))
    # model.to(device)

       ###### LOAD BEST MODEL ##########
    print("Loading best model for evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()

    ######## plots?
    # Validation probabilities
    val_probs = (val_logits)

    # Plots
    plot_training_curves(train_losses, val_errors)
    plot_auc_curves(train_aucs, val_aucs)
    # plot_roc_curve(val_labels, val_probs, val_weights, "validation")
    plot_score_distributions(val_labels, val_probs, val_weights)

    

    # with torch.no_grad():
    # for batch in val_loader:  # could be train_loader, test_loader too
    #     batch = batch.to(device)
    #     out = model(batch)
    #     probs = torch.softmax(out, dim=1)
    #     for i, data in enumerate(batch.to_data_list()):
    #         df_val.loc[data.idx, 'score'] = probs[i, 1].item()
    
    # plot_kinematic_vs_score_2d(df_val, val_probs, )
    # ===== FINAL TEST EVALUATION =====
    test_error, test_logits, test_labels, test_weights = test(
        model, test_loader, device
    )

    test_probs = (test_logits)

    test_auc = roc_auc_score(
        test_labels.numpy(),
        test_probs.numpy(),
        sample_weight=test_weights.numpy()
    )
    # plot_roc_curve(test_labels, test_probs, test_weights,"Test")
    plot_roc_curves([
    {"labels": test_labels, "probs": test_probs, "weights": test_weights, "name": "Test"},
    {"labels": train_labels, "probs": train_probs, "weights": train_weights, "name": "Train"}], 
    title="Train vs Test ROC", 
    filename="roc_test_vs_train_test_2018.png")

    print("\n===== FINAL TEST RESULTS =====")
    print(f"Test Error: {test_error:.4f}")
    print(f"Test AUC:   {test_auc:.4f}")

    df_test = df.loc[[data.idx for data in test_list]]
    # df_test['score'] = test_probs.numpy() 
    print(df_test.head())
    plot_kinematic_vs_score_2d(df_test, test_probs, ["mass_jet1_jet2", "mass_jet3_jet4", "jet1_pt","jet2_pt", "events_mt_w_leptonic", "vbsjets_mass", "w_had_jets_mass", "vbsjets_delta_eta","jet1_qgl","jet2_qgl", "jet3_qgl","jet4_qgl", "events_nCentralJetsGood", "events_nJetGood"])
    plot_score_distributions_stacked(df_test, test_probs)
    plot_roc_per_background(df_test, test_probs)

if __name__ == "__main__":
    main()


