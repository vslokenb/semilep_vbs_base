### Optimus Prime in lingerie
import re
import sys
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
)
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch_geometric.transforms as T

from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, GPSConv, global_add_pool, HeteroConv, TransformerConv
from torch_geometric.nn.attention import PerformerAttention

import mplhep as hep
hep.style.use("CMS")


def load_category_from_coffea(coffea_file, category_name):
    #Load and combine all processes for a given category from a .coffea file
    ####### DEBUGGING?

    if 'coffea.processor.accumulator' not in sys.modules:
        sys.modules['coffea.processor.accumulator'] = types.ModuleType('accumulator')
    
    # You can optionally map classes if needed:
    try:
        from coffea.processor import processor
        sys.modules['coffea.processor.accumulator'].ProcessorAccumulator = getattr(processor, 'ProcessorAccumulator', None)
    except ImportError:
        pass  # if ProcessorAccumulator no longer exists, keep as None
    

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
    return df_all,gen_weight_normalization

def get_graphical(df, norm_table):
    data_list = []

    df["label"] = df["process"].apply(lambda x: 1 if "EWK" in str(x) else 0) 

    labels = df["label"].to_numpy()
    # 3. Event weights
    if "weight" in df.columns:
        #print(df["year_tag"], " year and scaled correction ", norm_table)
        df["weight"] = df.apply( lambda row: row["weight"] / norm_table.get(row["year_tag"], 1.0), axis=1 )
        weights = df["weight"].to_numpy()
    else:
        weights = None

    
    exclude_cols = ["label", "year_tag", "category", "weight", "events_genWeight", "process",  'weight_variation_PileupWeightUp', 'weight_variation_PileupWeightDown', 'weight_variation_sf_mu_idUp', 'weight_variation_sf_mu_idDown', 'weight_variation_sf_mu_isoUp', 'weight_variation_sf_mu_isoDown', 'weight_variation_sf_ele_idUp', 'weight_variation_sf_ele_idDown', 'weight_variation_sf_ele_recoUp', 'weight_variation_sf_ele_recoDown', 'weight_variation_sf_L1prefiringUp', 'weight_variation_sf_L1prefiringDown', 'weight_variation_sf_mu_triggerUp', 'weight_variation_sf_mu_triggerDown', 'weight_variation_sf_jet_puIdUp', 'weight_variation_sf_jet_puIdDown', 'weight_variation_sf_partonshower_isrUp', 'weight_variation_sf_partonshower_isrDown', 'weight_variation_sf_partonshower_fsrUp', 'weight_variation_sf_partonshower_fsrDown']


    feature_cols = (
        df.drop(columns=[c for c in exclude_cols if c in df.columns])
        .select_dtypes(include=["number"])
        .columns.tolist()
    )
    # print("Features:", feature_cols)
    # print(df[feature_cols].dtypes)

    EDGE_PREFIXES = ("mass", "dR", "dphi", "deta")
    feature_cols = [
        col for col in feature_cols
        if not any(col.startswith(p + "_") for p in EDGE_PREFIXES)
    ]

    global_cols = [c for c in feature_cols if c.startswith("events")]
    feature_cols = [c for c in feature_cols if c not in global_cols]

    feature_groups = {}
    
    for col in feature_cols:
        match = re.match(r"([^_]+)_", col)
        prefix = match.group(1) if match else "global"
        feature_groups.setdefault(prefix, []).append(col)

    # print("\nFeature groups:")
    # for k, v in feature_groups.items():
    #     print(f"  {k}: {v}")

    row_num=0
    data_list = []
    for _, row in df.iterrows():

        data = HeteroData()
        # --------------------------------------------------------------
        # Create node types with their own feature vectors
        # One node per feature group
        # --------------------------------------------------------------
        for node_type, cols in feature_groups.items():
            vals = row[cols].values.astype(float)
            if np.all(vals == -999):
                continue
            vals[vals == -999] = 0.0
            # shape => (1, num_features_for_this_group)
            data[node_type].x = torch.tensor([vals], dtype=torch.float)

        # --------------------------------------------------------------
        # Build fully connected edges between all node types
        # With exactly 1 node in each type, the index is always [[0],[0]]
        # --------------------------------------------------------------
        node_types = list(feature_groups.keys())

        EDGE_QUANTITIES = auto_generate_mappings(df, node_types)

        EDGE_TYPES = ["mass", "dR", "dphi","deta"]

        for src in node_types:
            if src not in data.node_types:
                continue
            for dst in node_types:
                if dst not in data.node_types:
                    continue
                for etype in EDGE_TYPES:

                    # get correct event-based column
                    col = EDGE_QUANTITIES.get((src, etype, dst))
                    if col is None:
                        continue  # skip if not available

                    # 1-edge graph (0 → 0)
                    edge_index = torch.tensor([[0], [0]], dtype=torch.long)

                    # lookup precomputed scalar
                    val = float(row[col])
                    edge_attr = torch.tensor([[val]], dtype=torch.float)

                    data[src, etype, dst].edge_index = edge_index
                    data[src, etype, dst].edge_attr  = edge_attr
        if global_cols:
            global_vals = row[global_cols].values.astype(float)
            data.u = torch.tensor([global_vals], dtype=torch.float)

        # --------------------------------------------------------------
        # Attach labels and metadata
        # --------------------------------------------------------------
        data.y = torch.tensor([int(row["label"])], dtype=torch.long)
        data.weight = torch.tensor([float(row["weight"])], dtype=torch.float)
        data.process = row["process"]

        data.node_types = node_types 

        data = T.ToUndirected()(data)
        data = T.AddSelfLoops()(data)
        # data = T.AddRandomWalkPE(walk_length=20, attr_name='pe')(data)
        data_list.append(data)
        # if row_num > 3:
        #     continue
        # row_num+=1
        
    
    return data_list

def visualize_graph(graph, index):
    """
    Visualize a HeteroData graph event-by-event including:
      - Node types
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

class RedrawProjection:
    def __init__(self, model: torch.nn.Module,
                redraw_interval: Optional[int] = None):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            fast_attentions = [
                module for module in self.model.modules()
                if isinstance(module, PerformerAttention)
            ]
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix()
            self.num_last_redraw = 0
            return
        self.num_last_redraw += 1


class HeteroGraph(torch.nn.Module):
    def __init__(self, channels,node_types, edge_types, num_layers,num_node_categories, num_edge_categories, u_dim, attn_type='multihead'):
        super().__init__()

        self.channels = channels 
        input_dim = channels * len(node_types) + channels


        node_feature_dims = {
            "jet1": 3,
            "jet2": 3,
            "jet3": 3,
            "jet4": 3,
            "jet5": 3,
            "jet6": 3,
            "lepton1": 3,
            "PuppiMET": 2,
            "w": 2,
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

        # per-edge-type embeddings

        # print("DEBUG: edge_types entries and their types:")
        # for et in edge_types:
        #     print("  ", et, " type:", type(et))

        # self.edge_emb = torch.nn.ModuleDict({
        #     "__".join(et): Linear(1, channels)#edge_feature_dims[et[1]]
        #     for et in edge_types
        # })

        self.convs = torch.nn.ModuleList()
        for layer_idx in range(num_layers):
            if layer_idx == 0:
                conv = HeteroConv({
                    (src, rel, dst): TransformerConv(
                        in_channels=channels,  # match node_emb output
                        out_channels=channels,
                        heads=3,
                        dropout=0.1,
                        edge_dim=1,
                    )
                    for (src, rel, dst) in edge_types
                }, aggr='sum')
            else:
                conv = HeteroConv({
                    (src, rel, dst): TransformerConv(
                        in_channels=channels*3,  # previous out_channels * heads
                        out_channels=channels,
                        heads=3,
                        dropout=0.1,
                        edge_dim=1,
                    )
                    for (src, rel, dst) in edge_types
                }, aggr='sum')

            self.u_mlp = Sequential( Linear(u_dim, channels), ReLU(),)

            self.convs.append(conv)
        
        self.mlp = None
        self.node_types = node_types

        self.redraw_projection = RedrawProjection( self.convs,redraw_interval=1000 if attn_type == 'performer' else None        )
    
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
        edge_attr_dict={}
        for etype in data.edge_types:    # etype = (src, rel, dst)
            if 'edge_attr' not in data[etype]:
                continue
            key = "__".join(etype)
            edge_attr_dict[etype] = data[etype].edge_attr  #self.edge_emb[key](data[etype].edge_attr)
        # layers
        for conv in self.convs:
            h = conv(h, edge_index_dict=edge_index_dict, edge_attr_dict=edge_attr_dict)
        num_graphs = data.num_graphs 
        pooled = []

        for nt in self.node_types:  
            if nt in h:
                h_nt = h[nt]                      # [num_nodes, C]
                batch = data[nt].batch            # [num_nodes]

                # 1) pool over nodes - [num_graphs, C]
                pooled_nt = global_add_pool(
                    h_nt, batch, size=num_graphs
                )

                # 2) reduce over features - [num_graphs, 1]
                pooled_nt = pooled_nt.sum(dim=1, keepdim=True)

            else:
                # Node type missing entirely - zero scalar per graph
                pooled_nt = torch.zeros(
                    num_graphs, 1, device=out_device
                )

            pooled.append(pooled_nt)



        # Now safe to concatenate
        # for i, t in enumerate(pooled):
        #     print(i, t.shape)

        pooled_nodes = torch.cat(pooled, dim=-1)


        u = data['u']          # shape [batch, u_dim]
        u_emb = self.u_mlp(u)  # map to channel size

        out = torch.cat([pooled_nodes, u_emb], dim=-1)

        if not hasattr(self, "mlp") or self.mlp is None:
            C = self.channels
            input_dim = out.size(1) 
            self.mlp = Sequential(
                Linear(input_dim, C // 2),
                ReLU(),
                Linear(C // 2, C // 4),
                ReLU(),
                Linear(C // 4, 1)
            ).to(out.device)


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
    num_layers = 2
    attn_type = 'multihead'
    attn_kwargs = {'dropout': 0.5}

    model = HeteroGraph(
        node_types=node_types,
        edge_types=edge_types,
        channels=channels,
        num_layers=num_layers,
        num_node_categories=num_node_categories,
        num_edge_categories=num_edge_categories,
        u_dim=u_dim).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, min_lr=0.00001)
    return model, optimizer, scheduler, device

def plot_training_curves(train_losses, val_errors):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(7,5))
    plt.plot(epochs, train_losses, label="Training")
    plt.plot(epochs, val_errors, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("transform_training_curve.png")

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


def plot_roc_curve(labels, probs, weights=None,name='validation'):
    if weights is not None:
        fpr, tpr, _ = roc_curve(
            labels.numpy(),
            probs.numpy(),
            sample_weight=weights.numpy()
        )
    else:
        fpr, tpr, _ = roc_curve(
            labels.numpy(),
            probs.numpy()
        )

    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1], [0,1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve ({name})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"transform_roc_curve_{name}.png")


def plot_score_distributions(labels, probs, weights=None):
    plt.figure(figsize=(7,5))

    sig = labels == 1
    bkg = labels == 0

    if weights is None:
        plt.hist(probs[bkg], bins=50, alpha=0.6, label="Background")
        plt.hist(probs[sig], bins=50, alpha=0.6, label="Signal")
    else:
        plt.hist(probs[bkg], bins=50, weights=weights[bkg],
                 alpha=0.6, label="Background")
        plt.hist(probs[sig], bins=50, weights=weights[sig],
                 alpha=0.6, label="Signal")

    plt.xlabel("Model output (sigmoid)")
    plt.ylabel("Events")
    plt.title("Classifier Output Distribution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("transform_score_distrib.png")


def collect_graph_metadata(dataset):
    node_types = set()
    edge_types = set()
    num_node_categories = {}
    num_edge_categories = {}
    u_dim = None

    for g in dataset:
        # Node types
        for nt in g.node_types:
            node_types.add(nt)
            # Track max feature size per node type
            x_size = g[nt].x.size(1)
            if nt not in num_node_categories or x_size > num_node_categories[nt]:
                num_node_categories[nt] = x_size
        # Edge types
        for et in g.edge_types:
            edge_types.add(et)
            if hasattr(g[et], 'edge_attr'):
                e_size = g[et].edge_attr.size(1)
                if et not in num_edge_categories or e_size > num_edge_categories[et]:
                    num_edge_categories[et] = e_size
            else:
                if et not in num_edge_categories:
                    num_edge_categories[et] = 0

        # Global feature 'u'
        if u_dim is None:
            u_dim = g['u'].size(1)

    return list(node_types), list(edge_types), num_node_categories, num_edge_categories, u_dim



def balance_signal_background_weights_graphs(data_list,    balance_to="background"):
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


def main():

    parser = argparse.ArgumentParser(description="Convert specific category from .coffea to pandas DataFrame")
    parser.add_argument("coffea_file", help="Path to input .coffea file")
    parser.add_argument("category", help="Event category key (e.g. whad_withbveto_e)")
    parser.add_argument("--out", help="Optional output parquet/csv file", default=None)
    parser.add_argument(    '--attn_type', default='multihead',    help="Global attention type such as 'multihead' or 'performer'.")

    args = parser.parse_args()

    df, norm = load_category_from_coffea(args.coffea_file, args.category)
    data_list=get_graphical(df, norm)
    data_list = balance_signal_background_weights_graphs( data_list,    balance_to="background")
  
    from torch_geometric.loader import DataLoader
    train_val_list, test_list = train_test_split(data_list, test_size=0.1, random_state=42)
    train_list, val_list = train_test_split(train_val_list, test_size=0.5, random_state=42 )
    # 0.25 x 0.8 = 0.2, so final split is 60% train / 20% val / 20% test
    # current: 0.5 * 0.9 = 0.45, so split is 45, 45, 10 (need stats for debugging)

    train_loader = DataLoader(train_list, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_list, batch_size=64, shuffle=False)
    test_loader  = DataLoader(test_list, batch_size=64, shuffle=False)

    print(f"Train size: {len(train_list)}, Val size: {len(val_list)}, Test size: {len(test_list)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model, optimizer, scheduler, device = neural_net_initialization(data_list)
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

    @torch.enable_grad()
    def train(model, optimizer, train_loader, device):
        model.train()

        total_loss = 0.0
        total_weight = 0.0

        all_logits = []
        all_labels = []
        all_weights = []

        for data in train_loader:
            data = data.to(device)

            optimizer.zero_grad()
            model.redraw_projection.redraw_projections()

            out = model(data).squeeze()      # logits
            y   = data.y.float()

            loss_per_event = criterion(out, y)

            if hasattr(data, 'weight'):
                w = data.weight.float()

                loss = (loss_per_event * w).sum() / w.sum()

                total_loss   += (loss_per_event * w).sum().item()
                total_weight += w.sum().item()

                all_weights.append(w.detach().cpu())
            else:
                loss = loss_per_event.mean()

                total_loss   += loss_per_event.sum().item()
                total_weight += loss_per_event.numel()

                all_weights.append(torch.ones_like(y).cpu())

            loss.backward()
            optimizer.step()

            all_logits.append(out.detach().cpu())
            all_labels.append(y.detach().cpu())

        all_logits  = torch.cat(all_logits)
        all_labels  = torch.cat(all_labels)
        all_weights = torch.cat(all_weights)

        avg_loss = total_loss / total_weight

        return avg_loss, all_logits, all_labels, all_weights





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

            loss_per_event = criterion(out, y)

            if hasattr(data, 'weight'):
                w = data.weight.float()

                total_loss   += (loss_per_event * w).sum().item()
                total_weight += w.sum().item()

                all_weights.append(w.detach().cpu())
            else:
                total_loss   += loss_per_event.sum().item()
                total_weight += loss_per_event.numel()

                all_weights.append(torch.ones_like(y).cpu())

            all_logits.append(out.detach().cpu())
            all_labels.append(y.detach().cpu())

        all_logits  = torch.cat(all_logits)
        all_labels  = torch.cat(all_labels)
        all_weights = torch.cat(all_weights)

        avg_loss = total_loss / total_weight

        return avg_loss, all_logits, all_labels, all_weights



    num_epochs = 50
    train_losses = []
    val_errors   = []
    val_aucs     = []
    train_aucs   = []

    patience = 5  # how many epochs to wait for improvement
    best_val_error = float('inf')
    epochs_no_improve = 0
    best_model_path = "best_model.pt"  # path to save the best model

    for epoch in range(1, num_epochs + 1):
        # -------- TRAIN --------
        train_loss, train_logits, train_labels, train_weights = train(
            model, optimizer, train_loader, device
        )
        
        print("Logits range: ", train_logits.min().item(), train_logits.max().item())
        train_probs = torch.sigmoid(train_logits)
        print("probs range: ", train_probs.min().item(), train_probs.max().item())
        train_auc = roc_auc_score(
            train_labels.numpy(),
            train_probs.numpy(),
            sample_weight=train_weights.numpy()
        )

        # -------- VALIDATION --------
        val_error, val_logits, val_labels, val_weights = test(
            model, val_loader, device
        )

        val_probs = torch.sigmoid(val_logits)
        val_auc = roc_auc_score(
            val_labels.numpy(),
            val_probs.numpy(),
            sample_weight=val_weights.numpy()
        )

        # -------- SCHEDULER --------
        scheduler.step(val_error)

        # -------- LOGGING --------
        train_losses.append(train_loss)
        val_errors.append(val_error)
        train_aucs.append(train_auc)
        val_aucs.append(val_auc)

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train AUC: {train_auc:.4f} | "
            f"Val Error: {val_error:.4f} | "
            f"Val AUC: {val_auc:.4f}"
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
    val_probs = torch.sigmoid(val_logits)

    # Plots
    plot_training_curves(train_losses, val_errors)
    plot_auc_curves(train_aucs, val_aucs)
    plot_roc_curve(val_labels, val_probs, val_weights, "validation")
    plot_score_distributions(val_labels, val_probs, val_weights)

    
    # ===== FINAL TEST EVALUATION =====
    test_error, test_logits, test_labels, test_weights = test(
        model, test_loader, device
    )

    test_probs = torch.sigmoid(test_logits)

    test_auc = roc_auc_score(
        test_labels.numpy(),
        test_probs.numpy(),
        sample_weight=test_weights.numpy()
    )
    plot_roc_curve(test_labels, test_probs, test_weights,"Evaluation")
    print("\n===== FINAL TEST RESULTS =====")
    print(f"Test Error: {test_error:.4f}")
    print(f"Test AUC:   {test_auc:.4f}")



if __name__ == "__main__":
    main()


