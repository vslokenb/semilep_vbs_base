### Optimus Prime 
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
            category_dict = year_dict[category_name]['nominal']
            #print(category_dict)
            print(f"Processing {process_name} ({subkey}) for category {category_name}")

            data_dict = {}

            for var, arr in category_dict.items():
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
    # 3. Event weights
    if "weight" in df_all.columns:
        #print(df["year_tag"], " year and scaled correction ", norm_table)
        df_all["weight"] = df_all.apply( lambda row: row["weight"] / gen_weight_normalization.get(row["year_tag"], 1.0), axis=1 )
        weights = df_all["weight"].to_numpy()
    else:
        weights = None

    return df_all,gen_weight_normalization





def main():

    parser = argparse.ArgumentParser(description="Convert specific category from .coffea to pandas DataFrame")
    parser.add_argument("coffea_file", help="Path to input .coffea file")
    parser.add_argument("category", help="Event category key (e.g. whad_withbveto_e)")
    parser.add_argument("--out", help="Optional input parquet/csv tag", default="default")
    # parser.add_argument( '--path', default='..',    help="set directory where to dump csv file. default input is ..")

    args = parser.parse_args()

    df, norm = load_category_from_coffea(args.coffea_file, args.category)
    df.to_parquet(args.out, index=False)

if __name__ == "__main__":
    main()

