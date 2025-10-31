#!/usr/bin/env python3
import xgboost as xgb
import pandas as pd
import coffea.util
import awkward as ak
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
print(xgb.__version__)
def load_category_from_coffea(coffea_file, category_name):
    #Load and combine all processes for a given category from a .coffea file
    merged_file = coffea.util.load(coffea_file)
    columns = merged_file["columns"]
    all_dfs = []

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

            df = pd.DataFrame(data_dict)
            df["process"] = process_name
            df["year_tag"] = subkey
            df["category"] = category_name
            all_dfs.append(df)

    if not all_dfs:
        raise ValueError(f"No data found for category '{category_name}' in {coffea_file}")

    df_all = pd.concat(all_dfs, ignore_index=True)
    return df_all

def setup_dmatrix(input_df, label_column="process", weight_column="weight",test_size=0.1, val_size=0.1, random_state=43):
    if isinstance(input_df, str):
        if input_df.endswith(".csv"):
            df = pd.read_csv(input_df)
        elif input_df.endswith(".parquet"):
            df = pd.read_parquet(input_df)
        else:
            raise ValueError("Input must be a CSV or Parquet file.")
    else:
        df = input_df.copy()

    df["label"] = df[label_column].apply(lambda x: 1 if "EWK" in str(x) else 0)

    exclude_cols = [label_column, "year_tag", "category", weight_column, "label"]
    features = df.drop(columns=[c for c in exclude_cols if c in df.columns])

    # 2. Labels
    labels = df["label"].to_numpy()
    # 3. Event weights
    if weight_column in df.columns:
        weights = df[weight_column].to_numpy()
    else:
        weights = None

    X_trainval, X_test, y_trainval, y_test, w_trainval, w_test, idx_trainval, idx_test = train_test_split(
        features, labels, weights, df.index,
        test_size=test_size, random_state=random_state, stratify=labels
    )

    # --- Split Train vs Val ---
    val_fraction = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val, w_train, w_val, idx_train, idx_val = train_test_split(
        X_trainval, y_trainval, w_trainval, idx_trainval,
        test_size=val_fraction, random_state=random_state, stratify=y_trainval
    )

    # Create DMatrix objects
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
    dval   = xgb.DMatrix(X_val,   label=y_val,   weight=w_val)
    dtest  = xgb.DMatrix(X_test,  label=y_test,  weight=w_test)
    
    print(f"Train: {X_train.shape[0]} events, Val: {X_val.shape[0]} events, Test: {X_test.shape[0]} events")
    print(f"Number of features: {features.shape[1]}")
    
    if weights is not None:
        print(f"Using per-event weights from column '{weight_column}'")
    
    # np.save(f"{outdir}/idx_train.npy", idx_train)
    # np.save(f"{outdir}/idx_val.npy", idx_val)
    # np.save(f"{outdir}/idx_test.npy", idx_test)

    return dtrain, dval, dtest, idx_train, idx_val, idx_test

def training_bdt(dtrain, dval, num_round = 10, channel='whad_withbveto_mu'):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 3,
        "eta": 0.1,
        "verbosity": 1
    }
    evals_result = {}
    # num_round = 10
    evallist = [(dtrain, 'train'), (dval, 'eval')]
    bdt = xgb.train(params, dtrain, num_boost_round=num_round, evals=evallist, evals_result=evals_result, maximize=False, early_stopping_rounds=10)
    
    bdt_best = bdt[: bdt.best_iteration + 1]

    # Save only the *best* model
    model_path = f"bdt/bdt_{channel}_best_iter_{bdt.best_iteration}.json"
    bdt_best.save_model(model_path)
    print(f"Saved best model to {model_path}")

    return bdt, evals_result


def plot_training_curves(evals_result, outpath="bdt/training_curves.png"):
    epochs = len(evals_result["train"]["logloss"])
    x_axis = range(0, epochs)

    plt.figure(figsize=(8, 6))
    plt.plot(x_axis, evals_result["train"]["logloss"], label="Train")
    plt.plot(x_axis, evals_result["eval"]["logloss"], label="Validation")
    plt.xlabel("iteration")
    plt.ylabel("Log Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"Saved training curve to {outpath}")

def test_bdt(bdt, dtest, channel, df_test):
    """
    Evaluate trained BDT on a test set using the best iteration found during training.
    """
    # Use only trees up to the best iteration
    ypred = bdt.predict(dtest, iteration_range=(0, bdt.best_iteration + 1))

    ytrue = dtest.get_label()

    auc = roc_auc_score(ytrue, ypred)
    acc = accuracy_score(ytrue, ypred > 0.5)

    print(f" Test AUC (best iter {bdt.best_iteration}): {auc:.4f}")
    print(f" Test Accuracy: {acc:.4f}")

    fpr, tpr, _ = roc_curve(ytrue, ypred)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"bdt/roc_curve_{channel}.png", dpi=300)
    plt.close()
    print("Saved ROC curve as 'roc_curve.png'")


    df_plot = df_test.copy()
    df_plot["bdt_score"] = ypred

    processes = sorted(df_plot["process"].unique())
    bins = np.linspace(0, 1, 40)

    plt.figure(figsize=(8, 6))
    for proc in processes:
        sub = df_plot[df_plot["process"] == proc]
        plt.hist(
            sub["bdt_score"],
            bins=bins,
            weights=sub["weight"],
            histtype="step",
            linewidth=1.8,
            label=proc,
            #density=True,
        )

    plt.xlabel("BDT Discriminator Output")
    plt.ylabel("Events")
    plt.title(f"BDT Discriminator per Process ({channel})")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    outpath = f"bdt/bdt_discriminator_{channel}.png"
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"Saved discriminator plot: {outpath}")

    return auc, acc
def plot_feature_importance(bdt_best, outpath="bdt/feature_importance.png"):
    ax = xgb.plot_importance(bdt_best, max_num_features=100, importance_type="gain")
    fig = ax.figure
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved feature importance plot to {outpath}")


def print_signal_background_info(dmatrix, name="dataset"):
    """Print number of signal and background events before BDT training."""
    y = dmatrix.get_label()
    w = dmatrix.get_weight()
    if w is None or len(w) == 0:
        w = [1.0] * len(y)

    n_signal = sum(w[i] for i, label in enumerate(y) if label == 1)
    n_bkg = sum(w[i] for i, label in enumerate(y) if label == 0)

    print(f"📊 {name}:")
    print(f"   Total events: {len(y)}")
    print(f"   Weighted signal events: {n_signal:.2f}")
    print(f"   Weighted background events: {n_bkg:.2f}")
    print(f"   Signal fraction: {n_signal / (n_signal + n_bkg):.3f}")


def balance_signal_background_weights(dmatrix, balance_to="background"):

    # Rescale signal or background weights so that
    # sum(signal_weights) == sum(background_weights).


    y = dmatrix.get_label()
    w = dmatrix.get_weight()
    if w is None or len(w) == 0:
        w = np.ones_like(y)

    sig_mask = y == 1
    bkg_mask = y == 0

    sum_sig = np.sum(w[sig_mask])
    sum_bkg = np.sum(w[bkg_mask])

    if sum_sig == 0 or sum_bkg == 0:
        raise ValueError("Error: One of the classes has zero total weight!")

    if balance_to == "background":
        scale = sum_bkg / sum_sig
        w[sig_mask] *= scale
        print(f"⚖️  Scaled signal weights by {scale:.3f} to match background total.")
    elif balance_to == "signal":
        scale = sum_sig / sum_bkg
        w[bkg_mask] *= scale
        print(f"⚖️  Scaled background weights by {scale:.3f} to match signal total.")
    else:
        raise ValueError("balance_to must be either 'signal' or 'background'.")

    dmatrix.set_weight(w)

    print(f"After reweighting: sum(signal)={np.sum(w[sig_mask]):.3f}, sum(background)={np.sum(w[bkg_mask]):.3f}")
    return dmatrix

def main():
    parser = argparse.ArgumentParser(description="Convert specific category from .coffea to pandas DataFrame")
    parser.add_argument("coffea_file", help="Path to input .coffea file")
    parser.add_argument("category", help="Event category key (e.g. whad_withbveto_e)")
    parser.add_argument("--out", help="Optional output parquet/csv file", default=None)

    args = parser.parse_args()

    df = load_category_from_coffea(args.coffea_file, args.category)
    
    print(f"Combined DataFrame shape: {df.shape}")
    print(df.head())

    if args.out:
        if args.out.endswith(".parquet"):
            df.to_parquet(args.out, index=False)
        elif args.out.endswith(".csv"):
            df.to_csv(args.out, index=False)
        else:
            raise ValueError("Output file must end with .csv or .parquet")
        print(f"Saved DataFrame to {args.out}")


    dtrain,dval,dtest, idx_train, idx_val, idx_test = setup_dmatrix(args.out)
    df_test = df.loc[idx_test]
    print_signal_background_info(dtrain, "Training set")
    print_signal_background_info(dval, "Validation set")
    print_signal_background_info(dtest, "Test set")
    dtrain_pure= balance_signal_background_weights(dtrain)
    bdt, evals_result = training_bdt(dtrain_pure, dval, num_round=100, channel=args.category)
    plot_training_curves(evals_result, outpath=f"bdt/training_curves_{args.category}.png")
    test_bdt(bdt, dtest, args.category,df_test)
    plot_feature_importance(bdt,outpath=f"bdt/feature_importance_{args.category}.png")

if __name__ == "__main__":
    main()
