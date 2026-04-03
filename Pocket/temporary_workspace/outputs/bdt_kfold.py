#!/usr/bin/env python3
import xgboost as xgb
import pandas as pd
import coffea.util
import awkward as ak
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
import mplhep as hep
import os
print(xgb.__version__)
hep.style.use("CMS")
from matplotlib.ticker import StrMethodFormatter

process_groups = {
    "VBS_EWK": ["WminusTo2JZTo2LJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WminusToLNuWminusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WminusToLNuZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WplusTo2JWminusToLNuJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WplusTo2JZTo2LJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WplusToLNuWminusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WplusToLNuWplusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WplusToLNuZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "ZTo2LZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8"],

    "Top/ttbar": ["ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8",
            "ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8",
            "ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8",
            "ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8",
            "ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8",
            "ttWJets_TuneCP5_13TeV_madgraphMLM_pythia8",
            "ttZJets_TuneCP5_13TeV_madgraphMLM_pythia8",
            "TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8",
            "TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8"],

    "W+jets": ["WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8",
            "WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8",
            "WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8",
            "WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8",
            "WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8",
            "WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8",
            "WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8",
            "WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8",
            "WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8"],

    "DY": ["DYJetsToLL_M-10to50_TuneCP5_13TeV-amcatnloFXFX-pythia8",
           "DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8"],

    "QCD-VV": ["WminusTo2JZTo2LJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WminusToLNuWminusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WminusToLNuZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WplusTo2JZTo2LJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WplusTo2JWminusToLNuJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WplusToLNuWminusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WplusToLNuWplusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WplusToLNuZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "ZTo2LZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8"]
}

colors = {
    "VBS_EWK":   "#bd1f01",
    "Top/ttbar": "#832db6",
    "W+jets":    "#3f90da",
    "DY":        "#ffa90e",
    "QCD-VV":    "#b9ac70",
    "Other":     "#a96b59"
}


def average_predict(models, dmatrix):
    preds = np.zeros(dmatrix.num_row())
    for m in models:
        preds += m.predict(dmatrix)
    return preds / len(models)


def load_category_from_coffea(coffea_file, category_name):
    """Load and combine all processes for a given category from a .coffea file."""
    merged_file = coffea.util.load(coffea_file)
    columns = merged_file["columns"]
    gen_weight_normalization = merged_file["sum_genweights"]
    all_dfs = []
    table_absweight_to_weight = {}

    for process_name, process_dict in columns.items():
        for subkey, year_dict in process_dict.items():
            if not isinstance(year_dict, dict):
                continue
            if category_name not in year_dict:
                continue
            category_dict = year_dict[category_name]['nominal']
            print(f"Processing {process_name} ({subkey}) for category {category_name}")

            data_dict = {}
            for var, arr in category_dict.items():
                val = getattr(arr, "value", arr)
                try:
                    arr_np = np.asarray(val)
                    if arr_np.ndim == 2 and arr_np.shape[1] == 1:
                        arr_np = arr_np[:, 0]
                    if arr_np.size == 0:
                        print(f"Skipping empty array for {var}")
                        continue
                    if "jet5" in var or "jet6" in var:
                        continue
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

            table_absweight_to_weight[subkey] = (
                sum(df['weight']) / sum(abs(df['weight']))
                if len(df['weight']) > 0 else 0
            )
            print(
                "events", len(data_dict['weight']),
                ", sum_genweights", gen_weight_normalization[subkey],
                ", absweight_to_weight", table_absweight_to_weight[subkey]
            )

    if not all_dfs:
        raise ValueError(f"No data found for category '{category_name}' in {coffea_file}")

    df_all = pd.concat(all_dfs, ignore_index=True)
    return df_all, gen_weight_normalization, table_absweight_to_weight


def setup_dmatrix(input_df, norm_table, table_absweight_to_weight,
                  label_column="process", weight_column="weight",
                  test_size=0.2, random_state=43):
    if isinstance(input_df, str):
        if input_df.endswith(".csv"):
            df = pd.read_csv(input_df)
        elif input_df.endswith(".parquet"):
            df = pd.read_parquet(input_df)
        else:
            raise ValueError("Input must be a CSV or Parquet file.")
    else:
        df = input_df.copy()

    signal_group = "VBS_EWK"
    df["label"] = df[label_column].apply(
        lambda x: 1 if str(x) in process_groups[signal_group] else 0
    )

    labels = df["label"].to_numpy()

    if weight_column in df.columns:
        print(df["year_tag"], " year and scaled correction ", norm_table)
        df[weight_column] = df.apply(
            lambda row: abs(row[weight_column])
                        * table_absweight_to_weight.get(row["year_tag"], 1.0)
                        / norm_table.get(row["year_tag"], 1.0),
            axis=1
        )
        weights = df[weight_column].to_numpy()
    else:
        weights = None

    exclude_cols = (
        ["label", "year_tag", "category", "weight", "events_genWeight", "process",
         "w_had_jets_mass", "w_had_jets_N"]#, "events_nCentralJetsGood", "events_nJetGood"]
        + [c for c in df.columns if c.startswith("weight_variation")]
        + [c for c in df.columns if c.startswith("vbsjets")]
        # + [c for c in df.columns if "jet5" in c]
        # + [c for c in df.columns if "jet6" in c]
    )
    features = df.drop(columns=[c for c in exclude_cols if c in df.columns])

    scale_to_full_run2 = 1 / test_size  # no lumi rescaling by default; adjust as needed

    if test_size == 1:
        X_test   = features
        y_test   = labels
        w_test   = weights
        idx_test = df.index
        X_train = y_train = w_train = idx_train = dtrain = None
        dtest = xgb.DMatrix(X_test, label=y_test,
                            weight=scale_to_full_run2 * abs(w_test))
    else:
        X_train, X_test, y_train, y_test, w_train, w_test, idx_train, idx_test = \
            train_test_split(features, labels, weights, df.index,
                             test_size=test_size, random_state=random_state,
                             stratify=labels)

        print("start loading training matrix")
        dtrain = xgb.DMatrix(X_train, label=y_train,
                             weight=scale_to_full_run2 * abs(w_train))
        print("test weights: ", w_test)
        dtest  = xgb.DMatrix(X_test,  label=y_test,
                             weight=scale_to_full_run2 * abs(w_test))
        print(f"Train: {X_train.shape[0]} events, Test: {X_test.shape[0]} events")
        print(f"Number of features: {features.shape[1]}")

    if weights is not None:
        print(f"Using per-event weights from column '{weight_column}'")

    sig_mask = (df["label"] == 1)
    bkg_mask = (df["label"] == 0)
    sum_sig  = df.loc[sig_mask, "weight"].sum()
    sum_bkg  = df.loc[bkg_mask, "weight"].sum()
    print("sum_sig", sum_sig)
    print("sum_bkg", sum_bkg)
    if sum_sig == 0 or sum_bkg == 0:
        raise ValueError("Error: One of the classes has zero total weight!")

    scale = sum_bkg / sum_sig
    print(f"Scaled DF weights by {scale:.3f} to match background total for unitary.")
    df_unit = df.copy()
    df_unit.loc[sig_mask, "weight"] *= scale

    return dtrain, dtest, idx_train, idx_test, df, df_unit


def training_bdt(dtrain, dval, features, num_round=1000,
                 channel='whad_withbveto_mu', outdir='bdt', fold=0):
    params = {
        "objective":   "binary:logistic",
        "eval_metric": "logloss",
        "max_depth":   3,
        "eta":         0.1,
        "verbosity":   1
    }
    evals_result = {}
    evallist = [(dtrain, 'train'), (dval, 'eval')]
    bdt = xgb.train(params, dtrain, num_boost_round=num_round,
                    evals=evallist, evals_result=evals_result,
                    maximize=False, early_stopping_rounds=10)
    bdt.feature_names = features
    bdt_best = bdt[: bdt.best_iteration + 1]

    model_path = f"{outdir}/bdt_{channel}_{fold}_iter_{bdt.best_iteration}.json"
    bdt_best.save_model(model_path)
    print(f"Saved best model to {model_path}")

    return bdt, evals_result


def plot_training_curves(evals_result, outpath="bdt/training_curves.png"):
    epochs = len(evals_result["train"]["logloss"])
    x_axis = range(0, epochs)

    plt.figure(figsize=(8, 6))
    plt.plot(x_axis, evals_result["train"]["logloss"], label="Train")
    plt.plot(x_axis, evals_result["eval"]["logloss"],  label="Validation")
    plt.xlabel("iteration")
    plt.ylabel("Log Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"Saved training curve to {outpath}")


def test_bdt(models, dtest, dtrain, channel, year, df_test, df_train,
             tag="", unitary=True, outdir="bdt", test_size=0.2):
    """Evaluate ensemble of BDT models on test (and optionally train) sets."""

    ypred = average_predict(models, dtest)
    ytrue = dtest.get_label()
    auc   = roc_auc_score(ytrue, ypred)
    acc   = accuracy_score(ytrue, ypred > 0.5)
    fpr, tpr, _ = roc_curve(ytrue, ypred)

    if dtrain is not None:
        ypred_training  = average_predict(models, dtrain)
        ytrue_training  = dtrain.get_label()
        auc_t = roc_auc_score(ytrue_training, ypred_training)
        acc_t = accuracy_score(ytrue_training, ypred_training > 0.5)
        fpr_t, tpr_t, _ = roc_curve(ytrue_training, ypred_training)
        df_plot_t = df_train.copy()
        df_plot_t["bdt_score"] = ypred_training

    print(f" Test AUC: {auc:.4f}")
    print(f" Test Accuracy: {acc:.4f}")

    # --- Global ROC ---
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"Test AUC = {auc:.3f}")
    if dtrain is not None:
        plt.plot(fpr_t, tpr_t, linestyle='--', label=f"Train AUC = {auc_t:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Train vs Test ROC - BDT" if dtrain is not None else "Test ROC - BDT")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{outdir}/roc_curve_{channel}_{year}_{tag}.png", dpi=300)
    plt.close()
    print("Saved ROC curve")

    # --- Per-background ROC curves ---
    def map_to_group(proc):
        for group_name, members in process_groups.items():
            if proc in members:
                return group_name
        return "Other"

    df_plot = df_test.copy()
    df_plot["bdt_score"] = ypred
    df_plot["group"]     = df_plot["process"].apply(map_to_group)

    df_test_eval = df_test.copy()
    df_test_eval["bdt_score"] = ypred
    df_test_eval["group"]     = df_test_eval["process"].apply(map_to_group)

    if dtrain is not None:
        df_plot_t["group"]       = df_plot_t["process"].apply(map_to_group)
        df_train_eval            = df_train.copy()
        df_train_eval["bdt_score"] = ypred_training
        df_train_eval["group"]   = df_train_eval["process"].apply(map_to_group)

    signal_group = "VBS_EWK"
    background_groups = sorted(
        g for g in df_test_eval["group"].unique()
        if g != signal_group and g != "Other"
    )

    for bkg in background_groups:
        mask_test = df_test_eval["group"].isin([signal_group, bkg])
        ytrue_b   = (df_test_eval.loc[mask_test, "group"] == signal_group).astype(int).values
        ypred_b   = df_test_eval.loc[mask_test, "bdt_score"].values

        if len(np.unique(ytrue_b)) < 2:
            print(f"Skipping TEST {bkg}: only one class present")
            continue

        auc_b = roc_auc_score(ytrue_b, ypred_b)
        acc_b = accuracy_score(ytrue_b, ypred_b > 0.5)
        fpr_b, tpr_b, _ = roc_curve(ytrue_b, ypred_b)

        print(f"\n[{bkg}]")
        print(f" Test AUC : {auc_b:.4f}")
        print(f" Test Accuracy : {acc_b:.4f}")

        plt.figure(figsize=(6, 6))
        plt.plot(fpr_b, tpr_b, label=f"Test AUC = {auc_b:.3f}")

        if dtrain is not None:
            mask_train   = df_train_eval["group"].isin([signal_group, bkg])
            ytrue_bt     = (df_train_eval.loc[mask_train, "group"] == signal_group).astype(int).values
            ypred_bt     = df_train_eval.loc[mask_train, "bdt_score"].values

            if len(np.unique(ytrue_bt)) >= 2:
                auc_bt = roc_auc_score(ytrue_bt, ypred_bt)
                acc_bt = accuracy_score(ytrue_bt, ypred_bt > 0.5)
                fpr_bt, tpr_bt, _ = roc_curve(ytrue_bt, ypred_bt)
                print(f" Train AUC : {auc_bt:.4f}")
                print(f" Train Accuracy : {acc_bt:.4f}")
                plt.plot(fpr_bt, tpr_bt, "--", label=f"Train AUC = {auc_bt:.3f}")

        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"BDT: VBS_EWK vs {bkg}", fontsize=16)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        outname = f"{outdir}/roc_VBS_EWK_vs_{bkg.replace('+','').replace('/','_')}_{channel}_{year}_{tag}.png"
        plt.savefig(outname, dpi=300)
        plt.close()
        print(f"Saved ROC curve: {outname}")

    # --- Discriminator plot ---
    groups       = sorted(df_plot["group"].unique())
    bins         = np.linspace(0, 1, 40)
    bin_centers  = 0.5 * (bins[:-1] + bins[1:])
    hist_data    = {}
    hist_data_t  = {}
    plt.figure(figsize=(8, 6))

    scale_to_full_run2 = 1 / test_size

    if unitary:
        for group in groups:
            sub   = df_plot[df_plot["group"] == group]
            scale = 1 / np.sum(sub['weight'])
            values, edges = np.histogram(sub["bdt_score"], bins=bins,
                                         weights=scale * sub["weight"])
            w2, _ = np.histogram(sub["bdt_score"], bins=bins,
                                 weights=scale * sub["weight"] ** 2)
            hist_data[group] = {"yields": values, "stat_unc": np.sqrt(w2),
                                "edges": edges, "centers": bin_centers,
                                "sumw": np.sum(sub["weight"])}
            color = colors.get(group, "#a96b59")
            if group == "Data":
                plt.scatter(bin_centers, values, label=group, linewidth=1.8,
                            marker='.', color='black')
            else:
                plt.step(bin_centers, values, where='mid', label=group,
                         color=color, linewidth=1.8)
    else:
        stack_values, stack_labels, stack_colors = [], [], []
        signal_value = signal_color = None
        for group in groups:
            sub = df_plot[df_plot["group"] == group]
            values, edges = np.histogram(sub["bdt_score"], bins=bins,
                                         weights=scale_to_full_run2 * sub["weight"])
            w2, _ = np.histogram(sub["bdt_score"], bins=bins,
                                 weights=(scale_to_full_run2 * sub["weight"]) ** 2)
            hist_data[group] = {"yields": values, "stat_unc": np.sqrt(w2),
                                "edges": edges, "centers": bin_centers,
                                "sumw": scale_to_full_run2 * np.sum(sub["weight"])}
            if group == "Data":
                data_vals, data_err = values, hist_data[group]["stat_unc"]
            elif group != signal_group:
                stack_values.append(values)
                stack_labels.append(group)
                stack_colors.append(colors.get(group, "#a96b59"))
            else:
                signal_value = values
                signal_color = colors.get(group, "#bd1f01")

        if stack_values:
            plt.hist([bin_centers] * len(stack_values), bins=bins,
                     weights=stack_values, stacked=True,
                     color=stack_colors, label=stack_labels, edgecolor="None")
        if signal_value is not None:
            plt.hist(bin_centers, bins=bins, weights=signal_value * 50,
                     stacked=False, histtype="step",
                     color=signal_color, label=signal_group + "*50",
                     edgecolor=signal_color)
        if "data_vals" in locals():
            plt.errorbar(bin_centers, data_vals, yerr=data_err, label="Data",
                         linewidth=1.8, marker="o", color="black",
                         linestyle="none", zorder=10)

    # Training overlay (unitary only)
    if unitary and dtrain is not None:
        groups_t = sorted(df_plot_t["group"].unique())
        for group in groups_t:
            sub_t  = df_plot_t[df_plot_t["group"] == group]
            scale  = 1 / np.sum(sub_t['weight'])
            values_t, edges_t = np.histogram(sub_t["bdt_score"], bins=bins,
                                             weights=scale * sub_t["weight"])
            w2_t, _ = np.histogram(sub_t["bdt_score"], bins=bins,
                                   weights=scale * sub_t["weight"] ** 2)
            hist_data_t[group] = {"yields": values_t, "stat_unc": np.sqrt(w2_t),
                                  "edges": edges_t, "centers": bin_centers,
                                  "sumw": np.sum(sub_t["weight"])}
            color = colors.get(group, "#a96b59")
            if group == "Data":
                plt.scatter(bin_centers, values_t, label=group + " (train)",
                            linewidth=1.8, marker='.', color='black')
            else:
                plt.step(bin_centers, values_t, where='mid',
                         label=group + " (train)", color=color,
                         linewidth=1.8, linestyle='--')

    hep.cms.label("Preliminary", data=False, loc=0, fontsize=16, lumi=16.98, com=13)
    plt.xlabel("BDT Discriminator Output", fontsize=14)
    plt.ylabel("Yields", fontsize=14)
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    outpath = f"{outdir}/bdt_discriminator_{channel}_{year}_{tag}.png"
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"Saved discriminator plot: {outpath}")

    return auc, acc, hist_data


def plot_feature_importance(bdt_best, outpath="bdt/feature_importance.png"):
    ax  = xgb.plot_importance(bdt_best, max_num_features=25,
                              importance_type="gain", values_format='{v:.2f}')
    fig = ax.figure
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved feature importance plot to {outpath}")


def print_signal_background_info(dmatrix, name="dataset"):
    y = dmatrix.get_label()
    w = dmatrix.get_weight()
    if w is None or len(w) == 0:
        w = [1.0] * len(y)
    n_signal = sum(w[i] for i, label in enumerate(y) if label == 1)
    n_bkg    = sum(w[i] for i, label in enumerate(y) if label == 0)
    print(f"{name}:")
    print(f"   Total events: {len(y)}")
    print(f"   Weighted signal events: {n_signal:.2f}")
    print(f"   Weighted background events: {n_bkg:.2f}")
    print(f"   Signal fraction: {n_signal / (n_signal + n_bkg):.3f}")


def balance_signal_background_weights(dmatrix, balance_to="background"):
    y = dmatrix.get_label()
    w = dmatrix.get_weight()
    if w is None or len(w) == 0:
        w = np.ones_like(y)

    sig_mask = y == 1
    bkg_mask = y == 0
    sum_sig  = np.sum(w[sig_mask])
    sum_bkg  = np.sum(w[bkg_mask])

    if sum_sig == 0 or sum_bkg == 0:
        raise ValueError("Error: One of the classes has zero total weight!")

    if balance_to == "background":
        scale = sum_bkg / sum_sig
        w[sig_mask] *= scale
        print(f"Scaled signal weights by {scale:.3f} to match background total.")
    elif balance_to == "signal":
        scale = sum_sig / sum_bkg
        w[bkg_mask] *= scale
        print(f"Scaled background weights by {scale:.3f} to match signal total.")
    else:
        raise ValueError("balance_to must be either 'signal' or 'background'.")

    dmatrix.set_weight(w)
    print(f"After reweighting: sum(signal)={np.sum(w[sig_mask]):.3f}, "
          f"sum(background)={np.sum(w[bkg_mask]):.3f}")
    return dmatrix


def calculate_signal_strength(full_dict, n_bins=0):
    background = bkg_unc = signal = sig_unc = 0
    for i in full_dict:
        events = np.sum(full_dict[i]['yields'][-n_bins:])
        stat   = np.sum(np.square(full_dict[i]['stat_unc'][-n_bins:]))
        if "EWK" in i:
            signal  += events
            sig_unc += stat
            print("signal found! is ", i)
        else:
            background += events
            bkg_unc    += stat
    dx  = np.sqrt(sig_unc)
    dy  = np.sqrt(bkg_unc)
    print("signal: ", signal)
    print("background: ", background)
    sig = signal / np.sqrt(background)
    unc = (dx ** 2 / np.sqrt(background) ** 2
           + signal ** 2 * (1 / 2 / np.sqrt(background) ** 3) ** 2 * dy ** 2) ** 0.5
    return sig, unc


def optimize_sig(group_of_samples, channel, year, n=39, outdir="bdt"):
    strength, unc = [], []
    for i in range(n):
        calculate, stat_err = calculate_signal_strength(group_of_samples, i)
        print("S/sqrt(B) = ", calculate)
        strength.append(calculate)
        unc.append(stat_err)

    max_idx = np.nanargmax(strength)

    # Use the first available background group's bin centers as x-axis
    ref_group = next(
        (g for g in group_of_samples if "EWK" not in g),
        list(group_of_samples.keys())[0]
    )
    discrim_score = np.flip(group_of_samples[ref_group]['centers'])
    x_max = discrim_score[max_idx]
    y_max = strength[max_idx]

    plt.figure(figsize=(7, 5))
    plt.errorbar(discrim_score, strength, yerr=unc, color='royalblue',
                 marker='.', linestyle='None', ecolor='lightblue', label='Significance')
    plt.axvline(x_max, color='red', linestyle='--', alpha=0.7)
    plt.axhline(y_max, color='red', linestyle='--', alpha=0.7)
    plt.scatter(x_max, y_max, color='darkred', s=80, zorder=5)
    plt.text(x_max + 0.02, y_max, f"max = ({x_max:.2f}, {y_max:.2f})",
             color='darkred', fontsize=10, verticalalignment='bottom')

    plt.xlim(0, 1)
    try:
        plt.ylim(0, 1.05 * max(strength))
        plt.yticks(np.linspace(0, round(max(strength), 2), 6))
    except Exception:
        plt.ylim(0, 1)
        plt.yticks(np.linspace(0, 1, 6))

    plt.xlabel("BDT discrim score cut", fontsize=12)
    plt.ylabel("Significance", fontsize=12)
    plt.title(f"BDT variable optimization - {channel} {year}", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.xticks(np.linspace(0, 1, 10))
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:.1f}'))
    plt.tight_layout()
    plt.savefig(f"{outdir}/optimize_bdt_{channel}_{year}.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train BDT from .coffea file input with k-fold cross-validation"
    )
    parser.add_argument("coffea_file", help="Path to input .coffea file")
    parser.add_argument("--category",  help="Event category key (e.g. whad_withbveto_e)",
                        default="resolved_mu")
    parser.add_argument("--year",      help="Year label for output filenames", default="2017")
    parser.add_argument("--out",       help="Optional output parquet/csv file", default=None)
    parser.add_argument("--outdir",    help="Output directory", default="bdt")
    parser.add_argument("--nfold",     type=int,   default=5,
                        help="Number of k-fold splits (default: 5)")
    parser.add_argument("--testsize",  type=float, default=0.2,
                        help="Fraction of data held out as test set (default: 0.2)")
    parser.add_argument("--num-round", type=int,   default=1000,
                        help="Max boosting rounds per fold (default: 1000)")
    parser.add_argument('--evaluate-only', action='store_true',
                        help='Skip training; evaluate a saved model ensemble instead')
    parser.add_argument("--model",     help="Path to a saved .json model (evaluate-only mode)",
                        default=None)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Load data from coffea file                                          #
    # ------------------------------------------------------------------ #
    df, norm, factor_absweight_to_weight = load_category_from_coffea(
        args.coffea_file, args.category
    )
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

    # ------------------------------------------------------------------ #
    #  Build DMatrix and train / evaluate                                  #
    # ------------------------------------------------------------------ #
    if args.evaluate_only and args.model is not None:
        dtrain, dtest, _, idx_test, df_fixed, df_unit = setup_dmatrix(
            df, norm, factor_absweight_to_weight,
            test_size=1.0, random_state=43
        )
        model = xgb.XGBClassifier()
        model.load_model(args.model)
        models        = [model.get_booster()]
        dtrain_pure   = None
        df_train_unit = None
        df_train_xs   = None
        test_size     = 1.0
    else:
        dtrain, dtest, idx_train, idx_test, df_fixed, df_unit = setup_dmatrix(
            df, norm, factor_absweight_to_weight,
            test_size=args.testsize, random_state=43
        )
        df_train_xs   = df_fixed.loc[idx_train]
        df_train_unit = df_unit.loc[idx_train]

        print_signal_background_info(dtrain, "Training set")

        # Balance signal/background then extract arrays for k-fold splitting
        dtrain_pure  = balance_signal_background_weights(dtrain)
        X_trainval   = dtrain_pure.get_data()
        y_trainval   = dtrain_pure.get_label()
        w_trainval   = dtrain_pure.get_weight()
        feature_names = dtrain_pure.feature_names

        kf = StratifiedKFold(n_splits=args.nfold, shuffle=True, random_state=43)
        models = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_trainval, y_trainval)):
            print(f"\n===== Fold {fold} =====")

            dtrain_fold = xgb.DMatrix(
                X_trainval[train_idx],
                label=y_trainval[train_idx],
                weight=w_trainval[train_idx]
            )
            dval_fold = xgb.DMatrix(
                X_trainval[val_idx],
                label=y_trainval[val_idx],
                weight=w_trainval[val_idx]
            )

            bdt, evals_result = training_bdt(
                dtrain_fold, dval_fold, feature_names,
                num_round=args.num_round,
                channel=args.category,
                outdir=args.outdir,
                fold=fold
            )
            models.append(bdt)
            plot_training_curves(
                evals_result,
                outpath=f"{args.outdir}/training_curves_{args.category}_fold{fold}.png"
            )
            plot_feature_importance(
                bdt,
                outpath=f"{args.outdir}/feature_importance_{args.category}_fold{fold}.png"
            )

        test_size = args.testsize

    # ------------------------------------------------------------------ #
    #  Evaluate ensemble on test set                                       #
    # ------------------------------------------------------------------ #
    df_test_xs   = df_fixed.loc[idx_test]
    df_test_unit = df_unit.loc[idx_test]

    print_signal_background_info(dtest, "Test set")
    dtest_pure = balance_signal_background_weights(dtest)

    # Unitary (area-normalised) discriminator + ROC
    auc1, acc1, hist_data1 = test_bdt(
        models, dtest_pure, dtrain_pure,
        args.category, args.year,
        df_test_unit, df_train_unit,
        tag="unitary", unitary=True,
        outdir=args.outdir, test_size=test_size
    )

    # Cross-section weighted discriminator + ROC
    auc2, acc2, hist_data2 = test_bdt(
        models, dtest, dtrain,
        args.category, args.year,
        df_test_xs, df_train_xs,
        tag="xsec", unitary=False,
        outdir=args.outdir, test_size=test_size
    )

    optimize_sig(hist_data2, args.category, args.year, outdir=args.outdir)


if __name__ == "__main__":
    main()