#!/usr/bin/env python3
"""
BDT score / feature Data-MC comparison script.
Inputs: pocket-coffea .coffea files (MC + optional nonprompt) + pre-trained model list.
Genweight normalisation for MC:
    w_corrected = w / sum_genweights   (nominal weights, sign preserved)
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import StrMethodFormatter

import xgboost as xgb
import coffea.util
import mplhep as hep

hep.style.use("CMS")

# ──────────────────────────────────────────────────────────────────────────────
# Default config  (all values can be overridden via CLI)
# ──────────────────────────────────────────────────────────────────────────────
config = {
    "category":              "resolved_mu_WCR",
    "year":                  "2022_postEE",
    "coffea_file":           None,          # MC+Data coffea file  (required)
    "coffea_file_nonprompt": None,          # nonprompt coffea file (optional)
    "output":                "bdt_CR",
    "models": [],
}

colors = {
    "VBS_EWK":  "#bd1f01",
    "TT":       "#832db6",
    "SingleTop":"#F38AA5",
    "WJets":    "#3f90da",
    "DY":       "#ffa90e",
    "QCD-VV":   "#b9ac70",
    "Data":     "black",
    "Other":    "#a96b59",
}

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
            "ZTo2LZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8"],

    "Data": [
        "SingleElectorn_2016_PostVFP_EraH",
        "SingleElectron_2016_PostVFP_EraF",
        "SingleElectron_2016_PostVFP_EraG",
        "SingleMuon_2016_PostVFP_EraH",
        "SingleMuon_2016_PostVFP_EraF",
        "SingleMuon_2016_PostVFP_EraG",
        "SingleMuon",
        "SingleElectron"
    ],
    "nonprompt": [
        "nonprompt"
    ]
}

# flat set for quick look-up
DATA_PROCESSES = set(process_groups["Data"])
print(DATA_PROCESSES)

# ──────────────────────────────────────────────────────────────────────────────
# Coffea loading
# ──────────────────────────────────────────────────────────────────────────────

def load_category_from_coffea(coffea_file, category_name,
                               nonprompt=False, weight_sign=1.0):
    """
    Load all processes for *category_name* from a pocket-coffea .coffea file.

    Genweight normalisation (MC only):
        w_corrected = abs(w) * (sum_w / sum_abs_w) / sum_genweights

    Data events keep their raw weight (typically 1.0).

    Parameters
    ----------
    coffea_file   : str   path to .coffea file
    category_name : str   pocket-coffea category key
    nonprompt     : bool  if True, mark events as "nonprompt" and flip weight sign
    weight_sign   : float extra global sign applied to MC weights (+1 or -1)

    Returns
    -------
    df_all        : pd.DataFrame
    """
    merged_file          = coffea.util.load(coffea_file)
    columns              = merged_file["columns"]
    gen_weight_norm      = merged_file["sum_genweights"]   # dict  subkey -> float
    all_dfs              = []

    for process_name, process_dict in columns.items():
        is_data = process_name in DATA_PROCESSES

        for subkey, year_dict in process_dict.items():
            if not isinstance(year_dict, dict):
                continue
            if category_name not in year_dict:
                continue

            category_dict = year_dict[category_name]["nominal"]
            print(f"  [{coffea_file}] Loading {process_name} ({subkey})"
                  f" for category '{category_name}'")

            data_dict = {}
            for var, arr in category_dict.items():
                val = getattr(arr, "value", arr)
                try:
                    arr_np = np.asarray(val)
                    if arr_np.ndim == 2 and arr_np.shape[1] == 1:
                        arr_np = arr_np[:, 0]
                    if arr_np.size == 0:
                        continue
                    # drop extra jet columns that are not used as features
                    if "jet5" in var or "jet6" in var:
                        continue
                    data_dict[var] = arr_np
                except Exception as e:
                    print(f"    Could not process {var}: {e}")
                    continue

            if not data_dict:
                continue

            df = pd.DataFrame(data_dict)

            # ── Genweight normalisation for MC ────────────────────────────────
            if not is_data:
                sum_genw = gen_weight_norm.get(subkey, None)
                if sum_genw is None or sum_genw == 0:
                    print(f"    WARNING: sum_genweights missing or zero for {subkey},"
                          " skipping normalisation.")
                    sum_genw = 1.0

                df["weight"] = df["weight"] / sum_genw * weight_sign

                print(f"    events={len(df)}, sum_genw={sum_genw:.4g},"
                      f" weight_sign={weight_sign:+.0f}")
            else:
                # Data: keep weight as-is (usually 1.0)
                print(f"    events={len(df)} [DATA]")

            # ── Labels ───────────────────────────────────────────────────────
            if nonprompt:
                df["process"]  = "nonprompt"
                df["year_tag"] = f"nonprompt_{subkey}"
            else:
                df["process"]  = process_name
                df["year_tag"] = subkey if not is_data else "Data"

            df["category"] = category_name
            all_dfs.append(df)

    if not all_dfs:
        raise ValueError(
            f"No data found for category '{category_name}' in {coffea_file}"
        )

    return pd.concat(all_dfs, ignore_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def whist(values, weights, edges):
    """Weighted histogram (area-normalised) + statistical uncertainty."""
    h,  _ = np.histogram(values, bins=edges, weights=weights)
    h2, _ = np.histogram(values, bins=edges, weights=weights ** 2)
    norm   = np.sum(h) * np.diff(edges)
    norm[norm == 0] = 1
    return h / norm, np.sqrt(h2) / norm


def weighted_profile(sub_df, feat, edges):
    """Weighted mean BDT score and its uncertainty in each feature bin."""
    means, errs = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (sub_df[feat] >= lo) & (sub_df[feat] < hi)
        w = sub_df.loc[mask, "weight"].values
        s = sub_df.loc[mask, "bdt_score"].values
        if w.sum() > 0:
            mu  = np.average(s, weights=w)
            var = np.average((s - mu) ** 2, weights=w)
            means.append(mu)
            errs.append(np.sqrt(var / max(np.count_nonzero(w), 1)))
        else:
            means.append(np.nan)
            errs.append(np.nan)
    return np.array(means), np.array(errs)


# ──────────────────────────────────────────────────────────────────────────────
# Main plotting loop
# ──────────────────────────────────────────────────────────────────────────────

def run(cfg):
    os.makedirs(cfg["output"], exist_ok=True)

    # ── Load MC + Data from main coffea file ──────────────────────────────────
    print("\n=== Loading main coffea file ===")
    df_main = load_category_from_coffea(
        cfg["coffea_file"], cfg["category"], nonprompt=False, weight_sign=1.0
    )

    df_list = [df_main]

    # ── Load nonprompt correction (optional, negative weight) ────────────────
    if cfg.get("coffea_file_nonprompt"):
        print("\n=== Loading nonprompt coffea file ===")
        df_np = load_category_from_coffea(
            cfg["coffea_file_nonprompt"], cfg["category"],
            nonprompt=True, weight_sign=-1.0
        )
        df_list.append(df_np)

    df = pd.concat(df_list, ignore_index=True)
    print(f"\nCombined DataFrame: {df.shape[0]} rows, {df.shape[1]} columns")

    # ── Load models ───────────────────────────────────────────────────────────
    boosters = []
    for path in cfg["models"]:
        bst = xgb.Booster()
        bst.load_model(path)
        boosters.append(bst)
    if not boosters:
        raise ValueError("No models provided. Use --models or set config['models'].")

    feature_names = boosters[0].feature_names
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise ValueError(f"Features missing from DataFrame: {missing}")

    # ── Score all events ──────────────────────────────────────────────────────
    X    = df[feature_names]
    dmat = xgb.DMatrix(X, feature_names=feature_names)
    raw_scores     = np.array([bst.predict(dmat) for bst in boosters])
    df["bdt_score"] = raw_scores.mean(axis=0)

    # ── Split Data / MC ───────────────────────────────────────────────────────
    mask_data = df["year_tag"] == "Data"
    df_data   = df[mask_data].copy()
    df_mc     = df[~mask_data].copy()

    print(f"  Data events : {len(df_data)}")
    print(f"  MC   events : {len(df_mc)}")

    # ── Global score histogram edges ──────────────────────────────────────────
    N_BINS        = 30
    SCORE_BINS    = 20
    score_edges_1d   = np.linspace(df["bdt_score"].min(),
                                   df["bdt_score"].max(), SCORE_BINS + 1)
    score_centres_1d = 0.5 * (score_edges_1d[:-1] + score_edges_1d[1:])

    h_data_base, err_data_base = whist(
        df_data["bdt_score"], df_data["weight"], score_edges_1d
    )
    h_mc_base, err_mc_base = whist(
        df_mc["bdt_score"], df_mc["weight"], score_edges_1d
    )

    # ── Per-feature loop ──────────────────────────────────────────────────────
    for feat in feature_names:
        print(f"  Plotting feature: {feat}")

        combined       = pd.concat([df_data[feat], df_mc[feat]]).dropna()
        score_combined = pd.concat([df_data["bdt_score"], df_mc["bdt_score"]])

        feat_edges  = np.linspace(combined.quantile(0.01),
                                  combined.quantile(0.99), N_BINS + 1)
        score_edges = np.linspace(score_combined.min(),
                                  score_combined.max(), N_BINS + 1)

        # ── 2D histograms (score vs feature) ─────────────────────────────────
        def make_hist2d(sub_df):
            h, _, _ = np.histogram2d(
                sub_df["bdt_score"], sub_df[feat],
                bins=[score_edges, feat_edges],
                weights=sub_df["weight"],
            )
            return h

        h_data_2d = make_hist2d(df_data)
        h_mc_2d   = make_hist2d(df_mc)

        def norm_cols(h):
            col_sum = h.sum(axis=0, keepdims=True)
            col_sum[col_sum == 0] = 1
            return h / col_sum

        h_data_n = norm_cols(h_data_2d)
        h_mc_n   = norm_cols(h_mc_2d)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_2d = np.where(h_mc_n > 0, h_data_n / h_mc_n, np.nan)

        # ── 1D Data/MC ratio → per-event reweight ────────────────────────────
        h_data_1d, _ = np.histogram(df_data[feat], bins=feat_edges,
                                    weights=df_data["weight"])
        h_mc_1d,   _ = np.histogram(df_mc[feat],   bins=feat_edges,
                                    weights=df_mc["weight"])

        scale = h_data_1d.sum() / h_mc_1d.sum() if h_mc_1d.sum() > 0 else 1.0
        with np.errstate(divide="ignore", invalid="ignore"):
            rw_factors = np.where(h_mc_1d > 0,
                                  h_data_1d / (h_mc_1d * scale), 1.0)

        mc_feat_bin  = np.clip(np.digitize(df_mc[feat], feat_edges) - 1,
                               0, N_BINS - 1)
        mc_weight_rw = df_mc["weight"].values * rw_factors[mc_feat_bin]

        h_mc_rw, err_mc_rw = whist(
            df_mc["bdt_score"], mc_weight_rw, score_edges_1d
        )

        # ══ FIGURE 1: 2D ratio colormap ══════════════════════════════════════
        finite_vals = ratio_2d[np.isfinite(ratio_2d)]
        if finite_vals.size > 0:
            vmin = np.nanpercentile(finite_vals, 2)
            vmax = np.nanpercentile(finite_vals, 98)
        else:
            vmin, vmax = 0.5, 1.5
        vmin = min(vmin, 2.0 - vmax)
        vmax = max(vmax, 2.0 - vmin)
        norm_2d = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
        cmap    = plt.cm.RdBu_r.copy()
        cmap.set_bad("lightgrey")

        fig1, ax1 = plt.subplots(figsize=(9, 7))
        extent = [feat_edges[0], feat_edges[-1],
                  score_edges[0], score_edges[-1]]
        im = ax1.imshow(ratio_2d, aspect="auto", origin="lower",
                        extent=extent, cmap=cmap, norm=norm_2d)
        plt.colorbar(im, ax=ax1, label="Data / MC")

        feat_centres  = 0.5 * (feat_edges[:-1]  + feat_edges[1:])
        score_centres = 0.5 * (score_edges[:-1] + score_edges[1:])
        for i, sc in enumerate(score_centres):
            for j, fc in enumerate(feat_centres):
                val = ratio_2d[i, j]
                if np.isfinite(val):
                    text_color = "black" if 0.7 < val < 1.3 else "white"
                    ax1.text(fc, sc, f"{val:.2f}", ha="center", va="center",
                             fontsize=7, color=text_color)

        ax1.set_title(f"Data / MC  —  {feat}", fontsize=11)
        ax1.set_xlabel(feat)
        ax1.set_ylabel("BDT score")
        plt.tight_layout()
        fig1.savefig(os.path.join(cfg["output"], f"ratio_2d_{feat}.png"),
                     dpi=150, bbox_inches="tight")
        plt.close(fig1)

        # ══ FIGURE 2: BDT score before/after reweighting ═════════════════════
        fig2, axes2 = plt.subplots(
            2, 1, figsize=(7, 7),
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True,
        )
        ax_main, ax_ratio = axes2

        ax_main.errorbar(score_centres_1d, h_data_base, yerr=err_data_base,
                         fmt="o", color="black", markersize=3, linewidth=1.0,
                         label="Data", zorder=5)
        ax_main.step(score_centres_1d, h_mc_base, where="mid",
                     color="steelblue", linewidth=1.5, linestyle="--",
                     label="MC (nominal)")
        ax_main.fill_between(score_centres_1d,
                             h_mc_base - err_mc_base,
                             h_mc_base + err_mc_base,
                             step="mid", alpha=0.3, color="steelblue")
        ax_main.step(score_centres_1d, h_mc_rw, where="mid",
                     color="tomato", linewidth=1.5,
                     label=f"MC (rw {feat})")
        ax_main.fill_between(score_centres_1d,
                             h_mc_rw - err_mc_rw,
                             h_mc_rw + err_mc_rw,
                             step="mid", alpha=0.3, color="tomato")
        ax_main.set_ylabel("Normalised events / bin")
        ax_main.set_title(f"BDT score after reweighting on  {feat}")
        ax_main.legend(fontsize=9)

        with np.errstate(divide="ignore", invalid="ignore"):
            r_nom = np.where(h_mc_base > 0, h_data_base / h_mc_base, np.nan)
            r_rw  = np.where(h_mc_rw   > 0, h_data_base / h_mc_rw,   np.nan)
            err_r_nom = np.where(
                h_mc_base > 0,
                np.sqrt((err_data_base / h_mc_base) ** 2
                        + (h_data_base * err_mc_base / h_mc_base ** 2) ** 2),
                np.nan,
            )
            err_r_rw = np.where(
                h_mc_rw > 0,
                np.sqrt((err_data_base / h_mc_rw) ** 2
                        + (h_data_base * err_mc_rw / h_mc_rw ** 2) ** 2),
                np.nan,
            )

        ax_ratio.errorbar(score_centres_1d, r_nom, yerr=err_r_nom,
                          fmt="s", color="steelblue", markersize=3,
                          linewidth=1.0, linestyle="--", label="Data/MC nom.")
        ax_ratio.errorbar(score_centres_1d, r_rw, yerr=err_r_rw,
                          fmt="o", color="tomato", markersize=3,
                          linewidth=1.0, label="Data/MC rw")
        ax_ratio.axhline(1.0, color="black", linewidth=0.8, linestyle=":")
        ax_ratio.set_ylim(0.5, 1.5)
        ax_ratio.set_ylabel("Data / MC")
        ax_ratio.set_xlabel("BDT score")
        ax_ratio.legend(fontsize=8)

        plt.tight_layout()
        fig2.savefig(os.path.join(cfg["output"], f"score_reweight_{feat}.png"),
                     dpi=150, bbox_inches="tight")
        plt.close(fig2)

        # ══ FIGURE 3: Profile — mean BDT score vs feature ════════════════════
        feat_edges_prof  = np.linspace(combined.quantile(0.01),
                                       combined.quantile(0.99), N_BINS + 1)
        feat_centres_prof = 0.5 * (feat_edges_prof[:-1] + feat_edges_prof[1:])

        prof_data, err_data_p = weighted_profile(df_data, feat, feat_edges_prof)
        prof_mc,   err_mc_p   = weighted_profile(df_mc,   feat, feat_edges_prof)

        fig3, axes3 = plt.subplots(
            2, 1, figsize=(7, 7),
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True,
        )
        ax_prof, ax_diff = axes3

        ax_prof.errorbar(feat_centres_prof, prof_data, yerr=err_data_p,
                         fmt="o", color="black",    label="Data",  markersize=4)
        ax_prof.errorbar(feat_centres_prof, prof_mc,   yerr=err_mc_p,
                         fmt="s", color="steelblue", label="MC",    markersize=4)
        ax_prof.set_ylabel("Mean BDT score")
        ax_prof.set_title(f"Profile: mean BDT score vs  {feat}")
        ax_prof.legend(fontsize=9)

        diff     = prof_data - prof_mc
        err_diff = np.sqrt(err_data_p ** 2 + err_mc_p ** 2)
        ax_diff.errorbar(feat_centres_prof, diff, yerr=err_diff,
                         fmt="o", color="black", markersize=4)
        ax_diff.axhline(0.0, color="grey", linewidth=0.8, linestyle="--")
        ax_diff.set_ylabel("Data − MC")
        ax_diff.set_xlabel(feat)

        plt.tight_layout()
        fig3.savefig(os.path.join(cfg["output"], f"profile_{feat}.png"),
                     dpi=150, bbox_inches="tight")
        plt.close(fig3)

        print(f"    Saved plots for feature: {feat}")

    print(f"\nAll plots written to: {cfg['output']}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Data/MC BDT-score and feature comparison from coffea inputs"
    )
    parser.add_argument(
        "coffea_file",
        help="Path to the main pocket-coffea .coffea file (MC + Data)",
    )
    parser.add_argument(
        "--coffea-nonprompt",
        default=None,
        help="Path to a second .coffea file for the nonprompt estimate "
             "(weights will be negated)",
    )
    parser.add_argument(
        "--category", default="resolved_mu_WCR",
        help="Pocket-coffea category key  (default: resolved_mu_WCR)",
    )
    parser.add_argument(
        "--year", default="2022_postEE",
        help="Year label used in printouts  (default: 2022_postEE)",
    )
    parser.add_argument(
        "--models", nargs="+", required=True,
        help="Paths to one or more saved BDT .json model files",
    )
    parser.add_argument(
        "--outdir", default="bdt_CR",
        help="Output directory for plots  (default: bdt_CR)",
    )
    args = parser.parse_args()

    cfg = {
        "coffea_file":           args.coffea_file,
        "coffea_file_nonprompt": args.coffea_nonprompt,
        "category":              args.category,
        "year":                  args.year,
        "models":                args.models,
        "output":                args.outdir,
    }

    run(cfg)


if __name__ == "__main__":
    main()