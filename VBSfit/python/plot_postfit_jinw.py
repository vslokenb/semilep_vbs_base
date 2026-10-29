#!/usr/bin/env python3
"""
CMS-style post-fit stacked plots from FitDiagnostics output.

Reads shapes_fit_s/{channel}/{process} from fitDiagnostics_asimov_s1.root
(produced by run_fitdiagnostics.sh with --saveShapes --saveWithUncertainties).

Produces linear- and log-scale versions for each of the 4 signal regions,
matching the style of plots/stacked/ but with post-fit normalisations and
the covariance-propagated post-fit uncertainty band.

Data overlay:
  - Real data (from histograms ROOT file): filled circles, blinded above BDT > 0.6
  - Post-fit Asimov data (from FitDiagnostics): open circles with dashed error bars,
    full range (unblinded)
"""

import argparse
import os

import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

hep.style.use("CMS")

# ---------------------------------------------------------------------------
# Process groups — same merging as plot_stacked.py
# ---------------------------------------------------------------------------
GROUPS = [
    ("Diboson, Triboson",     "#92dadd",
     ["WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8",
      "WWZ_4F_TuneCP5_13TeV-amcatnlo-pythia8",
      "WZTo3LNu_mllmin01_NNPDF31_TuneCP5_13TeV_powheg_pythia8",
      "WZZ_TuneCP5_13TeV-amcatnlo-pythia8",
      "ZZZ_TuneCP5_13TeV-amcatnlo-pythia8",
      "WminusTo2JZTo2LJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
      "WminusToLNuWminusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
      "WminusToLNuZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
      "WplusTo2JZTo2LJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
      "WplusTo2JWminusToLNuJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
      "WplusToLNuWminusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
      "WplusToLNuWplusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
      "WplusToLNuZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
      "ZTo2LZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8"]),
    ("$V\\gamma$, VBF",       "#b9ac70",
     ["GluGluWWToLNuQQ_TuneCP5_13TeV_madgraph-pythia8",
      "ZGToLLG_01J_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8"]),
    ("DY",                    "#f89c20",
     [
      "DYJetsToLL",
      "DY",
      "DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8",
      "DYJetsToLL_M-10to50_TuneCP5_13TeV-amcatnloFXFX-pythia8"]),
    ("$t\\bar{t}$",           "#964a8b",
     [  "TTbar",
        "TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8",
      "TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8"]),
    ("Single Top",            "#94a4a2",
     ["ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8",
      "ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8",
      "ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8",
      "ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8",
      "ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8"]),
    ("Nonprompt",             "#a96b59",
     ["nonprompt"]),
    ("$t\\bar{t}$",           "#964a8b",
     ["TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8",
      "TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8"]),
    ("$W\\rightarrow l\\nu$", "#5790fc",
     [
      "WJets",
      "WJetsToLNu", 
      "WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8",
      "WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8",
      "WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8",
      "WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8",
      "WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8",
      "WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8",
      "WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8",
      "WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8"]),
]

SIGNAL_PROCS_RE = [
    "WminusToLNuWminusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
    "WplusToLNuWplusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
    "WplusTo2JWminusToLNuJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
    "WplusToLNuWminusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
    "WminusTo2JZTo2LJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
    "WminusToLNuZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
    "WplusTo2JZTo2LJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
    "WplusToLNuZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
    "ZTo2LZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
]
SIGNAL_COLOR    = "#e42536"
SIGNAL_LABEL    = "Signal $\\times$ 10"
SIGNAL_SCALE    = 10.0

SIGNAL_REGIONS  = ["resolved_mu", "resolved_e", "boosted_mu", "boosted_e"]
SR_LABELS       = {
    "resolved_mu": "Resolved $\\mu$",
    "resolved_e":  "Resolved $e$",
    "boosted_mu":  "Boosted $\\mu$",
    "boosted_e":   "Boosted $e$",
}
LUMI_LABEL   = "59.83 fb$^{-1}$ (13 TeV, 2018)"
BLIND_ABOVE = 0.6  # overridable via --blind-above CLI arg


def read_channel(f, channel, fit_dir="shapes_fit_s"):
    """Read all post-fit histograms for one channel from the fitDiagnostics file."""
    prefix  = f"{fit_dir}/{channel}/"
    procs   = {}      # name -> (vals, errs, edges)
    asimov_data   = None
    total_bkg     = None
    total_bkg_err = None
    total_sum     = None   # signal + background
    total_sum_err = None

    for key in f.keys(cycle=False):
        if not key.startswith(prefix):
            continue
        name = key[len(prefix):]
        obj  = f[key]

        if name == "data":
            # TGraphAsymmErrors — x are 0-indexed bin centres (0.5, 1.5, ...)
            asimov_data = {
                "x":   np.array(obj.member("fX")),
                "y":   np.array(obj.member("fY")),
                "eyl": np.array(obj.member("fEYlow")),
                "eyh": np.array(obj.member("fEYhigh")),
            }
        elif name in ("total_covar", "total_signal"):
            continue
        elif name == "total":
            vals  = obj.values()
            errs  = obj.errors()
            total_sum     = np.maximum(vals, 0.0)
            total_sum_err = errs
        elif name == "total_background":
            vals  = obj.values()
            errs  = obj.errors()
            _, edges = obj.to_numpy()
            total_bkg     = (np.maximum(vals, 0.0), edges)
            total_bkg_err = errs
        elif not name.startswith("total"):
            vals, edges = obj.to_numpy()
            errs = obj.errors()
            procs[name] = np.maximum(vals, 0.0), errs, edges

    return procs, total_bkg, total_bkg_err, total_sum, total_sum_err, asimov_data


def read_real_data(hist_file, channel):
    """Read data_obs from the pre-fit histograms ROOT file."""
    with uproot.open(hist_file) as g:
        key = f"{channel}/data_obs"
        if key not in g:
            return None, None
        h = g[key]
        vals, edges = h.to_numpy()
        return np.maximum(vals, 0.0), edges


def merge_group_postfit(procs, proc_list):
    """Sum post-fit histograms across processes in a group."""
    merged = None
    edges  = None
    for p in proc_list:
        if p not in procs:
            continue
        v, errs, e = procs[p]
        if merged is None:
            merged = v.copy()
            edges  = e
        else:
            merged += v
    return merged, edges


def read_edges_from_histfile(hist_file, channel):
    """Read bin edges from any nominal histogram in the pre-fit ROOT file."""
    if hist_file is None:
        return None
    with uproot.open(hist_file) as g:
        for key in g.keys(cycle=False):
            if not key.startswith(f"{channel}/"):
                continue
            name = key[len(f"{channel}/"):]
            if name in ("data_obs",) or name.endswith("Up") or name.endswith("Down"):
                continue
            h = g[key]
            if hasattr(h, "axes"):
                return np.asarray(h.axes[0].edges())
    return None


def make_postfit_figure(channel, f, hist_file, fit_dir="shapes_fit_s", scale="linear"):
    procs, total_bkg_info, total_bkg_err, total_sum, total_sum_err, asimov_data = read_channel(f, channel, fit_dir)

    if total_bkg_info is None:
        return None
    total_nom, _ = total_bkg_info

    n_bins = len(total_nom)

    # Use real bin edges from pre-fit histogram file (supports variable binning).
    # Fall back to uniform [0,1] if histogram file is unavailable.
    edges = read_edges_from_histfile(hist_file, channel)
    if edges is None:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif len(edges) - 1 < n_bins:
        # Combine pads all channels in the combined fit to the maximum bin count.
        # Trim FitDiag arrays to the actual bin count from the ROOT file.
        n_bins = len(edges) - 1
        total_nom = total_nom[:n_bins]
        if total_bkg_err is not None:
            total_bkg_err = total_bkg_err[:n_bins]
        if total_sum is not None:
            total_sum = total_sum[:n_bins]
        if total_sum_err is not None:
            total_sum_err = total_sum_err[:n_bins]
        for key in list(procs):
            v, errs, e = procs[key]
            procs[key] = v[:n_bins], errs[:n_bins], e
        if asimov_data is not None:
            mask = asimov_data["x"] < n_bins
            asimov_data = {k: v[mask] for k, v in asimov_data.items()}

    # Use signal+background total as reference for band and ratio denominator;
    # fall back to background-only total if signal histogram is absent.
    total_ref     = total_sum     if total_sum     is not None else total_nom
    total_ref_err = total_sum_err if total_sum_err is not None else total_bkg_err

    centers = 0.5 * (edges[:-1] + edges[1:])
    widths  = edges[1:] - edges[:-1]

    # --- build stacked background groups ---
    bkg_vals   = []
    bkg_colors = []
    bkg_labels = []

    for label, color, proc_list in GROUPS:
        v, e = merge_group_postfit(procs, proc_list)
        if v is None:
            continue
        bkg_vals.append(v)
        bkg_colors.append(color)
        bkg_labels.append(label)

    # --- signal: sum values and errors in quadrature ---
    sig_procs = [p for p in procs if any(p.startswith(pat) for pat in SIGNAL_PROCS_RE)]
    sig_nom   = None
    sig_var   = None
    for p in sig_procs:
        v, errs, _ = procs[p]
        if sig_nom is None:
            sig_nom = v.copy()
            sig_var = (errs ** 2).copy()
        else:
            sig_nom += v
            sig_var += errs ** 2
    sig_unc = np.sqrt(sig_var) if sig_var is not None else None

    sig_draw_scale = 1.0 if scale == "log" else SIGNAL_SCALE
    sig_label      = "Signal" if scale == "log" else SIGNAL_LABEL

    # --- real data (from pre-fit histograms, Poisson errors, blinded above BLIND_ABOVE) ---
    real_y = real_err = blind_mask = None
    if hist_file is not None:
        real_vals, _ = read_real_data(hist_file, channel)
        if real_vals is not None:
            real_y    = real_vals.astype(float)
            real_err  = np.sqrt(real_vals).astype(float)
            blind_mask = centers > BLIND_ABOVE
            real_y[blind_mask]   = np.nan
            real_err[blind_mask] = np.nan

    # --- Asimov data (from FitDiagnostics TGraph, full range) ---
    # x values are 0-indexed bin centres (0.5, 1.5, ...); map to actual bin centers
    asim_x = asim_y = asim_eyl = asim_eyh = None
    if asimov_data is not None:
        bin_indices = np.round(asimov_data["x"] - 0.5).astype(int)
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        asim_x   = centers[bin_indices]
        asim_y   = asimov_data["y"].astype(float)
        asim_eyl = asimov_data["eyl"].astype(float)
        asim_eyh = asimov_data["eyh"].astype(float)

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    fig, (ax, ax_r) = plt.subplots(
        2, 1, figsize=(8, 7),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    )

    hep.histplot(
        bkg_vals, edges,
        stack=True, histtype="fill",
        color=bkg_colors, label=bkg_labels,
        ax=ax, linewidth=0,
    )

    # post-fit uncertainty band centered on background
    unc = np.maximum(total_bkg_err, 0.0) if total_bkg_err is not None else np.zeros_like(total_nom)
    ax.bar(
        centers, 2 * unc,
        width=widths, bottom=total_nom - unc,
        color="gray", alpha=0.4,
        linewidth=0, label="Post-fit unc.",
        zorder=4,
    )

    # signal + uncertainty band
    if sig_nom is not None:
        sig_scaled = sig_nom * sig_draw_scale
        hep.histplot(
            sig_scaled, edges,
            histtype="step", color=SIGNAL_COLOR,
            linewidth=2, label=sig_label,
            ax=ax, zorder=5,
        )
        if sig_unc is not None:
            sig_unc_scaled = sig_unc * sig_draw_scale
            ax.bar(
                centers, 2 * sig_unc_scaled,
                width=widths, bottom=sig_scaled - sig_unc_scaled,
                color="none", edgecolor=SIGNAL_COLOR,
                hatch="\\\\\\\\", linewidth=0, alpha=0.5, zorder=5,
            )

    # Asimov data — open circles, dashed error bars, full range
    if asim_x is not None:
        eb = ax.errorbar(
            asim_x, asim_y, yerr=[asim_eyl, asim_eyh],
            fmt="o", color="black", markersize=4,
            markerfacecolor="none", markeredgewidth=1,
            linestyle="", capsize=0, linewidth=1,
            label="Asimov data", zorder=7,
        )
        # make error bar lines dashed
        for barline in eb[2]:
            barline.set_linestyle("--")

    # Real data — filled circles, blinded above BLIND_ABOVE
    if real_y is not None:
        ax.errorbar(
            centers, real_y, yerr=real_err,
            fmt="ko", markersize=4, linewidth=1,
            label="Data", zorder=8,
        )

    ax.set_ylabel("Events / bin", fontsize=13)
    ax.set_xlim(edges[0], edges[-1])
    ax.tick_params(labelbottom=False, labelsize=11)

    ymax = total_ref.max() if total_ref is not None else total_nom.max()
    if scale == "log":
        ax.set_yscale("log")
        ax.set_ylim(0.1, ymax * 500)
    else:
        ax.set_ylim(0, ymax * 1.6)

    ax.legend(ncol=2, fontsize=9, loc="upper right", framealpha=0.8)
    hep.cms.label("Work in Progress", data=True, rlabel=LUMI_LABEL, ax=ax, fontsize=12)
    ax.text(0.02, 0.96, SR_LABELS.get(channel, channel),
            transform=ax.transAxes, fontsize=11, va="top")

    # --- ratio panel: data / background (blinded) + Asimov / background (full range) ---
    if asim_x is not None:
        asim_ratio     = np.where(total_nom > 0, asim_y   / total_nom, np.nan)
        asim_ratio_eyl = np.where(total_nom > 0, asim_eyl / total_nom, np.nan)
        asim_ratio_eyh = np.where(total_nom > 0, asim_eyh / total_nom, np.nan)
        eb_r = ax_r.errorbar(
            asim_x, asim_ratio, yerr=[asim_ratio_eyl, asim_ratio_eyh],
            fmt="o", color="black", markersize=4,
            markerfacecolor="none", markeredgewidth=1,
            linestyle="", capsize=0, linewidth=1, zorder=3,
        )
        for barline in eb_r[2]:
            barline.set_linestyle("--")

    if real_y is not None:
        ratio     = np.where((total_nom > 0) & ~blind_mask, real_y   / total_nom, np.nan)
        ratio_err = np.where((total_nom > 0) & ~blind_mask, real_err / total_nom, np.nan)
        ax_r.errorbar(centers, ratio, yerr=ratio_err,
                      fmt="ko", markersize=4, linewidth=1, zorder=4)

    unc_rel = np.where(total_nom > 0, unc / total_nom, 0.0)  # unc already uses total_bkg_err
    ax_r.bar(centers, 2 * unc_rel, width=widths, bottom=1.0 - unc_rel,
             color="gray", alpha=0.4, linewidth=0, zorder=2)
    ax_r.axhline(1.0, color="black", lw=1)
    ax_r.axhline(1.2, color="gray", lw=0.5, ls="--")
    ax_r.axhline(0.8, color="gray", lw=0.5, ls="--")
    ax_r.set_ylim(0.5, 1.5)
    ax_r.set_ylabel("Data / Bkg", fontsize=11)
    ax_r.set_xlabel("BDT score", fontsize=13)
    ax_r.set_xlim(edges[0], edges[-1])
    ax_r.tick_params(labelsize=11)

    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fitfile",  default="plots/fitdiagnostics/fitDiagnostics_asimov_s1.root")
    parser.add_argument("--histfile", default="histograms/histograms.root",
                        help="Pre-fit histograms ROOT file for real data overlay")
    parser.add_argument("--fitdir",     default="shapes_fit_s",
                        help="Directory inside fitDiagnostics file (shapes_fit_s or shapes_prefit)")
    parser.add_argument("--outdir",     default="plots/postfit")
    parser.add_argument("--blind-above", type=float, default=None,
                        help="Hide data above this BDT score (default: use script constant)")
    args = parser.parse_args()

    global BLIND_ABOVE
    if args.blind_above is not None:
        BLIND_ABOVE = args.blind_above

    os.makedirs(args.outdir, exist_ok=True)

    hist_file = args.histfile if os.path.exists(args.histfile) else None
    if hist_file is None:
        print(f"[warn] histfile not found: {args.histfile} — real data overlay skipped")

    with uproot.open(args.fitfile) as f:
        for channel in SIGNAL_REGIONS:
            for scale in ("linear", "log"):
                fig = make_postfit_figure(channel, f, hist_file, args.fitdir, scale)
                if fig is None:
                    print(f"[skip] {channel} — no shapes found")
                    continue
                for ext in ("pdf", "png"):
                    out = os.path.join(args.outdir, f"{channel}_{scale}.{ext}")
                    fig.savefig(out, bbox_inches="tight", dpi=150)
                plt.close(fig)
                print(f"Saved: {channel}_{scale}.pdf/png")

    print("Done.")


if __name__ == "__main__":
    main()
