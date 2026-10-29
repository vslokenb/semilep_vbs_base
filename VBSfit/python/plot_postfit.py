#!/usr/bin/env python3
"""
CMS-style post-fit stacked plots from FitDiagnostics output.

Reads shapes_fit_s/{channel}/{process} from a fitDiagnostics ROOT file
(produced with --saveShapes --saveWithUncertainties).

WHAT THE UNCERTAINTY BAND CONTAINS
----------------------------------
The band is total_background's bin errors, which combine fills (only when
the fit ran with --saveShapes --saveWithUncertainties) by sampling the full
postfit covariance: every constrained nuisance, the rateParams, autoMCStats,
and all their correlations. Its statistical precision is set by
--numToysForShapes (combine's default is a few hundred; use e.g.
--numToysForShapes 1000 for smoother bands). A fitDiagnostics file produced
WITHOUT --saveWithUncertainties has all-zero bin errors: this script now
detects that, prints a loud warning, and stamps the band label on the plot
itself so such a figure cannot be mistaken for a complete one.

The signal band comes from total_signal (same covariance sampling), NOT
from summing per-process errors in quadrature -- the per-process errors
carry no process-to-process correlations, and the signal samples are
essentially fully correlated through the shared nuisances.

Produces linear- and log-scale versions for every channel in the fit --
the 4 signal regions and, when present, the ttbar and W+jets control regions.

TWO DATASET MODES (--dataset)
-----------------------------
The 'data' TGraph inside a fitDiagnostics file is whatever dataset the fit
actually used. What that means depends on how the fit was run, so it must be
labelled accordingly or the plots are misleading:

  --dataset asimov   the fit used '-t -1', so EVERY channel -- control regions
                     included -- holds the Asimov expectation. Drawn as open
                     circles labelled "Asimov data". The real data from
                     --histfile is overlaid separately for comparison
                     (blinded in the signal regions, see below).

  --dataset crdata   the fit used a hybrid card from make_hybrid_data.py:
                     control regions hold REAL DATA and signal regions hold MC
                     pseudo-data built at the CR-fit normalizations. The graph
                     is therefore labelled per region -- "Data" in the CRs,
                     "SR pseudo-data (CR-norm.)" in the SRs. The separate real
                     data overlay is suppressed in the CRs, where it would just
                     duplicate the points already drawn.

Either way the signal region shows pseudo-data, never an observation.

BLINDING
--------
The real-data overlay from --histfile is region-aware:

  signal regions   hidden above --blind-above (default 0.6). run_all.sh passes
                   0, which hides every SR point.
  control regions  always shown; control regions are unblinded by design.

A channel that is neither a known SR name nor a recognised CR prefix is treated
as a SIGNAL region, so an unfamiliar channel errs toward staying blind.
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
#
# NOTE: each physical group must appear EXACTLY ONCE. A process listed under
# two groups is merged into the stack twice, which inflates the stack while
# leaving 'total_background' (read straight from the fit) correct -- the plot
# then shows a stack that does not match its own uncertainty band or ratio
# denominator. See check_group_overlap() below, which now aborts on that.
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
     ["DYJetsToLL",
      "DY",
      "DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8",
      "DYJetsToLL_M-10to50_TuneCP5_13TeV-amcatnloFXFX-pythia8"]),
    ("$t\\bar{t}$",           "#964a8b",
     ["TTbar",
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
    ("$W\\rightarrow l\\nu$", "#5790fc",
     ["WJets",
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
# Control-region name prefixes written by make_datacards.py, and how to label
# them. A channel matching none of these and not in SIGNAL_REGIONS is treated
# as a signal region, i.e. it stays blind.
CR_PREFIXES = [
    ("ttbar_cr_", "$t\\bar{t}$ CR"),
    ("w_cr_",     "$W$+jets CR"),
]

LUMI_LABEL   = "59.83 fb$^{-1}$ (13 TeV, 2018)"
BLIND_ABOVE = 0.6  # overridable via --blind-above CLI arg

# Per-mode labelling of the dataset stored in the fitDiagnostics file.
DATASET_MODES = {
    "asimov": {
        "banner":    "Asimov dataset (MC norms)",
        "sr_label":  "Asimov data",
        "cr_label":  "Asimov data",
        "cr_overlay_real": True,
    },
    "crdata": {
        "banner":    "CR data + SR Asimov at CR norms",
        "sr_label":  "SR pseudo-data (CR-norm.)",
        "cr_label":  "Data",
        "cr_overlay_real": False,
    },
}


# ---------------------------------------------------------------------------
# region helpers
# ---------------------------------------------------------------------------

def is_signal_region(channel):
    """True for signal regions. Fail-safe: unknown channels count as SR."""
    for pref, _ in CR_PREFIXES:
        if channel.startswith(pref):
            return False
    return True


def region_label(channel):
    if channel in SR_LABELS:
        return SR_LABELS[channel]
    for pref, name in CR_PREFIXES:
        if channel.startswith(pref):
            base = channel[len(pref):]
            return "%s, %s" % (name, SR_LABELS.get(base, base))
    return channel


def check_group_overlap():
    """Abort if a process is listed under more than one group.

    A duplicated process is silently added to the stack twice: the drawn stack
    no longer matches total_background, so the uncertainty band and the ratio
    denominator disagree with the bars above them.
    """
    seen, dupes = {}, {}
    for label, _, procs in GROUPS:
        for p in procs:
            if p in seen and seen[p] != label:
                dupes.setdefault(p, {seen[p]}).add(label)
            seen[p] = label
    if dupes:
        lines = ["ERROR: these processes appear in more than one entry of GROUPS,",
                 "       which would double-count them in the stack:"]
        for p, labels in sorted(dupes.items()):
            lines.append("         %s  ->  %s" % (p, ", ".join(sorted(labels))))
        raise SystemExit("\n".join(lines))


def discover_channels(f, fit_dir):
    """List channels present under fit_dir, signal regions first."""
    found = []
    prefix = fit_dir + "/"
    for key in f.keys(cycle=False):
        if not key.startswith(prefix):
            continue
        rest = key[len(prefix):]
        if "/" not in rest:
            continue
        ch = rest.split("/")[0]
        if ch not in found:
            found.append(ch)
    srs = [c for c in SIGNAL_REGIONS if c in found]
    srs += [c for c in found if c not in srs and is_signal_region(c)]
    crs = [c for c in found if c not in srs]
    return srs + crs


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def read_channel(f, channel, fit_dir="shapes_fit_s"):
    """Read all post-fit histograms for one channel from the fitDiagnostics file."""
    prefix  = f"{fit_dir}/{channel}/"
    procs   = {}      # name -> (vals, errs, edges)
    fit_data      = None
    total_bkg     = None
    total_bkg_err = None
    total_sum     = None   # signal + background
    total_sum_err = None
    total_sig     = None   # summed signal WITH full correlations
    total_sig_err = None

    for key in f.keys(cycle=False):
        if not key.startswith(prefix):
            continue
        name = key[len(prefix):]
        if "/" in name:
            continue
        obj  = f[key]

        if name == "data":
            # TGraphAsymmErrors — x are 0-indexed bin centres (0.5, 1.5, ...)
            fit_data = {
                "x":   np.array(obj.member("fX")),
                "y":   np.array(obj.member("fY")),
                "eyl": np.array(obj.member("fEYlow")),
                "eyh": np.array(obj.member("fEYhigh")),
            }
        elif name == "total_covar":
            continue
        elif name == "total_signal":
            # combine's covariance-sampled signal sum: its bin errors carry the
            # full nuisance correlations across the signal processes, unlike a
            # quadrature sum of the per-process errors.
            vals = obj.values()
            errs = obj.errors()
            total_sig     = np.maximum(vals, 0.0)
            total_sig_err = errs
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

    return (procs, total_bkg, total_bkg_err, total_sum, total_sum_err,
            total_sig, total_sig_err, fit_data)


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


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

def make_postfit_figure(channel, f, hist_file, fit_dir="shapes_fit_s",
                        scale="linear", dataset="asimov"):
    mode   = DATASET_MODES[dataset]
    is_sr  = is_signal_region(channel)

    (procs, total_bkg_info, total_bkg_err,
     total_sum, total_sum_err,
     total_sig, total_sig_err, fit_data) = read_channel(f, channel, fit_dir)

    if total_bkg_info is None:
        return None
    total_nom, _ = total_bkg_info

    n_bins = len(total_nom)

    # Use real bin edges from pre-fit histogram file (supports variable binning).
    # Fall back to uniform [0,1] if histogram file is unavailable.
    edges = read_edges_from_histfile(hist_file, channel)
    if edges is None:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        if hist_file is not None:
            print(f"[warn] {channel}: no template found in histfile; bin edges "
                  f"and any padding are guesses")
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
        if total_sig is not None:
            total_sig = total_sig[:n_bins]
        if total_sig_err is not None:
            total_sig_err = total_sig_err[:n_bins]
        for key in list(procs):
            v, errs, e = procs[key]
            procs[key] = v[:n_bins], errs[:n_bins], e
        if fit_data is not None:
            mask = fit_data["x"] < n_bins
            fit_data = {k: v[mask] for k, v in fit_data.items()}

    # Systematics sanity check: with --saveShapes --saveWithUncertainties the
    # total_background errors are the full covariance-sampled postfit
    # uncertainty (all constrained nuisances + rateParams + autoMCStats, with
    # correlations). All-zero errors on a non-empty total mean the fit was run
    # WITHOUT --saveWithUncertainties -- the band would silently contain no
    # systematics at all, so say it loudly and stamp the plot.
    band_ok = (total_bkg_err is not None
               and np.max(total_bkg_err) > 0 and np.max(total_nom) > 0)
    if not band_ok:
        print(f"[WARN] {channel}: total_background has no bin errors -- the "
              f"fitDiagnostics file was made without --saveWithUncertainties; "
              f"the plotted band contains NO systematics. Re-run combine with "
              f"--saveShapes --saveWithUncertainties (and e.g. "
              f"--numToysForShapes 1000).")

    # Use signal+background total as reference for band and ratio denominator;
    # fall back to background-only total if signal histogram is absent.
    total_ref = total_sum if total_sum is not None else total_nom

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

    # --- signal: prefer combine's total_signal (errors carry the full
    # process-to-process correlations); fall back to the quadrature sum of
    # per-process errors only when total_signal is absent, which UNDERSTATES
    # the band since the signal samples share their nuisances.
    if total_sig is not None and np.max(total_sig) > 0:
        sig_nom = total_sig
        sig_unc = total_sig_err
    else:
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
        if sig_nom is not None:
            print(f"[warn] {channel}: no total_signal in fitfile; signal band "
                  f"is a per-process quadrature sum (correlations dropped)")

    sig_draw_scale = 1.0 if scale == "log" else SIGNAL_SCALE
    sig_label      = "Signal" if scale == "log" else SIGNAL_LABEL

    # --- real data overlay from the pre-fit histograms ---------------------
    # Signal regions: masked above BLIND_ABOVE (run_all.sh passes 0 -> all
    # masked). Control regions: shown in full, but only when the fit itself
    # used Asimov -- in --dataset crdata the fit data graph below already IS
    # the real CR data, and drawing both would double up the points.
    real_y = real_err = None
    blind_mask = np.zeros(n_bins, dtype=bool)
    want_real = is_sr or mode["cr_overlay_real"]
    if hist_file is not None and want_real:
        real_vals, _ = read_real_data(hist_file, channel)
        if real_vals is not None:
            real_vals = np.asarray(real_vals, dtype=float)[:n_bins]
            if len(real_vals) == n_bins:
                real_y   = real_vals.copy()
                real_err = np.sqrt(real_vals)
                if is_sr:
                    blind_mask = centers > BLIND_ABOVE
                    real_y[blind_mask]   = np.nan
                    real_err[blind_mask] = np.nan
            else:
                print(f"[warn] {channel}: data_obs has {len(real_vals)} bins but "
                      f"the fit shapes have {n_bins}; overlay skipped")

    # --- the dataset the fit actually used (TGraph), full range ------------
    fit_x = fit_y = fit_eyl = fit_eyh = None
    if fit_data is not None:
        bin_indices = np.round(fit_data["x"] - 0.5).astype(int)
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        fit_x   = centers[bin_indices]
        fit_y   = fit_data["y"].astype(float)
        fit_eyl = fit_data["eyl"].astype(float)
        fit_eyh = fit_data["eyh"].astype(float)
    fit_data_label = mode["sr_label"] if is_sr else mode["cr_label"]

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
    band_label = ("Post-fit unc. (all syst.)" if band_ok
                  else "Post-fit unc. MISSING (no --saveWithUncertainties)")
    ax.bar(
        centers, 2 * unc,
        width=widths, bottom=total_nom - unc,
        color="gray", alpha=0.4,
        linewidth=0, label=band_label,
        zorder=4,
    )

    # signal + uncertainty band (signal is absent from the CR templates)
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

    # dataset used by the fit — open circles, dashed error bars, full range
    if fit_x is not None:
        eb = ax.errorbar(
            fit_x, fit_y, yerr=[fit_eyl, fit_eyh],
            fmt="o", color="black", markersize=4,
            markerfacecolor="none", markeredgewidth=1,
            linestyle="", capsize=0, linewidth=1,
            label=fit_data_label, zorder=7,
        )
        for barline in eb[2]:
            barline.set_linestyle("--")

    # real data — filled circles, blinded in the SR
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
    ax.text(0.02, 0.96, region_label(channel),
            transform=ax.transAxes, fontsize=11, va="top")
    # Say on the figure itself which dataset this is, so a stray PDF can never
    # be mistaken for the other running mode.
    ax.text(0.02, 0.90, mode["banner"],
            transform=ax.transAxes, fontsize=8, va="top", color="#555555")
    if not band_ok:
        ax.text(0.02, 0.86, "band has NO systematics (fit lacked --saveWithUncertainties)",
                transform=ax.transAxes, fontsize=8, va="top", color="#c00000")

    # --- ratio panel -------------------------------------------------------
    if fit_x is not None:
        r     = np.where(total_nom > 0, fit_y   / total_nom, np.nan)
        r_eyl = np.where(total_nom > 0, fit_eyl / total_nom, np.nan)
        r_eyh = np.where(total_nom > 0, fit_eyh / total_nom, np.nan)
        eb_r = ax_r.errorbar(
            fit_x, r, yerr=[r_eyl, r_eyh],
            fmt="o", color="black", markersize=4,
            markerfacecolor="none", markeredgewidth=1,
            linestyle="", capsize=0, linewidth=1, zorder=3,
        )
        for barline in eb_r[2]:
            barline.set_linestyle("--")

    if real_y is not None:
        ok = (total_nom > 0) & ~blind_mask
        ratio     = np.where(ok, real_y   / total_nom, np.nan)
        ratio_err = np.where(ok, real_err / total_nom, np.nan)
        ax_r.errorbar(centers, ratio, yerr=ratio_err,
                      fmt="ko", markersize=4, linewidth=1, zorder=4)

    unc_rel = np.where(total_nom > 0, unc / total_nom, 0.0)
    ax_r.bar(centers, 2 * unc_rel, width=widths, bottom=1.0 - unc_rel,
             color="gray", alpha=0.4, linewidth=0, zorder=2)
    ax_r.axhline(1.0, color="black", lw=1)
    ax_r.axhline(1.2, color="gray", lw=0.5, ls="--")
    ax_r.axhline(0.8, color="gray", lw=0.5, ls="--")
    ax_r.set_ylim(0.5, 1.5)
    ax_r.set_ylabel("Data / Bkg", fontsize=11)
    ax_r.set_xlabel("BDT score" if is_sr else "Counting bin", fontsize=13)
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
                        help="Hide SIGNAL-REGION data above this BDT score "
                             "(default: use script constant). Control regions "
                             "are never blinded by this option.")
    parser.add_argument("--dataset", choices=sorted(DATASET_MODES), default="asimov",
                        help="What the fitted dataset is: 'asimov' for a -t -1 "
                             "fit, 'crdata' for a hybrid card (real CR data + "
                             "SR pseudo-data at CR norms). Controls labelling "
                             "only; it does not change any numbers.")
    parser.add_argument("--channels", default="auto",
                        help="Comma-separated channels, or 'auto' to plot every "
                             "channel in the fit file (SRs first, then CRs), or "
                             "'sr' for the 4 signal regions only.")
    parser.add_argument("--tag", default="",
                        help="Suffix appended to output filenames, e.g. 'crdata'")
    args = parser.parse_args()

    check_group_overlap()

    global BLIND_ABOVE
    if args.blind_above is not None:
        BLIND_ABOVE = args.blind_above

    os.makedirs(args.outdir, exist_ok=True)

    hist_file = args.histfile if os.path.exists(args.histfile) else None
    if hist_file is None:
        print(f"[warn] histfile not found: {args.histfile} — real data overlay skipped")

    suffix = f"_{args.tag}" if args.tag else ""

    print(f"Dataset mode: {args.dataset} — {DATASET_MODES[args.dataset]['banner']}")
    print(f"SR data from --histfile hidden above BDT {BLIND_ABOVE}")

    with uproot.open(args.fitfile) as f:
        if args.channels == "auto":
            channels = discover_channels(f, args.fitdir)
        elif args.channels == "sr":
            channels = list(SIGNAL_REGIONS)
        else:
            channels = [c for c in args.channels.split(",") if c]

        if not channels:
            print(f"[warn] no channels found under {args.fitdir} in {args.fitfile}")

        for channel in channels:
            for scale in ("linear", "log"):
                fig = make_postfit_figure(channel, f, hist_file, args.fitdir,
                                          scale, args.dataset)
                if fig is None:
                    print(f"[skip] {channel} — no shapes found")
                    continue
                for ext in ("pdf", "png"):
                    out = os.path.join(args.outdir, f"{channel}_{scale}{suffix}.{ext}")
                    fig.savefig(out, bbox_inches="tight", dpi=150)
                plt.close(fig)
                print(f"Saved: {channel}_{scale}{suffix}.pdf/png")

    print("Done.")


if __name__ == "__main__":
    main()
