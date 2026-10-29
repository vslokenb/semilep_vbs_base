#!/usr/bin/env python3
"""
CMS-style stacked background plots with systematic + statistical uncertainty band,
and merged signal overlaid at ×10.

Produces linear- and log-scale versions for each of the 4 signal regions.
"""

import argparse
import os
from collections import defaultdict

import numpy as np
import uproot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep

hep.style.use("CMS")

# ---------------------------------------------------------------------------
# Process groups: (display label, hex color, list of ROOT-file process names)
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
      "DY",
      "DYJetsToLL"
      "DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8",
      "DYJetsToLL_M-10to50_TuneCP5_13TeV-amcatnloFXFX-pythia8"]),
    ("Single Top",            "#94a4a2",
     ["ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8",
      "ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8",
      "ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8",
      "ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8",
      "ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8"]),
    ("Nonprompt",             "#a96b59",
     ["nonprompt"]),
    ("$t\\bar{t}$",           "#964a8b",
     [  "TTbar",
        "TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8",
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
BLIND_ABOVE  = 0.6   # mask data points with BDT score > this value (overridable via CLI)


# ---------------------------------------------------------------------------
# ROOT file reading
# ---------------------------------------------------------------------------
def read_root(path):
    """
    Returns:
      data[channel][proc_name]["nominal"] = (values, variances, edges)
      data[channel][proc_name][syst+"Up"]  = (values, variances, edges)
      data[channel][proc_name][syst+"Down"]= (values, variances, edges)
    """
    raw = defaultdict(dict)  # raw[channel][key] = (vals, var, edges)

    with uproot.open(path) as f:
        for key in f.keys(cycle=False):
            parts = key.split("/")
            if len(parts) != 2:
                continue
            channel, name = parts
            h    = f[key]
            vals = h.values()
            var  = h.variances() if h.variances() is not None else np.zeros_like(vals)
            _, edges = h.to_numpy()
            raw[channel][name] = (vals, var, edges)

    # Attach Up/Down keys to their owning process
    structured = defaultdict(lambda: defaultdict(dict))
    for channel, keys in raw.items():
        # nominal process names (no Up/Down suffix, not data_obs)
        nom_names = sorted(
            [k for k in keys
             if not k.endswith("Up") and not k.endswith("Down")
             and k != "data_obs"],
            key=len, reverse=True
        )
        for proc in nom_names:
            structured[channel][proc]["nominal"] = keys[proc]
        if "data_obs" in keys:
            structured[channel]["data_obs"]["nominal"] = keys["data_obs"]
        for key, vh in keys.items():
            if not (key.endswith("Up") or key.endswith("Down")):
                continue
            match = next((p for p in nom_names if key.startswith(p + "_")), None)
            if match:
                var_name = key[len(match) + 1:]
                structured[channel][match][var_name] = vh
    return structured


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def merge_group(structured, channel, proc_list, proc_scales=None):
    """
    Sum nominal values and variances across a list of process names.
    proc_scales: optional dict {proc_name: scale_factor}
    Returns (nom_vals, nom_var, edges, per_syst_deltas)
    where per_syst_deltas[syst] = (delta_up, delta_dn)
    """
    nom_vals = None
    nom_var  = None
    edges    = None
    syst_deltas = defaultdict(lambda: [None, None])  # [delta_up, delta_dn]
    if proc_scales is None:
        proc_scales = {}

    for proc in proc_list:
        if proc not in structured[channel]:
            continue
        pvars = structured[channel][proc]
        if "nominal" not in pvars:
            continue
        v, var, e = pvars["nominal"]
        sf = proc_scales.get(proc, 1.0)
        v   = np.maximum(v, 0.0) * sf
        var = var * sf * sf
        if nom_vals is None:
            nom_vals = v.copy()
            nom_var  = var.copy()
            edges    = e
        else:
            nom_vals += v
            nom_var  += var

        # collect shape systematics
        for key in pvars:
            if key in ("nominal", "_raw"):
                continue
            if key.endswith("Up"):
                syst = key[:-2]
                vup, _, _ = pvars[key]
                vup = np.maximum(vup, 0.0) * sf
                delta = vup - v
                if syst_deltas[syst][0] is None:
                    syst_deltas[syst][0] = delta.copy()
                else:
                    syst_deltas[syst][0] += delta
            elif key.endswith("Down"):
                syst = key[:-4]
                vdn, _, _ = pvars[key]
                vdn = np.maximum(vdn, 0.0) * sf
                delta = v - vdn
                if syst_deltas[syst][1] is None:
                    syst_deltas[syst][1] = delta.copy()
                else:
                    syst_deltas[syst][1] += delta

    return nom_vals, nom_var, edges, syst_deltas


def total_uncertainty(nom_vals, nom_var, syst_deltas):
    """Combine stat + syst uncertainties. Returns (unc_up, unc_dn)."""
    stat = np.sqrt(np.maximum(nom_var, 0.0))
    syst_up2 = np.zeros_like(stat)
    syst_dn2 = np.zeros_like(stat)
    for syst, (d_up, d_dn) in syst_deltas.items():
        if d_up is not None:
            syst_up2 += d_up ** 2
        if d_dn is not None:
            syst_dn2 += d_dn ** 2
    return (np.sqrt(stat**2 + syst_up2),
            np.sqrt(stat**2 + syst_dn2))


def make_figure(channel, structured, scale="linear", proc_scales=None):
    """
    proc_scales: dict {process_name: scale_factor} applied at merge time.
    Intended for WJet (sf_wjet) and TTtoLNu2Q (sf_top) from scale_factors.json.
    Note: if histograms/histograms_SRonly.root was generated with pre-applied SFs
    (default make_histograms_SRonly.py behaviour), pass proc_scales=None to avoid
    double-scaling.  Pass the actual dict when histograms are unscaled.
    """
    if proc_scales is None:
        proc_scales = {}
    edges = None

    # --- build background groups ---
    bkg_vals  = []
    bkg_colors = []
    bkg_labels = []
    total_nom  = None
    total_var  = None
    all_syst_deltas_up = defaultdict(lambda: None)
    all_syst_deltas_dn = defaultdict(lambda: None)

    for label, color, procs in GROUPS:
        v, var, e, syst_d = merge_group(structured, channel, procs, proc_scales)
        if v is None:
            continue
        if edges is None:
            edges = e
        bkg_vals.append(v)
        bkg_colors.append(color)
        bkg_labels.append(label)
        if total_nom is None:
            total_nom = v.copy()
            total_var = var.copy()
        else:
            total_nom += v
            total_var += var
        for syst, (d_up, d_dn) in syst_d.items():
            if d_up is not None:
                all_syst_deltas_up[syst] = (
                    d_up if all_syst_deltas_up[syst] is None
                    else all_syst_deltas_up[syst] + d_up
                )
            if d_dn is not None:
                all_syst_deltas_dn[syst] = (
                    d_dn if all_syst_deltas_dn[syst] is None
                    else all_syst_deltas_dn[syst] + d_dn
                )

    if total_nom is None or edges is None:
        return None

    # combine syst deltas for uncertainty
    combined_systs = {
        s: (all_syst_deltas_up.get(s), all_syst_deltas_dn.get(s))
        for s in set(list(all_syst_deltas_up) + list(all_syst_deltas_dn))
    }
    unc_up, unc_dn = total_uncertainty(total_nom, total_var, combined_systs)

    # --- signal ---
    sig_procs = [p for p in structured[channel]
                 if any(p.startswith(pat) for pat in SIGNAL_PROCS_RE)]
    sig_nom  = None
    sig_var  = None
    sig_systs = defaultdict(lambda: [None, None])

    for proc in sig_procs:
        pvars = structured[channel][proc]
        if "nominal" not in pvars:
            continue
        v, var, _ = pvars["nominal"]
        v = np.maximum(v, 0.0)
        if sig_nom is None:
            sig_nom = v.copy()
            sig_var = var.copy()
        else:
            sig_nom += v
            sig_var += var
        for key in pvars:
            if key in ("nominal", "_raw"):
                continue
            if key.endswith("Up"):
                syst = key[:-2]
                vup, _, _ = pvars[key]
                delta = np.maximum(vup, 0.0) - v
                if sig_systs[syst][0] is None:
                    sig_systs[syst][0] = delta.copy()
                else:
                    sig_systs[syst][0] += delta
            elif key.endswith("Down"):
                syst = key[:-4]
                vdn, _, _ = pvars[key]
                delta = v - np.maximum(vdn, 0.0)
                if sig_systs[syst][1] is None:
                    sig_systs[syst][1] = delta.copy()
                else:
                    sig_systs[syst][1] += delta

    sig_draw_scale = 1.0 if scale == "log" else SIGNAL_SCALE
    sig_label      = "Signal" if scale == "log" else SIGNAL_LABEL
    sig_scaled = sig_nom * sig_draw_scale if sig_nom is not None else None
    if sig_nom is not None:
        sig_unc_up, sig_unc_dn = total_uncertainty(sig_nom, sig_var, sig_systs)
        sig_unc_up *= sig_draw_scale
        sig_unc_dn *= sig_draw_scale

    # --- data_obs ---
    data_vals = None
    if "data_obs" in structured[channel]:
        dv, dvar, _ = structured[channel]["data_obs"]["nominal"]
        data_vals = np.maximum(dv, 0.0)
        data_var  = np.maximum(dvar, 0.0)

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths  = edges[1:] - edges[:-1]

    fig, (ax, ax_r) = plt.subplots(
        2, 1, figsize=(8, 7),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    )

    # stacked backgrounds (no inter-stack boundaries)
    hep.histplot(
        bkg_vals, edges,
        stack=True, histtype="fill",
        color=bkg_colors, label=bkg_labels,
        ax=ax, linewidth=0,
    )

    # uncertainty band on total background — solid gray area
    ax.bar(
        centers, unc_up + unc_dn,
        width=widths, bottom=total_nom - unc_dn,
        color="gray", alpha=0.4,
        linewidth=0, label="Unc. (stat$\\oplus$syst)",
        zorder=4,
    )

    # signal ×10
    if sig_scaled is not None:
        hep.histplot(
            sig_scaled, edges,
            histtype="step", color=SIGNAL_COLOR,
            linewidth=2, label=sig_label,
            ax=ax, zorder=5,
        )
        # signal uncertainty band
        ax.bar(
            centers, (sig_unc_up + sig_unc_dn),
            width=widths, bottom=sig_scaled - sig_unc_dn,
            color="none", edgecolor=SIGNAL_COLOR,
            hatch="\\\\\\\\", linewidth=0, alpha=0.5, zorder=5,
        )

    # data points — blinded above BLIND_ABOVE
    if data_vals is not None:
        data_err   = np.sqrt(data_var)
        blind_mask = centers > BLIND_ABOVE
        d_plot     = data_vals.copy().astype(float)
        e_plot     = data_err.copy().astype(float)
        d_plot[blind_mask] = np.nan
        e_plot[blind_mask] = np.nan
        ax.errorbar(
            centers, d_plot,
            yerr=e_plot, fmt="ko",
            markersize=4, linewidth=1,
            label="Data", zorder=6,
        )

    ax.set_ylabel("Events / bin", fontsize=13)
    ax.set_xlim(edges[0], edges[-1])
    ax.tick_params(labelbottom=False, labelsize=11)

    if scale == "log":
        ax.set_yscale("log")
        ax.set_ylim(0.1, np.max(total_nom) * 500)
    else:
        ax.set_ylim(0, np.max(total_nom) * 1.6)

    ax.legend(ncol=2, fontsize=9, loc="upper right", framealpha=0.8)

    hep.cms.label(
        "Work in Progress",
        data=True,
        rlabel=LUMI_LABEL,
        ax=ax,
        fontsize=12,
    )

    # --- ratio panel ---
    if data_vals is not None:
        ratio      = np.where((total_nom > 0) & ~blind_mask, data_vals / total_nom, np.nan)
        ratio_err  = np.where((total_nom > 0) & ~blind_mask, data_err  / total_nom, np.nan)
        unc_up_rel = np.where(total_nom > 0, unc_up / total_nom, 0.0)
        unc_dn_rel = np.where(total_nom > 0, unc_dn / total_nom, 0.0)

        ax_r.errorbar(
            centers, ratio, yerr=ratio_err,
            fmt="ko", markersize=4, linewidth=1, zorder=3,
        )
        ax_r.bar(
            centers, unc_up_rel + unc_dn_rel,
            width=widths, bottom=1.0 - unc_dn_rel,
            color="gray", alpha=0.4, linewidth=0, zorder=2,
        )
    else:
        # no data: just show the uncertainty band around 1
        unc_up_rel = np.where(total_nom > 0, unc_up / total_nom, 0.0)
        unc_dn_rel = np.where(total_nom > 0, unc_dn / total_nom, 0.0)
        ax_r.bar(
            centers, unc_up_rel + unc_dn_rel,
            width=widths, bottom=1.0 - unc_dn_rel,
            color="gray", alpha=0.4, linewidth=0,
        )

    ax_r.axhline(1.0, color="black", lw=1)
    ax_r.axhline(1.2, color="gray", lw=0.5, ls="--")
    ax_r.axhline(0.8, color="gray", lw=0.5, ls="--")
    ax_r.set_ylim(0.5, 1.5)
    ax_r.set_ylabel("Data / MC", fontsize=11)
    ax_r.set_xlabel("BDT score", fontsize=13)
    ax_r.set_xlim(edges[0], edges[-1])
    ax_r.tick_params(labelsize=11)

    # channel label in top panel
    ax.text(
        0.02, 0.96, SR_LABELS.get(channel, channel),
        transform=ax.transAxes, fontsize=11,
        verticalalignment="top",
    )

    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rootfile",    default="histograms/histograms_SRonly.root")
    parser.add_argument("--sffile",      default="histograms/scale_factors.json")
    parser.add_argument("--outdir",      default="plots/stacked")
    parser.add_argument("--blind-above", type=float, default=None,
                        help="Hide data above this BDT score (default: use script constant)")
    args = parser.parse_args()

    global BLIND_ABOVE
    if args.blind_above is not None:
        BLIND_ABOVE = args.blind_above

    os.makedirs(args.outdir, exist_ok=True)

    # Load scale factors from CR-derived JSON.
    # These are already pre-applied in histograms_SRonly.root by
    # make_histograms_SRonly.py (Step 3).  Pass proc_scales=None below so
    # we do NOT double-scale.  The dict is built here for documentation and
    # can be activated if regenerating from unscaled histograms.
    proc_scales = {}
    if os.path.exists(args.sffile):
        import json
        with open(args.sffile) as fh:
            sf_info = json.load(fh)
        sf_top  = sf_info["SF_top"]["value"]
        sf_wjet = sf_info["SF_wjet"]["value"]
        print(f"Scale factors loaded: SF_top={sf_top:.4f}, SF_wjet={sf_wjet:.4f}")
        print("  (already pre-applied in ROOT file — proc_scales left empty)")
        # To apply here instead of in make_histograms_SRonly.py, uncomment:
        # proc_scales["WtoLNu-2Jets"] = sf_wjet
        # proc_scales["TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8"] = sf_top

    structured = read_root(args.rootfile)

    for channel in SIGNAL_REGIONS:
        if channel not in structured:
            print(f"[skip] {channel} not found in ROOT file")
            continue
        for scale in ("linear", "log"):
            fig = make_figure(channel, structured, scale=scale,
                              proc_scales=proc_scales)
            if fig is None:
                print(f"[skip] {channel}: no backgrounds found")
                continue
            for ext in ("pdf", "png"):
                out = os.path.join(args.outdir, f"{channel}_{scale}.{ext}")
                fig.savefig(out, bbox_inches="tight", dpi=150)
            plt.close(fig)
            print(f"Saved: {channel}_{scale}.pdf/png")

    print("Done.")


if __name__ == "__main__":
    main()
