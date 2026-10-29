#!/usr/bin/env python3
"""
Stacked prefit plots with systematic ratio panels below.

Reads histograms/histograms_kinematics.root (produced by make_histograms_kinematics.py).
ROOT file structure: {channel}__{var_name}/{process}[_{syst}Up/Down]

Produces one PDF+PNG per (channel, variable, linear/log).

SIGNAL AND BACKGROUND ARE SHOWN SEPARATELY (--components, default "both")
------------------------------------------------------------------------
Each systematic gets TWO curves per ratio panel: its effect on the summed
background and its effect on the summed signal, each divided by its OWN
nominal. A 5% shift on the signal therefore reads as 5%, instead of being
diluted by the much larger background it sits under -- which is what a single
curve normalised to the total would show.

The consequence of that choice: the two curves are NOT addable, and neither is
directly comparable to the grey band, which is the background total
uncertainty. They answer "how much does this systematic move each process",
not "how much does it move the stack".

  background   solid (Up) / dashed (Down)
  signal       dotted (Up) / dash-dot (Down)

Only Up curves are given legend entries, so the legend stays half the size;
the panel title states the Up/Down convention. A systematic with no signal
template (the nonprompt ones, for example) simply draws no signal curve.

For variables named "bdt_*", bins are rebinned onto explicit custom edges
defined in BDT_REBIN_EDGES (near the top of this file) -- e.g. to merge
low-stat tail bins by hand. Each original bin is reassigned to whichever
custom bin contains its center, applied consistently to every process,
systematic variation, and data_obs histogram in that (channel, var_name)
group so everything stays on the same binning. Variables with no entry in
BDT_REBIN_EDGES are left on their original binning.

Usage:
  python3 python/plot_syst_ratios.py \
      --rootfile histograms/histograms_kinematics.root \
      --outdir plots/syst_ratios
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
# Process groups (same style as plot_stacked.py)
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
      "DYJetsToLL",
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
     [ "TTbar",
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

SIGNAL_REGIONS = ["resolved_mu", "resolved_e", "boosted_mu", "boosted_e"]
SR_LABELS = {
    "resolved_mu": "Resolved $\\mu$",
    "resolved_e":  "Resolved $e$",
    "boosted_mu":  "Boosted $\\mu$",
    "boosted_e":   "Boosted $e$",
}
LUMI_LABEL   = "59.83 fb$^{-1}$ (13 TeV, 2018)"

VAR_XLABELS = {
    'pt_w_lep':                  r'$p_T^{W_{\ell\nu}}$ [GeV]',
    'eta_w_lep':                 r'$\eta^{W_{\ell\nu}}$',
    'mjj_vbs':                   r'$m_{jj}^{\rm VBS}$ [GeV]',
    'mjj_vbs_boost':             r'$m_{jj}^{\rm VBS}$ [GeV]',
    'deepmet_resolution_tune':   r'DeepMET$_{\rm res}$ [GeV]',
    'mt_w_lep_deepresolution':   r'$m_T^{W_{\ell\nu}}$ (DeepMET) [GeV]',
    'z_lep':                     r'$z_{\ell}$',
    'z_fat':                     r'$z_{\rm fat}$',
    'centrality_resolved':       r'Centrality',
    'centrality_boosted':        r'Centrality',
}

# ---------------------------------------------------------------------------
# Ratio-curve styling: colour = systematic, linestyle = (component, direction)
# ---------------------------------------------------------------------------
BKG_STYLE = {"up": "-",  "dn": "--",  "lw": 1.4, "alpha": 1.00, "tag": "bkg"}
SIG_STYLE = {"up": ":",  "dn": "-.",  "lw": 1.3, "alpha": 0.85, "tag": "sig"}

# Ratio panels never shrink below this, and never grow beyond RATIO_YLIM_MAX.
# Signal variations on a low-statistics template can be far larger than the
# background ones, and a fixed window silently clips them off the panel.
RATIO_YLIM_BASE = (0.75, 1.25)
RATIO_YLIM_MAX  = (0.30, 1.70)

# ---------------------------------------------------------------------------
# Variable-binning for BDT distributions
# ---------------------------------------------------------------------------
# Explicit target bin edges per BDT variable. Keys are matched first against
# "{channel}/{var_name}" (for a channel-specific override), then against
# just "{var_name}" (shared across all channels/categories).
#
# Edges don't need to exactly coincide with the original histogram's edges;
# each original bin is assigned to the new bin that contains its center.
# They DO need to span the same range as the original binning (start at the
# same low edge, end at the same high edge) or you'll silently lose/merge
# under/overflow-adjacent bins.
BDT_REBIN_EDGES = {
    "resolved_mu": [0.0, 0.4, 0.6, 0.8, 0.9, 1.0],
    "resolved_e":  [0.0, 0.4, 0.6, 0.8, 0.9, 1.0],
    "boosted_mu":  [0.0, 0.4, 0.6, 0.8, 0.9, 1.0],
    "boosted_e":   [0.0, 0.4, 0.6, 0.8, 0.9, 1.0],
}


def short_syst(name):
    """Collapse a repeated leading token run, for legend readability.

    The variation names repeat the weight name inside themselves --
    'sf_qgtagging_sf_qgtagging_herwig' -> 'sf_qgtagging_herwig'. Display only;
    nothing is looked up by the shortened name.
    """
    tok = name.split("_")
    for k in range(len(tok) // 2, 0, -1):
        if tok[:k] == tok[k:2 * k]:
            return "_".join(tok[k:])
    return name


def rebin_values(vals, variances, edges, new_edges):
    """
    Merge a fine-binned (vals, variances, edges) histogram onto `new_edges`.
    Each original bin's content is assigned to whichever new bin contains
    that original bin's center.
    """
    n_new = len(new_edges) - 1
    new_vals = np.zeros(n_new)
    new_vars = np.zeros(n_new)

    old_centers = 0.5 * (np.asarray(edges[:-1]) + np.asarray(edges[1:]))
    idx = np.searchsorted(new_edges, old_centers, side="right") - 1
    idx = np.clip(idx, 0, n_new - 1)

    for i in range(len(vals)):
        new_vals[idx[i]] += vals[i]
        new_vars[idx[i]] += variances[i]

    return new_vals, new_vars


def apply_bdt_rebinning(data, edges_map):
    """
    For every (channel, var_name) where var_name starts with 'bdt_' and a
    matching entry exists in `edges_map` (checked as "channel/var_name" then
    "var_name"), rebin every process/systematic/data histogram in that group
    onto the custom edges. Variables with no matching entry are left on
    their original binning (with a printed note).

    Mutates and returns `data` in place.
    """
    for channel, by_var in data.items():
        for var_name, structured_var in by_var.items():
            if not var_name.startswith("bdt_"):
                continue

            key = f"{channel}/{var_name}"
            if key in edges_map:
                new_edges = np.asarray(edges_map[key], dtype=float)
            elif var_name in edges_map:
                new_edges = np.asarray(edges_map[var_name], dtype=float)
            else:
                print(f"[rebin] {channel}/{var_name}: no custom edges defined, "
                      f"keeping original binning")
                continue

            # figure out original edges from any nominal histogram present
            edges = None
            for proc, pvars in structured_var.items():
                if "nominal" in pvars:
                    edges = pvars["nominal"][2]
                    break
            if edges is None:
                continue

            for proc, pvars in structured_var.items():
                for syst_key, (v, var_, e) in list(pvars.items()):
                    nv, nvar = rebin_values(v, var_, e, new_edges)
                    pvars[syst_key] = (nv, nvar, new_edges)

            n_old, n_new = len(edges) - 1, len(new_edges) - 1
            print(f"[rebin] {channel}/{var_name}: {n_old} -> {n_new} bins (custom edges)")


# ---------------------------------------------------------------------------
# 6 ratio panels: [(panel_label, [(syst_name, color), ...]), ...]
# ---------------------------------------------------------------------------
RATIO_PANELS = [
    ("AK4 JES (year-correlated)", [
        ("AK4PFchs_JES_Regrouped_FlavorQCD",    "#1f77b4"),
        ("AK4PFchs_JES_Regrouped_RelativeBal",  "#ff7f0e"),
        ("AK4PFchs_JES_Regrouped_HF",           "#2ca02c"),
        ("AK4PFchs_JES_Regrouped_BBEC1",        "#d62728"),
        ("AK4PFchs_JES_Regrouped_EC2",          "#9467bd"),
        ("AK4PFchs_JES_Regrouped_Absolute",     "#8c564b"),
    ]),
    ("AK4 JES (2018) + JER", [
        ("AK4PFchs_JES_Regrouped_Absolute_2018",       "#1f77b4"),
        ("AK4PFchs_JES_Regrouped_HF_2018",             "#ff7f0e"),
        ("AK4PFchs_JES_Regrouped_EC2_2018",            "#2ca02c"),
        ("AK4PFchs_JES_Regrouped_RelativeSample_2018", "#d62728"),
        ("AK4PFchs_JES_Regrouped_BBEC1_2018",          "#9467bd"),
        ("AK4PFchs_JER",                     "#000000"),
    ]),
    ("Nonprompt uncert", [
        ("electron_inverttight_to_fake", "#1f77b4"),
        ("muon_inverttight_to_fake", "#ff7f0e"),
        ("electron_inverttight_to_fake_boosted", "#2ca02c"),
        ("muon_inverttight_to_fake_boosted", "#3273a8")

    ]),
    ("PS uncert", [
        ("sf_partonshower_fsr", "#1f77b4"),
        ("sf_partonshower_isr", "#ff7f0e"),
    ]),
    ("AK8 JES (year-correlated)", [
        ("AK8PFPuppi_JES_Regrouped_FlavorQCD",    "#1f77b4"),
        ("AK8PFPuppi_JES_Regrouped_RelativeBal",  "#ff7f0e"),
        ("AK8PFPuppi_JES_Regrouped_HF",           "#2ca02c"),
        ("AK8PFPuppi_JES_Regrouped_BBEC1",        "#d62728"),
        ("AK8PFPuppi_JES_Regrouped_EC2",          "#9467bd"),
        ("AK8PFPuppi_JES_Regrouped_Absolute",     "#8c564b"),
    ]),
    ("AK8 JES (2018) + JER", [
        ("AK8PFPuppi_JES_Regrouped_Absolute_2018",       "#1f77b4"),
        ("AK8PFPuppi_JES_Regrouped_HF_2018",             "#ff7f0e"),
        ("AK8PFPuppi_JES_Regrouped_EC2_2018",            "#2ca02c"),
        ("AK8PFPuppi_JES_Regrouped_RelativeSample_2018", "#d62728"),
        ("AK8PFPuppi_JES_Regrouped_BBEC1_2018",          "#9467bd"),
        ("AK8PFPuppi_JER",                     "#000000"),
    ]),
]


# ---------------------------------------------------------------------------
# ROOT file reading
# ---------------------------------------------------------------------------
def read_root(path):
    """
    Returns: data[channel][var_name][proc][var_key] = (vals, variances, edges)
    """
    raw = {}  # raw[(channel, var_name, proc_key)] = (vals, var, edges)

    with uproot.open(path) as f:
        for key in f.keys(cycle=False):
            parts = key.split("/")
            if len(parts) != 2:
                continue
            dir_name, proc_key = parts
            if "__" not in dir_name:
                continue
            channel, var_name = dir_name.split("__", 1)
            h    = f[key]
            vals = h.values()
            var  = h.variances() if h.variances() is not None else np.zeros_like(vals)
            _, edges = h.to_numpy()
            raw[(channel, var_name, proc_key)] = (vals, var, edges)

    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    # First pass: collect nominal proc names per (channel, var_name)
    cv_nom_procs = defaultdict(set)  # (channel, var_name) -> set of nominal proc names
    for (channel, var_name, proc_key) in raw:
        if proc_key == "data_obs":
            cv_nom_procs[(channel, var_name)].add("data_obs")
        elif not proc_key.endswith("Up") and not proc_key.endswith("Down"):
            cv_nom_procs[(channel, var_name)].add(proc_key)

    # Second pass: attach variations
    for (channel, var_name, proc_key), vh in raw.items():
        cv = (channel, var_name)
        nom_procs = sorted(cv_nom_procs[cv], key=len, reverse=True)

        if proc_key in cv_nom_procs[cv]:
            data[channel][var_name][proc_key]["nominal"] = vh
        else:
            match = next((p for p in nom_procs if proc_key.startswith(p + "_")), None)
            if match:
                syst_key = proc_key[len(match) + 1:]
                data[channel][var_name][match][syst_key] = vh
            else:
                data[channel][var_name][proc_key]["nominal"] = vh

    return data


# ---------------------------------------------------------------------------
# Helpers (same logic as plot_stacked.py)
# ---------------------------------------------------------------------------
def merge_group(structured_var, proc_list):
    """Returns (nom_vals, nom_var, edges, syst_deltas)."""
    nom_vals = nom_var = edges = None
    syst_deltas = defaultdict(lambda: [None, None])

    for proc in proc_list:
        if proc not in structured_var:
            continue
        pvars = structured_var[proc]
        if "nominal" not in pvars:
            continue
        v, var, e = pvars["nominal"]
        v   = np.maximum(v, 0.0)
        var = np.maximum(var, 0.0)
        if nom_vals is None:
            nom_vals = v.copy(); nom_var = var.copy(); edges = e
        else:
            nom_vals += v; nom_var += var

        for key in pvars:
            if key in ("nominal",):
                continue
            if key.endswith("Up"):
                syst = key[:-2]
                vup = np.maximum(pvars[key][0], 0.0)
                delta = vup - v
                syst_deltas[syst][0] = delta if syst_deltas[syst][0] is None \
                                       else syst_deltas[syst][0] + delta
            elif key.endswith("Down"):
                syst = key[:-4]
                vdn = np.maximum(pvars[key][0], 0.0)
                delta = v - vdn
                syst_deltas[syst][1] = delta if syst_deltas[syst][1] is None \
                                       else syst_deltas[syst][1] + delta

    return nom_vals, nom_var, edges, syst_deltas


def total_uncertainty(nom_vals, nom_var, all_syst_deltas_up, all_syst_deltas_dn):
    stat = np.sqrt(np.maximum(nom_var, 0.0))
    up2 = np.zeros_like(stat); dn2 = np.zeros_like(stat)
    for syst in set(list(all_syst_deltas_up) + list(all_syst_deltas_dn)):
        d_up = all_syst_deltas_up.get(syst)
        d_dn = all_syst_deltas_dn.get(syst)
        if d_up is not None: up2 += d_up ** 2
        if d_dn is not None: dn2 += d_dn ** 2
    return np.sqrt(stat**2 + up2), np.sqrt(stat**2 + dn2)


def make_figure(channel, var_name, structured_var, scale="linear",
                components="both"):
    """Build one figure: stacked main panel + systematic ratio panels."""
    bkg_vals = []; bkg_colors = []; bkg_labels = []
    total_nom = total_var = edges = None
    all_up = defaultdict(lambda: None)
    all_dn = defaultdict(lambda: None)

    for label, color, procs in GROUPS:
        v, var, e, syst_d = merge_group(structured_var, procs)
        if v is None:
            continue
        if edges is None:
            edges = e
        bkg_vals.append(v); bkg_colors.append(color); bkg_labels.append(label)
        total_nom = v if total_nom is None else total_nom + v
        total_var = var if total_var is None else total_var + var
        for syst, (d_up, d_dn) in syst_d.items():
            if d_up is not None:
                all_up[syst] = d_up if all_up[syst] is None else all_up[syst] + d_up
            if d_dn is not None:
                all_dn[syst] = d_dn if all_dn[syst] is None else all_dn[syst] + d_dn

    if total_nom is None or edges is None:
        return None

    unc_up, unc_dn = total_uncertainty(total_nom, total_var, all_up, all_dn)

    # Signal, now through merge_group so its systematic deltas are collected
    # too -- the ratio panels need them, not just the nominal for the stack.
    sig_procs = [p for p in structured_var
                 if any(p.startswith(pat) for pat in SIGNAL_PROCS_RE)]
    sig_nom, _sig_var, _sig_edges, sig_syst_d = merge_group(structured_var, sig_procs)

    sig_draw_scale = 1.0 if scale == "log" else SIGNAL_SCALE
    sig_label      = "Signal" if scale == "log" else SIGNAL_LABEL

    centers = 0.5 * (edges[:-1] + edges[1:])
    widths  = edges[1:] - edges[:-1]

    # ---------------------------------------------------------------------------
    # Layout: 1 main + N ratio panels
    # ---------------------------------------------------------------------------
    n_panels = len(RATIO_PANELS)
    height_ratios = [4] + [1.2] * n_panels
    fig, axes = plt.subplots(
        1 + n_panels, 1,
        figsize=(8, 4 + 2.0 * n_panels),
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.08},
        sharex=True,
    )
    ax   = axes[0]
    ax_rs = axes[1:]

    # --- main panel ---
    hep.histplot(bkg_vals, edges, stack=True, histtype="fill",
                 color=bkg_colors, label=bkg_labels, ax=ax, linewidth=0)
    ax.bar(centers, unc_up + unc_dn, width=widths, bottom=total_nom - unc_dn,
           color="gray", alpha=0.4, linewidth=0, label="Unc. (stat$\\oplus$syst)", zorder=4)
    if sig_nom is not None:
        hep.histplot(sig_nom * sig_draw_scale, edges, histtype="step",
                     color=SIGNAL_COLOR, linewidth=2, label=sig_label, ax=ax, zorder=5)
    ax.set_ylabel("Events / bin", fontsize=13)
    ax.set_xlim(edges[0], edges[-1])
    ax.tick_params(labelbottom=False, labelsize=11)
    if scale == "log":
        ax.set_yscale("log")
        ax.set_ylim(0.1, total_nom.max() * 500)
    else:
        ax.set_ylim(0, total_nom.max() * 1.6)
    ax.legend(ncol=2, fontsize=8, loc="upper right", framealpha=0.8)
    hep.cms.label("Work in Progress", data=False, rlabel=LUMI_LABEL, ax=ax, fontsize=11)
    ax.text(0.02, 0.96, SR_LABELS.get(channel, channel),
            transform=ax.transAxes, fontsize=10, va="top")

    # --- ratio panels ---
    # The grey band is the BACKGROUND total uncertainty. Signal curves are
    # ratios to the signal's own nominal, so they are not comparable to it.
    unc_up_rel = np.where(total_nom > 0, unc_up / total_nom, 0.0)
    unc_dn_rel = np.where(total_nom > 0, unc_dn / total_nom, 0.0)

    want_bkg = components in ("both", "bkg")
    want_sig = components in ("both", "sig") and sig_nom is not None \
        and float(np.sum(sig_nom)) > 0.0

    for i, (panel_label, syst_list) in enumerate(RATIO_PANELS):
        ax_r = ax_rs[i]

        ax_r.bar(centers, unc_up_rel + unc_dn_rel, width=widths,
                 bottom=1.0 - unc_dn_rel, color="gray", alpha=0.35, linewidth=0, zorder=1)
        ax_r.axhline(1.0, color="black", lw=0.8)

        legend_handles = []
        drawn = []          # every ratio array drawn, for the y-range

        for syst_name, color in syst_list:
            series = []
            if want_bkg:
                series.append((total_nom, all_up.get(syst_name),
                               all_dn.get(syst_name), BKG_STYLE))
            if want_sig:
                s_up, s_dn = sig_syst_d.get(syst_name, (None, None))
                series.append((sig_nom, s_up, s_dn, SIG_STYLE))

            for ref, d_up, d_dn, style in series:
                if ref is None:
                    continue
                if d_up is not None:
                    r = np.where(ref > 0, (ref + d_up) / ref, 1.0)
                    line, = ax_r.step(edges, np.append(r, r[-1]), where="post",
                                      color=color, lw=style["lw"],
                                      linestyle=style["up"], alpha=style["alpha"],
                                      zorder=3)
                    legend_handles.append(
                        (line, f"{short_syst(syst_name)} [{style['tag']}]"))
                    drawn.append(r)
                if d_dn is not None:
                    r = np.where(ref > 0, (ref - d_dn) / ref, 1.0)
                    ax_r.step(edges, np.append(r, r[-1]), where="post",
                              color=color, lw=style["lw"],
                              linestyle=style["dn"], alpha=style["alpha"],
                              zorder=3)
                    drawn.append(r)

        # Expand the window to hold what was actually drawn, within a cap, so a
        # large signal variation is visible rather than silently clipped.
        lo, hi = RATIO_YLIM_BASE
        clipped = False
        if drawn:
            stacked = np.concatenate(drawn)
            stacked = stacked[np.isfinite(stacked)]
            if stacked.size:
                dmin, dmax = float(stacked.min()), float(stacked.max())
                lo = min(lo, max(RATIO_YLIM_MAX[0], dmin - 0.02))
                hi = max(hi, min(RATIO_YLIM_MAX[1], dmax + 0.02))
                clipped = dmin < RATIO_YLIM_MAX[0] or dmax > RATIO_YLIM_MAX[1]
        ax_r.set_ylim(lo, hi)

        ax_r.set_ylabel("Ratio", fontsize=8, labelpad=2)
        ax_r.tick_params(labelsize=8)
        ax_r.axhline(1.1, color="gray", lw=0.4, ls=":")
        ax_r.axhline(0.9, color="gray", lw=0.4, ls=":")
        ax_r.set_xlim(edges[0], edges[-1])

        if legend_handles:
            title = panel_label
            if want_bkg and want_sig:
                title += "\nbkg solid/dashed, sig dotted/dash-dot (Up/Down)"
            else:
                title += "  (solid = Up, dashed = Down)"
            if clipped:
                title += "\n[some curves clipped]"
            handles, labels = zip(*legend_handles)
            ax_r.legend(handles, labels, fontsize=6, ncol=2,
                        loc="upper right", framealpha=0.7,
                        title=title, title_fontsize=6,
                        handlelength=1.8, borderpad=0.4, labelspacing=0.2)

    # x-axis label on last panel
    x_label = VAR_XLABELS.get(var_name, var_name)
    if var_name.startswith("bdt_"):
        x_label = "BDT score"
    ax_rs[-1].set_xlabel(x_label, fontsize=12)

    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rootfile",   default="histograms/histograms_kinematics.root")
    parser.add_argument("--outdir",     default="plots/syst_ratios")
    parser.add_argument("--scales",     default="linear,log",
                        help="Comma-separated scales to produce (linear,log)")
    parser.add_argument("--components", choices=["both", "bkg", "sig"],
                        default="both",
                        help="Which ratio curves to draw per systematic. "
                             "'both' (default) draws the background and the "
                             "signal separately, each relative to its own "
                             "nominal; 'bkg' is the previous behaviour.")
    args = parser.parse_args()

    scales = [s.strip() for s in args.scales.split(",")]

    os.makedirs(args.outdir, exist_ok=True)
    data = read_root(args.rootfile)
    apply_bdt_rebinning(data, BDT_REBIN_EDGES)

    for channel in SIGNAL_REGIONS:
        if channel not in data:
            print(f"[skip] {channel} not in ROOT file")
            continue
        for var_name in sorted(data[channel].keys()):
            structured_var = data[channel][var_name]
            for scale in scales:
                fig = make_figure(channel, var_name, structured_var, scale=scale,
                                  components=args.components)
                if fig is None:
                    print(f"[skip] {channel}/{var_name}: no backgrounds")
                    continue
                for ext in ("pdf", "png"):
                    out = os.path.join(args.outdir, f"{channel}__{var_name}_{scale}.{ext}")
                    fig.savefig(out, bbox_inches="tight", dpi=150)
                plt.close(fig)
                print(f"Saved: {channel}__{var_name}_{scale}.pdf/png")

    print("Done.")


if __name__ == "__main__":
    main()