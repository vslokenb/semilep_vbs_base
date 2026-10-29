#!/usr/bin/env python3
"""
CR closure check: does the postfit nuisance configuration also describe the CR
*shape*, which the fit never saw?

The CRs enter the fit as single counting bins, so the fit constrains only their
total yield. This script rebuilds each CR at full BDT binning straight from the
coffea files and compares real CR data against the MC prediction built two ways:

  pre-fit   nuisances at 0
  post-fit  nuisances at their fitted pulls, applied by linear (chord)
            interpolation between the Down / Nominal / Up templates

WHICH NORMALIZATION THE PRE-FIT PANEL USES (--prefit-norms)
-----------------------------------------------------------
Default "postfit": the pre-fit panel uses the SAME norm_* and r as the post-fit
panel, and differs from it ONLY by having every nuisance at 0. That is what
makes the two panels comparable -- anything that changes between them is the
shape effect of the nuisance pulls, which is the thing being checked.

"cr-only" instead takes norm_* from the step-5c CR-only fit. Those numbers can
sit ~20% away from the full fit's, because step 5c freezes
allConstrainedNuisances, so its rateParam absorbs the entire data/MC difference
on its own, while in the full fit that difference is shared between the
rateParam and the nuisance pulls. Step 5c is also per-category, whereas
norm_top_resolved in the full fit is shared across resolved_mu and resolved_e.
With that option the panels differ by normalization AND nuisances at once, so a
shape change cannot be attributed to either.

"one" leaves every normalization at 1.0: the raw MC prediction. Useful for
seeing the total size of the correction, not for isolating the nuisances.

The post-fit band comes from propagating the fit's postfit covariance through
the same linear shape model (Var = J . Cov . J^T), not from combine's
shapes_fit_s -- which for a collapsed CR has no shape-level uncertainty to give.

Normalization: the coffea histograms are already lumi x cross-section weighted
(make_histograms.py applies no scaling of its own), so the only scalings applied
here are the rateParams and r, both shown in the legend.

Run extract_postfit_params.py first (CMSSW/PyROOT), then this (apptainer):

  python3 python/extract_postfit_params.py \
      --fitfile plots_cr/crdata_srasimov/fitdiagnostics/fitDiagnostics_hybrid.root \
      --region boosted_mu \
      --output plots_cr/crdata_srasimov/cr_closure/postfit_params_boosted_mu.json

  python3 python/plot_cr_closure.py \
      --config config/inputs_2018.yaml \
      --binning config/binning_example.yaml \
      --postfit-json plots_cr/crdata_srasimov/cr_closure/postfit_params_boosted_mu.json \
      --region boosted_mu \
      --outdir plots_cr/crdata_srasimov/cr_closure
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep

sys.path.insert(0, str(Path(__file__).parent))
from make_histograms import (
    build_process_hists, load_process_hists, merge_processes,
    rebin_hists_to_edges, rebin_hists_dict,
    coffea_cat, bdt_var_for_region, is_electron_region,
    TTBAR_MERGED_NAME, TTBAR_PROCESSES,
    WJET_MERGED_NAME, WJET_PROCESSES,
    DY_MERGED_NAME, DY_PROCESSES,
    ELECTRON_DATA_PROCESSES, MUON_DATA_PROCESSES,
    SIGNAL_REGIONS, CR_SUFFIXES, REBIN_TARGETS,
)
from make_datacards import mc_group, MC_GROUP_RULES, MC_GROUP_DEFAULT

hep.style.use("CMS")

SIGNAL_KEY = 'signal'
SIGNAL_MERGED_NAME = 'VBS'
LUMI_LABEL = "59.83 fb$^{-1}$ (13 TeV, 2018)"

# Process names as they appear AFTER make_histograms.py's merging: TTbar, WJets
# and DY are merged; single top and the diboson samples keep individual names.
#
# VBS first: mplhep stacks list[0] from the baseline upward, and the signal is
# by far the smallest yield in any CR -- placed on top it would be an invisible
# sliver on a log axis.
CR_GROUPS = [
    ("VBS (signal)", "#e42536", [SIGNAL_MERGED_NAME]),
    ("Diboson, Triboson", "#92dadd",
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
    ("$V\\gamma$, VBF", "#b9ac70",
     ["GluGluWWToLNuQQ_TuneCP5_13TeV_madgraph-pythia8",
      "ZGToLLG_01J_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8"]),
    ("DY", "#f89c20", [DY_MERGED_NAME]),
    ("Single Top", "#94a4a2",
     ["ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8",
      "ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8",
      "ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8",
      "ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8",
      "ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8"]),
    ("Nonprompt", "#a96b59", ["nonprompt"]),
    ("$t\\bar{t}$", "#964a8b", [TTBAR_MERGED_NAME]),
    ("$W\\rightarrow l\\nu$", "#5790fc", [WJET_MERGED_NAME]),
]

CR_TITLES = {'ttbar_cr': "$t\\bar{t}$ CR", 'w_cr': "$W$+jets CR"}
SR_TITLES = {'resolved_mu': "Resolved $\\mu$", 'resolved_e': "Resolved $e$",
             'boosted_mu': "Boosted $\\mu$", 'boosted_e': "Boosted $e$"}


def cr_label(cr_suffix, sr):
    return f"{CR_TITLES.get(cr_suffix, cr_suffix)}, {SR_TITLES.get(sr, sr)}"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_cr_hists(config, sr, cr_suffix, include_signal=True):
    """Un-collapsed CR histograms straight from coffea, at native BDT binning."""
    region = f'{cr_suffix}_{sr}'
    cat = coffea_cat(region)          # honours CATEGORY_OVERRIDES (w_cr_mu ...)
    var_name = bdt_var_for_region(region)
    electron = is_electron_region(region)

    years = list(config['years'].keys())
    yc = config['years'][years[0]]
    ch = {}

    for proc, pcfg in yc.get('background', {}).items():
        flist = pcfg.get('files', [pcfg.get('file')])
        h = build_process_hists(flist, proc, var_name, cat)
        if h:
            ch[proc] = h

    # make_histograms.py deliberately omits signal from the CR templates, but
    # for a closure plot the few signal events are part of the prediction the
    # data should be compared against.
    if include_signal:
        sig = {}
        for proc, pcfg in yc.get('signal', {}).items():
            flist = pcfg.get('files', [pcfg.get('file')])
            h = build_process_hists(flist, proc, var_name, cat, is_signal=True)
            if h:
                sig[proc] = h
        if sig:
            ch.update(sig)
            merge_processes(ch, set(sig.keys()), SIGNAL_MERGED_NAME)

    data_acc = None
    for proc, pcfg in yc.get('data', {}).items():
        if electron and proc not in ELECTRON_DATA_PROCESSES:
            continue
        if not electron and proc not in MUON_DATA_PROCESSES:
            continue
        flist = pcfg.get('files', [pcfg.get('file')])
        raw = load_process_hists(flist, proc, var_name, cat)
        if not raw or 'nominal' not in raw:
            continue
        v, e, edges = raw['nominal']
        if data_acc is None:
            data_acc = [v.copy(), e.copy(), edges]
        else:
            data_acc[0] += v
            data_acc[1] += e
    if data_acc is not None:
        ch['data_obs'] = {'nominal': tuple(data_acc)}

    merge_processes(ch, TTBAR_PROCESSES, TTBAR_MERGED_NAME)
    merge_processes(ch, WJET_PROCESSES, WJET_MERGED_NAME)
    merge_processes(ch, DY_PROCESSES, DY_MERGED_NAME)
    return ch


# ---------------------------------------------------------------------------
# Fitted-name -> template-name resolution
#
# make_datacards.py decorrelates nuisances by RENAMING them: per MC group for
# the parton shower / scale families (sf_partonshower_isr_TTbar), and per year
# in multi-year mode (sf_btag_hf_2017; _yr2017 for the Regrouped JES base
# components whose _2017 name is taken by the year-specific component). The
# histogram file keeps the RAW template names, so every fitted name must be
# mapped back before any template lookup -- otherwise the pulls and the
# covariance band silently do nothing.
# ---------------------------------------------------------------------------
_YEAR_TAG_RE = re.compile(r'_(yr)?20\d{2}(_(Pre|Post)VFP)?$')
_MC_GROUP_TAGS = tuple(name for name, _ in MC_GROUP_RULES) + (MC_GROUP_DEFAULT,)


def channel_raw_systs(ch_hists):
    """Raw template systematic names present anywhere in this channel."""
    out = set()
    for pvars in ch_hists.values():
        for key in pvars:
            if key.endswith('Up'):
                out.add(key[:-2])
    return out


def resolve_template_syst(fit_name, raw_names):
    """(raw template name, MC group or None) for a fitted nuisance, or None.

    Tries the exact name first -- the year-specific JES families carry their
    year natively -- then peels the decorrelation tags in reverse order of
    application: the year tag, then the per-MC-group tag, then the year tag
    again. Returns None for parameters with no template here (autoMCStats,
    nuisances of other channels).
    """
    cand, group = fit_name, None
    for _ in range(4):
        if cand in raw_names:
            return cand, group
        stripped = _YEAR_TAG_RE.sub('', cand)
        if stripped != cand:
            cand = stripped
            continue
        for g in _MC_GROUP_TAGS:
            if cand.endswith(f'_{g}'):
                group, cand = g, cand[: -len(g) - 1]
                break
        else:
            return None
    return (cand, group) if cand in raw_names else None


def rateparam_process(name):
    """(merged process, topology or None) when `name` is a normalization
    rateParam: either extractor-mapped ('TTbar'/'WJets') or a raw fit name
    (norm_top_boosted_2017, norm_wjet_resolved, norm_top, ...)."""
    if name in (TTBAR_MERGED_NAME, WJET_MERGED_NAME):
        return name, None
    m = re.match(r'^norm_(top|wjet)(?:_(resolved|boosted))?(?:_[A-Za-z0-9]+)*$',
                 name)
    if m:
        return (TTBAR_MERGED_NAME if m.group(1) == 'top'
                else WJET_MERGED_NAME), m.group(2)
    return None


def read_cr_only_norms(path, sr):
    """norm_top / norm_wjet for this SR from the step-5c cr_only_norms.txt."""
    out = {}
    if not path or not os.path.exists(path):
        # Only needed for --prefit-norms cr-only and for the comparison
        # printout, so this is a note rather than a warning.
        print(f'  [note] no CR-only norms file ({path}); '
              f'the CR-only vs full-fit comparison will be skipped')
        return out
    topology = 'boosted' if sr.startswith('boosted') else 'resolved'
    wanted = {TTBAR_MERGED_NAME: (f'norm_top_{sr}', f'norm_top_{topology}',
                                  'norm_top'),
              WJET_MERGED_NAME: (f'norm_wjet_{sr}', f'norm_wjet_{topology}',
                                 'norm_wjet')}
    seen = {}
    for line in open(path):
        tok = line.split()
        for i, t in enumerate(tok):
            if t.startswith('norm_') and i + 1 < len(tok):
                try:
                    seen.setdefault(t, float(tok[i + 1]))
                except ValueError:
                    pass
    # Year-decorrelated cards suffix the params (norm_top_resolved_2017);
    # fall back to the year-stripped name when the exact one is absent.
    lookup = dict(seen)
    for k, v in seen.items():
        lookup.setdefault(_YEAR_TAG_RE.sub('', k), v)
    for proc, names in wanted.items():
        hit = next((n for n in names if n in lookup), None)
        if hit:
            out[proc] = lookup[hit]
        else:
            print(f'  [warn] {path} has no rateParam for {proc}')
    return out


# ---------------------------------------------------------------------------
# Postfit morphing
# ---------------------------------------------------------------------------

def apply_postfit_shape(ch_hists, theta):
    """Morph nominal by the fitted pulls, linear between Down/Nominal/Up.

    Exact at theta = 0 and +/-1, which is where the templates are defined.
    Fitted names are mapped back to the raw template names, and a
    group-tagged pull (sf_partonshower_isr_TTbar_2017, ...) is applied only
    to the processes of that MC group.
    """
    raw_names = channel_raw_systs(ch_hists)
    resolved, unmatched = {}, []
    for syst, th in theta.items():
        if th == 0:
            continue
        res = resolve_template_syst(syst, raw_names)
        if res is None:
            if not syst.startswith('prop_bin'):
                unmatched.append(syst)
            continue
        resolved[syst] = res
    if unmatched:
        print(f'  [note] {len(unmatched)} fitted nuisance(s) have no template '
              f'in this channel (e.g. {unmatched[0]}); their pulls are not '
              f'applied here')

    out = {}
    for proc, pvars in ch_hists.items():
        if proc == 'data_obs' or 'nominal' not in pvars:
            out[proc] = pvars
            continue
        nom_v, nom_e, edges = pvars['nominal']
        new_v = nom_v.copy()
        pgroup = mc_group(proc)
        for syst, (base, group) in resolved.items():
            th = theta[syst]
            if group is not None and group != pgroup:
                continue
            if th > 0 and f'{base}Up' in pvars:
                new_v = new_v + th * (np.maximum(pvars[f'{base}Up'][0], 0.0) - nom_v)
            elif th < 0 and f'{base}Down' in pvars:
                new_v = new_v + th * (nom_v - np.maximum(pvars[f'{base}Down'][0], 0.0))
        out[proc] = dict(pvars)
        out[proc]['nominal'] = (np.maximum(new_v, 0.0), nom_e, edges)
    return out


def _merge_group(ch_hists, proc_list, proc_scales):
    """Sum a group's nominal (scaled) and collect its per-syst up/down deltas."""
    nom = var = edges = None
    syst_d = defaultdict(lambda: [None, None])
    for proc in proc_list:
        if proc not in ch_hists or 'nominal' not in ch_hists[proc]:
            continue
        v, ve, e = ch_hists[proc]['nominal']
        sf = proc_scales.get(proc, 1.0)
        v = np.maximum(v, 0.0) * sf
        ve = ve * sf * sf
        if nom is None:
            nom, var, edges = v.copy(), ve.copy(), e
        else:
            nom += v
            var += ve
        for key in ch_hists[proc]:
            if key in ('nominal', '_raw'):
                continue
            if key.endswith('Up'):
                syst = key[:-2]
                d = np.maximum(ch_hists[proc][key][0], 0.0) * sf - v
                syst_d[syst][0] = d if syst_d[syst][0] is None else syst_d[syst][0] + d
            elif key.endswith('Down'):
                syst = key[:-4]
                d = v - np.maximum(ch_hists[proc][key][0], 0.0) * sf
                syst_d[syst][1] = d if syst_d[syst][1] is None else syst_d[syst][1] + d
    return nom, var, edges, syst_d


def build_jacobian(ch_hists, cov_names, proc_scales, topology=None):
    """(nominal total, J) with J[:, j] = d(total)/d(parameter j).

    The up/down template difference for a shape nuisance (fitted names
    resolved to raw template names; a group-tagged nuisance differentiates
    only its own MC group's templates), and the process's own unscaled yield
    for a rateParam, since the prediction is linear in it. A rateParam of the
    other topology does not enter this channel and its column stays 0.
    """
    all_procs = [p for _, _, procs in CR_GROUPS for p in procs]
    nom, _v, _e, syst_d_all = _merge_group(ch_hists, all_procs, proc_scales)
    if nom is None or not cov_names:
        return None, None

    raw_names = channel_raw_systs(ch_hists)
    deltas = {None: syst_d_all}

    def syst_d_for(group):
        if group not in deltas:
            procs = [p for p in all_procs if mc_group(p) == group]
            deltas[group] = _merge_group(ch_hists, procs, proc_scales)[3]
        return deltas[group]

    n_bins, n_par = len(nom), len(cov_names)
    J = np.zeros((n_bins, n_par))
    for j, name in enumerate(cov_names):
        rp = rateparam_process(name)
        if rp is not None:
            proc, topo = rp
            if topo is not None and topology is not None and topo != topology:
                continue
            g, _, _, _ = _merge_group(ch_hists, [proc], proc_scales)
            if g is not None:
                # d(total)/d(norm) = yield / norm, since yield already includes it
                sf = proc_scales.get(proc, 1.0)
                J[:, j] = g / sf if sf else g
            continue
        res = resolve_template_syst(name, raw_names)
        if res is None:
            continue  # autoMCStats / absent here; column stays 0
        base, group = res
        du, dd = syst_d_for(group).get(base, (None, None))
        if du is not None and dd is not None:
            J[:, j] = 0.5 * (du + dd)
        elif du is not None:
            J[:, j] = du
        elif dd is not None:
            J[:, j] = dd
    return nom, J


def covariance_band(ch_hists, cov_names, cov, proc_scales, topology=None):
    """Propagate the postfit covariance through the linear shape model."""
    nom, J = build_jacobian(ch_hists, cov_names, proc_scales, topology)
    if J is None:
        return None
    var = np.einsum('bi,ij,bj->b', J, np.asarray(cov, dtype=float), J)
    return np.sqrt(np.maximum(var, 0.0))


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_figure(ch_hists, cr_suffix, sr, proc_scales, subtitle,
                scale='linear', cov_unc=None, xlabel='BDT score'):
    vals, colors, labels = [], [], []
    total = tot_var = edges = None

    for base_label, color, procs in CR_GROUPS:
        label = base_label
        if TTBAR_MERGED_NAME in procs:
            label = f"$t\\bar{{t}}$ $\\times$ {proc_scales.get(TTBAR_MERGED_NAME, 1.0):.3f}"
        elif WJET_MERGED_NAME in procs:
            label = f"$W\\rightarrow l\\nu$ $\\times$ {proc_scales.get(WJET_MERGED_NAME, 1.0):.3f}"
        elif SIGNAL_MERGED_NAME in procs:
            label = f"VBS (r = {proc_scales.get(SIGNAL_MERGED_NAME, 1.0):.2f})"
        v, ve, e, _ = _merge_group(ch_hists, procs, proc_scales)
        if v is None:
            continue
        if edges is None:
            edges = e
        vals.append(v)
        colors.append(color)
        labels.append(label)
        total = v.copy() if total is None else total + v
        tot_var = ve.copy() if tot_var is None else tot_var + ve

    if total is None:
        return None

    data = derr = None
    if 'data_obs' in ch_hists and 'nominal' in ch_hists['data_obs']:
        dv, de, _ = ch_hists['data_obs']['nominal']
        data = dv.copy()
        derr = np.sqrt(np.maximum(de, np.maximum(dv, 0.0)))

    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = edges[1:] - edges[:-1]

    fig, (ax, ax_r) = plt.subplots(
        2, 1, figsize=(8, 7),
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})
    hep.histplot(vals, edges, stack=True, histtype='fill',
                 color=colors, label=labels, ax=ax, linewidth=0)

    stat = np.sqrt(np.maximum(tot_var, 0.0))
    if cov_unc is not None:
        unc = np.sqrt(stat ** 2 + cov_unc ** 2)
        band = 'Post-fit unc. (MC stat $\\oplus$ covariance)'
    else:
        unc = stat
        band = 'MC stat. unc.'
    ax.bar(centers, 2 * unc, width=widths, bottom=total - unc,
           color='gray', alpha=0.4, linewidth=0, label=band, zorder=4)

    if data is not None:
        ax.errorbar(centers, data, yerr=derr, fmt='ko', markersize=4,
                    linewidth=1, label='Data', zorder=6)

    ax.set_ylabel('Events / bin', fontsize=13)
    ax.set_xlim(edges[0], edges[-1])
    ax.tick_params(labelbottom=False, labelsize=11)
    ymax = max(np.max(total), np.max(data) if data is not None else 0)
    if scale == 'log':
        ax.set_yscale('log')
        ax.set_ylim(max(0.01, ymax * 1e-5), ymax * 50)
    else:
        ax.set_ylim(0, ymax * 1.6)
    ax.legend(ncol=2, fontsize=9, loc='upper right', framealpha=0.8)
    hep.cms.label('Work in Progress', data=True, rlabel=LUMI_LABEL, ax=ax,
                  fontsize=12)
    ax.text(0.02, 0.96, f'{cr_label(cr_suffix, sr)}\n{subtitle}',
            transform=ax.transAxes, fontsize=11, va='top')

    if data is not None:
        ratio = np.where(total > 0, data / total, np.nan)
        rerr = np.where(total > 0, derr / total, np.nan)
        ax_r.errorbar(centers, ratio, yerr=rerr, fmt='ko', markersize=4,
                      linewidth=1, zorder=3)
    rel = np.where(total > 0, unc / total, 0.0)
    ax_r.bar(centers, 2 * rel, width=widths, bottom=1.0 - rel,
             color='gray', alpha=0.4, linewidth=0, zorder=2)
    ax_r.axhline(1.0, color='black', lw=1)
    ax_r.axhline(1.2, color='gray', lw=0.5, ls='--')
    ax_r.axhline(0.8, color='gray', lw=0.5, ls='--')
    ax_r.set_ylim(0.5, 1.5)
    ax_r.set_ylabel('Data / MC', fontsize=11)
    ax_r.set_xlabel(xlabel, fontsize=13)
    ax_r.set_xlim(edges[0], edges[-1])
    ax_r.tick_params(labelsize=11)
    return fig


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--postfit-json', required=True,
                    help='written by extract_postfit_params.py')
    ap.add_argument('--region', required=True, choices=SIGNAL_REGIONS)
    ap.add_argument('--binning', default=None,
                    help='SR bin-edge YAML; the CR is rebinned the same way')
    ap.add_argument('--cr-only-norms', default=None,
                    help='cards_cr/cr_only/cr_only_norms.txt. Used for the '
                         'CR-only vs full-fit comparison printed to the log, '
                         'and as the panel norms with --prefit-norms cr-only.')
    ap.add_argument('--prefit-norms', choices=['postfit', 'cr-only', 'one'],
                    default='postfit',
                    help="What the pre-fit panel normalizes to. 'postfit' "
                         "(default) reuses the post-fit norm_* and r so the two "
                         "panels differ ONLY by the nuisance pulls -- the "
                         "comparison this check exists for. 'cr-only' uses the "
                         "step-5c CR-only fit values, which can sit ~20%% away. "
                         "'one' leaves the raw MC at 1.0.")
    ap.add_argument('--outdir', default='plots_cr/cr_closure')
    ap.add_argument('--no-signal', action='store_true',
                    help='leave the (tiny) signal out of the CR prediction')
    ap.add_argument('--xlabel', default='BDT score')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    edges = None
    if args.binning and os.path.exists(args.binning):
        raw = yaml.safe_load(open(args.binning)) or {}
        if args.region in raw:
            edges = np.asarray(raw[args.region], dtype=float)

    with open(args.postfit_json) as fh:
        d = json.load(fh)
    theta = d['theta']
    postfit_rates = dict(d['rates'])
    cov_names, cov = d.get('cov_names'), d.get('cov')

    # 'signal' in the JSON is r; the closure stack calls the merged signal VBS.
    if SIGNAL_KEY in postfit_rates:
        postfit_rates[SIGNAL_MERGED_NAME] = postfit_rates.pop(SIGNAL_KEY)

    print(f'Postfit rates: {postfit_rates}')
    top = sorted(theta.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
    print(f'Postfit theta: {len(theta)} nuisances; largest pulls: '
          + ', '.join(f'{k}={v:+.2f}' for k, v in top))
    if not cov:
        print('  [warn] no covariance in the JSON -- postfit band is MC-stat only.')

    # Read the CR-only norms whenever available, even when not used for the
    # panel, so the two fits can be compared in the log.
    cr_only_rates = read_cr_only_norms(args.cr_only_norms, args.region)

    if args.prefit_norms == 'postfit':
        prefit_rates = dict(postfit_rates)
        prefit_sub = 'Pre-fit (same norms as post-fit, nuisances at 0)'
    elif args.prefit_norms == 'cr-only':
        prefit_rates = dict(cr_only_rates)
        prefit_rates.setdefault(SIGNAL_MERGED_NAME, 1.0)
        prefit_sub = 'Pre-fit (CR-only fit norms, nuisances at 0)'
    else:
        prefit_rates = {}
        prefit_sub = 'Pre-fit (MC, all norms = 1)'
    print(f'Prefit rates:  {prefit_rates or "all 1.0"}  '
          f'[--prefit-norms {args.prefit_norms}]')

    if cr_only_rates:
        print('\n  CR-only fit vs full fit normalizations:')
        print(f'    {"process":<10}{"CR-only":>10}{"full fit":>10}{"ratio":>9}')
        for proc in (TTBAR_MERGED_NAME, WJET_MERGED_NAME):
            a, b = cr_only_rates.get(proc), postfit_rates.get(proc)
            if a is None or b is None or a == 0:
                continue
            print(f'    {proc:<10}{a:>10.4f}{b:>10.4f}{b / a:>9.3f}')
        print('    (they differ because step 5c freezes every constrained')
        print('     nuisance, so its rateParam absorbs the whole data/MC')
        print('     difference; the full fit shares it with the pulls.)')

    for cr_suffix in CR_SUFFIXES:
        print(f'\n=== {cr_suffix}_{args.region} ===')
        raw_h = load_cr_hists(config, args.region, cr_suffix,
                              include_signal=not args.no_signal)
        if not raw_h:
            print('  [skip] no histograms found')
            continue

        if edges is not None:
            reb = {p: rebin_hists_to_edges(v, edges) for p, v in raw_h.items()}
        else:
            n = REBIN_TARGETS[args.region]
            reb = {p: rebin_hists_dict(v, n) for p, v in raw_h.items()}

        post_h = apply_postfit_shape(reb, theta)
        topology = 'boosted' if args.region.startswith('boosted') else 'resolved'
        cov_unc = (covariance_band(post_h, cov_names, cov, postfit_rates,
                                   topology)
                   if cov else None)

        # Total yields of the two panels, so the size of the pure nuisance
        # effect is quotable and not just eyeballed off the plot.
        all_procs = [p for _, _, procs in CR_GROUPS for p in procs]
        pre_tot, _, _, _ = _merge_group(reb, all_procs, prefit_rates)
        post_tot, _, _, _ = _merge_group(post_h, all_procs, postfit_rates)
        if pre_tot is not None and post_tot is not None and pre_tot.sum() > 0:
            print(f'  total MC: pre-fit {pre_tot.sum():.1f}, '
                  f'post-fit {post_tot.sum():.1f} '
                  f'(ratio {post_tot.sum() / pre_tot.sum():.4f})')

        for tag, hists, rates, sub, cu in (
                ('prefit', reb, prefit_rates, prefit_sub, None),
                ('postfit', post_h, postfit_rates, 'Post-fit', cov_unc)):
            for scale in ('linear', 'log'):
                fig = make_figure(hists, cr_suffix, args.region, rates, sub,
                                  scale=scale, cov_unc=cu, xlabel=args.xlabel)
                if fig is None:
                    continue
                stem = f'{cr_suffix}_{args.region}_{tag}_{scale}'
                for ext in ('pdf', 'png'):
                    fig.savefig(os.path.join(args.outdir, f'{stem}.{ext}'),
                                bbox_inches='tight', dpi=150)
                plt.close(fig)
                print(f'  Saved: {stem}.pdf/png')

    print('\nDone.')


if __name__ == '__main__':
    main()
