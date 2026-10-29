#!/usr/bin/env python3
"""
Postfit distributions of the DeepFlav quark-vs-gluon tagger score
(jet1..jet4 btagDeepFlavQG -- resolved_mu/resolved_e, SR + both CRs) with
nested postfit uncertainty bands, to check whether the
QvG tagger SF uncertainty (found to be the single largest systematic group
in the marginal uncertainty breakdown for resolved_e, see
uncertainty_breakdown_resolved_e/uncertainty_breakdown.txt) is over-
estimated relative to what data actually supports:

  stat only    -- MC statistical uncertainty alone
  stat + QvG   -- stat (+) postfit-covariance-propagated QvG-tagger SF
                  nuisances only, matched from the postfit covariance by
                  name prefix (QVG_PREFIXES below) rather than hardcoded:
                  in the current datacard these are the
                  sf_qgtagging_sf_qgtagging_{L1prefiring,PDF,fsr,herwig,
                  isr,jer,jes,pu,scale,stat,total} shape sources. Every
                  other nuisance is held fixed at its fitted value, i.e.
                  treated as if it had zero uncertainty.
  stat + all   -- stat (+) full postfit covariance (all shape nuisances +
                  norm_top/norm_wjet), same definition as
                  plot_cr_closure.py's single postfit band

  stat + all, QvG xN  -- (--qvg-scale N, default 2) the same full
                  propagation with the QvG Jacobian columns scaled by N:
                  QvG variances enter xN^2 and their correlations with
                  every other nuisance xN. Drawn as the outermost band, it
                  answers 'would an artificially doubled QvG SF uncertainty
                  cover the data/MC discrepancy?'. A per-bin coverage
                  printout (bins outside 1/2 sigma of data(+)band, per
                  band) makes the answer quotable. --qvg-scale 1 disables
                  the extra band.

The central curve (postfit-morphed prediction: nuisances at their fitted
theta, norm_top/norm_wjet at their fitted rate) is identical across all
three bands -- only the uncertainty width differs, so the three bands can
be read directly as a decomposition of the total postfit uncertainty.

Each channel is also plotted PRE-FIT (files tagged _prefit_): un-morphed
templates with the same postfit norms and r (plot_cr_closure's default
convention -- the two panels then differ only by the nuisance pulls). The
prefit panel's QvG band uses the PREFIT width: unit constraints,
uncorrelated, chord J evaluated at theta = 0 -- i.e. the assigned QvG
uncertainty before the fit constrains it, directly comparable to the
postfit panel's fitted QvG band; --qvg-scale draws its inflated version
here too.

Reuses plot_cr_closure.py's postfit machinery directly: apply_postfit_shape,
_merge_group, the CR_GROUPS stack definition, cr_label, SIGNAL_MERGED_NAME
and LUMI_LABEL, and follows its CR naming convention: channels are named
f'{cr}_{sr}' (ttbar_cr_resolved_mu, w_cr_resolved_e, ...) and resolved to
the real coffea categories through make_histograms.coffea_cat -- not the
legacy f'{sr}_TTCR'/f'{sr}_WCR' names, which no longer exist as coffea
categories. plot_cr_closure.py reads its postfit JSON inline in main(), so
the equivalent reader (theta / rates / cov, with the JSON's 'signal' rate
renamed to the merged VBS process) lives here as read_postfit_params().
compute_covariance_band() is plot_cr_closure.covariance_band() generalized
with a nuisance-name whitelist, and -- like there -- is evaluated on the
postfit-morphed templates with the postfit rates, so the linearization
point matches the central curve.

Run in the PocketCoffea apptainer env (same as make_histograms.py):
  python3 python/plot_qvg_postfit.py \
      --config config/inputs_2018.yaml \
      --postfit-dir plots/merged_offset_cat_v9_renorm_jetsym \
      --outdir plots/merged_offset_cat_v9_renorm_jetsym/qvg_postfit_check
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep

sys.path.insert(0, str(Path(__file__).parent))
from make_histograms import (
    build_process_hists, load_process_hists,
    merge_processes,
    rebin_hists_dict,
    coffea_cat, is_electron_region,
    TTBAR_MERGED_NAME, TTBAR_PROCESSES,
    WJET_MERGED_NAME, WJET_PROCESSES,
    DY_MERGED_NAME, DY_PROCESSES,
    ELECTRON_DATA_PROCESSES, MUON_DATA_PROCESSES,
    CR_SUFFIXES,
)
from plot_stacked import SR_LABELS
from plot_cr_closure import (
    apply_postfit_shape, _merge_group, cr_label, build_jacobian,
    CR_GROUPS, SIGNAL_MERGED_NAME, LUMI_LABEL,
)

hep.style.use("CMS")

N_REBIN = 10
ALL_PROCS = [p for _, _, procs in CR_GROUPS for p in procs]

# QvG tagger SF nuisances are selected from the postfit covariance by name
# prefix instead of a hardcoded list: the datacard names them with the
# doubled group prefix sf_qgtagging_sf_qgtagging_<source> (same convention
# as sf_fj_WvsQCD_sf_fj_WvsQCD_* / sf_fj_tau21_sf_fj_tau21_*), with sources
# L1prefiring/PDF/fsr/herwig/isr/jer/jes/pu/scale/stat/total. Prefix
# matching picks up exactly the subset that actually made it into this
# fit's covariance, and 'sf_qvg' is kept as an alternate spelling.
QVG_PREFIXES = ('sf_qgtagging', 'sf_qvg')

# Channels per variable: '' is the SR itself; a CR is one of
# make_histograms.CR_SUFFIXES ('ttbar_cr', 'w_cr') and the channel is named
# f'{cr}_{sr}' (ttbar_cr_resolved_mu, w_cr_resolved_e, ...) -- the final
# naming convention shared with plot_cr_closure.py and the datacards.
SR_AND_CR_CHANNELS = [''] + list(CR_SUFFIXES)  # SR + both CRs
CR_ONLY_CHANNELS = list(CR_SUFFIXES)           # both CRs, no SR

# (variable base name -- str shared across regions, or {region: var_name}
#  when the coffea variable name is itself region-specific (e.g. the BDT
#  score) --, [SRs it applies to], x-axis label, [channels to make plots
#  for])
VARIABLES = [
    (f'jet{i}_btagDeepFlavQG',       ['resolved_mu', 'resolved_e'],
     f'Jet {i} DeepFlav QG', SR_AND_CR_CHANNELS)
    for i in (1, 2, 3, 4)
] + [
    # BDT score itself: CRs only, matching plot_cr_closure.py's
    # scope -- the SR BDT score is the fit's own discriminant and already
    # has its own postfit plots in postfit/ (from combine's shapes_fit_s
    # via plot_postfit.py), so it's deliberately not duplicated here.
    ({'resolved_mu': 'bdt_resolved_mu', 'resolved_e': 'bdt_resolved_e'},
     ['resolved_mu', 'resolved_e'], 'BDT score', CR_ONLY_CHANNELS),
]

_qvg_reported = set()


def read_postfit_params(path):
    """theta / rates / covariance from extract_postfit_params.py's JSON --
    the same fields plot_cr_closure.py reads inline in its main(), with the
    JSON's 'signal' rate (r) renamed to the merged VBS process name."""
    with open(path) as fh:
        d = json.load(fh)
    theta = d['theta']
    rates = dict(d['rates'])
    if 'signal' in rates:
        rates[SIGNAL_MERGED_NAME] = rates.pop('signal')
    return theta, rates, d.get('cov_names'), d.get('cov')


def find_qvg_nuisances(cov_names, report_key=None):
    """Names in the postfit covariance belonging to the QvG SF group
    (prefix match against QVG_PREFIXES). Reports the matched set once per
    report_key, and flags the double-counting trap of a '_total' source
    coexisting with the decorrelated components -- if both are in the fit,
    the fit itself (and therefore this band) counts the QvG uncertainty
    twice, which is exactly the kind of overestimation this script is
    meant to expose."""
    matched = {n for n in (cov_names or []) if n.startswith(QVG_PREFIXES)}
    if report_key is not None and report_key not in _qvg_reported:
        _qvg_reported.add(report_key)
        if matched:
            print(f'  [{report_key}] QvG nuisances in covariance '
                  f'({len(matched)}): {", ".join(sorted(matched))}')
            totals = {n for n in matched if n.endswith('_total')}
            if totals and matched - totals:
                print(f'  [{report_key}] WARNING: {sorted(totals)} present '
                      f'alongside decorrelated components -- the QvG '
                      f'uncertainty is double-counted in this fit')
        else:
            print(f'  [{report_key}] WARNING: no QvG nuisances matched '
                  f'prefixes {QVG_PREFIXES} in the covariance -- '
                  f'stat+QvG band will be skipped')
    return matched


def load_hists_merged(config, var_name, base_sr, cr_suffix):
    """Load background+signal+data histograms for one channel (the SR when
    cr_suffix='', the CR f'{cr_suffix}_{base_sr}' otherwise) and one
    variable. Same loading and naming as plot_cr_closure.load_cr_hists:
    build_process_hists for MC (is_signal=True for the signal section,
    merged into the single SIGNAL_MERGED_NAME process), load_process_hists
    for data."""
    region = f'{cr_suffix}_{base_sr}' if cr_suffix else base_sr
    cat = coffea_cat(region)          # honours CATEGORY_OVERRIDES (w_cr_mu ...)
    is_electron = is_electron_region(region)

    years = list(config['years'].keys())
    yc = config['years'][years[0]]

    ch_hists = {}

    for proc, pcfg in yc.get('background', {}).items():
        flist = pcfg.get('files', [pcfg.get('file')])
        hists = build_process_hists(flist, proc, var_name, cat)
        if hists:
            ch_hists[proc] = hists

    sig = {}
    for proc, pcfg in yc.get('signal', {}).items():
        flist = pcfg.get('files', [pcfg.get('file')])
        hists = build_process_hists(flist, proc, var_name, cat,
                                    is_signal=True)
        if hists:
            sig[proc] = hists
    if sig:
        ch_hists.update(sig)
        merge_processes(ch_hists, set(sig.keys()), SIGNAL_MERGED_NAME)

    data_acc = None
    for proc, pcfg in yc.get('data', {}).items():
        if is_electron and proc not in ELECTRON_DATA_PROCESSES:
            continue
        if not is_electron and proc not in MUON_DATA_PROCESSES:
            continue
        flist = pcfg.get('files', [pcfg.get('file')])
        hists = load_process_hists(flist, proc, var_name, cat)
        if not hists or 'nominal' not in hists:
            continue
        v, e, edges = hists['nominal']
        if data_acc is None:
            data_acc = [v.copy(), e.copy(), edges]
        else:
            data_acc[0] += v
            data_acc[1] += e
    if data_acc is not None:
        ch_hists['data_obs'] = {'nominal': tuple(data_acc)}

    merge_processes(ch_hists, TTBAR_PROCESSES, TTBAR_MERGED_NAME)
    merge_processes(ch_hists, WJET_PROCESSES, WJET_MERGED_NAME)
    merge_processes(ch_hists, DY_PROCESSES, DY_MERGED_NAME)

    return ch_hists


def rebin_ch(ch_hists, n_target):
    any_hists = next(iter(ch_hists.values()), None)
    if not any_hists or 'nominal' not in any_hists:
        return ch_hists
    n_orig = len(any_hists['nominal'][0])
    eff_target = n_target if n_orig % n_target == 0 else n_orig
    return {proc: rebin_hists_dict(pvars, eff_target) for proc, pvars in ch_hists.items()}


def compute_covariance_band(ch_hists, cov_names, cov, proc_scales, only_names=None,
                            scale_names=None, scale_factor=1.0, topology=None):
    """plot_cr_closure.covariance_band generalized with an optional
    nuisance-name whitelist: columns of J for names not in only_names are
    left at zero, so their own variance and their correlations with
    everything else drop out of J.Cov.J^T -- exactly 'what if this nuisance
    had zero uncertainty and stayed fixed at its fitted value' for every
    name outside the whitelist. only_names=None keeps every column (the
    full postfit band, identical to plot_cr_closure's). Like there, this is
    meant to be evaluated on the postfit-morphed histograms so J is the
    chord around the fitted point.

    scale_names/scale_factor artificially inflate a subset of nuisances:
    their J columns are multiplied by scale_factor, so their variance
    contribution scales by scale_factor^2 and their correlation terms with
    everything else by scale_factor -- 'what if these uncertainties were N
    times larger', with the fitted correlations kept.

    J itself comes from plot_cr_closure.build_jacobian, which maps the
    FITTED nuisance names back to the raw template names (year and
    MC-group decorrelation tags) and handles the rateParams; only_names and
    scale_names keep matching the fitted names, as before."""
    nom, J = build_jacobian(ch_hists, cov_names, proc_scales, topology)
    if J is None:
        return None
    if only_names is not None:
        for j, name in enumerate(cov_names):
            if name not in only_names:
                J[:, j] = 0.0
    if scale_names and scale_factor != 1.0:
        for j, name in enumerate(cov_names):
            if name in scale_names:
                J[:, j] *= scale_factor
    var = np.einsum('bi,ij,bj->b', J, np.asarray(cov, dtype=float), J)
    return np.sqrt(np.maximum(var, 0.0))


def coverage_report(postfit_hists, proc_scales, bands):
    """Per-band data coverage: bins whose |data - prediction| exceeds 1/2
    sigma, with sigma = data stat (+) band width. This is the number the
    'is doubled QvG enough?' question needs."""
    tot, _v, _e, _sd = _merge_group(postfit_hists, ALL_PROCS, proc_scales)
    if tot is None or 'data_obs' not in postfit_hists:
        return
    dv, de, _ = postfit_hists['data_obs']['nominal']
    derr = np.sqrt(np.maximum(de, np.maximum(dv, 0.0)))
    print('  coverage (|data-MC| vs data-stat (+) band):')
    for label, _c, _a, unc in bands:
        sigma = np.sqrt(derr ** 2 + unc ** 2)
        pulls = np.abs(dv - tot) / np.maximum(sigma, 1e-9)
        clean = (label.replace('$\\oplus$', '+').replace('$', '')
                 .replace('\\times', 'x'))
        print(f'    {clean:32s} {int((pulls > 1).sum()):2d}/{len(pulls)} bins >1sig, '
              f'{int((pulls > 2).sum()):2d} >2sig, max pull {pulls.max():.2f}')


def make_three_band_figure(postfit_hists, channel_label, xlabel, proc_scales,
                           bands, scale='linear', subtitle='Post-fit'):
    """bands: list of (label, color, alpha, unc_array), widest/most-inclusive
    first so later (narrower) bars draw on top of it. Stack order and group
    labels follow plot_cr_closure's CR_GROUPS (VBS first, at the stack
    baseline, so it stays visible on a log axis)."""
    vals, colors, labels = [], [], []
    total_nom = edges = None

    for base_label, color, procs in CR_GROUPS:
        label = base_label
        if TTBAR_MERGED_NAME in procs:
            label = f"$t\\bar{{t}}$ $\\times$ {proc_scales.get(TTBAR_MERGED_NAME, 1.0):.3f}"
        elif WJET_MERGED_NAME in procs:
            label = f"$W\\rightarrow l\\nu$ $\\times$ {proc_scales.get(WJET_MERGED_NAME, 1.0):.3f}"
        elif SIGNAL_MERGED_NAME in procs:
            label = f"VBS (r = {proc_scales.get(SIGNAL_MERGED_NAME, 1.0):.2f})"

        v, ve, e, _sd = _merge_group(postfit_hists, procs, proc_scales)
        if v is None:
            continue
        if edges is None:
            edges = e
        vals.append(v)
        colors.append(color)
        labels.append(label)
        total_nom = v.copy() if total_nom is None else total_nom + v

    if total_nom is None:
        return None

    data_vals = data_err = None
    if 'data_obs' in postfit_hists and 'nominal' in postfit_hists['data_obs']:
        dv, de, _ = postfit_hists['data_obs']['nominal']
        data_vals = dv.copy()
        data_err = np.sqrt(np.maximum(de, np.maximum(dv, 0.0)))

    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = edges[1:] - edges[:-1]

    fig, (ax, ax_r) = plt.subplots(
        2, 1, figsize=(8, 7),
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05},
    )
    hep.histplot(vals, edges, stack=True, histtype='fill',
                color=colors, label=labels, ax=ax, linewidth=0)

    for band_label, band_color, band_alpha, unc in bands:
        ax.bar(centers, 2 * unc, width=widths, bottom=total_nom - unc,
              color=band_color, alpha=band_alpha, linewidth=0,
              label=band_label, zorder=4)

    if data_vals is not None:
        ax.errorbar(centers, data_vals, yerr=data_err, fmt='ko',
                    markersize=4, linewidth=1, label='Data', zorder=6)

    ax.set_ylabel('Events / bin', fontsize=13)
    ax.set_xlim(edges[0], edges[-1])
    ax.tick_params(labelbottom=False, labelsize=11)
    ymax = max(np.max(total_nom), np.max(data_vals) if data_vals is not None else 0)
    if scale == 'log':
        ax.set_yscale('log')
        ax.set_ylim(max(0.01, ymax * 1e-5), ymax * 50)
    else:
        ax.set_ylim(0, ymax * 1.6)
    ax.legend(ncol=2, fontsize=8, loc='upper right', framealpha=0.8)
    hep.cms.label('Work in Progress', data=True, rlabel=LUMI_LABEL, ax=ax, fontsize=12)

    ratio = np.where(total_nom > 0, data_vals / total_nom, np.nan) if data_vals is not None else np.full_like(centers, np.nan)
    ratio_err = np.where(total_nom > 0, data_err / total_nom, np.nan) if data_err is not None else np.full_like(centers, np.nan)

    ax_r.errorbar(centers, ratio, yerr=ratio_err, fmt='ko', markersize=4, linewidth=1, zorder=6)
    for band_label, band_color, band_alpha, unc in bands:
        unc_rel = np.where(total_nom > 0, unc / total_nom, 0.0)
        ax_r.bar(centers, 2 * unc_rel, width=widths, bottom=1.0 - unc_rel,
                color=band_color, alpha=band_alpha, linewidth=0, zorder=4)
    ax_r.axhline(1.0, color='black', lw=1)
    ax_r.axhline(1.2, color='gray', lw=0.5, ls='--')
    ax_r.axhline(0.8, color='gray', lw=0.5, ls='--')
    ax_r.set_ylim(0.5, 1.5)
    ax_r.set_ylabel('Data / MC', fontsize=11)
    ax_r.set_xlabel(xlabel, fontsize=13)
    ax_r.set_xlim(edges[0], edges[-1])
    ax_r.tick_params(labelsize=11)

    ax.text(0.02, 0.96, f'{channel_label}\n{subtitle}', transform=ax.transAxes, fontsize=11, va='top')

    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True)
    parser.add_argument('--postfit-dir', required=True,
                        help='plots/<...> directory containing fitdiagnostics_<region>/postfit_params.json '
                             'for each region (written beforehand by extract_postfit_params.py)')
    parser.add_argument('--outdir', default='plots/qvg_postfit_check')
    parser.add_argument('--qvg-scale', type=float, default=2.0,
                        help='artificially inflate the QvG SF uncertainties by '
                             'this factor in an extra outermost band (their '
                             'covariance Jacobian columns are scaled, keeping '
                             'the fitted correlations). Default 2 = doubled; '
                             '1 disables the extra band.')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    for var_spec, regions, xlabel, channels in VARIABLES:
        for region in regions:
            var_name = var_spec[region] if isinstance(var_spec, dict) else var_spec
            postfit_json = os.path.join(args.postfit_dir, f'fitdiagnostics_{region}', 'postfit_params.json')
            if not os.path.isfile(postfit_json):
                print(f'  [skip] {region}: no postfit_params.json at {postfit_json}')
                continue
            theta, postfit_rates, cov_names, cov = read_postfit_params(postfit_json)
            # match against the covariance when there is one, else against the
            # theta names (enough for the prefit-width bands)
            qvg_names = find_qvg_nuisances(cov_names if cov else sorted(theta),
                                           report_key=region)

            for cr_suffix in channels:
                chan_key = f'{cr_suffix}_{region}' if cr_suffix else region
                print(f'=== {var_name} / {chan_key} ===')
                raw = load_hists_merged(config, var_name, region, cr_suffix)
                if not raw:
                    print('  [skip] no histograms found')
                    continue
                rebinned = rebin_ch(raw, N_REBIN)

                postfit_hists = apply_postfit_shape(rebinned, theta)
                topo = ('boosted' if region.startswith('boosted')
                        else 'resolved')

                # MC stat of the displayed (postfit) stack; morphing leaves
                # the variances untouched, only the rate scaling enters.
                _n, stat_var, _e, _sd = _merge_group(postfit_hists, ALL_PROCS, postfit_rates)
                if stat_var is None:
                    print('  [skip] no stack processes found')
                    continue
                stat_unc = np.sqrt(np.maximum(stat_var, 0.0))

                bands = []
                if cov:
                    # widest first: the QvG-inflated total, then the nominal
                    # total, then QvG alone, then stat -- so narrower bars
                    # draw on top of wider ones.
                    if qvg_names and args.qvg_scale != 1.0:
                        unc_all_x = compute_covariance_band(
                            postfit_hists, cov_names, cov, postfit_rates,
                            scale_names=qvg_names, scale_factor=args.qvg_scale,
                            topology=topo)
                        if unc_all_x is not None:
                            bands.append((f'Stat $\\oplus$ all syst. (QvG $\\times${args.qvg_scale:g})',
                                          '#55a868', 0.30,
                                          np.sqrt(stat_unc ** 2 + unc_all_x ** 2)))
                    unc_all = compute_covariance_band(
                        postfit_hists, cov_names, cov, postfit_rates,
                        only_names=None, topology=topo)
                    if unc_all is not None:
                        bands.append(('Stat $\\oplus$ all syst.', '#4c72b0', 0.35,
                                      np.sqrt(stat_unc ** 2 + unc_all ** 2)))
                    if qvg_names:
                        unc_qvg = compute_covariance_band(
                            postfit_hists, cov_names, cov, postfit_rates,
                            only_names=qvg_names, topology=topo)
                        if unc_qvg is not None:
                            bands.append(('Stat $\\oplus$ QvG SF', '#dd8452', 0.45,
                                          np.sqrt(stat_unc ** 2 + unc_qvg ** 2)))
                else:
                    print('  [warn] no covariance in postfit JSON -- bands are MC-stat only')
                bands.append(('Stat only', '#555555', 0.55, stat_unc))

                # Pre-fit panel: un-morphed templates with the SAME norms as
                # post-fit (plot_cr_closure's default convention, so the two
                # panels differ only by the nuisance pulls). Its QvG band is
                # the PREFIT width: unit constraints, uncorrelated (identity
                # covariance over the QvG names), chord J at theta = 0 -- the
                # assigned uncertainty before the fit constrains it.
                prefit_bands = []
                qvg_list = sorted(qvg_names)
                if qvg_list:
                    unc_pre = compute_covariance_band(
                        rebinned, qvg_list, np.eye(len(qvg_list)),
                        postfit_rates, topology=topo)
                    if unc_pre is not None:
                        if args.qvg_scale != 1.0:
                            prefit_bands.append(
                                (f'Stat $\\oplus$ QvG SF $\\times${args.qvg_scale:g} (prefit)',
                                 '#55a868', 0.30,
                                 np.sqrt(stat_unc ** 2 + (args.qvg_scale * unc_pre) ** 2)))
                        prefit_bands.append(('Stat $\\oplus$ QvG SF (prefit)',
                                             '#dd8452', 0.45,
                                             np.sqrt(stat_unc ** 2 + unc_pre ** 2)))
                prefit_bands.append(('Stat only', '#555555', 0.55, stat_unc))

                chan_label = (SR_LABELS.get(region, region) if not cr_suffix
                              else cr_label(cr_suffix, region))

                panels = (
                    ('prefit', rebinned, prefit_bands,
                     'Pre-fit (postfit norms, nuisances at 0)'),
                    ('postfit', postfit_hists, bands, 'Post-fit'),
                )
                for tag, hists, pbands, subtitle in panels:
                    print(f'  -- {tag}')
                    coverage_report(hists, postfit_rates, pbands)
                    for scale in ('linear', 'log'):
                        fig = make_three_band_figure(
                            hists, chan_label, xlabel, postfit_rates, pbands,
                            scale=scale, subtitle=subtitle)
                        if fig is None:
                            continue
                        stem = f'{var_name}_{chan_key}_{tag}_{scale}'
                        for ext in ('pdf', 'png'):
                            fig.savefig(os.path.join(args.outdir, f'{stem}.{ext}'),
                                       bbox_inches='tight', dpi=150)
                        plt.close(fig)
                        print(f'  Saved: {stem}.pdf/png')

    print('\nDone.')


if __name__ == '__main__':
    main()
