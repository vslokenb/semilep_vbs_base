#!/usr/bin/env python3
"""
quick_syst_prepost.py -- pre-fit vs post-fit effect of chosen nuisances.

For each requested FITTED nuisance name (year-tagged / group-tagged, exactly
as they appear in the fit and in extract_postfit_params.py's JSON), one
ratio panel per (channel, BDT variable):

  prefit   : the +-1 sigma template variations, (nom +- Delta)/nom
  postfit  : the same linear morph evaluated at theta-hat +- sigma-hat,
             with theta-hat from the JSON and sigma-hat from the diagonal
             of its covariance block; the theta-hat central shift is drawn
             too. This is "what the fit actually allows this nuisance to do",
             next to "what it was given".

Fitted names are mapped back to raw template names the same way
plot_cr_closure.py does (strip _2018 / _2016_PreVFP / _yr2018, then a
per-MC-group tag). A group-tagged nuisance (sf_partonshower_fsr_TTbar_2018)
is shown as its effect on THAT GROUP's summed templates; everything else is
shown on the summed background.

Run in the same directory layout as plot_syst_ratios.py (python/ on EOS):

  python3 python/quick_syst_prepost.py \
      --rootfile histograms/histograms_kinematics_cr_2018.root \
      --postfit-json plots_cr_2018/crdata_srasimov/cr_closure/postfit_params_resolved_mu.json \
      --binning config/binning_example.yaml \
      --outdir plots_cr_2018/syst_study

--binning mirrors the fit: each region's bdt_* variables are merged onto the
SAME edges the datacards use ({region: [edges...]} YAML -- binning_example or
a binning-scan variant), so the panels show the nuisance on the fit's bins,
not the native fine binning. Without it (or if the file is missing) the
plot_syst_ratios BDT_REBIN_EDGES fallback applies.
"""
import argparse
import fnmatch
import json
import os
import re
from collections import defaultdict

import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep

from plot_syst_ratios import (
    read_root, merge_group, apply_bdt_rebinning, short_syst,
    BDT_REBIN_EDGES, GROUPS, SIGNAL_PROCS_RE, SIGNAL_REGIONS, SR_LABELS,
    LUMI_LABEL,
)

hep.style.use("CMS")

# The nuisances under study (fitted names; override with --nuisances)
DEFAULT_NUISANCES = [
    'AK4PFchs_JES_Regrouped_BBEC1_yr2018',
    'sf_partonshower_fsr_TTbar_2018',
    'AK4PFchs_JES_Regrouped_HF_yr2018',
    'LHEScaleWeight_renorm_scale_DY_2018',
    'AK8PFPuppi_JES_Regrouped_RelativeSample_2018',
    'AK8PFPuppi_JES_Regrouped_RelativeBal_2018',
    'AK8PFPuppi_JES_Regrouped_FlavorQCD_2018',
    'AK8PFPuppi_JES_Regrouped_BBEC1_2018',
    'AK4PFchs_JES_Regrouped_Absolute_yr2018',
    'AK4PFchs_JER_2018',
    'AK4PFchs_JES_Regrouped_RelativeBal_2018',
    'electron_inverttight_to_fake_2018',
]

_YEAR_TAG_RE = re.compile(r'_(yr)?20\d{2}(_(Pre|Post)VFP)?$')

# fitted group tag -> display-group proc lists (built from plot_syst_ratios)
_LABEL_TO_TAG = {
    "$t\\bar{t}$": 'TTbar',
    "$W\\rightarrow l\\nu$": 'WJets',
    "DY": 'DY',
    "Single Top": 'SingleTop',
    "Nonprompt": 'Nonprompt',
}
GROUP_PROCS = {'Diboson': [], 'signal': list(SIGNAL_PROCS_RE)}
for _label, _color, _procs in GROUPS:
    tag = _LABEL_TO_TAG.get(_label)
    if tag:
        GROUP_PROCS[tag] = list(_procs)
    else:
        # everything without its own tag is 'Diboson' in make_datacards's
        # MC_GROUP_RULES (dibosons, tribosons, Vgamma, VBF QCD)
        GROUP_PROCS['Diboson'] += list(_procs)
ALL_BKG_PROCS = [p for _l, _c, procs in GROUPS for p in procs]
_GROUP_TAGS = ('signal', 'TTbar', 'WJets', 'DY', 'SingleTop', 'Nonprompt',
               'Diboson')


def resolve_template_syst(fit_name, raw_names):
    """(raw template name, MC group or None), or None -- same peeling order
    as plot_cr_closure.py: exact, year tag, group tag, year tag."""
    cand, group = fit_name, None
    for _ in range(4):
        if cand in raw_names:
            return cand, group
        stripped = _YEAR_TAG_RE.sub('', cand)
        if stripped != cand:
            cand = stripped
            continue
        for g in _GROUP_TAGS:
            if cand.endswith(f'_{g}'):
                group, cand = g, cand[: -len(g) - 1]
                break
        else:
            return None
    return (cand, group) if cand in raw_names else None


def read_postfit(path):
    """{fitted name: (theta_hat, sigma_hat or None)} from the JSON."""
    with open(path) as fh:
        d = json.load(fh)
    theta = d.get('theta', {})
    sig = {}
    names, cov = d.get('cov_names'), d.get('cov')
    if names and cov:
        for i, n in enumerate(names):
            try:
                sig[n] = float(cov[i][i]) ** 0.5
            except Exception:
                pass
    return {n: (v, sig.get(n)) for n, v in theta.items()}


def morph_ratio(nom, d_up, d_dn, s):
    """Linear chord morph of the component at pull value s, as a ratio.

    merge_group stores d_up = up - nom and d_dn = nom - down, so the morph
    is (nom + s*d_up)/nom for s >= 0 and (nom + s*d_dn)/nom for s < 0
    (s = -1 reproduces the down template exactly). When one direction's
    template is missing, the other is mirrored.
    """
    if s >= 0:
        delta = d_up if d_up is not None else d_dn
    else:
        delta = d_dn if d_dn is not None else d_up
    if delta is None:
        return None
    return np.where(nom > 0, (nom + s * delta) / nom, 1.0)


def make_figure(channel, var_name, structured_var, entries, postfit):
    """One figure: a ratio panel per requested nuisance."""
    n = len(entries)
    fig, axes = plt.subplots(n, 1, figsize=(8, 1.9 * n + 1.2),
                             sharex=True,
                             gridspec_kw={'hspace': 0.10})
    if n == 1:
        axes = [axes]
    edges = None
    for ax, (fit_name, raw, group) in zip(axes, entries):
        procs = GROUP_PROCS[group] if group else ALL_BKG_PROCS
        nom, _v, e, syst_d = merge_group(structured_var, procs)
        comp = group if group else 'bkg'
        if nom is None or raw is None:
            ax.text(0.5, 0.5, f'{fit_name}: no templates here',
                    transform=ax.transAxes, ha='center', fontsize=8)
            continue
        if edges is None:
            edges = e
        d_up, d_dn = syst_d.get(raw, (None, None))
        if d_up is None and d_dn is None:
            ax.text(0.5, 0.5, f'{fit_name}: variation absent for {comp}',
                    transform=ax.transAxes, ha='center', fontsize=8)
            continue

        drawn = []
        # prefit +-1 sigma templates
        if d_up is not None:
            r = np.where(nom > 0, (nom + d_up) / nom, 1.0)
            ax.step(e, np.append(r, r[-1]), where='post', color='#1f77b4',
                    lw=1.3, ls='-', label='prefit $+1\\sigma$')
            drawn.append(r)
        if d_dn is not None:
            r = np.where(nom > 0, (nom - d_dn) / nom, 1.0)
            ax.step(e, np.append(r, r[-1]), where='post', color='#1f77b4',
                    lw=1.3, ls='--', label='prefit $-1\\sigma$')
            drawn.append(r)

        # postfit: theta-hat and theta-hat +- sigma-hat along the same morph
        th, sg = postfit.get(fit_name, (None, None))
        title = f'{short_syst(fit_name)}  [{comp}]'
        if th is not None:
            title += f'   $\\hat\\theta$ = {th:+.2f}'
            if sg is not None:
                title += f', $\\hat\\sigma$ = {sg:.2f}'
            rc = morph_ratio(nom, d_up, d_dn, th)
            if rc is not None:
                ax.step(e, np.append(rc, rc[-1]), where='post', color='black',
                        lw=1.1, ls=':', label='postfit $\\hat\\theta$')
                drawn.append(rc)
            if sg is not None:
                for s, ls, lab in ((th + sg, '-', 'postfit $\\hat\\theta+\\hat\\sigma$'),
                                   (th - sg, '--', 'postfit $\\hat\\theta-\\hat\\sigma$')):
                    r = morph_ratio(nom, d_up, d_dn, s)
                    if r is not None:
                        ax.step(e, np.append(r, r[-1]), where='post',
                                color='#e42536', lw=1.3, ls=ls, label=lab)
                        drawn.append(r)
        else:
            title += '   [not in postfit JSON]'

        ax.axhline(1.0, color='black', lw=0.8)
        for yy in (0.9, 1.1):
            ax.axhline(yy, color='gray', lw=0.4, ls=':')
        lo, hi = 0.85, 1.15
        if drawn:
            st = np.concatenate(drawn)
            st = st[np.isfinite(st)]
            if st.size:
                lo = min(lo, max(0.3, float(st.min()) - 0.02))
                hi = max(hi, min(1.7, float(st.max()) + 0.02))
        ax.set_ylim(lo, hi)
        ax.set_xlim(e[0], e[-1])
        ax.set_ylabel('Ratio', fontsize=8)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=6, ncol=5, loc='upper left', framealpha=0.7,
                  title=title, title_fontsize=7, handlelength=1.6,
                  borderpad=0.3, labelspacing=0.2, columnspacing=0.8)

    if edges is None:
        plt.close(fig)
        return None
    axes[-1].set_xlabel('BDT score' if var_name.startswith('bdt_') else var_name,
                        fontsize=11)
    hep.cms.label('Work in Progress', data=False, rlabel=LUMI_LABEL,
                  ax=axes[0], fontsize=10)
    axes[0].set_title(SR_LABELS.get(channel, channel), fontsize=10, loc='left',
                      pad=24)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rootfile', required=True,
                    help='histograms_kinematics_cr_<year>.root')
    ap.add_argument('--postfit-json', required=True,
                    help='extract_postfit_params.py output (theta + cov)')
    ap.add_argument('--outdir', default='plots/syst_study')
    ap.add_argument('--nuisances', default=None,
                    help='comma-separated fitted names (default: the 11 '
                         'under study)')
    ap.add_argument('--binning', default='config/binning_example.yaml',
                    help='fit binning YAML ({region: [edges...]}); the '
                         'bdt_* variables are merged onto these edges per '
                         'region, mirroring the datacards. Pass a '
                         'binning-scan variant to study that setup.')
    ap.add_argument('--vars', default='bdt_*',
                    help='fnmatch pattern(s), comma-separated, for which '
                         'variables to plot')
    args = ap.parse_args()

    nuisances = ([n.strip() for n in args.nuisances.split(',') if n.strip()]
                 if args.nuisances else DEFAULT_NUISANCES)
    var_pats = [p.strip() for p in args.vars.split(',')]

    postfit = read_postfit(args.postfit_json)
    print(f'postfit JSON: {len(postfit)} nuisances; requested {len(nuisances)}')
    for n in nuisances:
        th, sg = postfit.get(n, (None, None))
        if th is None:
            print(f'  [warn] {n}: not in postfit JSON (prefit-only panel)')
        else:
            print(f'  {n}: theta={th:+.3f}'
                  + (f' sigma={sg:.3f}' if sg is not None else ' (no cov)'))

    os.makedirs(args.outdir, exist_ok=True)
    data = read_root(args.rootfile)

    # Merge bins exactly as the fit does: the binning YAML is keyed by
    # region, so point every bdt_* variable of that region at its edges
    # (apply_bdt_rebinning looks up "channel/var" keys).
    edges_map = None
    if args.binning and os.path.exists(args.binning):
        reg_edges = yaml.safe_load(open(args.binning)) or {}
        edges_map = {}
        for channel, by_var in data.items():
            if channel not in reg_edges:
                continue
            for var_name in by_var:
                if var_name.startswith('bdt_'):
                    edges_map[f'{channel}/{var_name}'] = reg_edges[channel]
        print(f'binning: {args.binning} -> '
              f'{len(edges_map)} (region, variable) group(s) merged onto '
              f'the fit edges')
    elif args.binning:
        print(f'  [warn] binning config not found: {args.binning} -- '
              f'falling back to BDT_REBIN_EDGES')
    apply_bdt_rebinning(data, edges_map if edges_map else BDT_REBIN_EDGES)

    for channel in SIGNAL_REGIONS:
        if channel not in data:
            print(f'[skip] {channel} not in ROOT file')
            continue
        for var_name in sorted(data[channel]):
            if not any(fnmatch.fnmatchcase(var_name, p) for p in var_pats):
                continue
            structured_var = data[channel][var_name]
            raw_names = set()
            for pvars in structured_var.values():
                for k in pvars:
                    if k.endswith('Up'):
                        raw_names.add(k[:-2])
            entries = []
            for fit_name in nuisances:
                res = resolve_template_syst(fit_name, raw_names)
                if res is None:
                    entries.append((fit_name, None, None))
                else:
                    entries.append((fit_name, res[0], res[1]))
            fig = make_figure(channel, var_name, structured_var, entries,
                              postfit)
            if fig is None:
                print(f'[skip] {channel}/{var_name}: nothing to draw')
                continue
            stem = f'prepost_{channel}__{var_name}'
            for ext in ('pdf', 'png'):
                fig.savefig(os.path.join(args.outdir, f'{stem}.{ext}'),
                            bbox_inches='tight', dpi=150)
            plt.close(fig)
            print(f'Saved: {stem}.pdf/png')

    print('Done.')


if __name__ == '__main__':
    main()
