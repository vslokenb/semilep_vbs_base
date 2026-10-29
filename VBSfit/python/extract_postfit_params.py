#!/usr/bin/env python3
"""
extract_postfit_params.py -- dump postfit nuisance pulls, rateParams and the
postfit covariance submatrix from a FitDiagnostics file into JSON, for
plot_cr_closure.py.

plot_cr_closure.py runs in the PocketCoffea apptainer image, which has no
PyROOT, so the fit result has to be handed over as plain JSON. This script runs
in the CMSSW environment.

WHY THE COVARIANCE IS NEEDED
The CRs enter the fit as single counting bins, so combine's own shapes_fit_s
output carries no shape-level postfit uncertainty for them -- it only ever saw
one number per CR. The closure plot rebuilds the CR at full BDT binning and
propagates the fit's actual covariance through that shape model
(Var = J . Cov . J^T), which is a correlated band rather than a naive
quadrature sum of each nuisance's own diagonal error.

RATEPARAM NAMES
Discovered from the fit result rather than assumed, trying in order:
    norm_top_{region}     per-category split
    norm_top_{topology}   topology split -- what make_datacards.py writes now
    norm_top              single shared parameter
each also matched with the year-decorrelation suffix stripped, so the
year-tagged parameters of multi-year cards (norm_top_boosted_2017, ...) are
found too. Only the rateParams affecting THIS region are kept; another
topology's normalization does not enter this region's prediction, and
carrying it would only pad the covariance matrix.

Usage:
  python3 python/extract_postfit_params.py \
      --fitfile plots_cr/crdata_srasimov/fitdiagnostics/fitDiagnostics_hybrid.root \
      --region boosted_mu \
      --output plots_cr/crdata_srasimov/cr_closure/postfit_params_boosted_mu.json
"""
import argparse
import json
import os
import re

import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kError

# Merged names used by make_histograms.py, and therefore by the closure plot.
TTBAR_MERGED_NAME = 'TTbar'
WJET_MERGED_NAME = 'WJets'
SIGNAL_KEY = 'signal'

# The year-decorrelation tag make_datacards.py appends in multi-year mode
# (_2017, _2016_PreVFP, and _yr2017 for names whose plain tag is taken).
_YEAR_TAG_RE = re.compile(r'_(yr)?20\d{2}(_(Pre|Post)VFP)?$')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fitfile', required=True)
    ap.add_argument('--region', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--fit', default='fit_s')
    args = ap.parse_args()

    topology = 'boosted' if args.region.startswith('boosted') else 'resolved'
    candidates = {
        TTBAR_MERGED_NAME: (f'norm_top_{args.region}',
                            f'norm_top_{topology}', 'norm_top'),
        WJET_MERGED_NAME: (f'norm_wjet_{args.region}',
                           f'norm_wjet_{topology}', 'norm_wjet'),
    }

    f = ROOT.TFile.Open(args.fitfile)
    if not f or f.IsZombie():
        raise SystemExit(f'ERROR: cannot open {args.fitfile}')
    fr = f.Get(args.fit)
    if not fr:
        raise SystemExit(f'ERROR: no {args.fit} in {args.fitfile}')
    pars = fr.floatParsFinal()

    all_names = {pars.at(i).GetName() for i in range(pars.getSize())}
    # Each fitted name is also findable under its year-stripped alias, so the
    # candidate lists above match both plain and year-tagged parameters.
    by_stripped = {}
    for n in sorted(all_names):
        by_stripped.setdefault(_YEAR_TAG_RE.sub('', n), n)

    chosen = {}
    for proc, names in candidates.items():
        hit = next((n for n in names if n in all_names), None)
        if hit is None:
            base = next((b for b in names if b in by_stripped), None)
            hit = by_stripped[base] if base else None
        if hit is None:
            print(f'  [warn] no rateParam found for {proc} '
                  f'(tried {", ".join(names)}, each also with a year '
                  f'suffix); it will be treated as 1.0')
        else:
            chosen[hit] = proc
    print(f'  rateParams for region {args.region}: '
          + (', '.join(f'{n} -> {p}' for n, p in chosen.items()) or 'none'))

    # Every norm_* that is NOT this region's is dropped: it does not affect this
    # region's prediction, so it belongs neither in the shape morphing nor in
    # the covariance block.
    other_norms = {n for n in all_names
                   if n.startswith(('norm_top', 'norm_wjet'))
                   and n not in chosen}
    if other_norms:
        print(f'  ignoring other regions\' rateParams: {", ".join(sorted(other_norms))}')

    theta, rates = {}, {}
    for i in range(pars.getSize()):
        p = pars.at(i)
        name, val = p.GetName(), p.getVal()
        if name == 'mh' or name.startswith('prop_bin'):
            continue
        if name == 'r':
            rates[SIGNAL_KEY] = val
        elif name in chosen:
            rates[chosen[name]] = val
        elif name in other_norms:
            continue
        else:
            theta[name] = val

    cov_full = fr.covarianceMatrix()
    keep_idx, keep_names = [], []
    for i in range(pars.getSize()):
        name = pars.at(i).GetName()
        if name in theta:
            keep_idx.append(i)
            keep_names.append(name)
        elif name in chosen:
            keep_idx.append(i)
            keep_names.append(chosen[name])
    n = len(keep_idx)
    cov = [[cov_full(keep_idx[a], keep_idx[b]) for b in range(n)]
           for a in range(n)]
    f.Close()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as fh:
        json.dump({'region': args.region,
                   'fitfile': os.path.abspath(args.fitfile),
                   'theta': theta, 'rates': rates,
                   'cov_names': keep_names, 'cov': cov}, fh)
    print(f'Wrote {args.output}: {len(theta)} nuisances, rates={rates}, '
          f'cov {n}x{n}')


if __name__ == '__main__':
    main()
