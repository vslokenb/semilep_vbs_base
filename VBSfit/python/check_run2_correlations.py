#!/usr/bin/env python3
"""
check_run2_correlations.py -- verify the cross-year correlation scheme of a
Run-2 combined datacard before any fit runs on it.

The scheme (set when the per-year input cards are made with
make_datacards.py --decorrelate-years none and combined with
combineCards.py Y<year>=...):

  1. Experimental nuisances whose names carry a year label
     (AK4PFchs_JES_Regrouped_EC2_2017, AK4PFchs_JER_2018, lumi_2016_PreVFP,
     ...) are DEcorrelated across years: each must act ONLY on its own
     year's channels.
  2. Experimental nuisances without a year label share one name across
     years: each must act on channels of MORE THAN ONE year (when the
     process exists there), i.e. be correlated.
  3. Theory uncertainties (scales, PDF, parton shower) must never carry a
     year label -- always correlated across years. The per-MC-group tags
     (sf_partonshower_isr_TTbar) are process decorrelation, not year.
  4. The norm_top/norm_wjet rateParams must carry no year suffix, so one
     parameter floats each background across all years of a topology.

Violations exit nonzero so the pipeline stops before fitting a card that
does not implement the intended scheme. Checks that depend on seeing more
than one year degrade to notes when only one year is present.

Usage:
  python3 check_run2_correlations.py --card cards_run2/resolved_mu/datacard.txt \
      --years "2018 2017 2016_PreVFP 2016_PostVFP"
"""
import argparse
import fnmatch
import re
import sys

THEORY_PATTERNS = [
    'LHEScaleWeight*', 'LHEPdfWeight*', 'sf_partonshower*',
    'QCDscale*', 'pdf_*', 'pdf', 'alphaS*', 'sf_lhe*',
]


def year_label_of(name, years):
    """(label token, set of years it allows) for a labeled name, or None.

    First tier: an exact year token ending the name (AK4..._2017,
    lumi_2016_PreVFP) or the collision-avoiding _yr<year> form -> allows
    exactly that year. Second tier: a bare era token like _2016 (the JME
    Regrouped components shared by both 2016 halves carry it) -> allows
    every year-era starting with that token, so acting on both 2016 halves
    is fine while leaking into 2017/2018 is a violation.
    """
    for y in sorted(years, key=len, reverse=True):
        if name.endswith(f'_{y}') or name.endswith(f'_yr{y}'):
            return y, {y}
    m = re.search(r'_(yr)?(20\d{2})$', name)
    if m:
        tok = m.group(2)
        return tok, {y for y in years if y.startswith(tok)}
    return None


def is_theory(name, years):
    base = name
    lab = year_label_of(name, years)
    if lab:  # strip the label so a (wrongly) year-tagged theory name still classifies
        tok = lab[0]
        suffix = f'_yr{tok}' if name.endswith(f'_yr{tok}') else None
        if suffix is None:
            for y in sorted(years, key=len, reverse=True):
                if name.endswith(f'_{y}'):
                    suffix = f'_{y}'
                    break
            else:
                suffix = f'_{tok}'
        base = base[: -len(suffix)]
    return any(fnmatch.fnmatchcase(base, p) or fnmatch.fnmatchcase(name, p)
               for p in THEORY_PATTERNS)


def parse_datacard(path):
    """(bin_of_column, nuisance rows, rateparam names) from a combine card."""
    bins_rows = []
    nuis = []      # (name, type, [cols])
    rateparams = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip('\n')
            if not line.strip() or line.lstrip().startswith(('#', '-')):
                continue
            tok = line.split()
            if tok[0] == 'bin':
                bins_rows.append(tok[1:])
            elif len(tok) >= 4 and tok[1] == 'rateParam':
                rateparams.append((tok[0], tok[2], tok[3]))
            elif len(tok) >= 2 and tok[1] == 'rateParam':
                raise SystemExit(
                    f'ERROR: malformed rateParam line in {path}: {line!r}')
            elif len(tok) >= 3 and tok[1] in ('lnN', 'shape', 'shapeN2', 'lnU'):
                nuis.append((tok[0], tok[1], tok[2:]))
    if len(bins_rows) < 2:
        raise SystemExit(f'ERROR: could not find the bin/process table in {path}')
    # first 'bin' row is the observation header; the second labels the columns
    return bins_rows[1], nuis, rateparams


def bin_year(bin_name, years):
    """Which year a (combineCards-prefixed) bin belongs to: Y<year>_..."""
    for y in sorted(years, key=len, reverse=True):
        if bin_name.startswith(f'Y{y}_'):
            return y
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--card', required=True)
    ap.add_argument('--years', required=True,
                    help='space-separated year-eras expected in the card')
    args = ap.parse_args()
    years = args.years.split()

    columns_bin, nuis, rateparams = parse_datacard(args.card)
    bin_years = {b: bin_year(b, years) for b in set(columns_bin)}
    unknown = sorted(b for b, y in bin_years.items() if y is None)
    if unknown:
        raise SystemExit(
            f'ERROR: {len(unknown)} bin(s) carry no Y<year>_ prefix, e.g. '
            f'{unknown[0]} -- was the card built with '
            f'combineCards.py Y<year>=... labels?')
    years_present = sorted({y for y in bin_years.values()})

    violations, notes = [], []
    n_dec = n_cor = n_theory = 0

    for name, typ, cols in nuis:
        active_years = {bin_years[columns_bin[i]]
                        for i, c in enumerate(cols)
                        if c != '-' and i < len(columns_bin)}
        label = year_label_of(name, years)
        theory = is_theory(name, years)

        if theory:
            n_theory += 1
            if label:
                violations.append(
                    f'theory nuisance {name} carries year label {label[0]} '
                    f'(theory must be correlated across years)')
            continue

        if label:
            tok, allowed = label
            n_dec += 1
            wrong = active_years - allowed
            if wrong:
                violations.append(
                    f'{name} is labeled {tok} (allows {sorted(allowed)}) '
                    f'but acts on {sorted(wrong)} channels too')
        else:
            n_cor += 1
            if len(years_present) > 1 and len(active_years) == 1:
                notes.append(
                    f'unlabeled {name} only acts in {sorted(active_years)} '
                    f'(fine if the source exists only there)')

    for pname, pbin, pproc in rateparams:
        if year_label_of(pname, years):
            violations.append(
                f'rateParam {pname} carries a year suffix (normalizations '
                f'must be correlated across years)')

    print(f'Correlation-scheme check: {args.card}')
    print(f'  years in card: {", ".join(years_present)}')
    print(f'  nuisances: {n_cor} correlated (unlabeled), '
          f'{n_dec} year-decorrelated (labeled), {n_theory} theory')
    print(f'  rateParams: ' + (', '.join(sorted({p for p, _, _ in rateparams}))
                               or 'none'))
    for note in notes:
        print(f'  [note] {note}')
    if violations:
        print(f'\n  {len(violations)} VIOLATION(S) of the correlation scheme:')
        for v in violations:
            print(f'    - {v}')
        sys.exit(1)
    if len(years_present) < 2:
        print('  [note] only one year present: cross-year correlation is '
              'not testable, naming checks passed')
    print('  OK: card implements the intended cross-year correlation scheme')


if __name__ == '__main__':
    main()
