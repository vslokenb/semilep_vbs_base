#!/usr/bin/env python3
"""
check_hybrid_norms.py -- audit EXISTING hybrid datasets: were the SR
pseudo-data actually built at the CR-fit normalizations, or silently at 1.0?

Standalone, outside run_all.sh; needs only uproot + numpy (the apptainer env).
It does NOT trust any log output: the injected norms are RECOVERED from the
files alone. The SR pseudo-data was built as

    data_obs = r * sum(signals) + sum_p scale_p * nominal_p

where scale_p is the fitted norm for processes covered by a rateParam and 1.0
for everything else. Every template on the right-hand side is in the same
shapes file as the pseudo-data, so with the card saying which rateParam covers
which (bin, process), the per-rateParam scales are the solution of a small
linear least-squares across the SR bins. A clean build reproduces the
pseudo-data to float precision; a build that silently fell back to 1.0
(the name-mismatch bug: year-tagged card rateParams vs bare-name --norms)
recovers scales ~= 1.0 instead of the CR values.

Usage (leaf directories only -- each must hold one datacard.txt whose shapes
file is the hybrid one; for the 'resolved'/'boosted' topology targets point at
their per-category SUBdirectories, which is where the hybrids are built):

  python3 python/check_hybrid_norms.py \
      cards_crdata_srasimov_2017/combined \
      cards_crdata_srasimov_2017/resolved_mu \
      cards_crdata_srasimov_2017/resolved/resolved_mu \
      --cr-norms cards_cr_2017/cr_only/cr_only_norms.txt

  # or compare against explicit values instead of the 5c table:
  ... --expected "norm_top_resolved_2017=0.9389,norm_wjet_resolved_2017=1.2421"

Which 5c case a directory is compared against:
  <base>/combined           -> the 'shared' fit
  <base>/<category>         -> that category's own fit
  <base>/<topology>/<cat>   -> the TOPOLOGY fit (that is what step 5d injects
                               when assembling topology cards)

Exit status 0 only if every directory passes.
"""
import argparse
import fnmatch
import os
import re
import sys

import numpy as np

INJECT_SIGNAL_DEFAULT = 1.0
REL_TOL = 2e-3     # recovered vs expected; the build is float-exact, the 5c
                   # table is printed with 4-6 decimals


# ---------------------------------------------------------------------------
# Datacard parsing (the subset this check needs)
# ---------------------------------------------------------------------------

def parse_card(path):
    shapes_generic = None
    bin_rows, proc_rows = [], []
    rate_params = []
    for raw in open(path):
        line = raw.strip()
        if not line or line.startswith(('#', '-')):
            continue
        tok = line.split()
        if tok[0] == 'shapes' and tok[1] == '*':
            shapes_generic = tok[3]
        elif tok[0] == 'bin':
            bin_rows.append(tok[1:])
        elif tok[0] == 'process':
            proc_rows.append(tok[1:])
        elif len(tok) > 4 and tok[1] == 'rateParam':
            rate_params.append((tok[0], tok[2], tok[3]))
    if shapes_generic is None or len(bin_rows) < 2 or len(proc_rows) != 2:
        raise SystemExit(f'{path}: not a single-card datacard this check '
                         f'understands (topology dirs: point at their '
                         f'per-category subdirectories instead)')
    names, indices = proc_rows
    columns = list(zip(bin_rows[1], names, [int(i) for i in indices]))
    return shapes_generic, columns, rate_params


# ---------------------------------------------------------------------------
# Expected norms
# ---------------------------------------------------------------------------

def parse_cr_norms_table(path):
    """{case: {param: value}} from cr_only_norms.txt (step 5c output)."""
    out, case = {}, None
    val_re = re.compile(r'^\d+\.\d+$')
    for raw in open(path):
        tok = raw.split()
        if not tok:
            continue
        if tok[0].startswith('norm_'):
            param, rest = tok[0], tok[1:]
        elif len(tok) >= 2 and tok[1].startswith('norm_'):
            case, param, rest = tok[0], tok[1], tok[2:]
        else:
            continue
        val = next((t for t in rest if val_re.match(t)), None)
        if case is not None and val is not None:
            out.setdefault(case, {})[param] = float(val)
    return out


def expected_case_for(dirpath):
    """Which 5c case step 5d injected for this hybrid directory."""
    base = os.path.basename(os.path.normpath(dirpath))
    parent = os.path.basename(os.path.dirname(os.path.normpath(dirpath)))
    if parent in ('resolved', 'boosted'):
        return parent
    if base == 'combined':
        return 'shared'
    return base


def parse_expected_arg(spec):
    out = {}
    for item in (spec or '').split(','):
        item = item.strip()
        if not item:
            continue
        k, _, v = item.partition('=')
        out[k.strip()] = float(v)
    return out


# ---------------------------------------------------------------------------
# Norm recovery
# ---------------------------------------------------------------------------

def read_file(path):
    import uproot
    out = {}
    with uproot.open(path) as f:
        for key, classname in f.classnames(cycle=False).items():
            if classname.startswith('TDirectory'):
                continue
            out[key] = np.asarray(f[key].values(), dtype=np.float64)
    return out


def solve_normal_equations(ata, atb):
    """Solve the K x K system by Gaussian elimination (K <= 4 here)."""
    n = len(atb)
    a = [row[:] + [atb[i]] for i, row in enumerate(ata)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[piv][col]) < 1e-12:
            raise SystemExit('ERROR: degenerate system -- the covered '
                             'processes have indistinguishable shapes here.')
        a[col], a[piv] = a[piv], a[col]
        for r in range(n):
            if r == col:
                continue
            f = a[r][col] / a[col][col]
            for c in range(col, n + 1):
                a[r][c] -= f * a[col][c]
    return [a[i][n] / a[i][i] for i in range(n)]


def recover_norms(dirpath, inject_signal):
    """(recovered {name: value}, max relative residual of the reconstruction)."""
    card_path = os.path.join(dirpath, 'datacard.txt')
    shapes_rel, columns, rate_params = parse_card(card_path)
    hists = read_file(os.path.normpath(os.path.join(dirpath, shapes_rel)))

    bins = []
    for b in dict.fromkeys(ch for ch, _, _ in columns):
        if any(ch == b and idx <= 0 for ch, _, idx in columns):
            bins.append(b)           # SR bins: where signal columns live
    if not bins:
        raise SystemExit(f'{card_path}: no signal-region bin found')

    rp_names = [n for n in dict.fromkeys(n for n, _, _ in rate_params)]

    def rps_for(b, p):
        return [n for n, bp, pp in rate_params
                if fnmatch.fnmatchcase(b, bp) and fnmatch.fnmatchcase(p, pp)]

    k = len(rp_names)
    ata = [[0.0] * k for _ in range(k)]
    atb = [0.0] * k
    resid_check = []   # (bin arrays) to evaluate the reconstruction afterwards

    for b in bins:
        data = hists[f'{b}/data_obs']
        fixed = np.zeros_like(data)
        cols = {n: np.zeros_like(data) for n in rp_names}
        for ch, p, idx in columns:
            if ch != b:
                continue
            nom = hists.get(f'{ch}/{p}')
            if nom is None:
                raise SystemExit(f'{card_path}: missing template {ch}/{p}')
            covering = rps_for(ch, p)
            if idx <= 0:
                fixed = fixed + inject_signal * nom
            elif len(covering) == 0:
                fixed = fixed + nom
            elif len(covering) == 1:
                cols[covering[0]] = cols[covering[0]] + nom
            else:
                raise SystemExit(f'{card_path}: {ch}/{p} covered by several '
                                 f'rateParams ({covering}); not supported')
        r = data - fixed
        for i, ni in enumerate(rp_names):
            atb[i] += float(np.sum(cols[ni] * r))
            for j, nj in enumerate(rp_names):
                ata[i][j] += float(np.sum(cols[ni] * cols[nj]))
        resid_check.append((b, r, cols, data))

    active = [i for i in range(k) if ata[i][i] > 0]
    sub_ata = [[ata[i][j] for j in active] for i in active]
    sub_atb = [atb[i] for i in active]
    sol = solve_normal_equations(sub_ata, sub_atb)
    recovered = {rp_names[i]: sol[n] for n, i in enumerate(active)}

    worst = 0.0
    for b, r, cols, data in resid_check:
        model = sum(recovered.get(n, 0.0) * cols[n] for n in rp_names)
        scale = max(float(np.sum(np.abs(data))), 1e-9)
        worst = max(worst, float(np.sum(np.abs(r - model))) / scale)
    return recovered, worst


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dirs', nargs='+',
                    help='hybrid output directories (datacard.txt + '
                         'histograms_hybrid.root each)')
    ap.add_argument('--cr-norms', default=None,
                    help='cards_cr_<year>/cr_only/cr_only_norms.txt; expected '
                         'values are taken from the matching 5c case')
    ap.add_argument('--expected', default=None,
                    help="explicit 'name=value,...' expectation, overriding "
                         "--cr-norms for every directory")
    ap.add_argument('--inject-signal', type=float,
                    default=INJECT_SIGNAL_DEFAULT,
                    help='r used when the pseudo-data was built (default 1)')
    args = ap.parse_args()

    table = parse_cr_norms_table(args.cr_norms) if args.cr_norms else {}
    override = parse_expected_arg(args.expected)

    ok = True
    for d in args.dirs:
        print(f'\n=== {d} ===')
        recovered, resid = recover_norms(d, args.inject_signal)
        if resid > 1e-6:
            print(f'  [warn] reconstruction residual {resid:.2e} -- the '
                  f'pseudo-data is not an exact linear combination of the '
                  f'templates (different inject-signal? edited file?)')
        expected = dict(override)
        case = expected_case_for(d)
        if not expected and table:
            expected = table.get(case, {})
            if not expected:
                print(f'  [warn] no 5c case {case!r} in the norms table; '
                      f'reporting recovered values only')
        for name in sorted(recovered):
            rec = recovered[name]
            line = f'  {name:<28} injected = {rec:.5f}'
            exp = expected.get(name)
            if exp is not None:
                diff = abs(rec - exp) / max(abs(exp), 1e-9)
                verdict = 'OK' if diff <= REL_TOL else 'MISMATCH'
                line += f'   5c ({case}) = {exp:.5f}   [{verdict}]'
                if verdict == 'MISMATCH':
                    ok = False
                    if abs(rec - 1.0) <= 5e-3 and abs(exp - 1.0) > 5e-3:
                        line += '  <- built at norm = 1.0 (the silent '
                        line += 'name-mismatch bug); rebuild this hybrid'
            elif abs(rec - 1.0) <= 5e-3:
                line += '   (~1.0: suspicious if the CR fit was not ~1.0)'
            print(line)
        if not recovered:
            print('  [warn] card carries no rateParams at all')

    print('\nAll hybrids consistent.' if ok
          else '\nMISMATCHES found -- those hybrids (and every fit built on '
               'them) must be regenerated.')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
