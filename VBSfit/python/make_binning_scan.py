#!/usr/bin/env python3
"""
make_binning_scan.py -- derive flat-signal BDT binnings for the binning scan.

For each signal region, the summed signal BDT distribution is read from a
NATIVE-binned histogram file (make_histograms.py run WITHOUT --binning keeps
the BDT's 40 native bins) and bin edges are chosen so every merged bin holds
(as nearly as possible) the same signal yield. Because the signal peaks at
high BDT score, equal-signal bins come out coarse at low score and fine at
high score automatically.

Edges always lie ON native bin edges -- rebin_hists_to_edges() can only merge
whole native bins -- so the requested bin count is an upper limit: if the
signal is concentrated in few native bins, adjacent quantile boundaries can
coincide and the variant comes out with fewer bins (reported).

One YAML per variant, in the format make_histograms.py --binning and
plot_cr_closure.py --binning expect:

    resolved_mu: [0.0, 0.4, 0.62, ...]
    resolved_e:  [...]
    boosted_mu:  [...]
    boosted_e:   [...]

Usage (inside the pocketcoffea apptainer, cwd = VBSfit):
  python3 python/make_binning_scan.py \
      --histfile histograms/histograms_native_2018.root \
      --outdir config/binning_scan_2018 \
      --nbins 4 5 6 7 8
"""
import argparse
import math
import os

import numpy as np
import uproot
import yaml

SIGNAL_REGIONS = ['resolved_mu', 'resolved_e', 'boosted_mu', 'boosted_e']
SIGNAL_MATCH = 'dipoleRecoil'
MAX_BINS = 8


def load_region(rf, region):
    """(signal values, signal variances, background values, background
    variances, native edges) -- everything summed over processes. The
    variances (sumw2) feed the MC-stat error on s/sqrt(b) that the shape
    fits' chi2 is measured against."""
    sig = sigv = bkg = bkgv = edges = None
    prefix = f'{region}/'
    for key, cls in rf.classnames(cycle=False).items():
        if not key.startswith(prefix) or cls.startswith('TDirectory'):
            continue
        name = key[len(prefix):]
        if name == 'data_obs' or name.endswith('Up') or name.endswith('Down'):
            continue
        h = rf[key]
        v = np.asarray(h.values(), dtype=np.float64)
        var = np.asarray(h.variances(), dtype=np.float64)
        if edges is None:
            e = h.axes[0].edges   # method on uproot axes, property on hist
            edges = np.asarray(e() if callable(e) else e, dtype=np.float64)
        if SIGNAL_MATCH in name:
            sig = v if sig is None else sig + v
            sigv = var if sigv is None else sigv + var
        else:
            bkg = v if bkg is None else bkg + v
            bkgv = var if bkgv is None else bkgv + var
    return sig, sigv, bkg, bkgv, edges


def flat_signal_edges(sig, edges, n_bins):
    """Native-aligned edges giving ~equal signal per bin.

    Exact dynamic programming over the native grid: of all ways to split the
    native bins into n_bins contiguous groups (each at least one native bin),
    it picks the one minimizing sum_i (signal_i - total/n)^2 -- the most
    equal-signal partition the grid allows. At <= 8 final bins over 40 native
    bins this is a few thousand candidate costs, instantaneous. A greedy
    quantile walk was tried first and left the last bin with the accumulated
    quantization error on steeply rising distributions.

    Final bins that come out with essentially no signal (pathologically
    concentrated inputs) are merged into their neighbor, so the result can
    have fewer than n_bins bins (reported by the caller).
    """
    vals = np.maximum(np.asarray(sig, dtype=np.float64), 0.0)
    n_native = len(vals)
    csum = np.concatenate([[0.0], np.cumsum(vals)])   # csum[i] = first i bins
    total = float(csum[-1])
    if total <= 0:
        return None
    nb = min(n_bins, n_native)
    tgt = total / nb

    INF = float('inf')
    # best[k][j]: minimal cost splitting the first j native bins into k groups
    best = [[INF] * (n_native + 1) for _ in range(nb + 1)]
    prev = [[0] * (n_native + 1) for _ in range(nb + 1)]
    best[0][0] = 0.0
    for k in range(1, nb + 1):
        # group k ends at native bin j; leave >= 1 native bin per later group
        for j in range(k, n_native - (nb - k) + 1):
            for i in range(k - 1, j):
                if best[k - 1][i] == INF:
                    continue
                c = best[k - 1][i] + (float(csum[j] - csum[i]) - tgt) ** 2
                if c < best[k][j]:
                    best[k][j] = c
                    prev[k][j] = i

    bounds = [n_native]
    for k in range(nb, 0, -1):
        bounds.append(prev[k][bounds[-1]])
    bounds = bounds[::-1]

    # merge away (essentially) signal-free bins
    kept = [bounds[0]]
    for b in bounds[1:-1]:
        if float(csum[b] - csum[kept[-1]]) > 1e-6 * total:
            kept.append(b)
    kept.append(bounds[-1])
    return [float(edges[i]) for i in sorted(set(kept))]


def _weighted_line(xs, ys, ws):
    """Weighted least-squares straight line. (intercept, slope) or None."""
    S = sum(ws)
    Sx = sum(w * x for w, x in zip(ws, xs))
    Sy = sum(w * y for w, y in zip(ws, ys))
    Sxx = sum(w * x * x for w, x in zip(ws, xs))
    Sxy = sum(w * x * y for w, x, y in zip(ws, xs, ys))
    det = S * Sxx - Sx * Sx
    if det <= 0:
        return None
    slope = (S * Sxy - Sx * Sy) / det
    return (Sy - slope * Sx) / S, slope


def fit_z_shapes(x, z, zerr):
    """Weighted fits of z(x) = s/sqrt(b) at the bin centers:

        linear       z = a + b*x
        exponential  z = A * exp(k*x)

        gaussian     z = A * exp(-(x - mu)^2 / (2 sigma^2)),
                     mu FIXED at the peak bin's center, fitted on the
                     rising side only (points with x <= mu)

    Returns {'lin': ((a, b), chi2, ndof), 'exp': ((A, k), chi2, ndof),
    'gaus': ((A, mu, sigma), chi2, ndof)} with None where a fit is
    impossible (fewer than 3 usable points; no positive z for the
    exponential; fewer than 3 rising-side points, or no curvature toward
    the peak, for the gaussian). chi2 is against the per-bin MC-stat error
    on z; bins with zero error (b = 0) are excluded and do not count toward
    ndof. The exponential and gaussian are seeded by weighted fits of log z
    (both are linear there) and refined with scipy.optimize.curve_fit in
    linear space when scipy is available.
    """
    pts = [(xi, zi, ei) for xi, zi, ei in zip(x, z, zerr) if ei > 0]
    out = {'lin': None, 'exp': None, 'gaus': None}
    if len(pts) < 3:
        return out

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ws = [1.0 / p[2] ** 2 for p in pts]
    lin = _weighted_line(xs, ys, ws)
    if lin is not None:
        a, b = lin
        chi2 = sum(w * (y - (a + b * xx)) ** 2
                   for w, xx, y in zip(ws, xs, ys))
        out['lin'] = ((a, b), float(chi2), len(pts) - 2)

    pos = [p for p in pts if p[1] > 0]
    if len(pos) >= 3:
        xp = [p[0] for p in pos]
        zp = [p[1] for p in pos]
        ep = [p[2] for p in pos]
        # log-space seed: sigma(log z) = sigma_z / z -> weight (z/sigma)^2
        seed = _weighted_line(xp, [math.log(v) for v in zp],
                              [(v / e) ** 2 for v, e in zip(zp, ep)])
        if seed is not None:
            A, k = math.exp(seed[0]), seed[1]
            try:
                from scipy.optimize import curve_fit
                popt, _ = curve_fit(
                    lambda xx, A_, k_: A_ * np.exp(k_ * np.asarray(xx)),
                    xp, zp, p0=(A, k), sigma=ep, absolute_sigma=True,
                    maxfev=10000)
                A, k = float(popt[0]), float(popt[1])
            except Exception:
                pass   # keep the log-space seed
            chi2 = sum(((v - A * math.exp(k * xx)) / e) ** 2
                       for xx, v, e in zip(xp, zp, ep))
            out['exp'] = ((A, k), float(chi2), len(pos) - 2)

    # Gaussian pinned at the peak, rising side only. In log space
    # ln z = ln A - u / (2 sigma^2) with u = (x - mu)^2: linear in u, so the
    # seed is exact; a non-negative slope means z does not fall away from
    # the peak and no gaussian describes the rise.
    mu = max(pts, key=lambda p: p[1])[0]
    rise = [p for p in pts if p[0] <= mu and p[1] > 0]
    if len(rise) >= 3:
        xr = [p[0] for p in rise]
        zr = [p[1] for p in rise]
        er = [p[2] for p in rise]
        u = [(xx - mu) ** 2 for xx in xr]
        seed = _weighted_line(u, [math.log(v) for v in zr],
                              [(v / e) ** 2 for v, e in zip(zr, er)])
        if seed is not None and seed[1] < 0:
            A = math.exp(seed[0])
            sigma = math.sqrt(-1.0 / (2.0 * seed[1]))
            try:
                from scipy.optimize import curve_fit
                popt, _ = curve_fit(
                    lambda xx, A_, s_: A_ * np.exp(
                        -(np.asarray(xx) - mu) ** 2 / (2.0 * s_ ** 2)),
                    xr, zr, p0=(A, sigma), sigma=er, absolute_sigma=True,
                    maxfev=10000)
                A, sigma = float(popt[0]), abs(float(popt[1]))
            except Exception:
                pass   # keep the log-space seed
            chi2 = sum(((v - A * math.exp(-(xx - mu) ** 2
                                          / (2.0 * sigma ** 2))) / e) ** 2
                       for xx, v, e in zip(xr, zr, er))
            out['gaus'] = ((A, mu, sigma), float(chi2), len(rise) - 2)
    return out


def plot_ssqrtb(variant_name, report, plotdir):
    """One figure per variant: nominal per-bin s/sqrt(b) in all four regions.

    The bar width is the actual bin width, so the coarse-at-low /
    fine-at-high structure is visible. Each panel's legend carries the
    quadrature total sqrt(sum z_i^2) -- the naive stat-only significance
    this binning retains, which is the number to compare across variants.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    for ax, region in zip(axes.flat, SIGNAL_REGIONS):
        e, s, b, z, zerr, fits = report[region]
        zq = math.sqrt(sum(x * x for x in z))
        centers = [0.5 * (e[i] + e[i + 1]) for i in range(len(z))]
        for i, zi in enumerate(z):
            ax.bar(e[i], zi, width=e[i + 1] - e[i], align='edge',
                   color='#5790fc', edgecolor='black', linewidth=0.8,
                   alpha=0.9)
            if b[i] <= 0:
                ax.text(centers[i], 0.02, 'b=0', ha='center', va='bottom',
                        color='#e42536', fontsize=8, rotation=90)
            else:
                ax.text(centers[i], zi + zerr[i], f'{zi:.2f}', ha='center',
                        va='bottom', fontsize=8)
        ax.errorbar(centers, z, yerr=zerr, fmt='none', ecolor='black',
                    elinewidth=1.0, capsize=2)
        # overlay the fitted shapes across the panel
        xs = [e[0] + (e[-1] - e[0]) * t / 200.0 for t in range(201)]
        if fits.get('lin'):
            (a, slope), chi2, ndof = fits['lin']
            ax.plot(xs, [a + slope * xx for xx in xs], '-', color='#e42536',
                    lw=1.5,
                    label=f'linear: $\\chi^2$/ndof = {chi2:.1f}/{ndof}')
        if fits.get('exp'):
            (A, k), chi2, ndof = fits['exp']
            ax.plot(xs, [A * math.exp(k * xx) for xx in xs], '--',
                    color='#964a8b', lw=1.5,
                    label=f'expo: $\\chi^2$/ndof = {chi2:.1f}/{ndof}')
        if fits.get('gaus'):
            (A, mu, sg), chi2, ndof = fits['gaus']
            xg = [xx for xx in xs if xx <= mu]   # drawn over the fit range only
            ax.plot(xg, [A * math.exp(-(xx - mu) ** 2 / (2 * sg ** 2))
                         for xx in xg], ':', color='#55a868', lw=2.0,
                    label=f'gaus rise ($\\mu$={mu:.2f}): '
                          f'$\\chi^2$/ndof = {chi2:.1f}/{ndof}')
        if any(fits.get(k) for k in ('lin', 'exp', 'gaus')):
            ax.legend(fontsize=8, loc='upper left', frameon=False)
        ax.set_xlim(e[0], e[-1])
        ax.set_ylim(0, max([zi + ei for zi, ei in zip(z, zerr)] + [0.1]) * 1.3)
        ax.set_xlabel('BDT score')
        ax.set_ylabel(r'$s/\sqrt{b}$ per bin')
        ax.set_title(f'{region}'
                     f'   $\\sqrt{{\\Sigma z^2}}$ = {zq:.2f}'
                     f'   ({len(z)} bins)', fontsize=11)
    fig.suptitle(f'{variant_name}: nominal per-bin $s/\\sqrt{{b}}$',
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(plotdir, exist_ok=True)
    paths = []
    for ext in ('png', 'pdf'):
        p = os.path.join(plotdir, f'ssqrtb_{variant_name}.{ext}')
        fig.savefig(p, dpi=150)
        paths.append(p)
    plt.close(fig)
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--histfile', required=True,
                    help='NATIVE-binned histograms (make_histograms.py '
                         'without --binning)')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--nbins', type=int, nargs='+', default=[4, 5, 6, 7, 8])
    ap.add_argument('--plotdir', default=None,
                    help='also save a per-variant plot of the nominal '
                         'per-bin s/sqrt(b) (png + pdf) here')
    args = ap.parse_args()

    bad = [n for n in args.nbins if not 2 <= n <= MAX_BINS]
    if bad:
        raise SystemExit(f'ERROR: --nbins must be in [2, {MAX_BINS}]; got {bad}')

    os.makedirs(args.outdir, exist_ok=True)
    regions = {}
    with uproot.open(args.histfile) as rf:
        for region in SIGNAL_REGIONS:
            sig, sigv, bkg, bkgv, edges = load_region(rf, region)
            if sig is None:
                raise SystemExit(f'ERROR: no signal ({SIGNAL_MATCH}) templates '
                                 f'in channel {region} of {args.histfile}')
            regions[region] = (sig, sigv, bkg, bkgv, edges)
            print(f'{region}: {len(edges) - 1} native bins, '
                  f'signal {sig.sum():.2f}, background {bkg.sum():.1f}')

    written = []
    chi2_rows = []
    for n in sorted(set(args.nbins)):
        variant = {}
        report = {}
        print(f'\n=== flat{n}bins ===')
        for region, (sig, sigv, bkg, bkgv, edges) in regions.items():
            e = flat_signal_edges(sig, edges, n)
            if e is None:
                raise SystemExit(f'ERROR: zero signal in {region}')
            variant[region] = e
            # per-bin report: the numbers the binning choice is judged on
            idx = np.searchsorted(edges, e)
            s = [float(sig[idx[i]:idx[i + 1]].sum()) for i in range(len(e) - 1)]
            b = [float(bkg[idx[i]:idx[i + 1]].sum()) for i in range(len(e) - 1)]
            vs = [float(sigv[idx[i]:idx[i + 1]].sum()) for i in range(len(e) - 1)]
            vb = [float(bkgv[idx[i]:idx[i + 1]].sum()) for i in range(len(e) - 1)]
            # z = s/sqrt(b) with the MC-stat error propagated from both:
            #   sigma_z^2 = var_s/b + s^2 * var_b / (4 b^3)
            z, zerr = [], []
            for si, bi, vsi, vbi in zip(s, b, vs, vb):
                if bi > 0:
                    z.append(si / math.sqrt(bi))
                    zerr.append(math.sqrt(vsi / bi
                                          + si * si * vbi / (4.0 * bi ** 3)))
                else:
                    z.append(0.0)
                    zerr.append(0.0)
            centers = [0.5 * (e[i] + e[i + 1]) for i in range(len(z))]
            fits = fit_z_shapes(centers, z, zerr)
            report[region] = (e, s, b, z, zerr, fits)
            chi2_rows.append((f'flat{n}bins', region,
                              fits.get('lin'), fits.get('exp'),
                              fits.get('gaus')))
            tag = '' if len(e) - 1 == n else f'  [only {len(e) - 1} bins possible]'
            print(f'  {region}{tag}')
            print('    edges: ' + ', '.join(f'{x:g}' for x in e))
            print('    sig/bin: ' + ', '.join(f'{x:.2f}' for x in s))
            print('    bkg/bin: ' + ', '.join(f'{x:.1f}' for x in b))
            print('    s/sqrt(b): ' + ', '.join(
                f'{zi:.2f}+-{ei:.2f}' if bi > 0 else 'b=0'
                for zi, ei, bi in zip(z, zerr, b)))
            for label, key in (('linear', 'lin'), ('expo', 'exp'),
                               ('gaus(rise)', 'gaus')):
                fit = fits.get(key)
                if fit:
                    pars, chi2, ndof = fit
                    ps = ', '.join(f'{p:+.3g}' for p in pars)
                    print(f'    {label} fit: ({ps})  '
                          f'chi2/ndof = {chi2:.2f}/{ndof}')
                else:
                    print(f'    {label} fit: not possible (too few usable bins)')

        if args.plotdir:
            for p in plot_ssqrtb(f'flat{n}bins', report, args.plotdir):
                print(f'  -> {p}')

        out = os.path.join(args.outdir, f'flat{n}bins.yaml')
        with open(out, 'w') as fh:
            fh.write('# Flat-signal BDT binning (~equal signal yield per bin)\n'
                     f'# derived from {args.histfile} by make_binning_scan.py\n')
            yaml.safe_dump(variant, fh, default_flow_style=None, sort_keys=True)
        written.append(out)
        print(f'  -> {out}')

    print('\nShape-fit chi2 of per-bin s/sqrt(b) vs bin center '
          '(MC-stat errors; gaussian fixed at the peak, rising side only):')
    print(f'  {"variant":<12}{"region":<13}'
          f'{"linear chi2/ndof":>18}{"expo chi2/ndof":>18}'
          f'{"gaus chi2/ndof":>18}')
    for variant_name, region, lin, expf, gaus in chi2_rows:
        cl = f'{lin[1]:.2f}/{lin[2]}' if lin else 'n/a'
        ce = f'{expf[1]:.2f}/{expf[2]}' if expf else 'n/a'
        cg = f'{gaus[1]:.2f}/{gaus[2]}' if gaus else 'n/a'
        print(f'  {variant_name:<12}{region:<13}{cl:>18}{ce:>18}{cg:>18}')

    print('\nWritten:')
    for w in written:
        print(f'  {w}')


if __name__ == '__main__':
    main()
