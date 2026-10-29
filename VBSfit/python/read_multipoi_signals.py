#!/usr/bin/env python3
"""
read_multipoi_singles.py -- read per-POI best fit and +/-1 sigma from the
output of  combine -M MultiDimFit --algo singles  with several POIs.

    read_multipoi_singles.py <higgsCombine*.MultiDimFit.*.root> r_WW,r_WZ

prints one line per POI:

    r_WW    1.0000  -0.3120 / +0.3405
    r_WZ    1.0000  -0.5533 / +0.6021

With N POIs the 'limit' tree holds 1 + 2N entries: the best fit first, then a
down/up pair for each POI, in the order the POIs were given to
--redefineSignalPOIs.

Each POI's interval comes from ITS OWN pair of entries. Taking the min/max of a
branch over all entries would be wrong: while combine scans r_WZ, r_WW is
profiled and can move well outside its own +/-1 sigma range, so the spread of
the r_WW branch across every entry is not the r_WW uncertainty.

Reads the tree instead of parsing stdout because the printed summary is rounded
to 3 decimals.
"""

import sys

import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kError


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    path, pois = sys.argv[1], [p for p in sys.argv[2].split(",") if p]

    f = ROOT.TFile.Open(path)
    if not f or f.IsZombie():
        sys.stderr.write("  [warn] cannot open %s\n" % path)
        return 1
    t = f.Get("limit")
    if not t:
        sys.stderr.write("  [warn] no 'limit' tree in %s\n" % path)
        f.Close()
        return 1

    n = t.GetEntries()
    expected = 1 + 2 * len(pois)
    if n < expected:
        sys.stderr.write("  [warn] %s: expected %d entries for %d POI(s), "
                         "found %d -- some scans failed\n"
                         % (path, expected, len(pois), n))

    have = {b.GetName() for b in t.GetListOfBranches()}
    missing = [p for p in pois if p not in have]
    if missing:
        sys.stderr.write("  [warn] %s: no branch for %s\n"
                         % (path, ", ".join(missing)))
        pois = [p for p in pois if p in have]
    if not pois or n == 0:
        f.Close()
        return 1

    def value_at(entry, poi):
        t.GetEntry(entry)
        return float(getattr(t, poi))

    best = {p: value_at(0, p) for p in pois}

    for i, p in enumerate(pois):
        lo_entry, hi_entry = 1 + 2 * i, 2 + 2 * i
        if hi_entry >= n:
            print("%-8s %8.4f   scan points missing" % (p, best[p]))
            continue
        v1, v2 = value_at(lo_entry, p), value_at(hi_entry, p)
        lo, hi = min(v1, v2), max(v1, v2)
        b = best[p]
        if not (lo <= b <= hi):
            sys.stderr.write("  [warn] %s: best fit %.4f is outside its scan "
                             "points [%.4f, %.4f]; the fit did not converge\n"
                             % (p, b, lo, hi))
        print("%-8s %8.4f  -%.4f / +%.4f" % (p, b, b - lo, hi - b))

    f.Close()
    return 0


if __name__ == "__main__":
    sys.exit(main())