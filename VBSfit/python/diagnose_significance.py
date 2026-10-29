#!/usr/bin/env python3
"""
diagnose_significance.py -- find out where a combined significance comes from.

Computes the STAT-ONLY Asimov significance directly from the datacard's
templates, bin by bin, and compares it with what the fit reports.

WHY THIS IS DECISIVE
--------------------
For an Asimov dataset built at r = 1, the per-bin contribution to q0 is

    q0(bin) = 2 * [ (s+b) * ln(1 + s/b) - s ]        Z = sqrt(sum q0)

(Cowan et al., the standard asymptotic result). Profiling nuisance parameters
can only make the significance SMALLER, so this number is an upper bound on
what `combine -M Significance -t -1 --expectSignal 1` may report.

That gives a clean split:

  reported Z  >  this bound      the workspace is not the model you think it
                                 is -- duplicated channels, signal counted
                                 twice, a card built from the wrong file.

  reported Z ~= this bound, and  the templates themselves produce it. Look at
  the bound is itself too high   the per-bin table: a single bin with b ~ 0 and
                                 s > 0 sends s/b -> infinity and dominates the
                                 sum. That is consistent with a sensible r and
                                 sigma(r) and a sensible impact plot, because
                                 those are driven by the bulk of the
                                 distribution while q0 is driven by the tail.

The second case is the common one for a BDT whose last bin is nearly pure
signal, and it is why the per-bin table below is sorted by q0.

Reads nothing but the datacard, its shapes file, and optionally a
FitDiagnostics result. PyROOT only -- runs in the same CMSSW environment as
combine.

Usage:
  python3 python/diagnose_significance.py \
      --datacard cards_cr/combined/datacard.txt \
      --fitfile plots_cr/fitdiagnostics/fitDiagnostics_simfit.root \
      --reported 4.31
"""

import argparse
import fnmatch
import math
import os
import sys

import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kError


# ---------------------------------------------------------------------------
# datacard parsing (same minimal reader as make_hybrid_data.py)
# ---------------------------------------------------------------------------

class Card(object):
    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.dir = os.path.dirname(self.path)
        self.lines = open(self.path).read().splitlines()
        self.shapes = []        # (process, bin, file, nominal, syst)
        self.columns = []       # (bin, process, index, rate)
        self.rate_params = []   # (name, bin_pat, proc_pat, init)
        self._parse()

    def _parse(self):
        idx_rate = None
        for i, line in enumerate(self.lines):
            tok = line.split()
            if not tok or line.lstrip().startswith("#"):
                continue
            if tok[0] == "shapes" and len(tok) >= 5:
                self.shapes.append((tok[1], tok[2], tok[3], tok[4],
                                    tok[5] if len(tok) > 5 else None))
            elif tok[0] == "rate" and idx_rate is None:
                idx_rate = i
            elif len(tok) > 1 and tok[1] == "rateParam" and len(tok) >= 5:
                try:
                    init = float(tok[4])
                except ValueError:
                    init = 1.0
                self.rate_params.append((tok[0], tok[2], tok[3], init))
        if idx_rate is None:
            raise SystemExit(f"ERROR: no 'rate' line in {self.path}")

        bin_line = None
        for i in range(idx_rate - 1, -1, -1):
            if self.lines[i].split() and self.lines[i].split()[0] == "bin":
                bin_line = i
                break
        proc_lines = [i for i in range(bin_line, idx_rate)
                      if self.lines[i].split()
                      and self.lines[i].split()[0] == "process"]
        if len(proc_lines) != 2:
            raise SystemExit(f"ERROR: expected 2 'process' lines in {self.path}")

        bins = self.lines[bin_line].split()[1:]
        a = self.lines[proc_lines[0]].split()[1:]
        b = self.lines[proc_lines[1]].split()[1:]
        try:
            [int(x) for x in a]
            names, idxs = b, a
        except ValueError:
            names, idxs = a, b
        rates = self.lines[idx_rate].split()[1:]
        for bb, p, k, r in zip(bins, names, idxs, rates):
            self.columns.append((bb, p, int(k), float(r)))

    def bins(self):
        out = []
        for b, _, _, _ in self.columns:
            if b not in out:
                out.append(b)
        return out

    def shape_entry(self, process, chan):
        for want in ((process, chan), (process, "*"), ("*", chan), ("*", "*")):
            for e in self.shapes:
                if (e[0], e[1]) == want:
                    return e
        return None

    def rate_factor(self, chan, process, norms):
        f = 1.0
        for name, bp, pp, init in self.rate_params:
            if fnmatch.fnmatchcase(chan, bp) and fnmatch.fnmatchcase(process, pp):
                f *= norms.get(name, init)
        return f


def subst(pattern, process, chan, mass):
    return (pattern.replace("$PROCESS", process)
                   .replace("$CHANNEL", chan)
                   .replace("$MASS", str(mass)))


# ---------------------------------------------------------------------------
# the Asimov significance
# ---------------------------------------------------------------------------

def bin_q0(s, b):
    """2*[(s+b) ln(1 + s/b) - s]; None when b <= 0 and s > 0 (unbounded)."""
    if s <= 0:
        return 0.0
    if b <= 0:
        return None
    return 2.0 * ((s + b) * math.log1p(s / b) - s)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datacard", required=True)
    ap.add_argument("--fitfile", default=None,
                    help="FitDiagnostics result, for the r +/- sigma comparison")
    ap.add_argument("--reported", type=float, default=None,
                    help="the significance combine printed, to compare against")
    ap.add_argument("--mass", type=int, default=125)
    ap.add_argument("--norms", default="",
                    help="rateParam values to apply, e.g. 'norm_top_resolved=0.93'")
    ap.add_argument("--top", type=int, default=15,
                    help="how many bins to list (default 15)")
    ap.add_argument("--min-b", type=float, default=1e-3,
                    help="flag bins with background below this (default 1e-3)")
    args = ap.parse_args()

    norms = {}
    for item in args.norms.split(","):
        if "=" in item:
            k, v = item.split("=", 1)
            norms[k.strip()] = float(v)

    card = Card(args.datacard)
    files = {}
    for _, _, f, _, _ in card.shapes:
        p = f if os.path.isabs(f) else os.path.join(card.dir, f)
        files.setdefault(os.path.normpath(p), None)
    for p in list(files):
        rf = ROOT.TFile.Open(p)
        if not rf or rf.IsZombie():
            raise SystemExit(f"ERROR: cannot open shapes file {p}")
        files[p] = rf

    print("=" * 78)
    print(f"Asimov significance decomposition: {args.datacard}")
    print("=" * 78)

    # --- duplicated columns are the first thing to rule out ----------------
    seen_cols = {}
    dupes = []
    for b, p, k, r in card.columns:
        if (b, p) in seen_cols:
            dupes.append((b, p))
        seen_cols[(b, p)] = r
    if dupes:
        print("\n  ERROR: the same (bin, process) appears more than once:")
        for b, p in dupes:
            print(f"    {b}  {p}")
        print("  That double-counts those templates. Everything below is"
              " computed on the card as written.")

    rows = []          # (q0, chan, ibin, s, b)
    per_chan = {}      # chan -> [q0_sum, s_sum, b_sum, n_unbounded]
    unbounded = []

    for chan in card.bins():
        s_tot = b_tot = 0.0
        q_tot = 0.0
        n_unb = 0
        s_arr = b_arr = None

        for cb, proc, kind, rate in card.columns:
            if cb != chan:
                continue
            entry = card.shape_entry(proc, chan)
            if entry is None:
                continue
            path = os.path.normpath(entry[2] if os.path.isabs(entry[2])
                                    else os.path.join(card.dir, entry[2]))
            h = files[path].Get(subst(entry[3], proc, chan, args.mass))
            if not h:
                print(f"  [warn] missing template for {chan}/{proc}")
                continue
            vals = [max(h.GetBinContent(i), 0.0)
                    for i in range(1, h.GetNbinsX() + 1)]
            integral = sum(vals)
            if rate >= 0 and integral > 0 and abs(rate - integral) / integral > 1e-6:
                vals = [v * rate / integral for v in vals]
            scale = 1.0 if kind <= 0 else card.rate_factor(chan, proc, norms)
            vals = [v * scale for v in vals]

            if s_arr is None:
                s_arr = [0.0] * len(vals)
                b_arr = [0.0] * len(vals)
            target = s_arr if kind <= 0 else b_arr
            for i, v in enumerate(vals):
                target[i] += v

        if s_arr is None:
            continue
        for i, (s, b) in enumerate(zip(s_arr, b_arr)):
            s_tot += s
            b_tot += b
            q = bin_q0(s, b)
            if q is None:
                n_unb += 1
                unbounded.append((chan, i + 1, s, b))
            else:
                q_tot += q
                rows.append((q, chan, i + 1, s, b))
        per_chan[chan] = [q_tot, s_tot, b_tot, n_unb]

    for rf in files.values():
        rf.Close()

    # --- per-channel table -------------------------------------------------
    print(f"\n{'channel':<26}{'signal':>12}{'background':>14}"
          f"{'S/sqrt(B)':>12}{'Z (stat)':>10}{'b<=0 bins':>11}")
    print("-" * 85)
    tot_q = 0.0
    for chan, (q, s, b, n_unb) in per_chan.items():
        tot_q += q
        srb = s / math.sqrt(b) if b > 0 else float("inf")
        print(f"{chan:<26}{s:>12.2f}{b:>14.2f}{srb:>12.3f}"
              f"{math.sqrt(q):>10.3f}{n_unb:>11d}")
    print("-" * 85)
    z_bound = math.sqrt(tot_q)
    print(f"{'COMBINED (stat-only bound)':<26}{'':>12}{'':>14}{'':>12}"
          f"{z_bound:>10.3f}")

    if unbounded:
        print(f"\n  {len(unbounded)} bin(s) have signal with ZERO background.")
        print("  Their q0 is unbounded, so the number above EXCLUDES them and")
        print("  the true asymptotic value is larger still. In combine these")
        print("  bins are handled by autoMCStats, but they are exactly what")
        print("  inflates a significance while leaving r and sigma(r) sane.")
        for chan, i, s, b in unbounded[:args.top]:
            print(f"    {chan:<26} bin {i:<4} s={s:.4f}  b={b:.6f}")

    # --- the bins that actually make the number ----------------------------
    rows.sort(reverse=True)
    print(f"\nTop {args.top} bins by q0 contribution:")
    print(f"  {'channel':<26}{'bin':>5}{'s':>10}{'b':>12}{'s/b':>10}"
          f"{'q0':>10}{'% of q0':>9}")
    print("  " + "-" * 82)
    for q, chan, i, s, b in rows[:args.top]:
        frac = 100.0 * q / tot_q if tot_q > 0 else 0.0
        sb = s / b if b > 0 else float("inf")
        print(f"  {chan:<26}{i:>5}{s:>10.3f}{b:>12.4f}{sb:>10.3f}"
              f"{q:>10.3f}{frac:>8.1f}%")

    thin = [r for r in rows if r[4] < args.min_b and r[3] > 0]
    if thin:
        print(f"\n  {len(thin)} bin(s) have background below {args.min_b} with"
              f" signal present -- these dominate q0 and are the first thing")
        print("  to rebin away if the significance looks too good.")

    # --- compare with the fit ----------------------------------------------
    if args.fitfile:
        f = ROOT.TFile.Open(args.fitfile)
        fr = f.Get("fit_s") if f and not f.IsZombie() else None
        if fr:
            p = fr.floatParsFinal().find("r")
            if p:
                lo, hi = abs(p.getErrorLo()), abs(p.getErrorHi())
                sig = max(lo, hi, p.getError())
                print(f"\nFitDiagnostics: r = {p.getVal():.4f} "
                      f"-{lo:.4f}/+{hi:.4f}")
                if sig > 0:
                    print(f"  naive r/sigma      = {p.getVal() / sig:.3f}")
                    print("  For an Asimov fit this should land near the")
                    print("  reported significance. A large gap means the two")
                    print("  are not describing the same likelihood.")
        if f:
            f.Close()

    if args.reported is not None:
        print(f"\nReported significance : {args.reported:.3f}")
        print(f"Stat-only bound       : {z_bound:.3f}")
        if unbounded:
            print("  (bound is a LOWER estimate -- zero-background bins excluded)")
        elif args.reported > z_bound * 1.02:
            print("\n  >> The reported significance EXCEEDS the stat-only bound.")
            print("     Systematics cannot increase it, so the workspace is not")
            print("     the model in this datacard. Check for duplicated")
            print("     channels, a card pointing at the wrong shapes file, or")
            print("     signal entering more than once.")
        else:
            print("\n  >> Consistent with the bound: the significance is what")
            print("     these templates imply. If it is still too high, the")
            print("     cause is in the per-bin table above, not in the fit.")


if __name__ == "__main__":
    main()