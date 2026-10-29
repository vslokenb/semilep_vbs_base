#!/usr/bin/env python3
"""
read_cr_norms.py -- extract the CR-fit normalizations from a FitDiagnostics file.

Prints exactly one line to STDOUT, ready for combine's --setParameters or for
make_hybrid_data.py --norms (names exactly as they float in the fit,
so the year-tagged parameters of multi-year cards come out as themselves):

    norm_top_resolved=0.9259,norm_wjet_resolved=1.2482
    norm_top_resolved_2017=0.9259,norm_wjet_resolved_2017=1.2482

All diagnostics go to STDERR so the stdout line stays machine-readable.
Exits nonzero (printing nothing to stdout) if the values are not usable, so a
caller doing `setpars=$(read_cr_norms.py ...) || skip` never silently falls
back to 1.0.


WHY status IS NOT THE ACCEPTANCE TEST
-------------------------------------
The CR-only fits are run with --robustFit 1 --robustHesse 1. RobustHesse
computes and inverts the Hessian itself instead of leaving it to Minuit's
HESSE, and the RooFitResult that FitDiagnostics saves then carries

    covQual = -1     no Minuit covariance matrix was stored
    status  = 300    the HESSE stage of the status word is nonzero

for a fit that minimized perfectly well -- you can see RobustHesse report
">> Matrix inverted" in the log. Rejecting on status==0 throws away every one
of these fits.

So the acceptance test is on the PARAMETERS, which is what actually matters:

  * present in the fit result
  * finite value and finite, strictly positive error
  * not pinned against the edge of the rateParam range (a boundary value means
    the fit ran out of room and the number is meaningless)
  * within --max-deviation of 1.0, as a guard against a runaway fit

status and covQual are reported loudly on stderr but are advisory. Pass
--strict-status to restore the old behaviour and require status == 0.

If you would rather have a meaningful status word, drop --robustHesse from the
CR-only fits: with allConstrainedNuisances frozen only norm_top and norm_wjet
float, so the Hessian is 2x2 and plain HESSE handles it fine and reports
covQual = 3. run_all.sh exposes that as CRFIT_ROBUST_HESSE=0.
"""

import argparse
import math
import sys

import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kError

BOUNDARY_TOL = 1e-4      # fraction of the range counted as "at the boundary"


def err(msg):
    sys.stderr.write("  [read_cr_norms] %s\n" % msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fitfile")
    ap.add_argument("--params", default="auto",
                    help="comma-separated parameters to extract, or 'auto' "
                         "(default): every FLOATING parameter named norm_*, "
                         "which follows the topology- and year-decorrelated "
                         "naming (norm_top_resolved, norm_top_resolved_2017, "
                         "...) without hardcoding it")
    ap.add_argument("--fit", default="fit_s",
                    help="RooFitResult name (falls back to fit_b)")
    ap.add_argument("--strict-status", action="store_true",
                    help="also require status==0 and covQual==3")
    ap.add_argument("--max-deviation", type=float, default=5.0,
                    help="reject if |value-1| exceeds this (runaway-fit guard)")
    args = ap.parse_args()

    names = ([] if args.params == "auto"
             else [p for p in args.params.split(",") if p])

    f = ROOT.TFile.Open(args.fitfile)
    if not f or f.IsZombie():
        err("cannot open %s" % args.fitfile)
        return 1

    fr = f.Get(args.fit)
    if not fr:
        fr = f.Get("fit_b")
        if fr:
            err("%s not found, using fit_b" % args.fit)
    if not fr:
        err("no RooFitResult in %s" % args.fitfile)
        return 1

    status, cov_qual = fr.status(), fr.covQual()

    pars = fr.floatParsFinal()
    if not names:   # --params auto
        names = sorted(pars.at(i).GetName() for i in range(pars.getSize())
                       if pars.at(i).GetName().startswith("norm_"))
        if not names:
            err("no floating norm_* parameter in %s" % args.fitfile)
            f.Close()
            return 1
    out, problems = [], []

    for name in names:
        p = pars.find(name)
        if not p:
            # A frozen or otherwise constant parameter is not a usable
            # measurement, but say which case it is.
            const = fr.constPars().find(name)
            problems.append("%s: %s" % (
                name, "held constant in this fit" if const else "not in the fit result"))
            continue

        val = p.getVal()
        e_lo, e_hi = abs(p.getErrorLo()), abs(p.getErrorHi())
        e_sym = p.getError()
        e_use = max(e_lo, e_hi, e_sym)

        if not math.isfinite(val):
            problems.append("%s: value is not finite (%r)" % (name, val))
            continue
        if not math.isfinite(e_use) or e_use <= 0.0:
            problems.append("%s: no usable uncertainty (err=%r)" % (name, e_use))
            continue
        if val <= 0.0:
            problems.append("%s: non-positive value %.6g" % (name, val))
            continue

        lo, hi = p.getMin(), p.getMax()
        if math.isfinite(lo) and math.isfinite(hi) and hi > lo:
            span = hi - lo
            if (val - lo) < BOUNDARY_TOL * span:
                problems.append("%s: pinned at the lower edge of [%g, %g]"
                                % (name, lo, hi))
                continue
            if (hi - val) < BOUNDARY_TOL * span:
                problems.append("%s: pinned at the upper edge of [%g, %g]"
                                % (name, lo, hi))
                continue

        if abs(val - 1.0) > args.max_deviation:
            problems.append("%s: %.4f is more than %.1f away from 1.0"
                            % (name, val, args.max_deviation))
            continue

        out.append((name, val, e_use))

    if problems:
        err("cannot use %s:" % args.fitfile)
        for p in problems:
            err("    " + p)
        f.Close()
        return 1

    # Advisory only -- see the module docstring.
    if status != 0 or cov_qual != 3:
        err("status=%d covQual=%d in %s" % (status, cov_qual, args.fitfile))
        if cov_qual == -1:
            err("    covQual=-1 means no Minuit covariance was stored, which is")
            err("    expected with --robustHesse; RobustHesse computed its own.")
        if args.strict_status:
            err("    --strict-status given: rejecting.")
            f.Close()
            return 1
        err("    parameters look sane, accepting. Use --strict-status to reject,")
        err("    or set CRFIT_ROBUST_HESSE=0 for a meaningful status word.")

    for name, val, e in out:
        err("%-10s = %.6f +/- %.6f" % (name, val, e))

    print(",".join("%s=%.6f" % (n, v) for n, v, _ in out))
    f.Close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
