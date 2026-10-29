#!/usr/bin/env python3
"""
Uncertainty breakdown for the combined VBS fit.

Runs MultiDimFit repeatedly, freezing different groups of nuisance parameters,
and reports each group's contribution to the total uncertainty on r.

Methodology: group contribution = sqrt(sigma_total^2 - sigma_with_group_frozen^2)

WHERE PARAMETERS ARE FROZEN (the snapshot)
------------------------------------------
Every fit runs off a post-fit SNAPSHOT, not the bare workspace. Step 0b does
one unconditional fit with --saveWorkspace; every later scan loads that file
with --snapshotName MultiDimFit, so --freezeParameters pins parameters at
their BEST-FIT values.

Freezing on the bare workspace pins parameters at their PREFIT values instead,
which is wrong for any parameter the data pulls. On the hybrid CR-data cards
the norm_* rateParams sit 0.15-0.25 away from 1.0: clamping them at 1.0 puts
the model in tension with ~350k CR events, the frozen scan comes out as wide
as the total (or wider), and the "Rate params" row reads exactly zero -- which
is how this bug was found. Constrained NPs with small pulls are only mildly
distorted, so the other rows looked plausible while being subtly wrong too
(the wider-than-total frozen scans were silently clamped to +0.0000).

HOW sigma IS MEASURED (--algo, default "grid")
----------------------------------------------
"grid" scans the profiled likelihood in r and reads the 1-sigma interval off
the 2*deltaNLL = 1 crossings. "singles" uses MultiDimFit's own error, which is
what this script used to do together with --robustFit 1.

singles is unreliable here specifically for the normalization rateParams. They
are unconstrained parameters with a hard range ([0.2,5]), and the error that
--algo singles reports for r comes from a local scan whose profiling of those
flat directions is fragile: freezing norm_* could come back with an error
LARGER than the unfrozen fit, making sqrt(sigma_total^2 - sigma_frozen^2)
collapse to zero and the "Rate params" row read as no contribution at all.

The grid scan does not depend on that machinery -- it evaluates the profiled
NLL on a fixed set of r values and interpolates the crossing, so a group that
genuinely matters always widens the curve.

The crossing is interpolated in t = sqrt(2*deltaNLL) rather than in deltaNLL
itself. For a parabolic NLL t is exactly linear in r, so linear interpolation
in t is essentially exact and the scan needs far fewer points than interpolating
deltaNLL directly would.

Run inside CMSSW environment (combine must be in PATH):

  # MC-only Asimov reference (the historical behaviour, unchanged)
  python3 python/uncertainty_breakdown.py \
      --datacard cards_cr/combined/datacard.txt \
      --outdir plots_cr/uncertainty_breakdown \
      --label "MC-only Asimov"

  # CR data included: fit the dataset stored in a hybrid card
  python3 python/uncertainty_breakdown.py \
      --datacard cards_crdata_srasimov/combined/datacard.txt \
      --outdir plots_cr/crdata_srasimov/uncertainty_breakdown \
      --toys "" --label "CR data + SR Asimov"

  # cross-check with the old estimator
  ... --algo singles --robust-fit

With --algo grid every scan is also saved, as likelihood_scans[_tag].json and
a two-panel overlay of the parabolas in likelihood_scans[_tag].pdf/png. That
plot is the direct check on the table: freezing a group that matters visibly
narrows its parabola relative to the total, and a curve that is not smooth is
one whose sigma should not be trusted.

--toys controls the DATASET and defaults to "-t -1 --expectSignal 1", i.e. what
this script has always done. Pass an EMPTY string to fit whatever dataset the
datacard carries. That empty form is required for the hybrid cards written by
make_hybrid_data.py: they already hold real data in the control regions and MC
pseudo-data in the signal regions, and re-adding "-t -1" would overwrite BOTH
with Asimov, throwing away the CR data the whole exercise is about. The signal
region stays blind either way -- its observation is MC by construction.
"""

import argparse
import json
import math
import os
import re
import subprocess

# Marker used by run_all.sh to detect that this script understands --toys.
# Do not remove: older copies that hardcode "-t -1" would silently produce an
# Asimov breakdown labelled as a CR-data one.
# UNCERTAINTY_TOYS_SUPPORTED

# ---------------------------------------------------------------------------
# NP group definitions — each entry is (label, freeze_params_string)
# freeze_params_string is passed directly to combine --freezeParameters.
# Regex patterns use the combine syntax: rgx{pattern}
#
# Groups must be DISJOINT. A nuisance covered by two groups is counted twice in
# the "[check] quadrature sum of groups" row, and neither group's number means
# what its label says. check_group_overlap() below reports any repeats.
#
# Every rgx{...} pattern is validated against the compiled workspace before the
# fit (filter_freeze): a pattern that matches nothing is dropped LOUDLY instead
# of silently freezing nothing -- that is how a renamed nuisance family (e.g. a
# new decorrelation scheme) would otherwise turn a group's row into a fake
# zero contribution.
# ---------------------------------------------------------------------------
GROUPS = [
    ("Scale weights",
     "rgx{LHEScaleWeight.*}"),
    ("Parton shower",
     "rgx{sf_partonshower.*}"),
    ("QvG",
     "rgx{sf_qgtagging.*}"),
    ("Btagging",
     "rgx{sf_btag.*}"),
    ("AK4 jet energy scale and corrections",
     "rgx{AK4PF.*}"),
    ("AK8 jet energy scale and corrections",
     "rgx{AK8PF.*}"),
    ("PDF",
     "rgx{LHEPdfWeight.*}"),
    # The four topology-decorrelated normalizations. Named explicitly rather
    # than as rgx{norm_.*} so the row says exactly what it covers.
    #
    # A per-category card carries only its own topology's pair, so two of these
    # four are absent from it. filter_freeze() drops the missing ones against
    # the workspace before the fit -- combine aborts on an unknown parameter
    # name, which would otherwise kill every per-category breakdown.
    ("Rate params (norm_*)",
     "norm_top_resolved,norm_top_boosted,norm_wjet_resolved,norm_wjet_boosted"),
    ("Integrated luminosity",
     "lumi_2018"),
    ("FatJet",
     "rgx{sf_fj.*}"),
    ("Lepton ID and non-prompt",
     "rgx{ele.*},rgx{muon.*},rgx{sf_ele.*},rgx{sf_mu.*}"),
    # Previously this repeated every Lepton ID pattern, so it was a SUPERSET of
    # that group rather than the complement of it: the lepton nuisances were
    # counted in both rows and "Other experimental" did not mean "other".
    # It now holds only what no other group claims.
    ("Other experimental uncertainties",
     "PileupWeight,rgx{sf_jet_puId.*}"),
]

MASS = 125
DEFAULT_TOYS = "-t -1 --expectSignal 1"

# Grid-scan defaults. The range must comfortably contain the +/-1 sigma
# crossings of the WIDEST fit in the table (the per-category cards, where
# sigma(r) is ~0.7), and the spacing sets how precisely each crossing is
# located. Contributions are quadrature DIFFERENCES of similar numbers, so
# spacing that is too coarse shows up as noise in the small rows.
DEFAULT_POINTS = 61
DEFAULT_RMIN = -1.0
DEFAULT_RMAX = 3.0

# A frozen scan can only be NARROWER than the total when both are anchored at
# the same best fit; wider-than-total beyond this fractional tolerance (scan
# interpolation noise) means the row is broken, and it is flagged instead of
# being clamped to a silent zero.
WIDER_TOL = 0.02

# Every grid scan, kept so the parabolas can be drawn together at the end.
# label -> {"best", "lo", "hi", "r": [...], "q": [2*deltaNLL, ...]}
# The breakdown reads sigma off these curves, so the plot is the direct visual
# check on every number in the table: a group that contributes should visibly
# narrow the parabola when frozen, and a curve that is not smooth means its
# sigma should not be trusted.
_SCANS = {}


def check_group_overlap():
    """Warn if the same freeze token appears in more than one group.

    Only catches literal repeats of a name or an rgx{...} pattern, which is the
    mistake that actually happens when groups are edited by copy-paste. Two
    different regexes that happen to match the same NP cannot be detected here.
    """
    bad = [g for g in GROUPS if not (isinstance(g, tuple) and len(g) == 2
                                     and all(isinstance(x, str) for x in g))]
    if bad:
        raise SystemExit(
            "ERROR: every GROUPS entry must be exactly (label, freeze_string).\n"
            "       These are not:\n         "
            + "\n         ".join(repr(g)[:100] for g in bad)
            + "\n       Several parameters go in ONE comma-separated string:\n"
              '         ("Rate params", "norm_top_resolved,norm_top_boosted")\n'
              "       not as separate tuple elements.")

    seen, dupes = {}, {}
    for label, freeze in GROUPS:
        for tok in (t.strip() for t in freeze.split(",") if t.strip()):
            if tok in seen and seen[tok] != label:
                dupes.setdefault(tok, {seen[tok]}).add(label)
            seen[tok] = label
    if dupes:
        print("  [warn] these freeze patterns appear in more than one group;")
        print("         the affected rows overlap and the quadrature-sum check")
        print("         will double-count them:")
        for tok, labels in sorted(dupes.items()):
            print("           %-28s %s" % (tok, " / ".join(sorted(labels))))
        print("")
    return dupes


def build_workspace(datacard, workdir, mass=MASS):
    """Compile the text datacard once; all fits then run off the workspace."""
    ws = "workspace_unc.root"
    ws_path = os.path.join(workdir, ws)
    cmd = f"text2workspace.py {datacard} -o {ws} -m {mass} 2>&1"
    res = subprocess.run(cmd, shell=True, cwd=workdir,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if not os.path.exists(ws_path):
        raise RuntimeError(f"text2workspace failed:\n{res.stdout[-2000:]}")
    return ws


def make_snapshot(workspace, workdir, mass=MASS, toys=DEFAULT_TOYS):
    """One unconditional fit with --saveWorkspace; returns the output filename.

    Every subsequent fit loads this file with --snapshotName MultiDimFit, so
    --freezeParameters pins parameters at their BEST-FIT values. Freezing on
    the bare workspace pins them at the PREFIT values instead, which is wrong
    for any pulled parameter: on the hybrid cards the norm_* rateParams sit
    0.15-0.25 away from 1.0, and a scan with them clamped at 1.0 is as wide as
    the total -- the "Rate params" row then reads exactly zero.

    Fails loudly if the snapshot is missing from the output file: running the
    breakdown off the bare workspace produces a table that LOOKS fine, which
    is exactly the failure this function exists to prevent.
    """
    safe = "snapshot"
    out = f"higgsCombine_unc_{safe}.MultiDimFit.mH{mass}.root"
    out_path = os.path.join(workdir, out)
    if os.path.exists(out_path):
        os.remove(out_path)   # never reuse a snapshot from a previous card
    cmd = (
        f"combine -M MultiDimFit"
        f" -d {workspace}"
        f" -m {mass}"
        f" {toys.strip()}"
        f" --algo none --saveWorkspace"
        f" --name _unc_{safe}"
        f" 2>&1"
    )
    res = subprocess.run(cmd, shell=True, cwd=workdir,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if not os.path.exists(out_path):
        raise RuntimeError(f"snapshot fit failed:\n{res.stdout[-2000:]}")

    import ROOT
    ROOT.gErrorIgnoreLevel = ROOT.kError
    f = ROOT.TFile.Open(out_path)
    w = f.Get("w") if f else None
    have = bool(w) and bool(w.getSnapshot("MultiDimFit"))
    if f:
        f.Close()
    if not have:
        raise RuntimeError(
            f"{out} carries no 'MultiDimFit' snapshot -- refusing to run the "
            f"breakdown off prefit parameter values.")
    return out


def constrained_nuis(workdir, workspace, exclude_prefix="prop_bin"):
    """All constrained nuisances except the Barlow-Beeston prop_bin params.

    Used instead of --freezeParameters allConstrainedNuisances + --floatParameters,
    because --floatParameters does NOT override the allConstrainedNuisances
    keyword (freeze wins), which silently makes the MC-stat fit identical to the
    stat-only fit.
    """
    import ROOT
    f = ROOT.TFile.Open(os.path.join(workdir, workspace))
    mc = f.Get("w").genobj("ModelConfig")
    names = [p.GetName() for p in mc.GetNuisanceParameters()]
    f.Close()
    keep = [n for n in names if not n.startswith(exclude_prefix)]
    print(f"  ({len(names)} constrained NPs, {len(names) - len(keep)} prop_bin excluded)")
    return keep


def workspace_params(workdir, workspace):
    """Every variable name in the compiled workspace."""
    import ROOT
    ROOT.gErrorIgnoreLevel = ROOT.kError
    f = ROOT.TFile.Open(os.path.join(workdir, workspace))
    w = f.Get("w")
    if not w:
        f.Close()
        return None
    names = {v.GetName() for v in w.allVars()}
    f.Close()
    return names


def filter_freeze(freeze, known, label):
    """Validate every freeze token against this workspace.

    Literal parameter names that do not exist are dropped: combine aborts on an
    unknown literal, so a group listing parameters from another category would
    take the whole breakdown down with it.

    rgx{...} tokens are checked too: a pattern that matches nothing in the
    workspace is dropped with a loud note. Combine itself treats an empty
    pattern as harmless, which is the OPPOSITE of harmless here -- the group's
    scan would silently freeze nothing and its row would read as a fake zero
    contribution. (The check uses python re.fullmatch on the workspace variable
    names; combine's rgx{} is anchored the same way.)

    Returns None if nothing is left, meaning the group does not apply to this
    workspace: the row is reported N/A instead of 0.
    """
    if not freeze or known is None:
        return freeze
    kept, dropped, empty_rgx = [], [], []
    for tok in (t.strip() for t in freeze.split(",") if t.strip()):
        if tok == "allConstrainedNuisances":
            kept.append(tok)
        elif tok.startswith("rgx{") and tok.endswith("}"):
            pat = tok[4:-1]
            try:
                n = sum(1 for name in known if re.fullmatch(pat, name))
            except re.error as exc:
                raise SystemExit(f"ERROR: bad regex in group '{label}': "
                                 f"{tok} ({exc})")
            if n == 0:
                empty_rgx.append(tok)
            else:
                kept.append(tok)
        elif tok in known:
            kept.append(tok)
        else:
            dropped.append(tok)
    if dropped:
        print(f"    [note] not in this workspace, dropped: {', '.join(dropped)}")
    if empty_rgx:
        print(f"    [!] rgx pattern(s) matching NO parameter in this workspace,")
        print(f"        dropped: {', '.join(empty_rgx)}")
        print(f"        If these nuisances were renamed (a decorrelation?),")
        print(f"        update GROUPS -- otherwise this row underestimates.")
    if not kept:
        print(f"    [skip] '{label}' has no parameter in this workspace")
        return None
    return ",".join(kept)


# ---------------------------------------------------------------------------
# Fit driver
# ---------------------------------------------------------------------------

def read_singles(workdir, safe_label, mass=MASS):
    """Read best fit and +/-1 sigma from the MultiDimFit output tree.

    Uses ROOT rather than uproot: this script runs inside CMSSW, where uproot
    is not available. Reading the tree instead of parsing combine's stdout
    matters because the summary line prints only 3 decimals.
    """
    import ROOT
    path = os.path.join(
        workdir, f"higgsCombine_unc_{safe_label}.MultiDimFit.mH{mass}.root")
    f = ROOT.TFile.Open(path)
    if not f or f.IsZombie():
        raise RuntimeError(f"cannot open {path}")
    t = f.Get("limit")
    if not t or t.GetEntries() < 3:
        n = t.GetEntries() if t else 0
        f.Close()
        raise RuntimeError(f"{path}: expected 3 entries in 'limit', found {n}")

    rs, best = [], None
    for i in range(t.GetEntries()):
        t.GetEntry(i)
        rs.append(float(t.r))
        if abs(t.quantileExpected + 1.0) < 1e-6:
            best = float(t.r)
    f.Close()

    if best is None:
        best = rs[0]
    return best, best - min(rs), max(rs) - best


def read_grid(workdir, safe_label, mass=MASS):
    """Read the 1-sigma interval off a --algo grid likelihood scan.

    Returns (best, err_lo, err_hi, points) where points is the list of
    (r, 2*deltaNLL) actually scanned, with deltaNLL measured from the minimum.
    An error is None when the 2*deltaNLL = 1 crossing falls outside the scanned
    range, which is reported rather than silently turned into a number.
    """
    import ROOT
    path = os.path.join(
        workdir, f"higgsCombine_unc_{safe_label}.MultiDimFit.mH{mass}.root")
    f = ROOT.TFile.Open(path)
    if not f or f.IsZombie():
        raise RuntimeError(f"cannot open {path}")
    t = f.Get("limit")
    if not t or t.GetEntries() < 3:
        n = t.GetEntries() if t else 0
        f.Close()
        raise RuntimeError(f"{path}: expected a scan in 'limit', found {n} entries")

    pts = []
    for i in range(t.GetEntries()):
        t.GetEntry(i)
        d = float(t.deltaNLL)
        if math.isfinite(d):
            pts.append((float(t.r), d))
    f.Close()

    if len(pts) < 3:
        raise RuntimeError(f"{path}: only {len(pts)} finite scan points")

    pts.sort()
    imin = min(range(len(pts)), key=lambda i: pts[i][1])
    best, dmin = pts[imin]

    def crossing(seq):
        """First point where t = sqrt(2*dNLL) reaches 1, interpolated in t."""
        prev_r, prev_t = best, 0.0
        for r, d in seq:
            tt = math.sqrt(max(2.0 * (d - dmin), 0.0))
            if tt >= 1.0:
                if tt == prev_t:
                    return None
                frac = (1.0 - prev_t) / (tt - prev_t)
                return prev_r + frac * (r - prev_r)
            prev_r, prev_t = r, tt
        return None

    hi = crossing(pts[imin + 1:])
    lo = crossing(list(reversed(pts[:imin])))
    curve = [(r, 2.0 * (d - dmin)) for r, d in pts]
    return (best,
            (best - lo) if lo is not None else None,
            (hi - best) if hi is not None else None,
            curve)


def run_multidimfit(target, workdir, freeze_params=None, label="total",
                    mass=MASS, toys=DEFAULT_TOYS, algo="grid",
                    points=DEFAULT_POINTS, rmin=DEFAULT_RMIN, rmax=DEFAULT_RMAX,
                    robust_fit=False, snapshot=None):
    """Run MultiDimFit and return (best, err_lo, err_hi).

    When `snapshot` is given it must be the output of make_snapshot(); the fit
    then loads it with --snapshotName MultiDimFit so every parameter -- frozen
    or floating -- starts (and freezes) at its best-fit value.
    """
    safe_label = re.sub(r"[^a-zA-Z0-9_]", "_", label)[:40]
    out = os.path.join(
        workdir, f"higgsCombine_unc_{safe_label}.MultiDimFit.mH{mass}.root")
    if os.path.exists(out):
        os.remove(out)   # never read a stale file from a previous run

    freeze_arg = f"--freezeParameters '{freeze_params}'" if freeze_params else ""
    toys_arg = toys.strip()
    if snapshot:
        target_arg = f"-d {snapshot} --snapshotName MultiDimFit"
    else:
        target_arg = f"-d {target}"

    if algo == "grid":
        # No --robustFit: it applies to the singles error estimate, and it is
        # the piece that misbehaves for the unconstrained norm_* rateParams.
        algo_arg = (f"--algo grid --points {points}"
                    f" --rMin {rmin} --rMax {rmax}")
    else:
        algo_arg = "--algo singles" + (" --robustFit 1" if robust_fit else "")

    cmd = (
        f"combine -M MultiDimFit"
        f" {target_arg}"
        f" -m {mass}"
        f" {toys_arg}"
        f" {algo_arg}"
        f" --name _unc_{safe_label}"
        f" {freeze_arg}"
        f" 2>&1"
    )
    res = subprocess.run(cmd, shell=True, cwd=workdir,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        if algo == "grid":
            best, lo, hi, curve = read_grid(workdir, safe_label, mass)
            _SCANS[label] = {"best": best, "lo": lo, "hi": hi,
                             "r": [c[0] for c in curve],
                             "q": [c[1] for c in curve]}
        else:
            best, lo, hi = read_singles(workdir, safe_label, mass)
    except Exception as e:
        print(f"  [warn] could not read output for '{label}': {e}")
        print(f"  Last 300 chars: {res.stdout[-300:]}")
        return None, None, None

    if algo == "grid" and (lo is None or hi is None):
        side = " and ".join(s for s, v in (("lower", lo), ("upper", hi))
                            if v is None)
        print(f"  [warn] '{label}': the {side} 1-sigma crossing is outside the"
              f" scanned range [{rmin}, {rmax}] (best fit r = {best:.4f}).")
        print(f"         Widen it with --rmin/--rmax.")
    return best, lo, hi


def quad_diff(a, b):
    """sqrt(a^2 - b^2); returns 0 if b >= a."""
    if a is None or b is None:
        return None
    return math.sqrt(max(a**2 - b**2, 0.0))


def wider_than_total(lo_f, hi_f, lo_tot, hi_tot):
    """Sides on which a frozen scan is WIDER than the total, beyond noise.

    A properly anchored frozen scan can only be narrower. Wider means the row
    is broken (frozen at the wrong values, or a bad crossing) and its
    contribution must not be read as "zero".
    """
    sides = []
    if lo_f is not None and lo_tot is not None and lo_f > lo_tot * (1.0 + WIDER_TOL):
        sides.append("down")
    if hi_f is not None and hi_tot is not None and hi_f > hi_tot * (1.0 + WIDER_TOL):
        sides.append("up")
    return sides


def fmt(v, sign):
    if v is None:
        return "  N/A  "
    return f"{sign}{v:.4f}"


# ---------------------------------------------------------------------------
# Likelihood-scan plot
# ---------------------------------------------------------------------------

# Fits that are not per-group: drawn thicker, since they set the scale the
# group curves are compared against.
_REFERENCE_LABELS = ("total", "stat_only", "stat_plus_mc", "no_mc_stat")


def write_scan_outputs(outdir, tag, label_run, algo, mass):
    """Dump every grid scan as JSON and overlay the parabolas in one figure.

    The JSON is written first and unconditionally: matplotlib is not guaranteed
    to be importable in the CMSSW environment this script runs in, and losing
    the scans because of a plotting dependency would be worse than losing the
    plot. If the import fails the points are still on disk to draw elsewhere.
    """
    if algo != "grid" or not _SCANS:
        return
    suffix = f"_{tag}" if tag else ""

    js = os.path.join(outdir, f"likelihood_scans{suffix}.json")
    with open(js, "w") as fh:
        json.dump({"label": label_run, "mass": mass, "scans": _SCANS}, fh)
    print(f"\nScan points written to {js}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                                   # noqa: BLE001
        print(f"  [note] matplotlib not usable here ({exc.__class__.__name__});")
        print(f"         no plot written. The points are in {js}.")
        return

    refs = [l for l in _REFERENCE_LABELS if l in _SCANS]
    groups = [l for l in _SCANS if l not in _REFERENCE_LABELS]
    # Widest first, so the legend reads in order of how much each group matters.
    groups.sort(key=lambda l: -(_SCANS[l]["hi"] or 0.0))

    fig, axes = plt.subplots(1, 2, figsize=(16.5, 6.5))

    def draw(ax, labels, title, thick, ymax, legend_outside):
        for label in labels:
            sc = _SCANS[label]
            lo, hi = sc["lo"], sc["hi"]
            if lo is None or hi is None:
                sig = "crossing outside range"
            else:
                sig = f"-{lo:.3f}/+{hi:.3f}"
            if label == "total":
                style = dict(color="black", lw=2.4, zorder=5)
            elif thick:
                style = dict(lw=1.8)
            else:
                style = dict(lw=1.2)
            ax.plot(sc["r"], sc["q"], label=f"{label}  {sig}", **style)

        ax.axhline(1.0, color="grey", lw=0.8, ls="--")
        ax.text(0.01, 1.02, r"$2\Delta$NLL = 1  (1$\sigma$)",
                transform=ax.get_yaxis_transform(),
                fontsize=8, color="grey", va="bottom")
        if ymax > 4.0:
            ax.axhline(3.84, color="grey", lw=0.8, ls=":")
            ax.text(0.01, 3.86, r"3.84  (2$\sigma$)",
                    transform=ax.get_yaxis_transform(),
                    fontsize=8, color="grey", va="bottom")

        # x-range from the curves in THIS panel, at this panel's y-zoom.
        xs = [r for l in labels
              for r, q in zip(_SCANS[l]["r"], _SCANS[l]["q"]) if q <= ymax]
        if xs:
            pad = 0.05 * ((max(xs) - min(xs)) or 1.0)
            ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(0.0, ymax)

        ax.set_xlabel("r")
        ax.set_title(title, fontsize=10)
        if legend_outside:
            ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                      framealpha=0.9, ncol=1, borderaxespad=0.0)
        else:
            ax.legend(fontsize=7, loc="upper right", framealpha=0.85, ncol=1)
        ax.grid(alpha=0.25, lw=0.5)

    # Left: the reference fits, which differ enough to need the full window.
    draw(axes[0], refs, "Reference fits", thick=True,
         ymax=5.0, legend_outside=False)

    # Right: freezing one group moves the curve only slightly, so this panel is
    # zoomed onto the 1-sigma crossing -- at the full y-range every curve lies
    # on top of the total and the panel says nothing. The legend goes outside
    # for the same reason: in-panel it covers the very region being compared.
    draw(axes[1], (["total"] if "total" in _SCANS else []) + groups,
         "One NP group frozen per curve (zoomed on the 1$\sigma$ crossing)",
         thick=False, ymax=1.6, legend_outside=True)

    for ax in axes:
        ax.set_ylabel(r"$2\Delta$NLL")

    fig.suptitle(f"Profiled likelihood scans -- {label_run}", fontsize=12)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        out = os.path.join(outdir, f"likelihood_scans{suffix}.{ext}")
        fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Scan plot written to "
          f"{os.path.join(outdir, 'likelihood_scans' + suffix + '.pdf')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datacard", required=True,
                        help="Path to the datacard (e.g. cards/combined/datacard.txt)")
    parser.add_argument("--outdir", default=".",
                        help="Directory to write uncertainty_breakdown.txt")
    parser.add_argument("--toys", default=DEFAULT_TOYS,
                        help="Dataset options passed to every combine call. "
                             "Default '%s'. Pass an empty string to fit the "
                             "dataset stored in the datacard (needed for the "
                             "hybrid CR-data cards)." % DEFAULT_TOYS)
    parser.add_argument("--label", default=None,
                        help="Run label written into the table header, e.g. "
                             "'CR data + SR Asimov'")
    parser.add_argument("--tag", default="",
                        help="Suffix for the output filename")
    parser.add_argument("--mass", type=int, default=MASS)
    parser.add_argument("--algo", choices=["grid", "singles"], default="grid",
                        help="how sigma(r) is measured: 'grid' reads the "
                             "2*deltaNLL=1 crossings off a likelihood scan "
                             "(default); 'singles' uses MultiDimFit's own "
                             "error, which is unreliable for the norm_* "
                             "rateParams. See the module docstring.")
    parser.add_argument("--points", type=int, default=DEFAULT_POINTS,
                        help="grid scan points (default %d)" % DEFAULT_POINTS)
    parser.add_argument("--rmin", type=float, default=DEFAULT_RMIN)
    parser.add_argument("--rmax", type=float, default=DEFAULT_RMAX)
    parser.add_argument("--robust-fit", action="store_true",
                        help="only with --algo singles: restore --robustFit 1")
    args = parser.parse_args()

    mass = args.mass
    toys = args.toys
    label_run = args.label or ("MC-only Asimov" if toys.strip() else "dataset in card")

    os.makedirs(args.outdir, exist_ok=True)
    workdir = os.path.dirname(os.path.abspath(args.datacard))
    datacard = os.path.basename(args.datacard)

    print("=" * 60)
    print(f"Uncertainty breakdown: {label_run}")
    print(f"  datacard : {args.datacard}")
    print(f"  dataset  : {toys.strip() or '(the one stored in the card)'}")
    if args.algo == "grid":
        print(f"  sigma(r) : --algo grid, {args.points} points over "
              f"[{args.rmin}, {args.rmax}]")
    else:
        print(f"  sigma(r) : --algo singles"
              f"{' --robustFit 1' if args.robust_fit else ''}")
    print("=" * 60)

    check_group_overlap()

    print("Step 0: Compiling workspace")
    ws = build_workspace(datacard, workdir, mass)
    print(f"  {ws}\n")

    print("Step 0b: Post-fit snapshot (all frozen fits anchor here)")
    snap = make_snapshot(ws, workdir, mass, toys)
    print(f"  {snap}\n")

    def fit(freeze=None, label="total"):
        return run_multidimfit(ws, workdir, freeze_params=freeze, label=label,
                               mass=mass, toys=toys, algo=args.algo,
                               points=args.points, rmin=args.rmin,
                               rmax=args.rmax, robust_fit=args.robust_fit,
                               snapshot=snap)

    # ------------------------------------------------------------------
    # 1. Full fit
    # ------------------------------------------------------------------
    print("Step 1: Full fit (no parameters frozen)")
    best, lo_tot, hi_tot = fit(label="total")
    if lo_tot is None:
        raise RuntimeError("Full fit failed — cannot proceed.")
    print(f"  r = {best:.4f}  -{lo_tot:.4f}/+{hi_tot:.4f}\n")

    # ------------------------------------------------------------------
    # 2. Stat-only fit (freeze all constrained nuisances)
    #
    # The norm_* rateParams are UNCONSTRAINED, so they keep floating here by
    # design: this row therefore contains the CR-statistics contribution that
    # flows through the normalizations, and it OVERLAPS the "Rate params" group
    # row. The two answer different questions and do not add in quadrature.
    # ------------------------------------------------------------------
    print("Step 2: Stat-only fit (all constrained NPs frozen; norm_* floats)")
    _, lo_stat, hi_stat = fit(freeze="allConstrainedNuisances", label="stat_only")
    if lo_stat is None:
        print("  [warn] stat-only fit failed; continuing without it.")
    else:
        print(f"  r = _  -{lo_stat:.4f}/+{hi_stat:.4f}\n")

    lo_syst = quad_diff(lo_tot, lo_stat)
    hi_syst = quad_diff(hi_tot, hi_stat)

    # ------------------------------------------------------------------
    # 2b. MC background statistics
    #     Freeze every constrained NP EXCEPT prop_bin*, so the only extra
    #     freedom relative to the stat-only fit is Barlow-Beeston.
    #     contrib = sqrt(sigma_stat+MC^2 - sigma_stat^2)
    # ------------------------------------------------------------------
    print("Step 2b: MC background statistics (all NPs frozen except prop_bin*)")
    freeze_no_mc = ",".join(constrained_nuis(workdir, ws))
    _, lo_stat_mc, hi_stat_mc = fit(freeze=freeze_no_mc, label="stat_plus_mc")
    if lo_stat_mc is None:
        print("  [warn] MC stat fit failed.")
        lo_mc = hi_mc = None
    else:
        print(f"  r = _  -{lo_stat_mc:.4f}/+{hi_stat_mc:.4f}")
        lo_mc = quad_diff(lo_stat_mc, lo_stat)
        hi_mc = quad_diff(hi_stat_mc, hi_stat)
        print(f"  MC bkg stat contrib = -{lo_mc:.4f}/+{hi_mc:.4f}")

    # Cross-check from the opposite direction: freeze only prop_bin* in the full fit
    print("Step 2c: MC stat cross-check (prop_bin* frozen in full fit)")
    _, lo_nomc, hi_nomc = fit(freeze="rgx{prop_bin.*}", label="no_mc_stat")
    if lo_nomc is not None:
        print(f"  complementary contrib = -{quad_diff(lo_tot, lo_nomc):.4f}"
              f"/+{quad_diff(hi_tot, hi_nomc):.4f}\n")

    # ------------------------------------------------------------------
    # 3. Per-group fits
    # ------------------------------------------------------------------
    print("Step 3: Per-group fits")
    known = workspace_params(workdir, ws)
    if known is None:
        print("  [warn] could not read the workspace parameter list; freeze "
              "strings will be passed to combine unfiltered.")
    group_results = []
    broken_rows = []
    for label, freeze in GROUPS:
        print(f"  Freezing: {label}")
        eff = filter_freeze(freeze, known, label)
        if eff is None:
            group_results.append((label, None, None))
            continue
        _, lo_f, hi_f = fit(freeze=eff, label=label)
        if lo_f is None:
            group_results.append((label, None, None))
            continue

        # A frozen scan can only be narrower than the total. Wider means the
        # row is broken (bad anchoring or a bad crossing): flag it in the
        # table instead of letting quad_diff clamp it to a silent zero.
        wider = wider_than_total(lo_f, hi_f, lo_tot, hi_tot)
        if wider:
            print(f"    [!] frozen scan is WIDER than the total on the "
                  f"{'/'.join(wider)} side")
            print(f"        (-{lo_f:.4f}/+{hi_f:.4f} vs total "
                  f"-{lo_tot:.4f}/+{hi_tot:.4f}).")
            print(f"        Impossible for a scan anchored at the best fit --")
            print(f"        this row is broken, NOT a zero contribution.")
            label = label + " (!)"
            broken_rows.append(label)

        lo_c = quad_diff(lo_tot, lo_f)
        hi_c = quad_diff(hi_tot, hi_f)
        if (lo_c is not None and hi_c is not None
                and lo_c < 1e-6 and hi_c < 1e-6 and not wider):
            print(f"    contrib = 0 — check that '{eff}' matches real NP names")
        elif lo_c is not None and hi_c is not None:
            print(f"    contrib = -{lo_c:.4f}/+{hi_c:.4f}")
        group_results.append((label, lo_c, hi_c))

    # ------------------------------------------------------------------
    # 4. Format table
    # ------------------------------------------------------------------
    W = 50
    header = f"{'Source':<{W}} {'sigma_down':>12} {'sigma_up':>12}"
    sep    = "-" * len(header)
    lines  = []
    lines.append(f"Uncertainty breakdown on r -- {label_run}")
    lines.append(f"  datacard: {args.datacard}")
    lines.append(f"  dataset : {toys.strip() or 'the one stored in the datacard'}")
    if args.algo == "grid":
        lines.append(f"  sigma   : --algo grid, {args.points} points over "
                     f"[{args.rmin}, {args.rmax}], 2*deltaNLL=1 crossings")
    else:
        lines.append("  sigma   : --algo singles"
                     + (" --robustFit 1" if args.robust_fit else ""))
    lines.append(f"  anchor  : frozen fits pinned at the post-fit snapshot")
    lines.append(f"  best fit: r = {best:.4f}")
    lines.append("")
    lines += [sep, header, sep]

    def row(label, lo, hi):
        return f"{label:<{W}} {fmt(lo, '-'):>12} {fmt(hi, '+'):>12}"

    lines.append(row("Total", lo_tot, hi_tot))
    lines.append(row("  Data statistical", lo_stat, hi_stat))
    lines.append(row("  MC background statistical", lo_mc, hi_mc))
    lines.append(row("  Total systematic", lo_syst, hi_syst))
    lines.append(sep)
    for label, lo, hi in group_results:
        lines.append(row(f"  {label}", lo, hi))

    # Quadrature sum of groups vs total systematic — sanity check on coverage
    got = [(lo, hi) for _, lo, hi in group_results if lo is not None]
    if got and lo_syst is not None:
        qlo = math.sqrt(sum(l**2 for l, _ in got))
        qhi = math.sqrt(sum(h**2 for _, h in got))
        lines.append(sep)
        lines.append(row("  [check] quadrature sum of groups", qlo, qhi))
    lines.append(sep)
    lines.append("Note: the norm_* rateParams are unconstrained and FLOAT in the")
    lines.append("'Data statistical' row; the CR-statistics contribution that flows")
    lines.append("through them appears in 'Rate params (norm_*)' and OVERLAPS the")
    lines.append("data-statistical number. The two rows do not add in quadrature.")
    if broken_rows:
        lines.append("")
        lines.append("(!) marks rows whose frozen scan came out WIDER than the total:")
        lines.append("    impossible for a properly anchored fit, so treat those")
        lines.append("    numbers as broken, not as small contributions:")
        for b in broken_rows:
            lines.append(f"      {b}")

    table = "\n".join(lines)
    print("\n" + table)

    write_scan_outputs(args.outdir, args.tag, label_run, args.algo, mass)

    name = f"uncertainty_breakdown{'_' + args.tag if args.tag else ''}.txt"
    out_path = os.path.join(args.outdir, name)
    with open(out_path, "w") as fout:
        fout.write(table + "\n")
    print(f"\nTable written to {out_path}")


if __name__ == "__main__":
    main()
