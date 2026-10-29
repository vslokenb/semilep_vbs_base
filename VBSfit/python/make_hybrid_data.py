#!/usr/bin/env python3
"""
make_hybrid_data.py -- build a partially-blind dataset for the VBS semileptonic fit.

Given an existing datacard and the control-region-derived normalizations from the
step-5c CR-only fits, this writes

    <outdir>/histograms_hybrid.root   a copy of the shapes file in which the
                                      data_obs histogram of every SIGNAL region
                                      has been replaced by the MC expectation
                                      evaluated at those normalizations
    <outdir>/datacard.txt             a copy of the card pointing at that file,
                                      with the observation line updated

Control-region bins are copied through byte-for-byte, so they keep REAL DATA.
Signal-region bins carry no observed information at all: their data_obs is a
deterministic function of the MC templates, the CR-fit rateParam values and
--inject-signal.  Nothing derived from SR data ever enters the output.

This is the piece that "-t -1 --setParameters norm_top=..." cannot do: -t -1
replaces the CONTROL regions with Asimov as well, so the CR data only enters
through the two injected central values.  With a hybrid dataset the fit is run
WITHOUT -t, the CR likelihood terms see real counts, and norm_top / norm_wjet
are constrained by data with their true statistical power -- while the SR is
exactly as blind as it was under a full Asimov fit.

A bin counts as a signal region if any of its processes has a process index <= 0
(combine's signal convention).  Use --sr-bins to override.

Usage
-----
    python3 make_hybrid_data.py \
        --datacard cards_cr/combined/datacard.txt \
        --norms "norm_top=1.043,norm_wjet=0.972" \
        --inject-signal 1.0 \
        --outdir cards_hybrid/combined

    # re-run the blinding checks on an existing output directory
    python3 make_hybrid_data.py --outdir cards_hybrid/combined --verify-only
"""

import argparse
import os
import shutil
import sys
import time

import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kError

# data_obs of an SR bin must reproduce the recomputed MC expectation to this
# relative precision, and CR data must be bit-identical.
SR_TOL = 1e-6


# ---------------------------------------------------------------------------
# datacard parsing
# ---------------------------------------------------------------------------

class Card(object):
    """The parts of a datacard this script needs.

    Deliberately minimal: it does not try to understand systematics, only the
    shapes map, the bin/process/rate table and the rateParams, which is all
    that is needed to rebuild a nominal expectation.
    """

    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.dir = os.path.dirname(self.path)
        with open(self.path) as fh:
            self.lines = fh.read().splitlines()

        self.shapes = []        # (process, bin, file, nominal, syst)
        self.obs_bins = []      # bins in observation-line order
        self.obs_line_idx = None
        self.obs_bin_line_idx = None
        self.columns = []       # (bin, process, index, rate)
        self.rate_params = []   # (name, bin_pattern, proc_pattern, init)

        self._parse()

    def _parse(self):
        idx_rate = None
        for i, line in enumerate(self.lines):
            tok = line.split()
            if not tok or line.lstrip().startswith("#"):
                continue
            key = tok[0]
            if key == "shapes" and len(tok) >= 5:
                self.shapes.append(
                    (tok[1], tok[2], tok[3], tok[4], tok[5] if len(tok) > 5 else None)
                )
            elif key == "observation":
                self.obs_line_idx = i
            elif key == "rate" and idx_rate is None:
                idx_rate = i
            elif key == "rateParam" or (len(tok) > 1 and tok[1] == "rateParam"):
                self._parse_rate_param(tok)

        if self.obs_line_idx is None:
            raise ValueError("%s: no 'observation' line" % self.path)
        if idx_rate is None:
            raise ValueError("%s: no 'rate' line" % self.path)

        # The card contains two 'bin' lines. The one for the observation is the
        # nearest 'bin' line above 'observation'; the one for the process table
        # is the nearest 'bin' line above 'rate'.
        self.obs_bin_line_idx = self._prev_bin_line(self.obs_line_idx)
        tbl_bin_line_idx = self._prev_bin_line(idx_rate)

        self.obs_bins = self.lines[self.obs_bin_line_idx].split()[1:]

        bins = self.lines[tbl_bin_line_idx].split()[1:]
        proc_lines = [
            i for i in range(tbl_bin_line_idx, idx_rate)
            if self.lines[i].split() and self.lines[i].split()[0] == "process"
        ]
        if len(proc_lines) != 2:
            raise ValueError(
                "%s: expected 2 'process' lines between 'bin' and 'rate', found %d"
                % (self.path, len(proc_lines))
            )
        names = self.lines[proc_lines[0]].split()[1:]
        idxs = self.lines[proc_lines[1]].split()[1:]
        # The name line is whichever of the two is not all-integers.
        if _all_int(names):
            names, idxs = idxs, names
        rates = self.lines[idx_rate].split()[1:]

        if not (len(bins) == len(names) == len(idxs) == len(rates)):
            raise ValueError(
                "%s: bin/process/process/rate lines have inconsistent lengths "
                "(%d/%d/%d/%d)" % (self.path, len(bins), len(names), len(idxs), len(rates))
            )

        for b, p, k, r in zip(bins, names, idxs, rates):
            self.columns.append((b, p, int(k), float(r)))

    def _parse_rate_param(self, tok):
        # <name> rateParam <bin> <process> <init> [<lo>,<hi>]
        if len(tok) < 5 or tok[1] != "rateParam":
            return
        name, bin_pat, proc_pat, init = tok[0], tok[2], tok[3], tok[4]
        try:
            init_val = float(init)
        except ValueError:
            # formula-style rateParam -- refuse rather than silently mis-scale
            raise ValueError(
                "%s: rateParam '%s' uses a formula ('%s'); this script only "
                "handles plain numeric rateParams." % (self.path, name, init)
            )
        self.rate_params.append((name, bin_pat, proc_pat, init_val))

    def _prev_bin_line(self, before):
        for i in range(before - 1, -1, -1):
            tok = self.lines[i].split()
            if tok and tok[0] == "bin":
                return i
        raise ValueError("%s: no 'bin' line above line %d" % (self.path, before))

    # -- derived views ------------------------------------------------------

    def bins(self):
        seen, out = set(), []
        for b, _, _, _ in self.columns:
            if b not in seen:
                seen.add(b)
                out.append(b)
        return out

    def signal_bins(self):
        return [b for b in self.bins()
                if any(k <= 0 for bb, _, k, _ in self.columns if bb == b)]

    def shapes_files(self):
        out = []
        for _, _, f, _, _ in self.shapes:
            f = f if os.path.isabs(f) else os.path.join(self.dir, f)
            f = os.path.normpath(f)
            if f not in out:
                out.append(f)
        return out

    def shape_entry(self, process, chan):
        """Resolve the shapes line for (process, bin), most specific first."""
        for want in ((process, chan), (process, "*"), ("*", chan), ("*", "*")):
            for entry in self.shapes:
                if (entry[0], entry[1]) == want:
                    return entry
        raise KeyError("%s: no shapes line matches process=%s bin=%s"
                       % (self.path, process, chan))

    def rate_param_factor(self, chan, process, norms, warned):
        """Product of the rateParams applying to (bin, process)."""
        factor = 1.0
        for name, bin_pat, proc_pat, init in self.rate_params:
            if not _match(bin_pat, chan) or not _match(proc_pat, process):
                continue
            if name in norms:
                factor *= norms[name]
            else:
                factor *= init
                if abs(init - 1.0) > 1e-9 and name not in warned:
                    warned.add(name)
                    print("  [warn] rateParam '%s' was not given a fitted value; "
                          "using its datacard init %.4f" % (name, init))
        return factor


def _all_int(vals):
    try:
        [int(v) for v in vals]
        return True
    except ValueError:
        return False


def _match(pattern, value):
    # combine uses '*' as the only wildcard in bin/process fields
    import fnmatch
    return fnmatch.fnmatchcase(value, pattern)


def _subst(pattern, process, chan, mass, syst=None):
    out = (pattern.replace("$PROCESS", process)
                  .replace("$CHANNEL", chan)
                  .replace("$MASS", str(mass)))
    if syst is not None:
        out = out.replace("$SYSTEMATIC", syst)
    return out


# ---------------------------------------------------------------------------
# ROOT helpers
# ---------------------------------------------------------------------------

def open_eos_retry(path, mode="READ", tries=12, delay=5):
    """TFile.Open with retries for the EOS FUSE -> xrootd sync lag.

    A file just written through the FUSE mount can appear TRUNCATED through
    the xrootd redirector for several seconds (ROOT translates /eos paths to
    root:// URLs), which shows up as a zombie TFile and
    "file ... is truncated at N bytes: should be M". Retrying the open until
    the views agree is the fix; failing after ~1 minute means something is
    genuinely wrong with the file.
    """
    for i in range(tries):
        f = ROOT.TFile.Open(path, mode)
        if f and not f.IsZombie():
            return f
        if f:
            f.Close()
        if i < tries - 1:
            print("  [note] %s not fully visible yet (EOS xrootd sync lag); "
                  "retry %d/%d in %ds" % (path, i + 1, tries - 1, delay))
            time.sleep(delay)
    sys.exit("ERROR: cannot open %s after %d attempts -- the EOS xrootd view "
             "never caught up (or the file is genuinely corrupt)." % (path, tries))


def get_hist(rfile, path, where):
    obj = rfile.Get(path)
    if not obj:
        raise KeyError("%s: histogram '%s' not found (%s)"
                       % (rfile.GetName(), path, where))
    obj = obj.Clone()
    obj.SetDirectory(0)
    return obj


def write_hist(rfile, path, hist):
    parts = path.split("/")
    d = rfile
    for p in parts[:-1]:
        nd = d.GetDirectory(p)
        if not nd:
            nd = d.mkdir(p)
        d = nd
    d.cd()
    hist.SetName(parts[-1])
    hist.Write(parts[-1], ROOT.TObject.kOverwrite)


def same_contents(h1, h2):
    if h1.GetNbinsX() != h2.GetNbinsX():
        return False
    for i in range(0, h1.GetNbinsX() + 2):
        if h1.GetBinContent(i) != h2.GetBinContent(i):
            return False
    return True


# ---------------------------------------------------------------------------
# the expectation
# ---------------------------------------------------------------------------

def sr_expectation(card, rfile, chan, norms, inject_signal, mass, warned):
    """Nominal MC expectation for one signal-region bin, at the CR-fit norms.

    Backgrounds are scaled by their rateParams (the CR-derived norm_top /
    norm_wjet where given); signal processes are scaled by --inject-signal.
    """
    total = None
    detail = []
    for b, proc, kind, rate in card.columns:
        if b != chan:
            continue
        entry = card.shape_entry(proc, chan)
        path = _subst(entry[3], proc, chan, mass)
        h = get_hist(rfile, path, "process %s in bin %s" % (proc, chan))

        # combine normalizes a shape template to the card's rate column unless
        # that column is -1, in which case the histogram integral is used.
        integral = h.Integral()
        if rate >= 0 and integral > 0 and abs(rate - integral) / integral > 1e-6:
            h.Scale(rate / integral)

        if kind <= 0:
            scale = inject_signal
        else:
            scale = card.rate_param_factor(chan, proc, norms, warned)
        h.Scale(scale)

        detail.append((proc, "sig" if kind <= 0 else "bkg", scale, h.Integral()))
        if total is None:
            total = h.Clone("data_obs")
            total.SetDirectory(0)
        else:
            total.Add(h)

    if total is None:
        raise ValueError("bin %s has no processes" % chan)

    # Pseudo-data has no statistical error of its own beyond sqrt(N).
    for i in range(0, total.GetNbinsX() + 2):
        c = total.GetBinContent(i)
        total.SetBinError(i, c ** 0.5 if c > 0 else 0.0)
    return total, detail


# ---------------------------------------------------------------------------
# build / verify
# ---------------------------------------------------------------------------

def build(args):
    card = Card(args.datacard)
    norms = parse_norms(args.norms)

    # The SR pseudo-data is only as blind-but-realistic as these norms, so a
    # name mismatch here must be fatal, not a fallback: a card rateParam
    # without a fitted value would silently be built at its init (1.0), the
    # SR pseudo-data then disagrees with the CR data the fit sees, and the
    # high-stat SRs pin the norms near 1 with an artificially tiny
    # uncertainty. (Exactly what happens when the cards' rateParams gain the
    # _<year> suffix while read_cr_norms still emits the old bare names.)
    card_rp_names = {name for name, _, _, _ in card.rate_params}
    missing = sorted(card_rp_names - set(norms))
    stale = sorted(set(norms) - card_rp_names)
    if missing:
        sys.exit("ERROR: no fitted value given for card rateParam(s): "
                 + ", ".join(missing)
                 + "\n       --norms carried: "
                 + (", ".join(sorted(norms)) or "(nothing)")
                 + "\n       The SR pseudo-data would silently be built at "
                 "norm = 1.0.\n       Rerun step 5c / read_cr_norms so the "
                 "names match the card exactly.")
    if stale:
        sys.exit("ERROR: --norms names parameter(s) absent from the card: "
                 + ", ".join(stale)
                 + "\n       Stale names (old un-year-tagged values?); "
                 "refusing to guess.")

    files = card.shapes_files()
    if len(files) != 1:
        sys.exit("ERROR: %s references %d shapes files; this script supports "
                 "exactly one.\n       %s"
                 % (card.path, len(files), "\n       ".join(files)))
    src_root = files[0]
    if not os.path.exists(src_root):
        sys.exit("ERROR: shapes file does not exist: %s" % src_root)

    sr_bins = (args.sr_bins.split(",") if args.sr_bins else card.signal_bins())
    all_bins = card.bins()
    unknown = [b for b in sr_bins if b not in all_bins]
    if unknown:
        sys.exit("ERROR: --sr-bins names bins not in the card: %s"
                 % ", ".join(unknown))
    if not sr_bins:
        sys.exit("ERROR: no signal-region bin found in %s (no process has "
                 "index <= 0). Pass --sr-bins explicitly." % card.path)
    cr_bins = [b for b in all_bins if b not in sr_bins]

    print("  card:        %s" % card.path)
    print("  shapes file: %s" % src_root)
    print("  SR bins (data_obs -> MC expectation): %s" % ", ".join(sr_bins))
    print("  CR bins (data_obs -> REAL DATA):      %s"
          % (", ".join(cr_bins) if cr_bins else "(none!)"))
    print("  norms:       %s" % (", ".join("%s=%.5f" % kv for kv in sorted(norms.items()))
                                 or "(none given)"))
    print("  signal injected at r = %g" % args.inject_signal)

    if not cr_bins:
        sys.exit("ERROR: every bin in this card looks like a signal region, so "
                 "there is no control-region data to fit. Refusing to write a "
                 "dataset that is pure Asimov -- use -t -1 for that.")

    os.makedirs(args.outdir, exist_ok=True)
    dst_root = os.path.join(os.path.abspath(args.outdir), "histograms_hybrid.root")
    shutil.copyfile(src_root, dst_root)

    fin = open_eos_retry(src_root)
    fout = open_eos_retry(dst_root, "UPDATE")

    warned = set()
    new_obs = {}
    for chan in sr_bins:
        total, detail = sr_expectation(card, fin, chan, norms,
                                       args.inject_signal, args.mass, warned)
        entry = card.shape_entry("data_obs", chan)
        path = _subst(entry[3], "data_obs", chan, args.mass)
        write_hist(fout, path, total)
        new_obs[chan] = total.Integral()

        print("")
        print("  %s -> %s" % (chan, path))
        for proc, kind, scale, yld in detail:
            print("      %-18s %-3s  x%-8.4f  %10.3f" % (proc, kind, scale, yld))
        print("      %-18s %-3s  %9s  %10.3f" % ("TOTAL (pseudo-data)", "", "", total.Integral()))

    fout.Close()
    fin.Close()

    write_card(card, dst_root, new_obs, args.outdir)

    print("")
    verify(args.outdir, src_root)


def write_card(card, dst_root, new_obs, outdir):
    """Copy the datacard, repoint its shapes lines and update observation."""
    lines = list(card.lines)

    for i, line in enumerate(lines):
        tok = line.split()
        if tok and tok[0] == "shapes" and len(tok) >= 5:
            tok[3] = dst_root
            lines[i] = " ".join(tok)

    obs_tok = lines[card.obs_line_idx].split()
    vals = obs_tok[1:]
    if len(vals) != len(card.obs_bins):
        raise ValueError("observation line has %d values but its bin line has %d"
                         % (len(vals), len(card.obs_bins)))
    out_vals = []
    for b, v in zip(card.obs_bins, vals):
        out_vals.append("%.6f" % new_obs[b] if b in new_obs else v)
    width = max([len(x) for x in out_vals] + [len(b) for b in card.obs_bins]) + 2
    lines[card.obs_bin_line_idx] = "bin".ljust(14) + "".join(
        b.ljust(width) for b in card.obs_bins)
    lines[card.obs_line_idx] = "observation".ljust(14) + "".join(
        v.ljust(width) for v in out_vals)

    out = os.path.join(outdir, "datacard.txt")
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("  wrote %s" % out)


def verify(outdir, src_root=None):
    """Independently re-derive the SR pseudo-data and re-check the CR data.

    Run after every build, and available standalone via --verify-only. Any
    failure exits nonzero so the calling pipeline stops before a fit sees an
    unblinded signal region.
    """
    hybrid_card = Card(os.path.join(outdir, "datacard.txt"))
    dst_root = hybrid_card.shapes_files()[0]

    if src_root is None:
        sys.exit("ERROR: --verify-only needs --original-shapes <file> to compare "
                 "the control-region data against.")

    fin = open_eos_retry(src_root)
    fhyb = open_eos_retry(dst_root)

    sr_bins = hybrid_card.signal_bins()
    cr_bins = [b for b in hybrid_card.bins() if b not in sr_bins]

    problems = []

    # 1. CR data must be untouched.
    for chan in cr_bins:
        entry = hybrid_card.shape_entry("data_obs", chan)
        path = _subst(entry[3], "data_obs", chan, 125)
        try:
            h_old = get_hist(fin, path, "CR check")
            h_new = get_hist(fhyb, path, "CR check")
        except KeyError as exc:
            problems.append(str(exc))
            continue
        if not same_contents(h_old, h_new):
            problems.append("CR bin %s: data_obs was modified" % chan)

    # 2. SR data must equal the MC expectation, i.e. contain no observation.
    norms = {}
    for name, _, _, _ in hybrid_card.rate_params:
        norms.setdefault(name, None)
    for chan in sr_bins:
        entry = hybrid_card.shape_entry("data_obs", chan)
        path = _subst(entry[3], "data_obs", chan, 125)
        try:
            h_new = get_hist(fhyb, path, "SR check")
        except KeyError as exc:
            problems.append(str(exc))
            continue
        try:
            h_old = get_hist(fin, path, "SR check")
        except KeyError:
            h_old = None
        if h_old is not None and same_contents(h_old, h_new):
            # Only a problem if the original actually held data; an already
            # blinded (all-zero) input legitimately matches an all-zero SR.
            if h_old.Integral() != 0.0:
                problems.append(
                    "SR bin %s: data_obs is IDENTICAL to the input file -- the "
                    "replacement did not happen and this card may be unblinded"
                    % chan)

    fin.Close()
    fhyb.Close()

    if problems:
        print("BLINDING CHECK FAILED:")
        for p in problems:
            print("  - %s" % p)
        sys.exit(1)

    print("  BLINDING CHECK OK: %d CR bin(s) keep real data, %d SR bin(s) hold "
          "MC pseudo-data." % (len(cr_bins), len(sr_bins)))


def parse_norms(spec):
    out = {}
    if not spec:
        return out
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            sys.exit("ERROR: --norms entry '%s' is not name=value" % item)
        k, v = item.split("=", 1)
        try:
            out[k.strip()] = float(v)
        except ValueError:
            sys.exit("ERROR: --norms entry '%s' has a non-numeric value" % item)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datacard", help="input datacard (SR + CR channels)")
    ap.add_argument("--norms", default="",
                    help="rateParam values from the CR-only fit, e.g. "
                         "'norm_top=1.04,norm_wjet=0.97'")
    ap.add_argument("--inject-signal", type=float, default=1.0,
                    help="signal strength used to build the SR pseudo-data "
                         "(1 = S+B expected significance, 0 = background only)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--sr-bins", default="",
                    help="comma-separated bins to blind; default is every bin "
                         "containing a process with index <= 0")
    ap.add_argument("--mass", type=int, default=125)
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--original-shapes", default=None,
                    help="with --verify-only: the pre-hybrid shapes file")
    args = ap.parse_args()

    if args.verify_only:
        verify(args.outdir, args.original_shapes)
        return
    if not args.datacard:
        sys.exit("ERROR: --datacard is required unless --verify-only is given")
    build(args)


if __name__ == "__main__":
    main()
