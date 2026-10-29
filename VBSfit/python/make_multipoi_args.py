#!/usr/bin/env python3
"""
make_multipoi_args.py -- build the text2workspace multiSignalModel arguments
that split the VBS signal into separate WW and WZ signal strengths.

Reads the SIGNAL processes straight out of a datacard (process index <= 0),
classifies each one, and prints the --PO map options, one token per line:

    --PO
    map=.*/WminusToLNuWminusTo2JJJ_...$:r_WW[1,-5,10]
    --PO
    map=.*/WminusToLNuZTo2JJJ_...$:r_WZ[1,-5,10]
    ...

so a shell can do

    mapfile -t PO < <(make_multipoi_args.py --datacard card.txt)
    text2workspace.py card.txt \\
        -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel \\
        "${PO[@]}" -o ws.root

One map per process, with the exact name anchored by '$', rather than a
catch-all regex per group. multiSignalModel gives a process that matches NO map
a constant scale of 1 -- it silently stops being signal and becomes a fixed
component of the expectation. Enumerating from the card makes that impossible:
every signal process gets exactly one map, and --list-pois / the stderr summary
show which POI each one landed on.

CLASSIFICATION
--------------
By diboson content of the process name:

    two 'ZTo'            -> ZZ    (e.g. ZTo2LZTo2JJJ_...)
    one 'ZTo'            -> WZ    (e.g. WminusToLNuZTo2JJJ_, WplusTo2JZTo2LJJ_)
    no 'ZTo'             -> WW    (e.g. WplusTo2JWminusToLNuJJ_)

--zz decides what happens to the ZZ processes:
    wz   (default) they scale with r_WZ, i.e. the second POI is really "WZ+ZZ"
    ww   they scale with r_WW
    own  they get their own r_ZZ, profiled when measuring r_WW / r_WZ

Use --show to print the classification without emitting arguments.
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from make_hybrid_data import Card
except ImportError as exc:                              # pragma: no cover
    sys.exit("ERROR: cannot import Card from make_hybrid_data.py "
             "(expected next to this script): %s" % exc)

POI_WW = "r_WW"
POI_WZ = "r_WZ"
POI_ZZ = "r_ZZ"

# Every signal process name must contain this.
#
# The QCD-induced diboson processes are BACKGROUND and their names differ from
# the EWK signal only by this token:
#
#   WminusTo2JZTo2LJJ_QCD_LO_SM_MJJ100PTJ10_...              background
#   WminusTo2JZTo2LJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_... signal
#
# The maps below use each process's exact name anchored with '$', so a QCD name
# can never match a signal map. The check exists for the other direction: if a
# QCD process ever carries a signal index in the datacard, or an EWK process is
# written as background, it would silently end up on the wrong side of the fit.
SIGNAL_REQUIRED_TOKEN = "dipoleRecoil"


def classify(process):
    """Return 'WW', 'WZ' or 'ZZ' for a signal process name."""
    # Counts diboson decay tokens, which appear before the _dipoleRecoil_EWK_
    # suffix and so are unaffected by it.
    n_z = len(re.findall(r"ZTo", process))
    if n_z >= 2:
        return "ZZ"
    if n_z == 1:
        return "WZ"
    return "WW"


def poi_for(group, zz_assign):
    if group == "WW":
        return POI_WW
    if group == "WZ":
        return POI_WZ
    return {"wz": POI_WZ, "ww": POI_WW, "own": POI_ZZ}[zz_assign]


def signal_processes(card):
    """Signal process names, in card order, deduplicated."""
    out = []
    for _, proc, idx, _ in card.columns:
        if idx <= 0 and proc not in out:
            out.append(proc)
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datacard", required=True)
    ap.add_argument("--zz", choices=["wz", "ww", "own"], default="wz",
                    help="which POI the ZZ processes scale with (default wz)")
    ap.add_argument("--range", default="[1,-5,10]",
                    help="POI declaration range, default '[1,-5,10]'")
    ap.add_argument("--show", action="store_true",
                    help="print the classification table only")
    ap.add_argument("--list-pois", action="store_true",
                    help="print the comma-separated POI names only")
    ap.add_argument("--require-token", default=SIGNAL_REQUIRED_TOKEN,
                    help="abort if a signal process name lacks this substring "
                         "(default %r); pass '' to disable" % SIGNAL_REQUIRED_TOKEN)
    args = ap.parse_args()

    card = Card(args.datacard)
    sigs = signal_processes(card)
    if not sigs:
        sys.exit("ERROR: %s has no process with index <= 0, so it declares no "
                 "signal." % args.datacard)

    token = args.require_token
    if token:
        bad = [p for p in sigs if token not in p]
        if bad:
            sys.exit(
                "ERROR: these processes carry a SIGNAL index in %s but their "
                "names do not contain %r:\n    %s\n"
                "  The QCD-induced diboson processes are background and differ "
                "from the EWK signal only by that token, so this usually means "
                "a QCD process was given a signal index.\n"
                "  Fix the datacard, or pass --require-token '' to skip this "
                "check." % (args.datacard, token, "\n    ".join(bad)))

        # The other direction: an EWK process sitting in the background block
        # would never be scaled by any POI, and the fit would quietly measure
        # only part of the signal.
        bkg_ewk = sorted({proc for _, proc, idx, _ in card.columns
                          if idx > 0 and token in proc})
        if bkg_ewk:
            sys.stderr.write(
                "  [warn] these processes contain %r but are declared as "
                "BACKGROUND, so no POI scales them:\n" % token)
            for b in bkg_ewk:
                sys.stderr.write("           %s\n" % b)

    assigned = [(p, classify(p), poi_for(classify(p), args.zz)) for p in sigs]

    # Keep POI order stable and predictable: WW, WZ, then ZZ if it has its own.
    pois = [p for p in (POI_WW, POI_WZ, POI_ZZ)
            if any(a[2] == p for a in assigned)]

    if args.list_pois:
        print(",".join(pois))
        return

    if args.show:
        print("signal processes in %s" % args.datacard)
        width = max(len(p) for p, _, _ in assigned)
        for proc, group, poi in assigned:
            print("  %-*s  %-3s -> %s" % (width, proc, group, poi))
        print("")
        print("POIs: %s" % ", ".join(pois))
        for poi in pois:
            n = sum(1 for a in assigned if a[2] == poi)
            print("  %-6s %d process(es)" % (poi, n))
        return

    # Summary to stderr so stdout stays a clean argument stream.
    sys.stderr.write("  multiSignalModel: %d signal process(es) -> %s\n"
                     % (len(assigned), ", ".join(pois)))
    for poi in pois:
        names = [p for p, _, q in assigned if q == poi]
        sys.stderr.write("    %-6s <- %s\n" % (poi, ", ".join(names)))
    if args.zz == "wz" and any(g == "ZZ" for _, g, _ in assigned):
        sys.stderr.write("    [note] ZZ processes are folded into %s "
                         "(--zz own gives them their own POI)\n" % POI_WZ)

    for proc, _, poi in assigned:
        print("--PO")
        print("map=.*/%s$:%s%s" % (re.escape(proc), poi, args.range))


if __name__ == "__main__":
    main()