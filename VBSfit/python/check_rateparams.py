#!/usr/bin/env python3
"""
check_rateparams.py -- confirm a datacard's rateParams reached the workspace.

make_datacards.py declares the normalization rateParams with a WILDCARD bin
field ("*resolved*" / "*boosted*") so they survive combineCards.py prefixing
bins with ch1_/ch2_. That relies on combine interpreting the pattern the way
fnmatch does. If it does not, the rateParam covers no bin at all: the fit still
runs, norm_* stays pinned at 1.0 with no constraint, and the control-region
data never reaches the background normalization.

Nothing in the fit output makes that visible, so this check runs once after the
cards are built. Exits nonzero if any declared rateParam is missing.

Usage: check_rateparams.py <workspace.root> <datacard.txt>
"""
import sys
import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kError

ws_path, card = sys.argv[1], sys.argv[2]

expected = []
for line in open(card):
    tok = line.split()
    if len(tok) > 1 and tok[1] == "rateParam" and tok[0] not in expected:
        expected.append(tok[0])

if not expected:
    print("  [note] %s declares no rateParams." % card)
    sys.exit(0)

f = ROOT.TFile.Open(ws_path)
w = f.Get("w") if f and not f.IsZombie() else None
if not w:
    print("  [warn] no RooWorkspace 'w' in %s; skipping check." % ws_path)
    sys.exit(0)

have = {v.GetName() for v in w.allVars()}
missing = [n for n in expected if n not in have]

print("  datacard declares: %s" % ", ".join(expected))
if missing:
    print("  ERROR: not in the workspace: %s" % ", ".join(missing))
    print("         combine did not match their wildcard bin pattern, so they")
    print("         constrain nothing. Either re-run make_datacards.py with")
    print("         --decorrelate-norms none, or change TOPOLOGY_PATTERNS to")
    print("         use explicit channel names instead of '*resolved*'.")
    sys.exit(1)
print("  all %d present in the workspace." % len(expected))