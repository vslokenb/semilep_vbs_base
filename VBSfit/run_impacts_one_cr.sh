#!/bin/bash
# Run impact plots for a single datacard.
#
# Usage: run_impacts_one_cr.sh <datacard_path> <outdir> [extra_combine_opts] [toy_opts]
#
# extra_combine_opts is passed verbatim to every combineTool call, e.g.
#   "--setParameters norm_top=1.043,norm_wjet=0.981"
#
# toy_opts controls the DATASET, and is also passed verbatim to every call.
# It defaults to "-t -1 --expectSignal 1", i.e. the MC-only Asimov behaviour
# this script has always had, so existing 2- and 3-argument callers are
# unaffected. Pass an EMPTY 4th argument to fit the dataset stored in the
# datacard instead:
#
#   run_impacts_one_cr.sh cards_hybrid/combined/datacard.txt out/impacts "" ""
#
# That empty form is what the hybrid cards from make_hybrid_data.py need.
# Those cards already contain real data in the control regions and MC
# pseudo-data in the signal regions; adding "-t -1" would overwrite BOTH with
# Asimov and silently throw the CR data away, which is the whole point of
# building them. The signal regions stay blind either way -- their observation
# is MC by construction -- so this does not unblind anything.
#
# IMPACTS_TOY_OPTS in the environment overrides the 4th argument.
#
# Note the ${4-default} form (no colon): an explicitly empty 4th argument is
# honoured as "no toy options", while an omitted one falls back to Asimov.
set -e
DATACARD="$1"
OUTDIR="$2"
EXTRA_OPTS="${3:-}"
TOY_OPTS="${IMPACTS_TOY_OPTS-${4--t -1 --expectSignal 1}}"
MASS=125

# Marker used by run_all.sh to detect that this script understands toy_opts.
# Do not remove: older copies without it are rejected rather than silently
# producing Asimov impacts for the hybrid cards.
# IMPACTS_TOY_OPTS_SUPPORTED

VBSFIT_DIR="/eos/user/v/vslokenb/vbs_semilep/sensitivities/VBSfit"

source /cvmfs/cms.cern.ch/cmsset_default.sh 2>/dev/null || true
cd "${VBSFIT_DIR}/CMSSW_14_1_0_pre4/src"
eval $(scramv1 runtime -sh)

mkdir -p "${OUTDIR}"
cd "${OUTDIR}"

if [ -n "${EXTRA_OPTS}" ]; then
    echo "[$(date '+%H:%M:%S')] Extra options: ${EXTRA_OPTS}"
fi
if [ -n "${TOY_OPTS}" ]; then
    echo "[$(date '+%H:%M:%S')] Dataset options: ${TOY_OPTS}"
else
    echo "[$(date '+%H:%M:%S')] Dataset options: (none) -- fitting the dataset in the card"
fi

# Step 0: compile text datacard to RooFit workspace
echo "[$(date '+%H:%M:%S')] Compiling workspace: ${OUTDIR}"
text2workspace.py "${DATACARD}" -o workspace.root -m ${MASS} 2>&1 | tail -3

echo "[$(date '+%H:%M:%S')] Initial fit: ${OUTDIR}"
combineTool.py -M Impacts \
    -d workspace.root \
    -m ${MASS} \
    ${TOY_OPTS} \
    ${EXTRA_OPTS} \
    --robustFit 1 \
    --doInitialFit \
    2>&1 | tail -3

echo "[$(date '+%H:%M:%S')] Scanning nuisances: ${OUTDIR}"
combineTool.py -M Impacts \
    -d workspace.root \
    -m ${MASS} \
    ${TOY_OPTS} \
    ${EXTRA_OPTS} \
    --robustFit 1 \
    --doFits --parallel 20 \
    2>&1 | tail -3

echo "[$(date '+%H:%M:%S')] Collecting + plotting: ${OUTDIR}"
combineTool.py -M Impacts \
    -d workspace.root \
    -m ${MASS} \
    ${TOY_OPTS} \
    ${EXTRA_OPTS} \
    --output impacts.json \
    2>&1 | tail -2

plotImpacts.py -i impacts.json -o impacts 2>&1 | tail -2
echo "[$(date '+%H:%M:%S')] Done: ${OUTDIR}/impacts.pdf"