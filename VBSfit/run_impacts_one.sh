#!/bin/bash
# Run impact plots for a single datacard.
# Usage: run_impacts_one.sh <datacard_path> <outdir>
set -e
DATACARD="$1"
OUTDIR="$2"
MASS=125

source /cvmfs/cms.cern.ch/cmsset_default.sh 2>/dev/null || true
cd /eos/user/v/vslokenb/vbs_semilep/sensitivities/VBSfit/CMSSW_14_1_0_pre4/src
eval $(scramv1 runtime -sh)

mkdir -p "${OUTDIR}"
cd "${OUTDIR}"

# Step 0: compile text datacard to RooFit workspace
echo "[$(date '+%H:%M:%S')] Compiling workspace: ${OUTDIR}"
text2workspace.py "${DATACARD}" -o workspace.root -m ${MASS} 2>&1 | tail -3

echo "[$(date '+%H:%M:%S')] Initial fit: ${OUTDIR}"
combineTool.py -M Impacts \
    -d workspace.root \
    -m ${MASS} \
    -t -1 --expectSignal 1 \
    --robustFit 1 \
    --doInitialFit \
    2>&1 | tail -3

echo "[$(date '+%H:%M:%S')] Scanning nuisances: ${OUTDIR}"
combineTool.py -M Impacts \
    -d workspace.root \
    -m ${MASS} \
    -t -1 --expectSignal 1 \
    --robustFit 1 \
    --doFits --parallel 20 \
    2>&1 | tail -3

echo "[$(date '+%H:%M:%S')] Collecting + plotting: ${OUTDIR}"
combineTool.py -M Impacts \
    -d workspace.root \
    -m ${MASS} \
    --output impacts.json \
    2>&1 | tail -2

plotImpacts.py -i impacts.json -o impacts 2>&1 | tail -2
echo "[$(date '+%H:%M:%S')] Done: ${OUTDIR}/impacts.pdf"
