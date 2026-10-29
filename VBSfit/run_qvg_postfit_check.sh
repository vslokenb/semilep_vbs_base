#!/usr/bin/env bash
# End-to-end: datacards -> per-region FitDiagnostics -> postfit param/
# covariance extraction -> QvG postfit plots (VBSjet1/2, w_had_jet1/2 PNet
# QvG score, three uncertainty bands: stat only / stat+QvG SF / stat+all).
#
# Three environments are needed and this script switches between them
# itself -- run it as a plain interactive shell (it sources everything it
# needs), don't pre-source CMSSW or the apptainer env yourself:
#   1. CMSSW/combine        for FitDiagnostics
#   2. CMSSW/PyROOT         for extract_postfit_params.py
#   3. PocketCoffea apptainer (coffea + mplhep) for plot_qvg_postfit.py
#
# Usage:
#   ./run_qvg_postfit_check.sh
#
# Re-running is safe/cheap: FitDiagnostics and the postfit-params JSON are
# both skipped if their output already exists. Delete the specific file(s)
# under $OUTBASE/fitdiagnostics_<region>/ to force a redo for one region
# (e.g. after the datacard changes -- stale workspaces are NOT
# auto-detected, this is a known combine/condor pitfall). The plotting step
# always reruns (it's fast, seconds per variable).
set -euo pipefail

# --- Config: edit these to point at a different production -------------
VBSFIT_DIR="/eos/user/v/vslokenb/vbs_semilep/sensitivities/VBSfit"
CONFIG=$VBSFIT_DIR/config/inputs_2018.yaml
CARDS="$VBSFIT_DIR/cards_crdata_srasimov"
OUTBASE=$VBSFIT_DIR/plots_cr/crdata_srasimov
QVGOUT=$OUTBASE/qvg_postfit_check
REGIONS=(resolved_mu resolved_e boosted_mu boosted_e)
MASS=125
FIT_TAG=offsetV9
# Offset study: FitDiagnostics on the card's own (already-Asimov-like)
# data_obs, no toy args. For the postfit-Asimov/realdata study instead, set
# TOY_ARGS="-t -1 --toysFrequentist --expectSignal 1" and point CARDS/
# OUTBASE/FIT_TAG at that study's cards/plots dir + naming instead.
TOY_ARGS=""
APPTAINER_IMG=/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/pocketcoffea:lxplus-el9-latest
# -------------------------------------------------------------------------

echo "=== Step 1/3: FitDiagnostics per region (CMSSW/combine env) ==="
source /cvmfs/cms.cern.ch/cmsset_default.sh
( cd "$VBSFIT_DIR/CMSSW_14_1_0_pre4/src" && eval "$(scramv1 runtime -sh)" )
eval "$(cd "$VBSFIT_DIR/CMSSW_14_1_0_pre4/src" && scramv1 runtime -sh)"

for region in "${REGIONS[@]}"; do
    od="$OUTBASE/fitdiagnostics_${region}"
    fitfile="$od/fitDiagnostics_${FIT_TAG}_${region}.root"
    if [ -f "$fitfile" ]; then
        echo "  [skip] ${region}: ${fitfile} already exists"
        continue
    fi
    card="$CARDS/${region}/datacard.txt"
    if [ ! -f "$card" ]; then
        echo "  [ERROR] ${region}: datacard not found at ${card}" >&2
        exit 1
    fi
    mkdir -p "$od"
    echo "  running FitDiagnostics for ${region} ..."
    ( cd "$od" && \
      combine -M FitDiagnostics -d "$card" -m "$MASS" $TOY_ARGS \
        --saveShapes --saveWithUncertainties --robustHesse 1 \
        --cminDefaultMinimizerStrategy 0 --robustFit 1 \
        -n "_${FIT_TAG}_${region}" > fitdiag.log 2>&1 )
    echo "  done: ${fitfile}"
done

echo
echo "=== Step 2/3: extract postfit params + covariance (CMSSW/PyROOT env) ==="
cd "$VBSFIT_DIR"
for region in "${REGIONS[@]}"; do
    od="$OUTBASE/fitdiagnostics_${region}"
    fitfile="$od/fitDiagnostics_${FIT_TAG}_${region}.root"
    outjson="$od/postfit_params.json"
    if [ -f "$outjson" ]; then
        echo "  [skip] ${region}: ${outjson} already exists"
        continue
    fi
    python3 python/extract_postfit_params.py \
        --fitfile "$fitfile" --region "$region" --output "$outjson"
done

echo
echo "=== Step 3/3: QvG postfit plots (PocketCoffea apptainer env) ==="
# pyrun()'s documented pitfall (see python/driver_common.py): the CMSSW env
# sourced above leaks LD_LIBRARY_PATH/PYTHONHOME/ROOTSYS into the container
# via inheritance and conflicts with its own libraries, so strip them here.
env -u LD_LIBRARY_PATH -u PYTHONHOME -u ROOTSYS \
    apptainer exec -B /eos -B /afs -B /cvmfs/cms.cern.ch -B /tmp -B /etc/sysconfig/ngbauth-submit -B ${XDG_RUNTIME_DIR} \
     --env KRB5CCNAME="FILE:${XDG_RUNTIME_DIR}/krb5cc" \
    "$APPTAINER_IMG" bash -c "
        source /afs/cern.ch/work/j/jinw/public/myenv/bin/activate &&
        export PYTHONPATH=/afs/cern.ch/work/j/jinw/public/PocketCoffea &&
        cd '$VBSFIT_DIR' &&
        python3 python/plot_qvg_postfit.py \
            --config '$CONFIG' \
            --postfit-dir '$OUTBASE' \
            --outdir '$QVGOUT'
    "

echo
echo "=== Done ==="
echo "  FitDiagnostics: ${OUTBASE}/fitdiagnostics_<region>/"
echo "  Plots:          ${QVGOUT}/"