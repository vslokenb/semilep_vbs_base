#!/usr/bin/env bash
# Full Run 2: all four eras x (config_validation, config_inverted), written to
# v31_* so the pre-BDT v30_* outputs are untouched and nothing is skipped by
# per-dataset resume. Sequential: one dask cluster per user.
#   bdt_nov13 classifiers + workflow_v9 jetcentral1..4 patch
set -u
cd /work/users/vsloken-fnal/semilep_vbs_base/Pocket/temporary_workspace
export SCALEOUT=${SCALEOUT:-50}
export TT_CAP=${TT_CAP:-25}
export PREFIX=v31
PY=/work/users/vsloken-fnal/envs/my-env/bin/python3.11
for Y in 2016_PreVFP 2016_PostVFP 2017 2018; do
  echo "########## ERA $Y  $(date -u +'%m-%d %H:%M') ##########"
  TS=$(date -u +%Y%m%d_%H%M%S)
  RUNLOG="logs/run_v31_${Y}_${TS}.log"
  echo "$RUNLOG" > .current_log_v30
  YEAR=$Y ./run_v30_year.sh > "$RUNLOG" 2>&1
  echo "########## ERA $Y DONE: val=$($PY v30_missing.py config_validation.py v31_validation_${Y} $Y --count) inv=$($PY v30_missing.py config_inverted.py v31_inverted_${Y} $Y --count) ##########"
done
echo "=== V31 ALL RUN2 ERAS COMPLETE ==="
