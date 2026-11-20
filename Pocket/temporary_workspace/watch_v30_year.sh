#!/usr/bin/env bash
# Year-aware read-only watcher: alerts on each dataset written, any dataset
# failure, and chain exit. Never kills, never deletes.
#   usage: YEAR=2017 ./watch_v30_year.sh [minutes]
set -u
cd /work/users/vsloken-fnal/semilep_vbs_base/Pocket/temporary_workspace
PY=/work/users/vsloken-fnal/envs/my-env/bin/python3.11
YEAR=${YEAR:-2018}
PREFIX=${PREFIX:-v30}
CHAIN=${CHAIN:-run_v31_all.sh}
MAX_MIN=${1:-1440}
DEADLINE=$(( $(date +%s) + MAX_MIN*60 ))
VAL="${PREFIX}_validation_${YEAR}"; INV="${PREFIX}_inverted_${YEAR}"
LOG=$(cat .current_log_v30)
say(){ echo "[$(date -u +'%m-%d %H:%M:%S')] $*"; }
count(){ ls -1 "$VAL"/output_*.coffea "$INV"/output_*.coffea 2>/dev/null | wc -l; }
LAST_OK=$(count); LAST_FAIL=0
say "watching $LOG  year=$YEAR chain=$CHAIN  (start: $LAST_OK outputs)"
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  if ! pgrep -f "$CHAIN" >/dev/null 2>&1; then
    say "ALERT: chain exited. validation=$($PY v30_missing.py config_validation.py "$VAL" "$YEAR" --count) inverted=$($PY v30_missing.py config_inverted.py "$INV" "$YEAR" --count)"
    exit 0
  fi
  N=$(count)
  F=$(tr '\r' '\n' < "$LOG" 2>/dev/null | grep -aic 'failed, moving')
  if [ "$N" -gt "$LAST_OK" ]; then
    say "completed: $N written (+$((N-LAST_OK)))  [val $($PY v30_missing.py config_validation.py "$VAL" "$YEAR" --count) | inv $($PY v30_missing.py config_inverted.py "$INV" "$YEAR" --count)]"
    LAST_OK=$N
  fi
  if [ "$F" -gt "$LAST_FAIL" ]; then
    say "ALERT: dataset FAILED (total=$F)"
    tr '\r' '\n' < "$LOG" | grep -a 'failed, moving' | tail -1 | sed 's/\x1b\[[0-9;]*m//g' | sed 's/^/    /'
    LAST_FAIL=$F
  fi
  sleep 180
done
say "window elapsed: $LAST_OK outputs, $LAST_FAIL failures"
