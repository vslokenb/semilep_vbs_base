#!/usr/bin/env bash
# config_validation + config_inverted, 2018 only, on latest_poco (patched JES).
# Sequential: the gateway allows one cluster per user.
# --process-separately => one output_<dataset>.coffea each, a failing dataset is
# logged to failed_jobs.json instead of aborting, and retries resume per dataset.
WORKDIR=/work/users/vsloken-fnal/semilep_vbs_base/Pocket/temporary_workspace
IMG=/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/pocketcoffea:lxplus-el9-latest
BINDS=(-B /work/users/vsloken-fnal/ -B /cvmfs -B /cvmfs/cms-griddata.cern.ch -B /cvmfs/cms.cern.ch -B /tmp -B /etc/sysconfig/)
PY=/work/users/vsloken-fnal/envs/my-env/bin/python3.11
export X509_USER_PROXY=/work/users/vsloken-fnal/x509up_u2070155
cd "$WORKDIR"
OPTS=custom_v30_2018.yaml
YEAR=${YEAR:-2018}         # data-taking year to process
PREFIX=${PREFIX:-v30}      # output dir prefix (v31 = bdt_nov13 classifiers)
SCALEOUT=${SCALEOUT:-50}   # CLI -s, overrides the yaml
TT_CAP=${TT_CAP:-50}       # --limit-files for the two ttbar datasets
MAX_ATTEMPTS=3

stop_clusters() {
  timeout 150 "$PY" -c "
from dask_gateway import Gateway
g=Gateway()
for c in g.list_clusters():
    try: g.connect(c.name).shutdown()
    except Exception: pass
" >/dev/null 2>&1
}

run_pass() {   # $1=config  $2=outdir
  local CFG=$1 OUT=$2
  mkdir -p "$OUT"
  for ATTEMPT in $(seq 1 $MAX_ATTEMPTS); do
    local MISS; MISS=$("$PY" v30_missing.py "$CFG" "$OUT" "$YEAR")
    local NMISS; NMISS=$(echo "$MISS" | tr ',' '\n' | grep -c .)
    [ "$NMISS" -eq 0 ] && break
    # TTToSemiLeptonic + TTTo2L2Nu are 53% of all 2018 events; they run capped at
    # TT_CAP files. --limit-files is global to one invocation, so they go in a
    # separate pass from everything else.
    local NOTT TT
    NOTT=$(echo "$MISS" | tr ',' '\n' | grep -v '^TT' | paste -sd, -)
    TT=$(echo "$MISS"   | tr ',' '\n' | grep    '^TT' | paste -sd, -)
    echo "=== $CFG | $YEAR | attempt $ATTEMPT | $NMISS left (non-tt=$(echo "$NOTT" | tr ',' '\n' | grep -c .), tt=$(echo "$TT" | tr ',' '\n' | grep -c .) capped at ${TT_CAP}) | $(date -u +'%m-%d %H:%M') ==="
    if [ -n "$NOTT" ]; then
      stop_clusters
      apptainer exec "${BINDS[@]}" "$IMG" bash "$WORKDIR/run_inside_v30.sh" "$CFG" "$OUT" "$NOTT" "$OPTS" "$SCALEOUT" "" "$YEAR"
      stop_clusters
    fi
    if [ -n "$TT" ]; then
      echo "=== $CFG | ttbar pass | --limit-files $TT_CAP | $(date -u +'%m-%d %H:%M') ==="
      stop_clusters
      apptainer exec "${BINDS[@]}" "$IMG" bash "$WORKDIR/run_inside_v30.sh" "$CFG" "$OUT" "$TT" "$OPTS" "$SCALEOUT" "$TT_CAP" "$YEAR"
      stop_clusters
    fi
    echo "=== $CFG after attempt $ATTEMPT: $("$PY" v30_missing.py "$CFG" "$OUT" "$YEAR" --count) ==="
  done
  local FIN; FIN=$("$PY" v30_missing.py "$CFG" "$OUT" "$YEAR")
  if [ -z "$FIN" ]; then echo "=== OK $CFG ($("$PY" v30_missing.py "$CFG" "$OUT" "$YEAR" --count)) ==="
  else echo "=== INCOMPLETE $CFG -- missing: $FIN ==="; fi
}

run_pass config_validation.py "${PREFIX}_validation_${YEAR}"
run_pass config_inverted.py  "${PREFIX}_inverted_${YEAR}"
echo "=== ${PREFIX^^} ${YEAR} COMPLETE ==="
