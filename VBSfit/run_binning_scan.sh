#!/bin/bash
# run_binning_scan.sh -- systematic scan of the SR BDT binning.
#
# For each variant, the FULL run_all.sh workflow is rerun with a different
# --binning config and a unique --tag, so every variant gets its own
# histograms / cards / plots tree (histograms_cr_<year>_flat6.root,
# cards_cr_<year>_flat6/, plots_cr_<year>_flat6/, ...) and the variants can
# run IN PARALLEL without touching each other's files.
#
# Binning variants: flat-signal binnings with N = 4..8 merged bins by
# default -- edges placed on native (40-bin) BDT edges so each merged bin
# holds ~the same signal yield. The signal peaks at high score, so this gives
# coarse bins at low score and fine bins at high score automatically.
# make_binning_scan.py prints the per-bin signal/background yields for each
# variant when it generates the YAMLs (kept in the scan log).
#
# Usage (interactively, inside tmux on lxplus):
#   ./run_binning_scan.sh                         # year 2018, N=4..8, 3 jobs
#   ./run_binning_scan.sh --year=2017 --jobs=2
#   ./run_binning_scan.sh --nbins="5 6 8"         # which variants to run
#   ./run_binning_scan.sh --summary-only          # just reprint the table
#
# Each parallel job is a full pipeline (coffea loading + combine fits);
# memory and EOS load are the real limits on an interactive node, hence the
# conservative default of 3 concurrent jobs.
#
# The per-variant uncertainty breakdown (steps 5b/5f) is SKIPPED by default:
# it is ~7000 likelihood-scan minimizations per variant and does not feed the
# binning choice. Re-run the winner alone afterwards for the full treatment:
#   ./run_all.sh --year=<Y> --binning=config/binning_scan_<Y>/flat<N>bins.yaml \
#                --tag=flat<N> --breakdown-only
#
# pipefail matters: the python steps run through "| tee <log>", and without it
# a failing step would exit with tee's status 0 and the scan would charge on
# (the same way a "| tail -5" once hid combine dropping the postfit bands).
set -eo pipefail

VBSFIT_DIR="/eos/user/v/vslokenb/vbs_semilep/sensitivities/VBSfit"
MASS=125

YEAR="2018"
NBINS="4 5 6 7 8"
JOBS=3
SUMMARY_ONLY=0
EXTRA_ARGS="--skip-breakdown"   # forwarded to every run_all.sh invocation

for arg in "$@"; do
    case "${arg}" in
        --year=*)      YEAR="${arg#*=}" ;;
        --nbins=*)     NBINS="${arg#*=}" ;;
        --jobs=*)      JOBS="${arg#*=}" ;;
        --summary-only) SUMMARY_ONLY=1 ;;
        --extra=*)     EXTRA_ARGS="${arg#*=}" ;;
        *) echo "Unknown option: ${arg}"; exit 1 ;;
    esac
done

cd "${VBSFIT_DIR}"
SCAN_CFG_DIR="config/binning_scan_${YEAR}"
NATIVE_HIST_REL="histograms/histograms_native_${YEAR}.root"
LOG_DIR="${VBSFIT_DIR}/logs_binning_scan_${YEAR}"
mkdir -p "${LOG_DIR}"

# --- Apptainer environment (same image as run_all.sh) for the python steps ---
_APT_IMG="/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/pocketcoffea:lxplus-el9-latest"
pyrun() {
    apptainer exec \
        -B /eos -B /afs -B /cvmfs/cms.cern.ch -B /tmp -B /eos/cms/ \
        -B /etc/sysconfig/ngbauth-submit \
        -B "${XDG_RUNTIME_DIR}" \
        -B /cvmfs/cms-griddata.cern.ch \
        --env KRB5CCNAME="FILE:${XDG_RUNTIME_DIR}/krb5cc" \
        "${_APT_IMG}" bash -c "cd '${VBSFIT_DIR}' && $*"
}

summary() {
    echo ""
    echo "==================================================================="
    echo "Binning scan summary -- year ${YEAR}"
    echo "==================================================================="
    for n in ${NBINS}; do
        tag="flat${n}"
        outdir="${VBSFIT_DIR}/plots_cr_${YEAR}_${tag}"
        sens="${outdir}/significances_m${MASS}.txt"
        crsens="${outdir}/crdata_srasimov/significances_crdata_m${MASS}.txt"
        echo ""
        echo "--- ${tag} (${SCAN_CFG_DIR}/flat${n}bins.yaml) ---"
        if [ -f "${sens}" ]; then
            echo "  MC Asimov:"
            sed 's/^/    /' "${sens}"
        else
            echo "  MC Asimov:   MISSING (${sens})"
        fi
        if [ -f "${crsens}" ]; then
            echo "  CR-data hybrid:"
            sed 's/^/    /' "${crsens}"
        else
            echo "  CR-data hybrid: MISSING (${crsens})"
        fi
    done
    echo ""
    echo "Per-bin signal/background yields of every variant:"
    echo "  ${LOG_DIR}/make_binning_scan.log"
    echo "Per-bin s/sqrt(b) plots (one per variant):"
    echo "  ${VBSFIT_DIR}/plots_binning_scan_${YEAR}/ssqrtb_flat<N>bins.png"
    echo "Full per-variant pipeline logs:"
    echo "  ${LOG_DIR}/scan_flat<N>.log"
}

if [ "${SUMMARY_ONLY}" = "1" ]; then
    summary
    exit 0
fi

echo "========================================"
echo "BDT binning scan"
echo "  Year:      ${YEAR}"
echo "  Variants:  flat-signal, N in {${NBINS}} bins"
echo "  Parallel:  ${JOBS} jobs"
echo "  run_all:   ${EXTRA_ARGS:-<no extra args>}"
echo "  Logs:      ${LOG_DIR}"
echo "========================================"

# -------------------------------------------------------------------
# Step 0: native-binned histograms (once per year) -- the signal BDT
# distribution at full 40-bin granularity that the quantile edges are
# derived from. Reused if already present and newer than the inputs config.
# -------------------------------------------------------------------
CONFIG_REL="config/inputs_v28_${YEAR}.yaml"
if [ ! -f "${CONFIG_REL}" ]; then
    echo "ERROR: inputs config not found: ${CONFIG_REL}"
    exit 1
fi
if [ ! -f "python/make_binning_scan.py" ]; then
    echo "ERROR: python/make_binning_scan.py not found in ${VBSFIT_DIR}."
    echo "       Copy make_binning_scan.py into ${VBSFIT_DIR}/python/ first."
    exit 1
fi
if ! grep -q -- "--tag=" run_all.sh; then
    echo "ERROR: this run_all.sh has no --tag support -- install the updated"
    echo "       run_all.sh (with --binning/--tag) before running the scan."
    exit 1
fi
if [ ! -f "${NATIVE_HIST_REL}" ] || [ "${CONFIG_REL}" -nt "${NATIVE_HIST_REL}" ]; then
    echo ""
    echo "--- Step 0: building native-binned histograms (no --binning) ---"
    pyrun python3 python/make_histograms.py \
        --config "${CONFIG_REL}" \
        --output "${NATIVE_HIST_REL}" \
        2>&1 | tee "${LOG_DIR}/make_native_histograms.log"
else
    echo "Native histfile up to date: ${NATIVE_HIST_REL}"
fi

# -------------------------------------------------------------------
# Step 1: derive the flat-signal edge sets and write one YAML per variant
# -------------------------------------------------------------------
echo ""
echo "--- Step 1: deriving flat-signal binnings ---"
PLOT_DIR="plots_binning_scan_${YEAR}"
pyrun python3 python/make_binning_scan.py \
    --histfile "${NATIVE_HIST_REL}" \
    --outdir "${SCAN_CFG_DIR}" \
    --plotdir "${PLOT_DIR}" \
    --nbins ${NBINS} \
    2>&1 | tee "${LOG_DIR}/make_binning_scan.log"

for n in ${NBINS}; do
    if [ ! -f "${SCAN_CFG_DIR}/flat${n}bins.yaml" ]; then
        echo "ERROR: expected ${SCAN_CFG_DIR}/flat${n}bins.yaml was not written."
        echo "       Last 20 lines of ${LOG_DIR}/make_binning_scan.log:"
        tail -20 "${LOG_DIR}/make_binning_scan.log" | sed 's/^/    /'
        exit 1
    fi
done

# -------------------------------------------------------------------
# Step 2: run the full workflow for every variant, JOBS at a time.
# Each variant is a separate run_all.sh process with its own --tag, so the
# output trees are disjoint and the CMSSW/apptainer environments are set up
# per process. set -e must not kill the scan when one variant fails, so
# failures are collected and reported at the end instead.
#
# PYRUN_LOCK serializes the coffea/python steps ACROSS variants: they are
# the memory-heavy part (tens of GB each) and the lxplus user memory cgroup
# fits only one -- three at once got every variant OOM-killed in step 1.
# With the lock, one variant loads histograms/plots while the others run
# their combine fits, which is where the parallelism actually pays.
# -------------------------------------------------------------------
export PYRUN_LOCK="${LOG_DIR}/.pyrun.lock"
touch "${PYRUN_LOCK}"
echo ""
echo "--- Step 2: running ${JOBS}-wide over variants ---"
declare -A SCAN_PID
for n in ${NBINS}; do
    tag="flat${n}"
    while [ "$(jobs -rp | wc -l)" -ge "${JOBS}" ]; do
        wait -n || true   # one slot freed; its exit code is judged below
    done
    echo "  launching ${tag}  (log: ${LOG_DIR}/scan_${tag}.log)"
    ./run_all.sh --year="${YEAR}" \
                 --binning="${SCAN_CFG_DIR}/flat${n}bins.yaml" \
                 --tag="${tag}" \
                 ${EXTRA_ARGS} \
                 > "${LOG_DIR}/scan_${tag}.log" 2>&1 &
    SCAN_PID[${tag}]=$!
done

FAILED=""
for n in ${NBINS}; do
    tag="flat${n}"
    if ! wait "${SCAN_PID[${tag}]}"; then
        FAILED="${FAILED} ${tag}"
        echo "  [FAIL] ${tag} -- last 15 lines of its log:"
        tail -15 "${LOG_DIR}/scan_${tag}.log" | sed 's/^/    /'
    else
        echo "  [done] ${tag}"
    fi
done

summary

if [ -n "${FAILED}" ]; then
    echo ""
    echo "FAILED variants:${FAILED}"
    exit 1
fi
