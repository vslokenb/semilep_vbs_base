#!/bin/bash
# run_run2_fits.sh -- cross-year (full Run 2) combination fits per category:
# resolved_mu across all year-eras, resolved_e across all year-eras, etc.,
# plus the all-category all-year combination.
#
# CORRELATION SCHEME (the whole point of this pipeline):
#   - experimental nuisances whose names natively carry a year label
#     (AK4PFchs_JES_Regrouped_*_2017, AK4PFchs_JER_2018, lumi_<year>, ...)
#     -> DEcorrelated across years (distinct names never merge)
#   - experimental nuisances without year labels (sf_btag_*, triggers, ...)
#     -> correlated across years (one shared nuisance)
#   - theory uncertainties (LHE scales, PDF, parton shower -- including their
#     per-MC-group split) -> correlated across years
#   - norm_top / norm_wjet rateParams -> correlated across years: one
#     parameter per topology floats that background in every year, and the
#     '*resolved*'/'*boosted*' wildcard bins deliberately span all years.
#
# The standard per-year cards (cards_cr_<year>) CANNOT be combined for this:
# they are year-tagged on every nuisance (the per-year default is full
# decorrelation), which would decorrelate everything. So step 1 regenerates
# lightweight per-year INPUT cards from the existing histogram files with
# make_datacards.py --decorrelate-years none -- exactly the scheme above --
# without touching the per-year trees or rerunning any coffea step.
#
# Steps:
#   1. per-year input cards (uproot only, fast)      -> cards_run2_input_<y>/
#   2. combineCards.py Y<year>=... per category      -> cards_run2/<cat>/
#   3. mechanical correlation-scheme check (aborts on violation)
#   4. fits per category, --jobs at a time:
#        text2workspace + Asimov Significance (+ FitDiagnostics unless
#        --sig-only), with the strategy-0 + robustHesse + robustFit method
#        and the zero-error-band guard from run_all.sh
#   5. significance summary                          -> plots_run2/
#
# Usage (lxplus, inside tmux):
#   ./run_run2_fits.sh                        # all 4 year-eras, all categories
#   ./run_run2_fits.sh --years="2018 2017"    # subset (>= 2 years)
#   ./run_run2_fits.sh --jobs=2 --sig-only    # significances only, no FitDiag
#   ./run_run2_fits.sh --cards-only           # stop after step 3
set -eo pipefail

VBSFIT_DIR="/eos/user/v/vslokenb/vbs_semilep/sensitivities/VBSfit"
MASS=125

YEARS="2018 2017 2016_PreVFP 2016_PostVFP"
CATEGORIES="resolved_mu resolved_e boosted_mu boosted_e combined"
JOBS=2
SIG_ONLY=0
CARDS_ONLY=0
# --allow-lumi-fallback: build cards even for years missing a
# LUMI_UNC_BY_YEAR entry, using the 2018 placeholder value. The resulting
# lumi_<year> constraints are WRONG for those years -- structure checks and
# timing studies only, never quotable numbers.
LUMI_FLAG="--require-lumi"

# Fit method settings, matching run_all.sh (see its history comments: plain
# HESSE cannot produce an acceptable covariance on these cards; judge fits
# by values and errors, not status words).
FIT_STRATEGY=0
FD_SHAPES_NTOYS=200
FD_RRANGE="--rMin -2 --rMax 2"
SIG_MINIMIZER="--cminDefaultMinimizerStrategy 2 --cminFallbackAlgo Minuit2,Migrad,2:0.1 --cminFallbackAlgo Minuit2,Migrad,1:0.1"

for arg in "$@"; do
    case "${arg}" in
        --years=*)      YEARS="${arg#*=}" ;;
        --categories=*) CATEGORIES="${arg#*=}" ;;
        --jobs=*)       JOBS="${arg#*=}" ;;
        --sig-only)     SIG_ONLY=1 ;;
        --allow-lumi-fallback) LUMI_FLAG="" ;;
        --cards-only)   CARDS_ONLY=1 ;;
        *) echo "Unknown option: ${arg}"; exit 1 ;;
    esac
done

# --- CMSSW setup (for combine) ---
source /cvmfs/cms.cern.ch/cmsset_default.sh 2>/dev/null || true
cd "${VBSFIT_DIR}/CMSSW_14_1_0_pre4/src"
eval $(scramv1 runtime -sh)
cd "${VBSFIT_DIR}"

# --- Apptainer environment (for the uproot card-making step) ---
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

# The card maker must be the updated one: an older copy without
# --require-lumi makes argparse exit 2 inside the tee pipe and, under
# set -eo pipefail, the whole script dies right after the first card log
# with the real error buried at its end.
if ! grep -q -- "--require-lumi" python/make_datacards.py; then
    echo "ERROR: python/make_datacards.py has no --require-lumi support --"
    echo "       install the updated make_datacards.py first."
    exit 1
fi

OUT_CARDS="${VBSFIT_DIR}/cards_run2"
OUT_PLOTS="${VBSFIT_DIR}/plots_run2"
LOG_DIR="${VBSFIT_DIR}/logs_run2"
mkdir -p "${OUT_CARDS}" "${OUT_PLOTS}" "${LOG_DIR}"

# ROOT cross-protocol redirect guard (see run_all.sh): a file written via the
# EOS FUSE mount can be opened by combine through xrootd before the flush has
# committed. Wait until the xrootd view matches the local file.
wait_eos_xrootd_sync() {
    local f="$1"
    if [ ! -f "${f}" ]; then
        echo "ERROR: wait_eos_xrootd_sync: missing file ${f}"
        exit 1
    fi
    local url host rpath size_local size_remote
    url=$(getfattr --only-values -n eos.url.xroot "${f}" 2>/dev/null) || url=""
    if [ -z "${url}" ] || ! command -v xrdfs >/dev/null 2>&1; then
        return 0
    fi
    host="${url#root://}"; host="${host%%/*}"
    rpath="${url#root://${host}/}"
    size_local=$(stat -c %s "${f}")
    local errf; errf=$(mktemp)
    local seen_remote=0 i
    for i in $(seq 1 60); do
        # '|| true': under set -eo pipefail a failing xrdfs would otherwise
        # kill the whole script silently through this command substitution,
        # before the warn-and-skip logic below ever runs.
        size_remote=$(xrdfs "${host}" stat "${rpath}" 2>"${errf}" \
                      | awk '/Size:/ {print $2; exit}' || true)
        if [ -n "${size_remote}" ]; then
            seen_remote=1
            if [ "${size_remote}" = "${size_local}" ]; then
                rm -f "${errf}"; return 0
            fi
        elif [ "${i}" -ge 3 ] && [ "${seen_remote}" = "0" ]; then
            echo "  [warn] cannot stat ${url} via xrdfs; skipping flush check"
            rm -f "${errf}"; return 0
        fi
        sleep 2
    done
    rm -f "${errf}"
    echo "ERROR: xrootd view of ${f} stuck at ${size_remote:-?} bytes" \
         "(local ${size_local})"
    exit 1
}

run_combine_logged() {
    local logf="$1"; shift
    if ! stdbuf -oL -eL combine "$@" > "${logf}" 2>&1; then
        echo "ERROR: combine failed -- last 20 lines of ${logf}:"
        tail -20 "${logf}"
        return 1
    fi
    tail -3 "${logf}"
}

check_band_saved() {
    local logf="$1"
    if grep -q "saveWithUncertainties will be ignored" "${logf}"; then
        echo "ERROR: combine ignored --saveWithUncertainties in ${logf}"
        grep -m1 "saveWithUncertainties will be ignored" "${logf}"
        return 1
    fi
}

# Abort at card-building time if any shapes file a card references does not
# exist (absolute paths checked as-is; relative ones against the card's dir).
check_shapes_resolve() {
    local card="$1" dir f missing=0
    dir=$(dirname "${card}")
    while read -r f; do
        case "${f}" in
            /*) [ -f "${f}" ] || { echo "ERROR: shapes file not found: ${f} (referenced by ${card})"; missing=1; } ;;
            *)  [ -f "${dir}/${f}" ] || { echo "ERROR: shapes file not found: ${dir}/${f} (referenced by ${card})"; missing=1; } ;;
        esac
    done < <(awk '$1=="shapes" {for(i=2;i<=NF;i++) if ($i ~ /\.root$/) {print $i; break}}' "${card}" | sort -u)
    [ "${missing}" = "0" ] || exit 1
}

echo "========================================"
echo "Run-2 cross-year combination"
echo "  Years:      ${YEARS}"
echo "  Categories: ${CATEGORIES}"
echo "  Fits:       Significance$( [ "${SIG_ONLY}" = "1" ] || echo ' + FitDiagnostics' )  (${JOBS} parallel)"
echo "  Cards:      ${OUT_CARDS}"
echo "  Output:     ${OUT_PLOTS}"
echo "========================================"

# -------------------------------------------------------------------
# Step 1: per-year input cards with the cross-year naming scheme
# -------------------------------------------------------------------
AVAILABLE_YEARS=""
for y in ${YEARS}; do
    hist="histograms/histograms_cr_${y}.root"
    cfg="config/inputs_v28_${y}.yaml"
    if [ ! -f "${hist}" ] || [ ! -f "${cfg}" ]; then
        echo "  [skip] ${y}: missing ${hist} or ${cfg} (run ./run_all.sh --year=${y} first)"
        continue
    fi
    echo ""
    echo "--- Step 1: input cards for ${y} (--decorrelate-years none) ---"
    rm -rf "${VBSFIT_DIR}/cards_run2_input_${y}"   # never combine stale cards
    if ! pyrun python3 python/make_datacards.py \
            --histfile "${hist}" \
            --config "${cfg}" \
            --year "${y}" \
            --outdir "cards_run2_input_${y}" \
            --decorrelate-years none \
            ${LUMI_FLAG} \
            2>&1 | tee "${LOG_DIR}/make_cards_${y}.log"; then
        echo ""
        echo "ERROR: card building FAILED for ${y} -- last 5 lines of" \
             "${LOG_DIR}/make_cards_${y}.log:"
        tail -5 "${LOG_DIR}/make_cards_${y}.log" | sed 's/^/    /'
        echo "If it says 'no luminosity uncertainty set', fill in this"
        echo "year's value in LUMI_UNC_BY_YEAR (python/make_datacards.py)."
        echo "To proceed anyway with the WRONG 2018 placeholder (structure"
        echo "tests only, numbers not quotable): --allow-lumi-fallback"
        exit 1
    fi
    if [ -f "cards_run2_input_${y}/histograms_decorrelated.root" ]; then
        wait_eos_xrootd_sync "${VBSFIT_DIR}/cards_run2_input_${y}/histograms_decorrelated.root"
    fi
    AVAILABLE_YEARS="${AVAILABLE_YEARS} ${y}"
done
AVAILABLE_YEARS="${AVAILABLE_YEARS# }"
N_YEARS=$(echo "${AVAILABLE_YEARS}" | wc -w)
if [ "${N_YEARS}" -lt 2 ]; then
    echo "ERROR: need input cards for at least 2 years, have: '${AVAILABLE_YEARS}'"
    exit 1
fi
echo ""
echo "Combining year-eras: ${AVAILABLE_YEARS}"

# -------------------------------------------------------------------
# Step 2: cross-year card per category. The Y<year>= labels make
# combineCards prefix every bin with its year, which the correlation
# checker and the per-year channel bookkeeping both rely on. Nuisances
# correlate purely by name: the scheme set in step 1 carries through.
# -------------------------------------------------------------------
for cat in ${CATEGORIES}; do
    inputs=()
    for y in ${AVAILABLE_YEARS}; do
        # ABSOLUTE input paths: combineCards rewrites each shapes line by
        # prefixing the input card's directory exactly as given here, so
        # relative inputs would produce a combined card whose shapes paths
        # only resolve from this cwd -- and step 4 cd's into the card dir.
        card="${VBSFIT_DIR}/cards_run2_input_${y}/${cat}/datacard.txt"
        if [ ! -f "${card}" ]; then
            echo "ERROR: missing ${card}"
            exit 1
        fi
        inputs+=("Y${y}=${card}")
    done
    mkdir -p "${OUT_CARDS}/${cat}"
    echo "--- Step 2: ${cat}: combineCards over ${N_YEARS} year-eras ---"
    combineCards.py "${inputs[@]}" > "${OUT_CARDS}/${cat}/datacard.txt"
    check_shapes_resolve "${OUT_CARDS}/${cat}/datacard.txt"
done

# -------------------------------------------------------------------
# Step 3: verify the correlation scheme mechanically; abort on violation
# -------------------------------------------------------------------
echo ""
for cat in ${CATEGORIES}; do
    python3 python/check_run2_correlations.py \
        --card "${OUT_CARDS}/${cat}/datacard.txt" \
        --years "${AVAILABLE_YEARS}"
done

if [ "${CARDS_ONLY}" = "1" ]; then
    echo ""
    echo "Cards built and checked (--cards-only). Fits not run."
    exit 0
fi

# -------------------------------------------------------------------
# Step 4: fits per category, JOBS at a time
# -------------------------------------------------------------------
fit_one_category() {
    local cat="$1"
    local dir="${OUT_CARDS}/${cat}"
    cd "${dir}"

    text2workspace.py datacard.txt -o workspace.root -m ${MASS} \
        > t2w.log 2>&1 || {
        echo "ERROR: text2workspace failed for ${cat} -- tail of t2w.log:"
        tail -10 t2w.log
        return 1
    }
    wait_eos_xrootd_sync "${dir}/workspace.root"

    echo "--- Expected significance (MC Asimov, Run 2): ${cat} ---"
    run_combine_logged sig.log -M Significance workspace.root \
        -m ${MASS} -t -1 --expectSignal 1 ${SIG_MINIMIZER} \
        -n "_run2_${cat}" || return 1
    grep -h "Significance:" sig.log \
        | sed "s/^/${cat}: /" >> "${OUT_PLOTS}/significances_run2_m${MASS}.txt"

    if [ "${SIG_ONLY}" != "1" ]; then
        echo "--- FitDiagnostics (MC Asimov, Run 2): ${cat} ---"
        run_combine_logged fitdiag.log -M FitDiagnostics workspace.root \
            -m ${MASS} -t -1 --expectSignal 1 \
            --saveShapes --saveWithUncertainties \
            --numToysForShapes ${FD_SHAPES_NTOYS} \
            --cminDefaultMinimizerStrategy ${FIT_STRATEGY} \
            --robustHesse 1 --robustFit 1 ${FD_RRANGE} \
            -n "_run2_${cat}" || return 1
        check_band_saved fitdiag.log || return 1
        wait_eos_xrootd_sync "${dir}/fitDiagnostics_run2_${cat}.root"
        # The correlated norms are the headline numbers of this combination:
        # one parameter per topology across all years.
        python3 - "fitDiagnostics_run2_${cat}.root" <<'PYEOF' || true
import sys
import ROOT
ROOT.gROOT.SetBatch(True)
f = ROOT.TFile.Open(sys.argv[1])
fr = f.Get('fit_s')
if fr:
    pars = fr.floatParsFinal()
    rows = []
    for i in range(pars.getSize()):
        p = pars.at(i)
        if p.GetName().startswith(('norm_top', 'norm_wjet')) or p.GetName() == 'r':
            rows.append(f"  {p.GetName():<24} = {p.getVal():+.4f} +/- {p.getError():.4f}")
    print('Run-2 correlated normalizations (fit_s):')
    for r in sorted(rows):
        print(r)
PYEOF
    fi
    cd "${VBSFIT_DIR}"
}

echo ""
: > "${OUT_PLOTS}/significances_run2_m${MASS}.txt"
# Success is judged by per-category .ok marker files, NOT by per-pid wait:
# the throttle's 'wait -n' reaps whichever job finishes first, after which
# 'wait $pid' on that job errors ("not a child") and a SUCCESSFUL category
# would be reported as failed.
for cat in ${CATEGORIES}; do
    while [ "$(jobs -rp | wc -l)" -ge "${JOBS}" ]; do
        wait -n || true
    done
    echo "  launching fits: ${cat}  (log: ${LOG_DIR}/fit_${cat}.log)"
    rm -f "${LOG_DIR}/fit_${cat}.ok"
    { fit_one_category "${cat}" > "${LOG_DIR}/fit_${cat}.log" 2>&1 \
        && touch "${LOG_DIR}/fit_${cat}.ok"; } &
done
wait || true

FAILED=""
for cat in ${CATEGORIES}; do
    if [ ! -f "${LOG_DIR}/fit_${cat}.ok" ]; then
        FAILED="${FAILED} ${cat}"
        echo "  [FAIL] ${cat} -- last 15 lines of its log:"
        tail -15 "${LOG_DIR}/fit_${cat}.log" | sed 's/^/    /'
    else
        echo "  [done] ${cat}"
    fi
done

# -------------------------------------------------------------------
# Step 5: summary
# -------------------------------------------------------------------
echo ""
echo "==================================================================="
echo "Run-2 combination summary (years: ${AVAILABLE_YEARS})"
echo "==================================================================="
echo "Expected significances (MC Asimov, -t -1 --expectSignal 1):"
sed 's/^/  /' "${OUT_PLOTS}/significances_run2_m${MASS}.txt" 2>/dev/null \
    || echo "  (none recorded)"
echo ""
echo "Cards:            ${OUT_CARDS}/<category>/datacard.txt"
echo "Fit logs/results: ${OUT_CARDS}/<category>/  and  ${LOG_DIR}/"
if [ "${SIG_ONLY}" != "1" ]; then
    echo "Correlated norms: grep 'norm_' ${LOG_DIR}/fit_<category>.log"
fi

if [ -n "${FAILED}" ]; then
    echo ""
    echo "FAILED categories:${FAILED}"
    exit 1
fi
