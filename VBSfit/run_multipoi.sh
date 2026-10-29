#!/bin/bash
# Multi-POI sensitivity for the VBS semileptonic fit.
#
# Produces three sensitivity numbers from the SAME datacards run_all.sh built:
#
#   combined   one signal strength r scaling every signal process (this is the
#              number run_all.sh already reports; recomputed here so the three
#              live in one table)
#   r_WW       WW signal strength, from a multiSignalModel workspace
#   r_WZ       WZ signal strength, from the same workspace
#
# and for each POI a significance, a 1D likelihood scan, MultiDimFit --algo
# singles intervals, impacts, plus a 2D r_WW vs r_WZ scan.
#
# WHAT "SENSITIVITY TO WW" MEANS
# ------------------------------
# With two POIs the answer depends on what the other signal is doing, so both
# variants are computed (OTHER_POI_MODE below):
#
#   profiled  the other signal strength floats freely. This is the honest
#             "measure WW without assuming WZ" number, and is the larger
#             uncertainty / smaller significance of the two.
#   fixed     the other signal strength is frozen at the SM value 1. This is
#             "measure WW given the SM WZ rate" and is more optimistic.
#
# Quote 'profiled' unless you specifically mean the other process is known.
#
# CARD SETS
# ---------
# Runs over whichever of these exist, keeping their results separate:
#   mconly   cards_cr/            Asimov everywhere, MC normalizations
#   crdata   cards_crdata_srasimov/  real CR data + blind SR pseudo-data
# The signal region stays blind in both: the crdata cards carry MC pseudo-data
# in the SR by construction (see make_hybrid_data.py), and mconly is pure
# Asimov. No dataset here contains signal-region observations.
#
# Usage:
#   ./run_multipoi.sh                      # both card sets, all targets
#   ./run_multipoi.sh --mconly-only        # skip the CR-data cards
#   ./run_multipoi.sh --crdata-only        # skip the MC-only cards
#   ./run_multipoi.sh --targets "combined" # one category (much faster)
#   ./run_multipoi.sh --zz own             # give ZZ its own POI
#   ./run_multipoi.sh --no-impacts         # skip the slowest step
#   ./run_multipoi.sh --no-scan            # skip the 1D and 2D scans
#
# Intended to be run inside a tmux session. Copy this file to a new name before
# editing it while a run is in progress -- bash reads a script incrementally and
# overwriting it mid-run makes it resume at a stale byte offset.
set -e

VBSFIT_DIR="/eos/user/v/vslokenb/vbs_semilep/sensitivities/VBSfit"
MASS=125

# Which POI the ZZ signal processes scale with: wz | ww | own.
#
# The signal list has 4 WW processes, 4 WZ, and one ZZ (ZTo2LZTo2JJJ). "Two
# signals" leaves the ZZ process needing a home; by default it is folded into
# r_WZ, so that POI means "WZ + ZZ". Use 'own' to give it a third POI which is
# then profiled when measuring r_WW and r_WZ.
ZZ_ASSIGN=wz

# profiled | fixed | both -- see the header.
OTHER_POI_MODE=both

TARGETS="combined resolved boosted resolved_mu resolved_e boosted_mu boosted_e"

# 2D scan grid points, and the scan window for each POI.
SCAN2D_POINTS=2500
SCAN1D_POINTS=60
POI_SCAN_RANGE="-1,3"

RUN_MCONLY=1
RUN_CRDATA=1
RUN_IMPACTS=1
RUN_SCAN=1

while [ $# -gt 0 ]; do
    case "$1" in
        --mconly-only) RUN_CRDATA=0 ;;
        --crdata-only) RUN_MCONLY=0 ;;
        --no-impacts)  RUN_IMPACTS=0 ;;
        --no-scan)     RUN_SCAN=0 ;;
        --targets)     TARGETS="$2"; shift ;;
        --zz)          ZZ_ASSIGN="$2"; shift ;;
        --other-poi)   OTHER_POI_MODE="$2"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

# --- CMSSW setup (for combine) ---
source /cvmfs/cms.cern.ch/cmsset_default.sh 2>/dev/null || true
cd "${VBSFIT_DIR}/CMSSW_14_1_0_pre4/src"
eval $(scramv1 runtime -sh)
cd "${VBSFIT_DIR}"

OUTDIR="${VBSFIT_DIR}/plots_multipoi"
MULTIPOI_ARGS="${VBSFIT_DIR}/python/make_multipoi_args.py"
READ_SINGLES="${VBSFIT_DIR}/python/read_multipoi_singles.py"

declare -A CARDROOT
CARDROOT[mconly]="${VBSFIT_DIR}/cards_cr"
CARDROOT[crdata]="${VBSFIT_DIR}/cards_crdata_srasimov"

declare -A SETLABEL
SETLABEL[mconly]="MC-only Asimov"
SETLABEL[crdata]="CR data + SR Asimov"

mkdir -p "${OUTDIR}"

SETS=""
[ "${RUN_MCONLY}" = "1" ] && SETS="${SETS} mconly"
[ "${RUN_CRDATA}" = "1" ] && SETS="${SETS} crdata"

echo "========================================"
echo "VBSfit multi-POI sensitivity"
echo "  Card sets:   ${SETS}"
echo "  Targets:     ${TARGETS}"
echo "  ZZ assigned: ${ZZ_ASSIGN}"
echo "  Other POI:   ${OTHER_POI_MODE}"
echo "  Output:      ${OUTDIR}"
echo "========================================"

if [ ! -f "${MULTIPOI_ARGS}" ]; then
    echo "ERROR: ${MULTIPOI_ARGS} not found."
    exit 1
fi


# ---------------------------------------------------------------------------
# Dataset options.
#
# mconly: nothing in the card is data, so generate the Asimov here. With a
#   multiSignalModel workspace there is no parameter called 'r', so
#   --expectSignal does not apply -- every POI is set to 1 explicitly instead.
# crdata: the card already holds real CR data and blind SR pseudo-data built at
#   r=1 for every signal process, which is exactly r_WW=r_WZ=1. Adding -t -1
#   here would regenerate Asimov for the control regions and throw that away.
# ---------------------------------------------------------------------------
dataset_opts() {
    local set="$1" pois="$2"
    if [ "${set}" = "mconly" ]; then
        echo "-t -1 --setParameters $(echo "${pois}" | sed 's/,/=1,/g')=1"
    else
        echo ""
    fi
}

# Single-POI equivalent for the combined number.
dataset_opts_single() {
    if [ "$1" = "mconly" ]; then echo "-t -1 --expectSignal 1"; else echo ""; fi
}


# Build the multiSignalModel workspace and verify the POIs really exist.
#
# If a map pattern matched nothing, multiSignalModel gives that process a
# constant scale of 1: it stops being signal, no POI is created, and the fit
# still runs and reports a plausible-looking number for whatever POIs remain.
# make_multipoi_args.py enumerates the processes from the card so every one is
# mapped, and this check confirms the workspace agrees.
build_multipoi_ws() {
    local card="$1" ws="$2" pois="$3"
    if [ -f "${ws}" ]; then
        return 0
    fi
    local po=()
    mapfile -t po < <(python3 "${MULTIPOI_ARGS}" --datacard "${card}" --zz "${ZZ_ASSIGN}")
    if [ "${#po[@]}" -eq 0 ]; then
        echo "  [skip] no signal map produced for ${card}"
        return 1
    fi
    text2workspace.py "${card}" \
        -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel \
        "${po[@]}" \
        -o "${ws}" -m ${MASS} 2>&1 | tail -3
    if [ ! -f "${ws}" ]; then
        echo "  ERROR: text2workspace produced no workspace for ${card}"
        echo "         Rerun without '| tail -3' to see the failure."
        return 1
    fi
    python3 - "${ws}" "${pois}" <<'PYEOF'
import sys
import ROOT
ROOT.gErrorIgnoreLevel = ROOT.kError
ws_path, pois = sys.argv[1], sys.argv[2].split(",")
f = ROOT.TFile.Open(ws_path)
w = f.Get("w")
if not w:
    print("  ERROR: no RooWorkspace 'w' in %s" % ws_path)
    sys.exit(1)
missing = [p for p in pois if not w.var(p)]
if missing:
    have = sorted(v.GetName() for v in w.allVars() if v.GetName().startswith("r"))
    print("  ERROR: POIs not in workspace: %s" % ", ".join(missing))
    print("         workspace has: %s" % ", ".join(have))
    print("         A multiSignalModel map matched no process, so that signal")
    print("         is now a fixed component instead of a measured one.")
    sys.exit(1)
print("  POIs in workspace: %s" % ", ".join(pois))
PYEOF
}


# significance <label> <workspace> <poi> <other pois> <mode> <dataset opts> <outdir>
run_poi_significance() {
    local label="$1" ws="$2" poi="$3" others="$4" mode="$5" dopts="$6" out="$7"
    local freeze=""
    if [ "${mode}" = "fixed" ] && [ -n "${others}" ]; then
        freeze="--freezeParameters ${others}"
    fi
    local res
    res=$(cd "${out}" && combine -M Significance \
        -d "${ws}" -m ${MASS} \
        --redefineSignalPOIs "${poi}" \
        ${dopts} ${freeze} \
        --cminDefaultMinimizerStrategy 0 \
        --name "_${label}_${poi}_${mode}" \
        2>&1 | grep -E "Significance:|ERROR" | head -3)
    echo "${res}"
}


process_target() {
    local set="$1" target="$2"
    local card="${CARDROOT[$set]}/${target}/datacard.txt"
    local out="${OUTDIR}/${set}/${target}"

    if [ ! -f "${card}" ]; then
        echo "  [skip] ${set}/${target}: no datacard at ${card}"
        return 0
    fi
    mkdir -p "${out}"

    echo ""
    echo "========================================"
    echo "${SETLABEL[$set]} -- ${target}"
    echo "========================================"

    local pois
    pois=$(python3 "${MULTIPOI_ARGS}" --datacard "${card}" --zz "${ZZ_ASSIGN}" --list-pois)
    echo "  POIs: ${pois}"
    python3 "${MULTIPOI_ARGS}" --datacard "${card}" --zz "${ZZ_ASSIGN}" --show \
        > "${out}/signal_grouping.txt"

    local ws="${out}/workspace_multipoi.root"
    build_multipoi_ws "${card}" "${ws}" "${pois}" || return 0

    local dopts dopts1
    dopts=$(dataset_opts "${set}" "${pois}")
    dopts1=$(dataset_opts_single "${set}")

    local sumfile="${out}/summary.txt"
    {
        echo "Multi-POI sensitivity -- ${SETLABEL[$set]}, ${target}"
        echo "  datacard : ${card}"
        echo "  dataset  : ${dopts:-the one stored in the datacard}"
        echo "  POIs     : ${pois}   (ZZ -> ${ZZ_ASSIGN})"
        echo "  Generated: $(date)"
        echo "============================================================"
        echo ""
        echo "SIGNIFICANCE"
    } > "${sumfile}"

    # --- combined: the ordinary single-r workspace -------------------------
    echo ""
    echo "--- significance: combined (single r) ---"
    local ws1="${out}/workspace_single.root"
    if [ ! -f "${ws1}" ]; then
        text2workspace.py "${card}" -o "${ws1}" -m ${MASS} 2>&1 | tail -2
    fi
    local res
    res=$(cd "${out}" && combine -M Significance -d "${ws1}" -m ${MASS} \
        ${dopts1} --cminDefaultMinimizerStrategy 0 \
        --name "_combined" 2>&1 | grep -E "Significance:|ERROR" | head -3)
    echo "  ${res}"
    printf '  %-28s %s\n' "combined (single r)" "${res}" >> "${sumfile}"

    # --- per-POI ------------------------------------------------------------
    local modes="${OTHER_POI_MODE}"
    [ "${OTHER_POI_MODE}" = "both" ] && modes="profiled fixed"

    local poi others
    for poi in $(echo "${pois}" | tr ',' ' '); do
        others=$(echo "${pois}" | tr ',' '\n' | grep -v "^${poi}$" | paste -sd, -)
        for mode in ${modes}; do
            echo ""
            echo "--- significance: ${poi} (${mode}: others=${others:-none}) ---"
            res=$(run_poi_significance "${target}" "${ws}" "${poi}" \
                                       "${others}" "${mode}" "${dopts}" "${out}")
            echo "  ${res}"
            printf '  %-28s %s\n' "${poi} (${mode})" "${res}" >> "${sumfile}"
        done
    done

    # --- intervals on every POI at once ------------------------------------
    echo ""
    echo "--- MultiDimFit --algo singles (all POIs) ---"
    (cd "${out}" && combine -M MultiDimFit \
        -d "${ws}" -m ${MASS} \
        --redefineSignalPOIs "${pois}" \
        ${dopts} --algo singles --robustFit 1 \
        --cminDefaultMinimizerStrategy 0 \
        --name "_singles" 2>&1 | tail -4)

    {
        echo ""
        echo "68% INTERVALS (MultiDimFit --algo singles, other POIs profiled)"
    } >> "${sumfile}"
    python3 "${READ_SINGLES}" \
        "${out}/higgsCombine_singles.MultiDimFit.mH${MASS}.root" "${pois}" \
        | tee -a "${sumfile}" || true

    # --- likelihood scans ---------------------------------------------------
    if [ "${RUN_SCAN}" = "1" ]; then
        for poi in $(echo "${pois}" | tr ',' ' '); do
            echo ""
            echo "--- 1D scan: ${poi} ---"
            (cd "${out}" && combine -M MultiDimFit \
                -d "${ws}" -m ${MASS} \
                --redefineSignalPOIs "${pois}" \
                -P "${poi}" --floatOtherPOIs 1 \
                ${dopts} --algo grid --points ${SCAN1D_POINTS} \
                --setParameterRanges "${poi}=${POI_SCAN_RANGE}" \
                --cminDefaultMinimizerStrategy 0 \
                --name "_scan_${poi}" 2>&1 | tail -3) || true
        done

        # 2D contour input. Only for the multi-POI case; a single POI has none.
        if [ "$(echo "${pois}" | tr ',' ' ' | wc -w)" -ge 2 ]; then
            local p1 p2 ranges
            p1=$(echo "${pois}" | cut -d, -f1)
            p2=$(echo "${pois}" | cut -d, -f2)
            ranges="${p1}=${POI_SCAN_RANGE}:${p2}=${POI_SCAN_RANGE}"
            echo ""
            echo "--- 2D scan: ${p1} vs ${p2} ---"
            (cd "${out}" && combine -M MultiDimFit \
                -d "${ws}" -m ${MASS} \
                --redefineSignalPOIs "${pois}" \
                -P "${p1}" -P "${p2}" --floatOtherPOIs 1 \
                ${dopts} --algo grid --points ${SCAN2D_POINTS} \
                --setParameterRanges "${ranges}" \
                --cminDefaultMinimizerStrategy 0 \
                --name "_scan2d" 2>&1 | tail -3) || true
        fi
    fi

    # --- impacts per POI -----------------------------------------------------
    if [ "${RUN_IMPACTS}" = "1" ]; then
        for poi in $(echo "${pois}" | tr ',' ' '); do
            echo ""
            echo "--- impacts: ${poi} ---"
            local idir="${out}/impacts_${poi}"
            mkdir -p "${idir}"
            (
                cd "${idir}"
                combineTool.py -M Impacts -d "${ws}" -m ${MASS} \
                    --redefineSignalPOIs "${pois}" \
                    ${dopts} --robustFit 1 --doInitialFit 2>&1 | tail -2
                combineTool.py -M Impacts -d "${ws}" -m ${MASS} \
                    --redefineSignalPOIs "${pois}" \
                    ${dopts} --robustFit 1 --doFits --parallel 20 2>&1 | tail -2
                combineTool.py -M Impacts -d "${ws}" -m ${MASS} \
                    --redefineSignalPOIs "${pois}" \
                    ${dopts} -o "impacts_${poi}.json" 2>&1 | tail -2
                plotImpacts.py -i "impacts_${poi}.json" -o "impacts_${poi}" \
                    --POI "${poi}" 2>&1 | tail -2
            ) || echo "  [warn] impacts for ${poi} did not complete"
        done
    fi

    echo ""
    echo "  Summary: ${sumfile}"
}


for set in ${SETS}; do
    if [ ! -d "${CARDROOT[$set]}" ]; then
        echo ""
        echo "[skip] ${set}: ${CARDROOT[$set]} does not exist"
        continue
    fi
    for target in ${TARGETS}; do
        process_target "${set}" "${target}"
    done
done


# ---------------------------------------------------------------------------
# One table across everything that ran
# ---------------------------------------------------------------------------
ALLSUM="${OUTDIR}/summary_all.txt"
{
    echo "Multi-POI sensitivity, all card sets and categories"
    echo "  ZZ processes assigned to: ${ZZ_ASSIGN}"
    echo "  'profiled' = other signal strength floats (quote this one)"
    echo "  'fixed'    = other signal strength frozen at the SM value 1"
    echo "Generated: $(date)"
    echo "============================================================"
} > "${ALLSUM}"

for set in ${SETS}; do
    for target in ${TARGETS}; do
        f="${OUTDIR}/${set}/${target}/summary.txt"
        [ -f "${f}" ] || continue
        {
            echo ""
            echo "--- ${SETLABEL[$set]} / ${target} ---"
            sed -n '/^SIGNIFICANCE/,$p' "${f}"
        } >> "${ALLSUM}"
    done
done

cat > "${OUTDIR}/README.txt" <<EOF
MULTI-POI SENSITIVITY

Three numbers per category, from the datacards run_all.sh built:
  combined   single signal strength r over every signal process
  r_WW       WW signal strength   (multiSignalModel)
  r_WZ       WZ signal strength   (ZZ processes assigned to: ${ZZ_ASSIGN})

Each POI is reported two ways:
  profiled   the other signal strength floats -- the honest number
  fixed      the other signal strength frozen at 1 -- assumes the SM rate

Card sets, kept separate:
  ${OUTDIR}/mconly/   Asimov everywhere, MC normalizations
  ${OUTDIR}/crdata/   real control-region data, blind SR pseudo-data

The signal region is blind in both. Per category:
  signal_grouping.txt  which signal process feeds which POI
  summary.txt          significances and 68% intervals
  higgsCombine_scan_<poi>.MultiDimFit.*  1D scan, plot with plot1DScan.py
  higgsCombine_scan2d.MultiDimFit.*      2D r_WW vs r_WZ grid
  impacts_<poi>/                         per-POI impacts

Generated: $(date)
EOF

echo ""
echo "========================================"
echo "Multi-POI run complete"
echo "  Combined table: ${ALLSUM}"
echo "  Per category:   ${OUTDIR}/<set>/<target>/summary.txt"
echo "  What is what:   ${OUTDIR}/README.txt"
echo "========================================"