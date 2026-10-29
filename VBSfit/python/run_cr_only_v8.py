#!/usr/bin/env python3
"""
CR-only fits against v8 CR data, using combine's native channel-masking
(--channel-masks) to cleanly exclude the SR bin(s) from the likelihood,
rather than hand-editing the datacard text (the old v7 cr_only_per_sr/
approach, which left r floating unconstrained against a masked SR and hit
a negative-eigenvalue drop). r is frozen (irrelevant once SR is masked).

7 fits, all on real data (no -t -1):
  percat_{boosted_mu,boosted_e,resolved_mu,resolved_e}: 4 independent
      single-region CR-only fits -> norm_top/norm_wjet per region
      (feeds the _offset_cat_v8 per-category offset construction)
  topo_{boosted,resolved}: boosted_mu+boosted_e combined / resolved_mu+
      resolved_e combined, shared norm_top/norm_wjet per topology
      (feeds the _offset_topo_v8 offset construction)
  shared: all 4 regions combined, single global norm_top/norm_wjet
      (feeds the _offset_v8 offset construction)

Run with CMSSW python3 (requires PyROOT, and combine/text2workspace.py/
combineCards.py/condor_submit on PATH):
  source /cvmfs/cms.cern.ch/cmsset_default.sh
  cd /afs/cern.ch/work/j/jinw/public/VBSfit/CMSSW_14_1_0_pre4/src
  eval $(scramv1 runtime -sh) && cd /afs/cern.ch/work/j/jinw/public/VBSfit

  python3 python/run_cr_only_v8.py            # Phase 1: submit
  python3 python/run_cr_only_v8.py --collect  # Phase 2: collect
"""
import argparse
import os
import subprocess

VBSFIT = "/eos/user/v/vslokenb/vbs_semilep/sensitivities/VBSfit"
MASS = 125
CARDS_IN = f"{VBSFIT}/cards/"
OUTBASE = f"{VBSFIT}/cards/cr_only"
CMSSW_DIR = f"{VBSFIT}/CMSSW_14_1_0_pre4/src"
CONDOR_FLAVOUR = "tomorrow"

# case_name -> (datacard path, [mask_<bin> names to set =1 / freeze])
CASES = {
    "percat_boosted_mu":  (f"{CARDS_IN}/boosted_mu/datacard.txt",  ["mask_boosted_mu"]),
    "percat_boosted_e":   (f"{CARDS_IN}/boosted_e/datacard.txt",   ["mask_boosted_e"]),
    "percat_resolved_mu": (f"{CARDS_IN}/resolved_mu/datacard.txt", ["mask_resolved_mu"]),
    "percat_resolved_e":  (f"{CARDS_IN}/resolved_e/datacard.txt",  ["mask_resolved_e"]),
    "topo_boosted":  (f"{OUTBASE}/topo_boosted/datacard.txt",
                      ["mask_ch1_boosted_mu", "mask_ch2_boosted_e"]),
    "topo_resolved": (f"{OUTBASE}/topo_resolved/datacard.txt",
                      ["mask_ch1_resolved_mu", "mask_ch2_resolved_e"]),
    "shared": (f"{CARDS_IN}/combined/datacard.txt",
              ["mask_ch1_boosted_mu", "mask_ch2_boosted_e",
               "mask_ch3_resolved_mu", "mask_ch4_resolved_e"]),
}

# case_name -> per-region datacards to combineCards.py together (only for
# the topo_* cases -- percat_* and shared already have their datacard built
# by the main production chain).
COMBINE_CARDS_INPUTS = {
    "topo_boosted":  [f"{CARDS_IN}/boosted_mu/datacard.txt",  f"{CARDS_IN}/boosted_e/datacard.txt"],
    "topo_resolved": [f"{CARDS_IN}/resolved_mu/datacard.txt", f"{CARDS_IN}/resolved_e/datacard.txt"],
}


def outdir_for(case_name):
    return f"{OUTBASE}/{case_name}"


def build_merged_datacard(case_name):
    """combineCards.py the two per-region datacards for a topo_* case."""
    out_path = CASES[case_name][0]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        subprocess.run(["combineCards.py"] + COMBINE_CARDS_INPUTS[case_name],
                       stdout=fh, check=True)


def build_workspace(datacard, ws_path):
    if os.path.exists(ws_path):
        return
    subprocess.run(
        ["text2workspace.py", datacard, "--channel-masks",
         "-o", ws_path, "-m", str(MASS)],
        check=True,
    )

AFS_SUBMIT_DIR = f"/afs/cern.ch/user/v/vslokenb/vbs_semilep/condor_submit"

def submit_crfit(case_name):
    datacard, masks = CASES[case_name]
    outdir = outdir_for(case_name)
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(AFS_SUBMIT_DIR, exist_ok=True)
    ws_path = f"{outdir}/workspace_crmask.root"
    build_workspace(datacard, ws_path)

    set_masks = ",".join(f"{m}=1" for m in masks)
    freeze_masks = ",".join(["r"] + masks)

    wrapper = f"{AFS_SUBMIT_DIR}/run_{case_name}.sh"
    with open(wrapper, "w") as fh:
        fh.write(f"""#!/bin/bash
cd /eos/user/v/vslokenb/vbs_semilep/sensitivities/VBSfit
source /cvmfs/cms.cern.ch/cmsset_default.sh 2>/dev/null || true
cd "{CMSSW_DIR}" && eval $(scramv1 runtime -sh 2>/dev/null)
cd ../../
combine -M FitDiagnostics \\
    -d "{ws_path}" -m {MASS} \\
    --setParameters {set_masks} \\
    --freezeParameters {freeze_masks} \\
    --robustFit 1 --robustHesse 1 --cminDefaultMinimizerStrategy 0 \\
    --name "_cronly_{case_name}" \\
    > "{outdir}/combine_{case_name}.log" 2>&1
""")
    os.chmod(wrapper, 0o755)
    os.makedirs(AFS_SUBMIT_DIR, exist_ok=True)
    jdl = f"{AFS_SUBMIT_DIR}/condor_{case_name}.jdl"
    with open(jdl, "w") as fh:
        fh.write(f"""executable = {wrapper}
output     = {AFS_SUBMIT_DIR}/condor.stdout
error      = {AFS_SUBMIT_DIR}/condor.stderr
log        = {AFS_SUBMIT_DIR}/condor.log
request_memory = 16000
+JobFlavour = "{CONDOR_FLAVOUR}"
queue
""")
    subprocess.run(["condor_submit", jdl], check=True, cwd=AFS_SUBMIT_DIR)
    print(f"  submitted: {case_name}")


def collect(case_name):
    import ROOT
    outdir = outdir_for(case_name)
    f_path = f"{outdir}/fitDiagnostics_cronly_{case_name}.root"
    if not os.path.exists(f_path):
        print(f"  MISS  {case_name:<20} <- job may still be running")
        return
    f = ROOT.TFile.Open(f_path)
    fr = f.Get("fit_s")
    pars = fr.floatParsFinal()
    top = pars.find("norm_top")
    wjet = pars.find("norm_wjet")
    print(f"  {case_name:<20} norm_top = {top.getVal():.4f} "
          f"-{abs(top.getErrorLo()):.4f}/+{top.getErrorHi():.4f}   "
          f"norm_wjet = {wjet.getVal():.4f} "
          f"-{abs(wjet.getErrorLo()):.4f}/+{wjet.getErrorHi():.4f}   "
          f"(status={fr.status()}, covQual={fr.covQual()})")
    f.Close()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--collect", action="store_true",
                        help="Collect results instead of submitting jobs")
    args = parser.parse_args()

    if not args.collect:
        print("=== Submitting v8 CR-only fits ===")
        for case_name in COMBINE_CARDS_INPUTS:
            build_merged_datacard(case_name)
        for case_name in CASES:
            submit_crfit(case_name)
        print(f"\n=== {len(CASES)} jobs submitted ===")
        print("  Collect: python3 python/run_cr_only_v8.py --collect")
    else:
        print("=== Collecting v8 CR-only fit results ===")
        for case_name in CASES:
            collect(case_name)


if __name__ == "__main__":
    main()
