# Shipping manifest: config_validation.py / config_inverted.py

Every SF / correction / dataset file the two configs depend on, traced from the
config `parameters` lists, the params YAMLs they load, and the dataset JSON
lists. Paths are relative to this directory
(`/work/users/vsloken-fnal/semilep_vbs_base/Pocket/temporary_workspace`).
Generated 2026-10-29; all files verified to exist on disk unless marked MISSING.

## Param YAMLs (loaded directly by the configs)

Used by both configs:

- `params/object_preselection_run2_v9.yaml`
- `params/triggers.yaml`
- `params/plotting.yaml`
- `params/pileup.yaml`
- `params/jet_scale_factors.yaml`
- `params/lepton_scale_factors.yaml`
- `params/classifiers.yaml`
- `params/variations.yaml`
- `params/fakelepton_weights_noiso_3j.yaml`
- `params/qgtagging.yaml`
- `params/fj_taggers.yaml`

config_validation only:

- `params/jets_calibration.yaml` (commented out in config_inverted; references
  only cvmfs jsonpog files, nothing local to ship beyond the YAML itself)

config_inverted only:

- `params/dphi_weights.yaml`

## Dataset definition JSONs (both configs, identical lists)

- `datasets/rare_skim.json` (144 KB)
- `datasets/skimmed_rescale.json` (373 KB)
- `datasets/skimmed_dataset_definition_rare1617_fixed.json` (86 KB)
- `datasets/skimmed_dataset_definition_top1617.json` (44 KB)
- `datasets/TTToSemiLeptonic_2016_PreVFP_skim.json` (17 KB)

## SF / correction files referenced by the param YAMLs

### lepton_scale_factors.yaml — electron trigger SF + EGM scale uncertainty

- `ElectronTriggerSF_UL16preVFP_pocketcoffea.json.gz`
- `ElectronTriggerSF_UL16postVFP_pocketcoffea.json.gz`
- `ElectronTriggerSF_UL17_pocketcoffea.json.gz`
- `ElectronTriggerSF_UL18_pocketcoffea.json.gz`
- `EGM_ScaleUnc_2016preVFP.json.gz`
- `EGM_ScaleUnc_2016postVFP.json.gz`
- `EGM_ScaleUnc_2017.json.gz`
- `EGM_ScaleUnc_2018.json.gz`

(muon/electron ID+reco SFs and btag/ctag/JEC come from cvmfs jsonpog and
pocket-coffea defaults — nothing to ship.)

### qgtagging.yaml — quark/gluon shape SFs

- `QGSF-Run2/UL16_preVFP_shape_sf.json.gz`
- `QGSF-Run2/UL16_postVFP_shape_sf.json.gz`
- `QGSF-Run2/UL17_shape_sf.json.gz`
- `QGSF-Run2/UL18_shape_sf.json.gz`

### fj_taggers.yaml — fat-jet tau21 / WvsQCD SFs

- `output_boosted_control_16pre/fj_tau21_SF.json.gz`
- `output_boosted_control_16pre/fj_WvsQCD_SF_tau21cut.json.gz`
- `output_boosted_control_16post/fj_tau21_SF.json.gz`
- `output_boosted_control_16post/fj_WvsQCD_SF_tau21cut.json.gz`
- `output_boosted_control_17/fj_tau21_SF.json.gz`
- `output_boosted_control_17/fj_WvsQCD_SF_tau21cut.json.gz`
- `fj_tau21_SF.json.gz` (2018)
- `fj_WvsQCD_SF_tau21cut.json.gz` (2018)

### fakelepton_weights_noiso_3j.yaml — fake-lepton weights (both configs)

- `fakeweight_2016preVFP_v1.root` (20 MB)
- `fakeweight_2016postVFP_v1.root` (20 MB)
- `fakeweight_2017_v1.root` (20 MB)
- `fakeweight_2018_v1.root` (20 MB)

### dphi_weights.yaml — nonprompt dphi SFs (config_inverted only)

- `nonprompt_sf_lo1_e.json`
- `nonprompt_sf_lo1_mu.json`

### classifiers.yaml — XGBoost BDT models

Resolved, e/mu, 10 folds each (~7.3 MB total):

- `bdt_qg/bdt_resolved_e_{0..9}_iter_*.json` (283, 264, 267, 324, 229, 244, 310, 229, 306, 376)
- `bdt_qg/bdt_resolved_mu_{0..9}_iter_*.json` (381, 256, 352, 264, 380, 285, 410, 362, 301, 340)

Boosted 2018, e/mu, 10 folds each (~3.1 MB total):

- `outputs/bdt_model_2018_boosted_e/bdt_boosted_e_{0..9}_iter_*.json` (98, 106, 86, 46, 193, 82, 140, 82, 159, 101)
- `outputs/bdt_model_2018_boosted_mu/bdt_boosted_mu_{0..9}_iter_*.json` (199, 74, 133, 108, 127, 120, 146, 171, 183 (145), 116)

### pileup.yaml — **MISSING FILE**

- `puWeights_17.json` — **does not exist on disk.** `params/pileup.yaml` points
  at it for 2017. Candidates present: `params/puWeights_2017.json` (18 KB,
  plain JSON) and `puWeights.json.gz` (2.6 KB). Fix the YAML or restore the
  file before shipping.

## Path portability (done 2026-10-29)

All absolute / Purdue-specific paths in the param YAMLs were rewritten to
`${config_dir:}/...`. Both configs register that resolver from their own
location before merging parameters
(`defaults.register_configuration_dir("config_dir", localdir)` with
`localdir = os.path.dirname(os.path.abspath(__file__))`), so the whole tree is
relocatable: check it out anywhere, paths resolve relative to the config file.
Rewritten: `pileup.yaml`, `fakelepton_weights_noiso_3j.yaml`,
`dphi_weights.yaml`, `qgtagging.yaml`, `fj_taggers.yaml` (also removed the
`/home/<user>/work` symlink prefix that was invisible to Dask/Slurm/Ray
workers), `classifiers.yaml`, `lepton_scale_factors.yaml`.
Verified: both configs import in the run environment and all resolved SF paths
exist (except the pre-existing `puWeights_17.json`, below).

## Environment

Reproduced by `environment.yml` (minimal spec: python + pip wheels — a full
pinned export of `my-env` does not solve off the Purdue image, e.g. on lxplus
EL9) plus the PocketCoffea fork installed editable:

```bash
# inside apptainer/singularity, source the conda hook first —
# the container shell never reads ~/.bashrc, so `conda init` does nothing:
source /usr/local/etc/profile.d/conda.sh
conda env create -f environment.yml          # add -p <path> to choose where
conda activate vbs-semilep                   # or the <path> given to -p
git clone -b nonprompt https://github.com/vslokenb/PocketCoffea.git
pip install -e ./PocketCoffea                # replaces export PYTHONPATH=...
```

A missing module at run time means one more `pip install <name>` — add it to
`environment.yml` too so the file stays the source of truth.

PocketCoffea pin (what was on PYTHONPATH when this was validated):

- repo: https://github.com/vslokenb/PocketCoffea.git
- branch: `nonprompt`
- commit: `a4704f09655faf382c1718c68f04832647e808aa`
  (2026-10-26 "Support both JERC SF-uncertainty layouts in get_jersmear_SFunc")
- **WARNING — uncommitted local changes** in that checkout:
  `pocket_coffea/executors/executors_purdue_af.py` and
  `pocket_coffea/scripts/runner.py` are modified but not committed. Commit and
  push them to the fork before shipping, or the pinned commit will not
  reproduce the running setup.

## Known issues to fix before git add

1. **Missing**: `puWeights_17.json` (see above) — now referenced as
   `${config_dir:}/puWeights_17.json`, still absent on disk.
2. **Size**: the four `fakeweight_*_v1.root` files are ~80 MB combined —
   consider git-lfs or external hosting if the repo should stay small.
3. **Dataset JSONs read from FNAL EOS**: all 2407 file entries across the five
   `datasets/*.json` use `root://cmseos.fnal.gov` — already site-neutral XRootD
   (nothing Purdue-specific), readable from anywhere with a valid VOMS proxy.

## Totals

- 13 param YAMLs + 5 dataset JSONs + 66 SF/correction/model files = 84 files
- ~92 MB total, of which 80 MB is the four fakeweight ROOT files
