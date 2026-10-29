# Pocket/temporary_workspace — PocketCoffea event processing

The live PocketCoffea workspace of the VBS semileptonic analysis: object definitions, event selections,
weights and systematics, the BDT scoring, the data-driven nonprompt estimate, and the scripts that run all
of it on the **Purdue Analysis Facility (Purdue AF)**. Its outputs (`.coffea` histograms) are the input of
[`../../VBSfit`](../../VBSfit/README.md).

> **The main workflow uses exactly three configs: `config_semilep.py`, `config_validation.py` and
> `config_inverted.py`.** Everything else in this directory (26 further `config_*.py`, six alternative
> workflows and cut modules, study scripts) is development history, kept for reference — see
> [§12 Legacy and study configs](#12-legacy-and-study-configs).

## Table of contents

1. [What the production workflow is](#1-what-the-production-workflow-is)
2. [Running it](#2-running-it)
3. [The processing code](#3-the-processing-code)
4. [Selections: skim, preselection, regions](#4-selections-skim-preselection-regions)
5. [Weights, calibrations and systematic variations](#5-weights-calibrations-and-systematic-variations)
6. [BDT: training, scoring, tooling](#6-bdt-training-scoring-tooling)
7. [Nonprompt (fake-lepton) estimate](#7-nonprompt-fake-lepton-estimate)
8. [Datasets](#8-datasets)
9. [Parameter files (`params/`)](#9-parameter-files-params)
10. [Outputs, merging, plotting, hand-off to VBSfit](#10-outputs-merging-plotting-hand-off-to-vbsfit)
11. [Purdue AF operating notes (measured)](#11-purdue-af-operating-notes-measured)
12. [Legacy and study configs](#12-legacy-and-study-configs)
13. [Supporting scripts and utilities](#13-supporting-scripts-and-utilities)
14. [Run-provenance directories](#14-run-provenance-directories)
15. [Troubleshooting history worth remembering](#15-troubleshooting-history-worth-remembering)
16. [What is (not) tracked in this repository](#16-what-is-not-tracked-in-this-repository)

---

## 1. What the production workflow is

| Config | Role | Workflow class | Categories filled | Output |
|---|---|---|---|---|
| **`config_semilep.py`** | Signal-region pass with a large **columnar output** — the training/validation input for the BDT | `workflow_v9.VBSSemileptonicProcessor` | `baseline`, `resolved_mu`, `resolved_e`, `boosted_mu`, `boosted_e` | histograms **and** per-event columns (`ColOut`: jets 1–6 kinematics + QG/b-tag scores, all pairwise mass/ΔR/Δφ/Δη, fat-jet taggers and τ_N, VBS-jet variables, MET, centrality, Zeppenfeld, …) |
| **`config_validation.py`** | The **nominal data/MC pass**: SR + t̄t CR + W CR + validation-region histograms with all weights and shape systematics | `workflow_v9.VBSSemileptonicProcessor` | `baseline`, `w_cr_mu`, `w_cr_e`, `w_cr_boosted_{mu,e}`, `ttbar_cr_{boosted,resolved}_{mu,e}`, `vr_{mu,e}`, `vr_boosted_{mu,e}`, `boosted_{mu,e}`, `resolved_{mu,e}` | histogram-only (no populated `columns` block) → `cr_v28_<YEAR>_gNN/…` |
| **`config_inverted.py`** | The **nonprompt pass**: the same selections/categories as validation, but with the lepton ID **inverted** (loose-not-tight leptons) and reweighted by fake-lepton weights | `workflow_invertlepton_noiso.VBSSemileptonicProcessor` | same list as `config_validation.py` | histogram-only → `nonp_v28_<YEAR>_gNN/…` intermediates, later merged to the per-year nonprompt files |

How they fit together:

```
config_semilep ──columns──► outputs/bdt_kfold.py ──model JSONs──► params/classifiers.yaml
                                                                          │ (scored in the workflow)
config_validation ─► nominal data + MC (all years, per sample group) ◄────┤
config_inverted   ─► inverted-lepton data × fake weights  ─merge─► nonp_v28_<YEAR>_<tag>.coffea
```

* `config_validation` and `config_inverted` are run back-to-back by `run_chain.sh`.
* `config_semilep` is run on its own (`run_columns.sh`) when columns are needed (BDT retraining, studies).
* Both histogram configs carry the full systematic machinery (§5); the inverted pass reweights with the
  nonprompt weights (§7).

---

## 2. Running it

All processing runs **inside the PocketCoffea Apptainer image on the Purdue AF**, with the Dask Gateway
executor `dask@purdue-af`. The repository scripts encode the exact recipe:

```bash
# run_chain.sh  (driver; edit WORKDIR/BINDS for your account)
apptainer exec -B /work/users/<user>/ -B /cvmfs -B /cvmfs/cms-griddata.cern.ch \
               -B /cvmfs/cms.cern.ch -B /tmp -B /etc/sysconfig/ \
  /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/pocketcoffea:lxplus-el9-latest \
  bash run_inside_validation.sh        # config_validation.py
# ... stop dask clusters ...
apptainer exec ... bash run_inside_inverted.sh                                  # config_inverted.py
```

| Script | What it does |
|---|---|
| `run_chain.sh` | Stops any existing Dask clusters (only **one active cluster per user** is allowed), runs the validation pass, stops the clusters, runs the inverted pass, stops the clusters again. |
| `run_inside_validation.sh` | `conda activate …/envs/my-env`, sets `PYTHONPATH` to the local PocketCoffea checkout and `LD_LIBRARY_PATH`, then `pocket-coffea run --cfg config_validation.py -o debugging_cr_chain_vNN -e dask@purdue-af --custom-run-options custom_full_opt.yaml --overwrite --process-separately --limit-files 50`. |
| `run_inside_inverted.sh` | Same for `config_inverted.py` (`-o debugging_nonp_chain_vNN`, `custom_nonprompt_opt.yaml`). |
| `run_columns.sh` | Recipe for the columnar `config_semilep.py` pass (`-o SR_2018_<date> … --limit-files 20`). It starts with `apptainer shell`; paste/run its remaining commands inside that shell. |
| `build_datasets.sh [dir]` | `pocket-coffea build-datasets --cfg <f> --overwrite` for every discovery JSON (skips `*_replicas.json`). |
| `run_mix_years.sh <dir>`, `submit_mix_years.sh` | Run `mix_years.py <dir>` (merge all `.coffea` of a directory) locally / as a SLURM array job. |

Run-option files (`custom_full_opt.yaml`, `custom_nonprompt_opt.yaml`, `my_run_options.yaml`) set
`scaleout`, `cores-per-worker`, `mem-per-worker`, `chunksize`, … — **read §11 before changing them**: several
settings in the committed files predate the measurements in §11 (e.g. `queue:` and `custom-setup-commands:`
are inert for `dask@purdue-af`, `--overwrite` is silently ignored, and workers × cores above ~50 overloads
the FNAL data origin).

Typical launch (adapt the output name; the CLI `-s/--scaleout` overrides the YAML):

```bash
export PYTHONUNBUFFERED=1
export X509_USER_PROXY=/work/users/<user>/x509up_u<uid>     # worker-visible proxy, see §11.1
pocket-coffea run --cfg config_validation.py -o cr_v28_2018_g05 -e dask@purdue-af \
    --custom-run-options custom_full_opt.yaml --process-separately --limit-files 50
```

---

## 3. The processing code

### `workflow_v9.py` — `VBSSemileptonicProcessor` (used by `config_semilep` and `config_validation`)

* **Gen-weight patch**: replaces `genWeight` by `sign(genWeight)` (the NanoAODv9 UL `genWeight` is
  unreliable); normalization then comes from `sum_genweights`.
* **Objects**: good muons (`dxy<0.2`, `dz<0.5`, `|η|<2.4`), good electrons (`pT>38`, `cutBased≥3`, barrel/endcap
  `dxy/dz` cuts), loose/veto leptons, good AK4 jets (lepton-cleaned, with `nJetGood30`, central
  `|η|<2.4` / middle `2.4–3.0` / forward `>3.0` collections), good AK8 fat jets, fat-jet **candidates**
  (`τ21<0.45`, `m_SD<250`; `candidate_boost180`, `candidate_boost` with `pT>200`), b-tag working points
  (AK4 DeepFlavB; AK8 particleNet-MD Xbb 0.9172 / 0.9880), HEM-affected jet/electron collections.
* **Event reconstruction**: **VBS tagging jets** = the jet pair with the highest m_jj (with
  `vbsjets.mass`, `.delta_eta`, `.delta_phi`); hadronic V candidate from the two non-VBS jets
  (`w_had_jets`) minimizing |m−80.4| (resolved) or the candidate fat jet (boosted); leptonic W with
  neutrino p_z solution; m_T, Zeppenfeld variables (`z_lep`, `z_fat`), centrality (`centrality_resolved`,
  `centrality_boosted`), q/g likelihood of the VBS and W jets, pairwise `mass/dR/dphi/deta` between
  jets 1–6, the lepton, MET (DeepMETResolutionTune) and the leading fat jet.
* **BDT scoring** (§6): for each of the four categories, loads the 10 XGBoost fold models of the event's
  data-taking year from `params/classifiers.yaml`, builds the feature matrix from the booster's own feature
  names, and stores `bdt_<category>` = mean of the fold predictions.

### `workflow_invertlepton_noiso.py` — the inverted-lepton workflow (`config_inverted`)

Identical object/event logic, but the *tight* leptons are replaced by **loose-but-not-tight** leptons
(`MuonLoose`/`ElectronLoose` minus those matching a tight lepton in ΔR), so the selection populates the
region where fake leptons live; MET-phi and m_T are recomputed with the loose lepton. The fake-lepton
weights are applied as extra weights in the config (§7). It also defines `veto_jer_forward_unmatched`
(disables JER smearing for gen-unmatched jets with 2.5<|η|<3.0); in this checkout it is defined but not
wired into the calls (the references are commented out).

### `custom_cut_functions.py` — selections

Skim cuts (`nLepton_skim_cut`, `nJet_skim_cut`, `met_skim_cut`), the VBS preselection
(`vbs_semileptonic_presel`), SR windows (`whad_window_cut_bveto_{mu,e}`, `msd_window_cut_{mu,e}`),
t̄t CRs, W CRs, validation regions, plus the QCD-enriched CR and recoil cuts used by the
fake-rate studies. Details in §4. `reweighting_st.py` holds a cubic-spline `ratio_function` of the
generator-jet scalar pT sum used by the `wjet_reweight` weight (registered in the configs but not part
of the active inclusive weight list of `config_validation`).

---

## 4. Selections: skim, preselection, regions

**Skim** (applied to every event): `nPVgood≥1`, event flags, golden JSON, ≥1 lepton (e pT>35 or μ pT>30),
(AK4 + AK8 jets with pT>30) ≥ 3, `DeepMETResolutionTune.pt>30`, HLT (`SingleMuon` or `EGamma` primary
dataset, per-year paths in `params/triggers.yaml`).

**Preselection** `vbs_semileptonic`: exactly one good lepton, `nJetGood30≥2`, `MET_pT>30`,
`m_T(W_lep)<185 GeV`, no b-veto at this level (`apply_b_veto=False`), W+jets inclusive sample restricted to
`LHE_HT ≤ 70` (HT-binned samples cover the rest), and the **HEM veto** for 2018 EraC/EraD data
(`run ≥ 319077` with a HEM-affected jet/electron).

| Region | Selection on top of the preselection |
|---|---|
| **Resolved SR** (`resolved_{mu,e}`) | `nMuonGood30==1` & `nElectronGood38==0` (or the electron mirror); `nJetGood30≥4`; `nBJetGood==0`; `nFatJetCandidate180==0`; `nLeptonVeto<2`; hadronic-W candidate `pT<200` and `&#124;m_jj−85&#124;<20`; VBS jets `m_jj>500`, `Δη>2.5`, `jet1 pT>50`, `jet2 pT>30`; `m_T>30` |
| **Boosted SR** (`boosted_{mu,e}`) | one fat-jet candidate (`nFatJetCandidate==1`) with `pT>200` and `&#124;m_SD−92.5&#124;<22.5`; `nBJetGood==0`; loose-lepton veto; same VBS-jet cuts; `m_T>30` |
| **t̄t CR** (`ttbar_cr_{resolved,boosted}_{mu,e}`) | the SR with the b-veto replaced by `nBJetTight≥1` |
| **W CR** (`w_cr_{mu,e}` resolved, `w_cr_boosted_{mu,e}`) | SR structure, hadronic-W window (resolved) / fat-jet m_SD window (boosted) **inverted**, `m_T>30` |
| **Validation region** (`vr_{mu,e}`, `vr_boosted_{mu,e}`) | SR structure exactly, `m_T∈[20,30]` |
| QCD-enriched CRs (`qcd_enriched_*`, recoil/m_T variants) | used only by the fake-rate derivation configs (§7, §12) |

The resolved W CR keeps the shorter names `w_cr_mu` / `w_cr_e` (no topology label) — a historical quirk the
fit side compensates for (`VBSfit` maps `w_cr_resolved_{mu,e}` → `w_cr_{mu,e}`).

---

## 5. Weights, calibrations and systematic variations

**Inclusive weights** (`config_validation.py`): `genWeight`, `lumi`, `XS`, `PileupWeight`, `sf_mu_id`,
`sf_mu_iso`, `sf_ele_id`, `sf_ele_reco`, `sf_mu_trigger`, `sf_ele_trigger_run2`, `sf_L1prefiring`,
`sf_jet_puId`, `sf_partonshower_isr/fsr`, `sf_btag`, `sf_fj_WvsQCD`, `sf_fj_tau21`, `sf_qgtagging`,
`LHEScaleWeight`, `LHEPdfWeight`; and, **for t̄t only** (`TTTo2L2Nu`, `TTToSemiLeptonic`),
`top_pt_reweight`. Each weight with variations is also listed under `variations.weights`.

**Shape variations**: `jet_calibration` (JES/JER, per-year correctionlib recipes in
`params/jets_calibration.yaml`), `muons_rochester`, `electron_scale_smearing_legacy`.

Custom weight classes defined in the configs:

| Weight | Notes |
|---|---|
| `MuonGoodLeadWeight`, `ElectronGoodLeadWeight`, `PileupWeight`, `SF_L1prefiring`, `SF_ele_trigger` | lepton/pileup scale factors; electron trigger SF via a correctionlib file with `sf` variation |
| `LHEScaleWeightWrapper`, `LHEPdfWeightWrapper` | built from the LHE replicas (scale: 9 members; PDF: up to 102 replicas + αS); samples lacking members pad to 1.0; PDF spread uses `ddof=1`; applies to the QCD-diboson and signal samples |
| `QGTaggingWeight` | per-jet **shape** SF (QvG score, parton flavour, \|η\|, pT) from per-year correctionlib files; QvG score inverted (1−score); only \|partonFlavour\|=1–3, 21 corrected; sources central, up/down, and `{stat,fsr,isr,pu,jes,jer,L1prefiring,herwig,scale,PDF}`; **renormalized to preserve the yield** per chunk |
| `FatJetTau21Weight`, `FatJetWvsQCDWeight` | per-event fat-jet tagger SFs from the leading candidate fat jet (matched/unmatched to a gen W, ΔR<0.8; `params/fj_taggers.yaml`) — same renormalization scheme |
| `TopPtReweight` | TOP-23-008 top-pT reweighting; up/down = double-reweight / no-reweight envelope; `renorm` toggles yield-preserving renormalization |
| `wjet_reweight` | W+jets reweighting vs the gen-jet pT sum (`reweighting_st.ratio_function`); registered, not active in the inclusive list |

Shape-SF renormalization is a **per-chunk constant computed on the nominal pass and reused on shape-variation
passes**. If a systematic's ratio panel shows a flat normalization shift, export `POCKET_NORM_DEBUG=1` to print
the constant per (weight, shape variation).

The `variation` axis of the output histograms is **inhomogeneous by design**: 153 variations for t̄t
(the extra `top_pt_reweight` up/down), 151 for the other MC, 1 for data, 5 for the merged nonprompt.

---

## 6. BDT: training, scoring, tooling

* Models: one 10-fold XGBoost ensemble per category and year, JSONs in
  `outputs/bdt_model_2018_{resolved_mu,resolved_e,boosted_mu,boosted_e}/bdt_<cat>_<fold>_iter_<N>.json`
  (also `outputs/bdt_results_nJet/` for an n-jet variant). `params/classifiers.yaml` lists the paths per
  year; **the file currently has blocks for `2017` and `2018` only** (the 2017 block reuses the 2018-trained
  boosted models and the `bdt_qg` resolved models) — any other year needs a block, otherwise
  PocketCoffea fails with `ConfigKeyError: Missing key 2016_PostVFP`.
* Feature-name convention used when scoring in the workflow: `events_n<X>` → `ak.num(events.<X>)`,
  `events_<X>` → `events.<X>`, `w_had_jets_<X>` → `events.w_had_jets.<X>`, other `<coll>_<field>` →
  `events.<coll>.<field>`.

| Script (`outputs/`) | Purpose |
|---|---|
| `bdt_kfold.py <coffea>` | Train the k-fold ensemble for one category/year from a `config_semilep` columnar output. Options: `--category`, `--year`, `--out`, `--outdir`, `--nfold` (default 10), `--testsize`, `--num-round`, `--evaluate-only --model`. Signal = the EWK `dipoleRecoil` group; backgrounds grouped (Top/t̄t, W+jets, …); class weights balanced. |
| `bdt_opt.py`, `apply_bdt_elsewhere.py` | Earlier optimisation / application variants of the training script. |
| `bdt_CR.py`, `bdt_linfit.py` | **Data/MC comparison** of the BDT score and input features in control regions with pre-trained models (`--coffea-nonprompt`, `--category`, `--year`, `--models`, `--outdir`; `bdt_linfit.py` adds a calibration, `--use_calib`). Genweight normalization `w/sum_genweights`, sign preserved. |
| `cutflow.py`, `compute_fake_rate_coffea.py`, `saving_df.py`, `skimpy_transformer.py`, `image_list.py`, `mix_years.py` | Small analysis helpers/older copies. |

> The BDT is trained with balanced class weights, so its raw output is **not** a calibrated P(signal)
> under the physics prior. A calibration map (isotonic/Platt/BBQ) is needed before reading the score as a
> probability.

---

## 7. Nonprompt (fake-lepton) estimate

The nonprompt background is **always data-driven**:

1. **Fake rate** `r = (data−MC)_tight / (data−MC)_loose` measured in QCD-enriched control regions
   (`qcd_enriched_*` categories of the `config_nonprompt*` configs, recoil/m_T/MET and jet-multiplicity
   variants) → `compute_fake_rate_coffea.py` / `_3d.py`. Merge scripts use a YAML `factors:` block
   (`+1.0` data, `−1.0` MC) so a plain accumulate yields `data − MC`; `ratio_pairs:` define numerator/denominator
   histograms (`nonprompt_subtraction_factors*.yaml`, `nonprompt_subtraction_3d.yaml` for the pT×|η|×Δφ version).
2. **Fake-lepton weights** `w = FR/(1−FR)` (+ up/down) → `compute_fakelepton_weights_coffea*.py`;
   `compute_fakeweight_contamination.py` is the extended version with robust nominal/up/down variations that
   accounts for prompt contamination; `compute_nonprompt_subtractMC.py` and `compute_nonpMC_AIO.py` are
   MC-subtraction variants. Results are stored as ROOT histograms (`fr_2018_<date>.root`) referenced by
   `params/fakelepton_weights_noiso_3j.yaml` with separate resolved (`Muon`, `Electron`) and boosted
   (`Boosted_Muon`, `Boosted_Electron`) entries.
3. **Apply** in `config_inverted.py`: weights `muon_inverttight_to_fake`, `electron_inverttight_to_fake`
   and their `_boosted` versions (nominal/up/down from step 2), plus the **Δφ correction** SF
   (`params/dphi_weights.yaml` → `nonprompt_sf_lo1_{mu,e}.json`, derived by `dphi_correction_sf.py` as
   `SF_i = (data_i − mc_other_i) / nonprompt_i` per Δφ bin).
4. **Merge** the raw inverted passes (`nonp_v28_<YEAR>_gNN/…`, never to be used directly) into **one file per
   year**, `nonp_v28_<YEAR>_<tag>.coffea` (sample `nonprompt`, dataset `nonprompt_<YEAR>`, five variations:
   nominal + `muon_inverttight_to_fake` and `electron_inverttight_to_fake` up/down). This is what VBSfit
   consumes.

Notes: the merged nonprompt lacks a separate `_boosted` fake-rate split, so the two topologies are correlated
unless the upstream production changes; the `qcd_enriched_mu` category was abandoned for the muon fake rate
(prompt W/Z contamination, no recoil cut).

---

## 8. Datasets

`datasets/` holds PocketCoffea dataset definitions (one JSON per sample; `*_redirector.json` variants use
the global redirector):

| Subdir | Content |
|---|---|
| `datasets/*.json` | built dataset definitions (per sample, per era; `_17` suffix for 2017 DY/W variants) |
| `datasets/discovery/` | inputs for `build-datasets`: `<sample>.json` (+ `<sample>_replicas.json`) listing DAS names |
| `datasets/storage/`, `datasets/filtered/` | site-restricted / reduced (`*_fast.json`) variants |

The configs read **skimmed** datasets (`rare_skim.json`, `skimmed_rescale.json`; the hadd'd
`skimmed_dataset_definition_hadd.json` is tracked) whose files live at
`root://cmseos.fnal.gov//eos/uscms/store/group/lnujj/semilep_vbs_skim/`; `config_skimmer.py` is the skim
producer. Utilities: `build_datasets.sh`, `update_datasets.py` (refresh JSONs via `dataset-discovery-cli`,
preserving xsecs and deriving the four Run-2 era patterns), `fix_redirector.py` (replace site-specific
xrootd URLs by the global redirector so failed skim jobs can be resubmitted), `find_buggy_datasets.py`,
`xsec_swapper.py`, `assign_2016_eras.py`, `easy_config_list.py`.

Gotchas:
* `Configurator.load_datasets` iterates the JSON list in order and **the last JSON wins** for a duplicate
  dataset key — append an override JSON last to replace one dataset without touching the others.
* The same dataset key can appear in several JSONs with different file counts/`nevents` (skimmed vs
  unskimmed vs redirector). Confirm which one the config actually loads.
* `SingleElectron` exists for 2016/2017 only; 2018 uses `EGamma`. Check each year's data samples, or the
  electron categories silently come out empty.
* W+jets: the inclusive sample is kept only at `LHE_HT < 70` (HT-stitching).

---

## 9. Parameter files (`params/`)

| File | Content |
|---|---|
| `classifiers.yaml` | BDT fold-model paths per year/category (§6) |
| `fakelepton_weights_noiso_3j.yaml`, `dphi_weights.yaml` | nonprompt weights (§7) |
| `object_preselection_run2_v9.yaml` (+ `run2`, `run2_invertlepton`, `run3`) | object-selection thresholds |
| `jets_calibration.yaml` (+ `_v15`) | per-year JEC/JER correctionlib tags (`${cvmfs:…}` resolver) |
| `fj_taggers.yaml`, `qgtagging.yaml`, `qg_tagging.yaml` | fat-jet tagger SFs and quark/gluon SF correctionlib files |
| `lepton_scale_factors.yaml`, `jet_scale_factors.yaml`, `pileup.yaml`, `puWeights_2017.json` | lepton/jet/pileup SFs |
| `triggers.yaml`, `lumi.yaml` | HLT paths and per-era luminosities (pb⁻¹) |
| `variations.yaml` | systematic weight-variation names (`sf_btag`: hf, lf, hfstats1/2, lfstats1/2, cferr1/2; `sf_ele_trigger`: `sf`; …) |
| `plotting.yaml`, `plotting2.yaml`, `nano_version.yaml` | plotting style, NanoAOD version |
| `convertPU_to_root.py` | pileup-histogram conversion helper |

> Several YAMLs contain **absolute Purdue AF paths** (`/work/users/vsloken-fnal/…`). Edit them when moving
> to another account or cluster.
>
> `variations.yaml` lists the correctionlib categories that exist in the SF files: listing a source that the
> file does not contain (e.g. `PileupWeight_up`) fails with `IndexError: Index not available in Category`.
> Remove it from the sources list rather than renaming the weight.

---

## 10. Outputs, merging, plotting, hand-off to VBSfit

**Output layout.** `pocket-coffea run -o <dir>` writes `config.json`, `parameters_dump.yaml` and either one
`output_all.coffea` (grouped run) or, with `--process-separately`, one `output_<dataset>.coffea` per dataset.
Top-level keys: `sum_genweights`, `sum_signOf_genweights` (by dataset); `sumw`, `sumw2`
(`[category][dataset][sample][variation]`); `cutflow` (by dataset); `variables`
(`[hist][sample][dataset]`, histogram objects); `columns` (**by sample** — a different key level);
`datasets_metadata` (`by_dataset`, `by_datataking_period`).

**v28 production layout** (the current Combine inputs): `<stage>_<YEAR>_<gNN>/output_all.coffea`.
Stages: `cr_v28` (nominal, i.e. `config_validation`) and `nonp_v28` (raw inverted passes — never use directly);
the merged per-year file is `nonp_v28_<YEAR>_<tag>.coffea` at the root. Years are never mixed in a file; the ten
sample groups `gNN` are fault isolation only (no physics meaning; `g04` = `EGamma`, 2018 only by design).
Histograms are **absolute expected yields** — do not re-apply lumi/xsec. A `limit_files` cap (50/30/20) only
costs MC statistics. The authoritative file→sample index is `v28_manifest.csv` (produced next to the outputs
by `make_v28_manifest.py`; columns `dataset,sample,year,stage,group,directory,file,n_variations,limit_files,
worker_config,file_size_bytes,file_mtime,periods_in_file,n_datasets_in_file`) — `VBSfit` is manifest-driven.

**Merging.** `mix_years.py <dir>` accumulates every `.coffea` of a directory; `latest_mix_years.py` (repo root)
merges pairwise to lower peak memory. Merging slices of one dataset works with `coffea.processor.accumulate`
as long as every slice keeps the **same dataset key** (see §11.12). Use the `my-env` interpreter to merge
(numpy 1.26): pickles written by a numpy-2 environment cannot be read downstream. Plain `accumulate` across
differing categorical axes fails with “axes not mergable”.

**Plotting.** `make_plots_multi.py` (repo root; `--v28 <root> -y <year>`), `2d_histo_plotter.py` (PocketCoffea has
no native 2D support — MC total, data total and data/MC per histogram/category/year; options `--input --plotdir
--cat --year --log --format --lumi --ratio-range --no-mc --no-data --no-ratio`), `outputs/cutflow.py` and
`VBSfit/python/make_cutflow_table.py` for cutflow tables, and `coffea_to_root.py --input --output` to convert
histograms to ROOT.

**Hand-off.** VBSfit reads these files via the manifest; see its README for the per-year → Run-2 flow.

---

## 11. Purdue AF operating notes (measured)

*Working notes from the v29 (2017 semilep/columnar) and v30 (2018 validation/inverted) campaigns, Oct 2026.
Everything below was measured in this environment, not taken from documentation. `<user>` stands for your
account name.*

### 11.1 Environment and invocation

| Item | Value |
|---|---|
| Container | `/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/pocketcoffea:lxplus-el9-latest` (via `apptainer exec`) |
| Binds | `-B /work/users/<user>/ -B /cvmfs -B /cvmfs/cms-griddata.cern.ch -B /cvmfs/cms.cern.ch -B /tmp -B /etc/sysconfig/` |
| Python env | `conda activate /work/users/<user>/envs/my-env` |
| Code | `PYTHONPATH=/work/users/<user>/latest_poco/PocketCoffea` — this local checkout **shadows the pip-installed `pocket_coffea` in `my-env`**; always verify which file is executing: `python -c "import pocket_coffea.scripts.runner as m; print(m.__file__)"` |

**Proxy.** `voms-proxy-init` writes to `/tmp/x509up_u<uid>`, which **Dask workers cannot see**. Copy it to a
worker-visible path and export it:

```bash
cp /tmp/x509up_u<uid> /work/users/<user>/
export X509_USER_PROXY=/work/users/<user>/x509up_u<uid>
voms-proxy-info -file /work/users/<user>/x509up_u<uid> -timeleft
```

The executor refuses a proxy that is not under `/work/` or `/depot/`. A stale copy there silently broke two
separate runs — check it with `voms-proxy-info -file … -timeleft`.

**Logging.** Set `PYTHONUNBUFFERED=1` in the launcher. stdout redirected to a file is block-buffered, so a run
that stalls *before* the progress bar emits nothing and looks identical to one that just started. This cost
~70 min of misdiagnosis twice.

### 11.2 `--process-separately` (`-ps`): the biggest structural choice

**Without `-ps`** (one fileset for all datasets in the invocation):

* \+ all datasets' chunks enter the graph at once → the cluster actually fills (measured: graph 24 tasks →
  2748 tasks for the same work);
* − one `output_all.coffea` per invocation, with a cross-dataset merge at the end;
* − a failure aborts the whole invocation;
* − results from completed datasets sit in worker memory until the merge, so a single lost worker forces mass
  recomputation (observed: 2520 → 2445 released, repeatedly, never converging).

**With `-ps`:**

* \+ one `output_<dataset>.coffea` per dataset; **no** cross-dataset merge at all;
* \+ runs with `exit_on_error=False`: a failed dataset is logged to `failed_jobs.json` and the loop continues;
* \+ a lost worker costs one dataset, not the whole group;
* \+ resume is trivial: re-run only datasets whose `.coffea` is missing;
* − only one dataset's chunks are runnable at a time, so small datasets cannot saturate the cluster;
* one Dask cluster is created per invocation but **reused across all datasets in it** (no per-dataset spin-up).

**Verdict.** `-ps` is what actually finishes. The grouped non-ps approach was measurably faster per task but
never produced a single output in ~20 h of trying; `-ps` produced 34/53 datasets in 18 h with zero errors.

### 11.3 Dask / cluster behaviour (hard-won)

* **`Client.scheduler_info()` is a cached snapshot** taken at connect time; on short-lived probe connections it
  reports whatever existed when you connected and never updates. It produced a fictitious "hard 5-worker cap"
  that several hours of work were built on. Always ask the scheduler itself:

  ```python
  def snap(dask_scheduler):                      # the parameter MUST be named dask_scheduler
      ws = dask_scheduler.workers
      return (len(ws), sum(w.nthreads for w in ws.values()))
  client.run_on_scheduler(snap)
  ```

* **The GIL means `worker_cores` buys almost nothing.** A Dask worker is one process; `worker_cores` becomes
  `nthreads` inside it, and coffea processing is Python-level enough that N threads deliver ~1 core of
  throughput. Measured: 12 workers × 8 cores = 96 cores → 13.5 busy (14 %); 50 workers × 1 core = 50 cores →
  49.4 busy (99 %). **Throughput tracks worker count, not core count — buy processes, not threads**
  ("Throughput per Worker" in the stats table is the meaningful column).
* **Only one active cluster per user.** Stop the old one before starting a new one, or the new run blocks.
* **Gateway options** (`list_dask_cluster_options`): `conda_env | pixi_project`, `worker_cores` (max 64),
  `worker_memory` (max 64 GiB), `env`. There is no walltime and no queue — `queue: workday` in a run-options
  YAML is **inert** for `dask@purdue-af`.
* **`dask@purdue-af` never reads `custom-setup-commands`** (those are for the condor/lxplus executors). The
  worker environment comes from a **fixed allowlist** in `executors_purdue_af.py`: `NB_UID`, `NB_GID`,
  `PYTHONPATH`, `LD_LIBRARY_PATH`, `X509_CERT_DIR`, `PATH` (+ `X509_USER_PROXY`). To forward any other
  variable you must add it to that list (`OMP_NUM_THREADS` etc. were added this way).
* A hand-rolled `Gateway().new_cluster()` requires `NB_UID` and `NB_GID` in `env=` or it is rejected; the
  executor supplies them automatically.

### 11.4 Run options that matter

| Option | Notes |
|---|---|
| `scaleout` | number of workers. The CLI `-s/--scaleout N` **overrides the YAML** (`runner.py`: `if scaleout != None: run_options["scaleout"] = …`). |
| `cores-per-worker` | `nthreads` per worker process — see the GIL note; **1 is often best**. |
| `mem-per-worker` | GiB per worker. Per-task budget = `mem / cores-per-worker`. 2 GiB/task caused `KilledWorker`; 4–6 GiB/task was safe. |
| `chunksize` | events per task; drives per-task memory directly. 50000 → `KilledWorker` under high concurrency; 20000 was stable. |
| `tree-reduction` | merge fan-in. The default 20 OOM'd on columnar output; 4 was safe. |
| `retries` | **Default is 10 and it hides failures.** On a dropped scheduler connection coffea silently *resubmits the whole computation*; symptom: progress bar pinned at 99 % for hours while the task counter resets (e.g. 291 → 0). Set `retries: 1` to see the real error in ~60 s; 3 is a reasonable compromise that absorbs a transient blip. |
| `limit-files` | takes the **first N files** of each dataset. **A cap ≥ the dataset's file count is a silent no-op** (three "retries" on a 24-file dataset with caps of 60 and 30 did nothing) — always derive the cap from the actual file count. It does **not** bias yields (`sum_genweights` is accumulated from the events actually processed, so normalization stays correct and only statistical precision is lost); capped and uncapped outputs can be mixed. |

Example run-options file assembled from the numbers above (illustrative — not a tracked file):

```yaml
scaleout: 50            # workers x cores-per-worker ~ 50 concurrent file opens (see 11.6)
cores-per-worker: 1
mem-per-worker: 6       # >= 4-6 GiB per task at 1 core per worker
chunksize: 20000
tree-reduction: 4
retries: 3
skip-bad-files: false
adaptive: false
```

### 11.5 A real upstream bug worth carrying

`runner.py` computes `adapted_chunksize = adapt_chunksize(...)` — whose documented purpose is to shrink the
chunk so every worker gets at least one — logs it, then passes the **unadapted** `run_options["chunksize"]`
to the runner. Both the `-ps` and non-ps branches do this. Effect: a 34k-event dataset ran as **one chunk on
one core**. Patch (in the `-ps` branch):

```python
adapted_chunksize = max(adapted_chunksize, 5000)   # floor: avoid tiny chunks
...
run = get_runner(..., chunksize=adapted_chunksize, ...)
```

Also: `--overwrite` is accepted but **silently ignored** (the click command sets `ignore_unknown_options=True`),
so it never did anything — remove the output directory yourself.

### 11.6 Data access

Skims live at `root://cmseos.fnal.gov//eos/uscms/store/group/lnujj/semilep_vbs_skim/`. Healthy single-stream
read: 1.29 GiB in ~8–18 s (~75–170 MB/s).

* **XCache does not serve these files.** `root://xcache.cms.rcac.purdue.edu/` fronts central CMS data under
  `/store/mc` and `/store/data` only; private group skims under `/store/group/` fail outright (cold and
  warm). Don't plan on it.
* **Concurrent-open limit.** Concurrent reads = workers × cores-per-worker. 100 concurrent opens →
  `OSError: XRootD error: [ERROR] Operation expired` on file open, killing 3 of 4 groups; 50 concurrent opens →
  zero errors over 18 h. Keep concurrent opens around **50** against this origin.
* **FNAL outage vs self-inflicted load.** Stat a central path through the redirector. During a real outage:

  ```
  xrdfs cmsxrootd.fnal.gov stat /store/mc            -> OK in 644 ms
  xrdfs cmseos.fnal.gov    stat <our skim>           -> timeout 50 s
  xrdfs cmseos.fnal.gov    ls /eos/uscms/store       -> timeout 40 s
  ```

  i.e. FNAL serves fine generally while *our* namespace hangs. TCP to port 1094 stayed open throughout, so a
  port check proves nothing.

### 11.7 Failure modes seen, and what each actually meant

| Symptom | What it meant / fix |
|---|---|
| `KilledWorker … on 4 different workers` on a `VBSSemileptonicProcessor-*` task | per-task OOM → lower `chunksize` and/or raise `mem-per-worker` (budget ≥ 4 GiB/task) |
| `KilledWorker … on a reduce-*` task | the **merge** is OOM'ing → lower `tree-reduction` (20 → 4) |
| `FutureCancelledError … reduce-* cancelled for reason: scheduler-connection-lost` | **intermittent scheduler connection drop**, not memory (peak 3–8 GiB of 32). With `retries: 10` it silently restarts everything. Identical at 219/60/30/25 files, so `--limit-files` does not fix it; shorter jobs reduce exposure |
| `OSError: XRootD error: [ERROR] Operation expired` on file open | too many concurrent file opens against FNAL → reduce workers × cores-per-worker |
| `ConfigKeyError: Missing key 2016_PostVFP` | a params YAML (e.g. `classifiers.yaml`) lacks a year block → add it |
| `IndexError: Index not available in Category … val: PileupWeight_up` | a correctionlib category key is listed in a variations/sources list that the SF file doesn't contain → remove it from the sources list; don't rename the weight |
| progress bar pinned at 99 % for hours | usually **not** the reduce; the bar is not scaled to task count. Check `released`/queued directly — we saw "99 %" with 456 ordinary processing tasks outstanding |

### 11.8 Monitoring that actually works

Trust, in order: (1) output files on disk, (2) the scheduler's task-state counters via `run_on_scheduler`,
(3) per-worker CPU/memory, (4) the log text. **Do not trust the progress bar for completion fraction.**

The one counter that separates progress from thrashing is `released`:

| `released` | Meaning |
|---|---|
| climbing | real progress |
| flat with high CPU | long tasks (fine) |
| flat with ~0 % CPU | blocked on I/O, or the origin is down |
| **decreasing** | work is being discarded and recomputed — a worker died or the whole computation was resubmitted |

Watchdog rules learned the hard way:

* **A watchdog must never delete or kill.** One early watcher compared `str(cluster.status)` (which returns
  `"2"`) against `"RUNNING"`, never matched, killed a healthy run at its timer, and its teardown `rm -rf`'d a
  *completed* output directory.
* Watch the thing that signals trouble, not the thing you're curious about. A watcher armed only on "worker
  count scales up" timed out saying "no change" while three groups failed unnoticed over the next seven hours.
* Verify a launch actually took (`pgrep` after starting) instead of assuming.
* `grep -c … || echo 0` can emit **two** lines and break every later integer comparison — quote and sanitise
  counters.

### 11.9 Measured throughput (use these, not guesses)

| Config / dataset | Events | Time | Total rate | Per worker |
|---|---|---|---|---|
| `config_validation` 2018 (histogram-only), one dataset, `-ps` | 2,559,441 | 32.05 min | 1331 ev/s | 26.6 ev/s/worker |
| `config_semilep` 2017 (columnar + classifier), one dataset | 4,702,538 | 45.90 min | 1708 ev/s | 34.2 ev/s/worker |

Steady-state `-ps` cadence at 50 workers: ~10 min (small) to ~2 h (large) per dataset; 30 datasets in ~14.5 h.

* **Do not extrapolate throughput from a single chunk run with `-t/--test` or `--limit-chunks 1`.** That pays
  all the one-time setup (correction loading, processor construction) and understated the real rate by 5.5×,
  producing a 550 h "ETA" against a true ~20 h.
* **Do not extrapolate from an early progress burst** either — small datasets in a group finish first and the
  tail is much slower.

### 11.10 Columnar output (`ColOut`) cost

`config_semilep` with a full `ColOut` block: **2.23 KB/event** — 33,908 events → 74 MB; 45 datasets →
77.6 GiB total. Histogram-only configs (validation/inverted have **no** populated `columns` block): ~115 MB per
dataset, 4.7 GB for 34 datasets. **If columnar output is on, size the disk first:** `/work` is 100 GB per user
and a full 2017 MC campaign projected to ~476 GB uncapped.

### 11.11 Script patterns in this workspace

| Script | Purpose |
|---|---|
| `run_v30_2018.sh` | chain: per-config passes, per-dataset resume; splits a separate capped pass for t̄t (`--limit-files` is global to one invocation, so capped datasets need their own) |
| `run_inside_v30.sh` | the actual `pocket-coffea run` call; sets the env that the executor allowlist forwards; takes `scaleout` and an optional `limit-files` |
| `v30_missing.py` | computes which **dataset keys** still lack an output file. Resume must work on dataset keys, not sample names: data samples expand into per-era datasets (`EGamma → EGamma_2018_EraA..D`) so they are not 1:1 |
| `watch_v30_ps.sh` | **read-only** watcher: alerts on each dataset written, any failure, and chain exit |
| `poll_fnal_and_launch.sh` | waits for FNAL to serve again (2 consecutive stats plus one real read), then launches; never deletes anything |
| `make_v29_manifest.py` | builds a manifest from the filesystem + dataset JSONs **without unpickling the outputs** (77.6 GiB is far too much to load just to read `sum_genweights`) |
| `push_v29_to_lxplus.sh` | rsync-if-available else a resumable `scp` loop; there is **no `rsync` anywhere on the AF** (not in the base env, global pixi, `my-env`, or the container) — only `scp` |

In this repository the tracked counterparts are `run_chain.sh`, `run_inside_validation.sh`,
`run_inside_inverted.sh`, `run_columns.sh` (see §2); the `v29`/`v30` scripts above live on the Purdue AF and
are not tracked at the time of writing (§16).

### 11.12 Output structure for downstream / Combine

With `-ps`: one `output_<dataset>.coffea` per dataset and **no** `output_all.coffea`. Top-level keys:
`sum_genweights`, `sum_signOf_genweights` (keyed by **dataset**); `sumw`, `sumw2`
(`[category][dataset][sample][variation]`); `cutflow` (keyed by dataset); `variables` (hist objects);
`columns` (keyed by **sample** — different key level!); `datasets_metadata` (`by_dataset` /
`by_datataking_period`).

Merging slices of one dataset: keep the **same dataset key** in each slice JSON and
`coffea.processor.accumulate` sums `sum_genweights`/`cutflow`/`sumw` and concatenates `columns` — the result
is structurally identical to one full run. Merge with the `my-env` interpreter (numpy 1.26); the global pixi
env's numpy 2 writes pickles the downstream scripts cannot read. (The VBSfit manifest generator is
manifest-driven, so per-dataset `-ps` outputs only need rows in the manifest.)

### 11.13 Dataset-definition gotchas

See [§8 Datasets](#8-datasets): last-JSON-wins on duplicate dataset keys; `SingleElectron` (2016/2017) vs
`EGamma` (2018); same dataset key appearing in several JSONs with different file counts/`nevents`.

---

## 12. Legacy and study configs

**None of these is part of the main workflow** (which is `config_semilep`, `config_validation`,
`config_inverted`). They document how the selections, scale factors and the nonprompt method were developed
and are kept as references or starting points. Many have outdated paths/parameters and may need the current
workflow/params to run.

| Group | Configs | What they were for |
|---|---|---|
| **Nonprompt / fake-rate derivation** | `config_nonprompt.py` (`workflow_v9`; `qcd_enriched_{mu,e}[_up/_down]`, boosted and recoil/low-jet variants), `config_nonprompt_3D.py` (`workflow_nonprompt`; DeepMET-resolution/response cuts 20–45 GeV, 2j/3j/4j, m_T variants), `config_nonprompt_invertlepton_noiso.py` and `_3D.py` (`workflow_invertlepton_noiso`) | QCD-enriched regions in which the fake rate is measured |
| **QCD validation** | `config_QCD_VR_3j.py`, `config_QCD_VR_invertlepton_noiso_3j.py`, `config_QCD_VR_invertlepton_noiso_3j_3d.py` | `QCD_VR[_0j][_e/_mu]` closure of the fake-rate method |
| **W+jets / t̄t CR studies** | `config_wjets_CR.py` (+ `_invertlepton_noiso`, `_invertlepton_noiso_3d`), `config_ttbar_CR.py`, `config_ttbar_CR_invertlepton_noiso.py`, `config_ttbar_cr_2017.py`, `config_ttbarCR_toppt_on.py` / `_off.py` | control-region development (including the top-pT reweighting on/off study); cut modules `custom_cut_functions_{wjets_CR,ttbar_CR,nonprompt}.py` |
| **Closure checks** | `config_closure_cr.py`, `config_closure_nonp.py` | CR and nonprompt closure of the main chain |
| **SR development (early)** | `config_semilep_SR.py` (`workflow.py`, 2017, electron `whad_*` categories), `config_semilep_pre.py` (`workflow_pre`), `config_semilep_invertlepton_noiso.py` (2016_PostVFP), `config_reweighting.py` (`workflow_reweight`; `w_check_*` categories; **header: lepton selections need fixing before reuse**) | early selection optimisation / reweighting checks |
| **Scale-factor derivation** | `config_boosted_control{,_data,_mc}.py` (`workflow_boosted_control`, `custom_cut_sf.py`; `TT_boosted`, `zjets_boosted` for 2017/2016) → `derive_fj_WvsQCD_SF.py`; `config_zpeak.py` (`custom_cut_zpeak.py`; `ele_z`, `mu_z`) | fat-jet tagger SF and Z-peak checks |
| **Skimming** | `config_skimmer.py` | skim production (not an analysis pass) |

Alternative processors and cut modules: `workflow.py` (older base), `workflow_pre.py`,
`workflow_reweight.py`, `workflow_nonprompt.py`, `workflow_boosted_control.py`, `workflow_v9_ttbar_CR.py`;
`custom_cut_functions_{SR,pre,reweight,nonprompt,ttbar_CR,wjets_CR}.py`, `custom_cut_sf.py`,
`custom_cut_zpeak.py`.

---

## 13. Supporting scripts and utilities

| Script | Purpose |
|---|---|
| `derive_fj_WvsQCD_SF.py` | Derive fat-jet tagger SFs: WvsQCD in two m_SD bins (inclusive τ21 and after τ21<0.45) and τ21 SFs in two coarse bins, by solving a 2×2 linear system per WvsQCD bin (`--mc --data --output --output-tau21cut --output-tau21sf`). Outputs `fj_WvsQCD_SF_tau21cut.json.gz`, `fj_tau21_SF.json.gz`. |
| `dphi_correction_sf.py` | Per-bin SF for the nonprompt contribution from a Δφ reference histogram (§7). |
| `optimization_1d.py`, `uncertainty_mc.py` | cut optimisation / total-uncertainty-per-category helpers. |
| `pileup.py`, `params/convertPU_to_root.py` | pileup weight lists and conversion. |
| `ElectronTriggerSF_*.json(.gz)`, `fatJet_jerc_*.json.gz`, `jet_jerc_*.json.gz`, `qgtagging*.json.gz`, `puWeights.json.gz`, `nonprompt_sf_lo1*.json`, `nlo_wj*.json` | correctionlib/other SF payloads referenced by `params/`. Electron trigger SF files come in a `_pocketcoffea` flavour matching the installed correctionlib input signature. |
| `environment.yml` | conda export of the Purdue AF environment (`/work/users/<user>/envs/my-env`, Python 3.11). |
| `.sites_map.json` | per-site LFN → xrootd rewrite rules (regex → `root://` URL; derived from the CVMFS SITECONF) used when rewriting dataset file URLs, e.g. by `check-jobs`. |
| `egam.json`, `eraD.json`, `redo.json`, `nlo_wj.json` (+ `*_replicas.json`) | dataset-**definition** inputs for `pocket-coffea build-datasets` (`sample`, `json_output`, `files` with DAS names): `egam`/`eraD` re-define the EGamma and SingleMuon era datasets, `redo` bundles six samples to rebuild, `nlo_wj` is the NLO W+jets sample. |

ML experiments (not part of the cut-based fit): `homogenous_transformer.py` (graph/transformer
classifier on columnar output; `--out --attn_type --skip_scan`), `hparam_scan.py` (Optuna scan with a shared
`JournalFileStorage` study — safe on EOS/AFS/NFS, unlike SQLite), `graph_builder.py`, and
`outputs/skimpy_transformer.py`.

---

## 14. Run-provenance directories

The many one-file directories in this folder (`debugging_cr_chain_v2…v15`, `debugging_nonp_chain_v*`,
`2018_wjet_CR_nonp_*`, `nonprompt_FR_*`, `revert_FR_recoil_*`, `FR_rescale_WJet_*`, `SR_2018_june16`, `btag_check`,
`test_skim`, `reweight_eval`, …) are **PocketCoffea output directories of which only `config.json` (the run's
frozen configuration) is tracked** — the `.coffea` files themselves are not in the repository. They document
which configuration produced which study. The naming convention is `debugging_cr_chain_vNN` (validation pass)
and `debugging_nonp_chain_vNN` (inverted pass) for chain iterations, and `cr_v28_<YEAR>_gNN` /
`nonp_v28_<YEAR>_gNN` for the v28 production.

---

## 15. Troubleshooting history worth remembering

* **Electron preselection near-total suppression**: `str.lstrip(prefix)` strips *characters*, not a prefix
  (`"Ele25_WPTight_Gsf".lstrip("HLT_")` → `"e25_WPTight_Gsf"`). Use `str.removeprefix`.
* **HEM masking bug**: `if "EGamma_2018_EraC" or "…EraD" in x` is always truthy; plus a typo and a scalar-vs-
  per-event mask. Use explicit `in` checks, `&`, and `ak.num(...)` masks.
* **Electron trigger SF correctionlib chain**: `map::at` mismatch → wrong input count (two out-of-sync `sf_ele_trigger`
  versions) → `Index not available … statUp` (YAML variation names must literally match the `{name}Up/Down`
  JSON keys) → a YAML block list (`- sf`) instead of inline `["sf"]`.
* **Skim normalization units trap**: for an `isSkim` dataset the metadata `sum_genweights` /
  `sum_signOf_genweights` are *authoritative overrides* of the per-event reconstruction. If they hold sign sums
  or event counts instead of Σ genWeight, every yield is scaled wrongly. Sniff: `sum_genweights ==
  sum_signOf_genweights` exactly on an amc@NLO sample is impossible for a true sum. Fix: strip both keys (the
  reconstruction from `skimRescaleGenWeight` is unit-correct) or refill from the checker script.
* **`weights["bysample"]` key trap**: a key that does not match a processed sample name only prints a warning
  and is silently skipped; the name must equal the weight class's `.name` and the class must be in
  `weights_classes`.
* **`hist.Stack.from_dict` "axes don't match"**: `variation` axes differ across samples (153/151/5/1,
  order and growth traits) — harmonize before stacking (`make_plots_multi.py` does).
* **MC always has a `variation` axis, data never does** → broadcast mismatches; drop the axis
  (`h[{"variation": "nominal"}]`) or `np.expand_dims(arr, axis=1)`.
* **`safe_divide` view mutation**: `num.values()[()]` returns a view; in-place edits corrupt the source — copy
  first.
* **Weight wrappers**: a fresh `WeightWrapper` instance is built per chunk and `nominal` is always processed
  first, so per-instance caches of renormalization constants are sound; `WeightLambda(has_variations=True)`
  expects `(nominal, [names], [ups], [downs])`.
* **OOM merging many `.coffea` files**: fold-and-`del` sequentially or use two-pass batching; the final `save()`
  can still exceed a per-user cgroup — use a batch job with explicit memory.
* **`extractor.add_weight_sets()` needs exactly three whitespace-separated tokens** — f-string alignment
  padding breaks the parser.
* **`import ROOT` at module level is not inherited by `multiprocessing` pool workers** — import inside the worker.
* **lxplus `condor_submit` OpenSSL crash** (`undefined symbol: BIO_new_dgram_sctp`) when running inside
  Apptainer+conda: strip `/usr/local/lib` from `LD_LIBRARY_PATH` before the subprocess.
* **xrootd `[3014] Network is unreachable`**: usually an IPv6-only redirector → `XRD_NETWORKSTACK=IPv4` in the
  job environment itself.
* **Lost-job triage on condor**: job state is the marker files `job_X.{idle,running,done,failed}`;
  a walltime removal transfers nothing back (0-byte `.err`, marker stuck at `.running`) — diagnose with the
  condor userlog, not `condor_history`.

---

## 16. What is (not) tracked in this repository

Tracked: all `config_*.py`, workflows, cut modules, `params/`, the SF payloads, the dataset JSONs under
`datasets/`, BDT model JSONs for 2018 (`outputs/bdt_model_2018_*`, `outputs/bdt_results_nJet`), and the
single-file run-provenance directories (§14).

**Not tracked here** (live on the Purdue AF or elsewhere):

* the `.coffea` outputs (`cr_v28_*`, `nonp_v28_*`, `debugging_*` payloads) and `v28_manifest.csv`
  (+ its generator `make_v28_manifest.py`);
* the skimmed dataset JSONs `datasets/rare_skim.json` and `datasets/skimmed_rescale.json` referenced by the main
  configs, and the QG SF payload directory `QGSF-Run2/`, the `bdt_qg/` resolved-channel models referenced by
  `classifiers.yaml`, and `fr_2018_<date>.root` (the fake-weight ROOT file);
* the v29/v30 campaign scripts of §11.11 (`run_v30_2018.sh`, `run_inside_v30.sh`, `v30_missing.py`,
  `watch_v30_ps.sh`, `poll_fnal_and_launch.sh`, `make_v29_manifest.py`, `push_v29_to_lxplus.sh`);
* the PocketCoffea fork itself (this workspace imports it from `PYTHONPATH`): the Purdue executor allowlist
  patch and the `adapted_chunksize` fix of §11.3/§11.5 are changes in that checkout, not in this repository.
