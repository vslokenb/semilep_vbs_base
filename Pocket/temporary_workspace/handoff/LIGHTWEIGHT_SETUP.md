# Lightweight PocketCoffea setup: building your own analysis from this workspace

This guide is for someone who wants the **framework pattern** of this workspace, not necessarily the VBS
semileptonic physics. It covers the six pieces you need to write your own analysis:

| Piece | Example file in this workspace | What you change for a new analysis |
|---|---|---|
| Skim (write slimmed NanoAOD once) | `config_skimmer.py` | loose cuts, datasets, output path |
| Cut functions | `custom_cut_functions.py` | your preselection and categories |
| Workflow (processor) | `workflow_v9.py` | object definitions, derived variables, optional ML scoring |
| Histogram config | `config_validation.py` | datasets, categories, weights, variations, histograms |
| Columnar (ntuple-like) config | `config_semilep.py` | which per-event columns to save, for example for ML training |
| Calibrations and custom weights | `config_validation.py` (weight classes), `params/*.yaml` | your corrections and scale factors |

Everything else in the workspace (nonprompt estimate, fat-jet SF derivation, the `run_v3*` campaign
scripts) is optional. For production-scale running see `PIPELINE_RUNBOOK.md`. For environment setup see
`handoff/AGENT_SETUP_HANDOFF.md` §3.

PocketCoffea docs: <https://pocketcoffea.readthedocs.io>. The version used here is the fork
`vslokenb/PocketCoffea@nonprompt` (0.9.9 based).

---

## 1. Minimal file layout

```
my_analysis/
├── config_skimmer.py          # optional: one-off skim to slimmed NanoAOD
├── config_hists.py            # histogram pass        (cf. config_validation.py)
├── config_columns.py          # columnar/ML pass      (cf. config_semilep.py)
├── workflow.py                # processor subclass    (cf. workflow_v9.py)
├── custom_cut_functions.py    # Cut objects
├── params/
│   ├── object_preselection.yaml   # pt/eta/id/iso per object
│   ├── triggers.yaml
│   ├── jets_calibration.yaml      # optional override of JEC/JER tags
│   └── <your SFs>.yaml            # paths written as ${config_dir:}/...
└── datasets/
    ├── datasets_definitions.yaml  # input to `pocket-coffea build-datasets`
    └── *.json                     # generated file lists with metadata (year, sample, xsec, isMC)
```

The header pattern that every config here uses (copy it verbatim):

```python
import os, cloudpickle
from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.parameters import defaults
import workflow, custom_cut_functions
from workflow import MyProcessor

# Ship local modules to Dask workers BY VALUE. Without this, workers fail with
# ModuleNotFoundError, because they don't have your working directory on sys.path.
cloudpickle.register_pickle_by_value(workflow)
cloudpickle.register_pickle_by_value(custom_cut_functions)

localdir = os.path.dirname(os.path.abspath(__file__))
defaults.register_configuration_dir("config_dir", localdir)   # enables ${config_dir:} in YAMLs
parameters = defaults.merge_parameters_from_files(
    defaults.get_default_parameters(),
    f"{localdir}/params/object_preselection.yaml",
    f"{localdir}/params/triggers.yaml",
    update=True,
)
```

Any helper module that a cut, weight or the workflow imports (for example `reweighting_st.py` here) also
needs `register_pickle_by_value`.

**Keep the tree relocatable.** Write every file path in `params/*.yaml` as `${config_dir:}/...`, never
absolute. Then the tree runs from any checkout and on any worker that can see it.

---

## 2. Skimming (`config_skimmer.py`)

Skim once, then run every later pass on the skims. A skim pass runs only the `skim` cuts and writes one
slimmed NanoAOD file per chunk. Its configuration has no weights, variations, variables or categories
beyond a passthrough.

```python
from pocket_coffea.lib.cut_functions import get_HLTsel, get_nPVgood, goldenJson, eventFlags
from pocket_coffea.parameters.cuts import passthrough

cfg = Configurator(
    parameters=parameters,
    datasets={"jsons": [...], "filter": {"samples": [...], "year": ["2018"]}},
    workflow=MyProcessor,
    skim=[get_nPVgood(1), eventFlags, goldenJson,
          get_HLTsel(primaryDatasets=["SingleMuon", "EGamma"]),
          nLepton_skim_cut, met_skim_cut, nJet_skim_cut],      # all AND-ed
    preselections=[passthrough],
    categories={"baseline": [passthrough]},
    weights_classes=[], weights={"common": {"inclusive": []}},
    variations={}, variables={},
    save_skimmed_files="root://<site-endpoint>//store/group/<yours>/skim/",   # XRootD destination
)
```

Rules that paid off here:
- **Skim loosely.** Make the skim looser than any region you might want later (here it is ≥1 lepton, MET > 30
  and ≥3 jets). Re-skimming costs far more than reading a few extra events.
- **Use raw NanoAOD collections in skim cuts** (`events.Muon`, `events.Jet`). Skim cuts run before
  `apply_object_preselection`, so `MuonGood` and similar collections don't exist yet.
- **After skimming, regenerate the dataset JSONs** so they point at the skim files. Keep the original
  `nevents` and `sum_genweights` normalisation in mind: PocketCoffea accumulates `sum_genweights` from the
  events it actually processes, and the skim files carry the pre-skim sums. Check this with your version
  before trusting the normalisation.
- **Data must cover each year.** For example, `SingleElectron` covers 2016/2017 and `EGamma` covers 2018. A
  missing primary dataset gives silently empty channels.
- **Private group-space skims are not served by XCache.** Read them directly from the origin, for example
  `root://cmseos.fnal.gov/`.

---

## 3. Cut functions (`custom_cut_functions.py`)

A cut is a plain function `f(events, params, **kwargs) -> boolean mask`, wrapped in a `Cut`. The params
travel with the cut, so the same function serves several regions:

```python
import awkward as ak, numpy as np
from pocket_coffea.lib.cut_definition import Cut

def in_mass_window(events, params, **kwargs):
    one_lep = (events.nMuonGood == 1) & (events.nElectronGood == 0)
    fj_msd  = ak.fill_none(ak.firsts(events.FatJetGood.msoftdrop), np.nan)
    window  = np.where(np.isnan(fj_msd), False,
                       np.abs(fj_msd - params["center"]) < params["half_width"])
    return ak.values_astype(one_lep & window, np.bool_)

sr_cut = Cut(name="sr", params={"center": 92.5, "half_width": 22.5}, function=in_mass_window)
cr_cut = Cut(name="cr", params={"center": 92.5, "half_width": 22.5},
             function=lambda ev, p, **kw: ~in_mass_window(ev, p))
```

Patterns used in this workspace:
- **Shared skeletons.** Write one function for the selection common to a family of regions (`_sr_skeleton_mu`
  here), and build each SR, CR and VR from it. Then the CR really is "the SR with one cut inverted".
- **Parameter dicts at module level** (`_SR_PARAMS`, `_W_CR_PARAMS = {**_SR_PARAMS, "mt_w_lo": 30.0}`). Change a
  window in one place.
- **Guard missing objects** with `ak.firsts`, then `fill_none(nan)`, then `np.where(isnan, False, ...)`.
  Comparing against `None` silently propagates `None` into the mask.
- **Run-dependent handling.** `hem_handling` checks `events.metadata["dataset"]` and `["year"]`, for
  example for the 2018 HEM veto.
- **Three stages of cuts:**
  - `skim` uses raw collections, before object selection
  - `preselections` runs after the workflow builds objects and is shared by all categories
  - `categories` holds one list of cuts per output category

---

## 4. Workflow (`workflow_v9.py`)

Subclass `BaseProcessorABC` and override the hooks you need. `workflow_v9.VBSSemileptonicProcessor` uses
three of them:

| Hook | Used for here | Notes |
|---|---|---|
| `process(events)` | sets `genWeight = sign(genWeight)` (UL NanoAODv9 fix), then `super().process` | Only per-file fixes belong here; the base class runs the pipeline. |
| `apply_object_preselection(variation)` | builds `MuonGood`, `ElectronGood`, `JetGood`, `FatJetGood`, loose and veto leptons, lepton–jet cleaning, b-tag collections, VBS-jet pair (max m_jj), hadronic-W candidate, indexed objects `jet1..N` / `lepton1` / `fatjet1` / `jetcentral1..4`, pairwise `deta`/`dphi`/`dR`/`mass` records, and the BDT scores | Called once per **shape variation**, so everything here is recomputed for each JES/JER shift. |
| `count_objects(variation)` | `nMuonGood`, `nJetGood`, `nBJetGood`, … | The cuts use these `n*` fields. |

Generic lessons:
- **Use the library selectors** (`lepton_selection`, `jet_selection`, `btagging` from
  `pocket_coffea.lib.objects`) driven by `params/object_preselection*.yaml`. Then add extra masks (dxy/dz,
  pileup jet ID) in the workflow.
- **Indexed objects** (`ev["jet1"] = ak.firsts(sorted[:, 0:1])`) make histogram and column definitions
  trivial (`Axis(coll="jet1", field="pt")`). Create as many as any downstream consumer expects; a BDT that
  wants `jetcentral4` crashes with `no field named ...` otherwise.
- **Nested records** (`ev["dR"] = {}`, then `ev["dR", "jet1_jet2"] = ...`) give pairwise variables their own
  collection name for `Axis(coll="dR", field="jet1_jet2")` and `ColOut("dR", [...])`.
- **In-workflow ML scoring.** The XGBoost k-fold models listed per year and region in
  `params/classifiers.yaml` are loaded, the feature names are read from the booster, and the event score is
  the mean of the folds (`ev[f"bdt_{region}"]`). **Cost:** models are loaded **per chunk** (40 here), which
  is why very small chunks are slow (§8).
- **Never mutate `events` collections that calibrators own.** Add new fields and collections instead.

---

## 5. Histogram config (`config_validation.py`)

The `Configurator` arguments, as used here:

```python
cfg = Configurator(
    parameters=parameters,
    datasets={"jsons": [...],                                  # last JSON wins on duplicate keys
              "filter": {"samples": [...], "year": [...],
                         "nfiles": {"TTToSemiLeptonic_...": 50}}},   # per-sample file cap
    workflow=MyProcessor,
    skim=[...],                       # can repeat the skim cuts on already-skimmed input (cheap)
    preselections=[presel_cut],
    categories={"baseline": [passthrough], "sr_mu": [sr_mu], "cr_mu": [cr_mu], ...},
    weights_classes=common_weights + [MyWeight, ...],          # must register custom classes
    weights={"common": {"inclusive": ["genWeight", "lumi", "XS", "sf_mu_id", ...]},
             "bysample": {"TTToSemiLeptonic_...": {"inclusive": ["top_pt_reweight"]}}},
    calibrators=default_calibrators_sequence + [MuonsRochesterCalibrator, ElectronsScaleSmearingLegacyCalibrator],
    variations={"weights": {"common": {"inclusive": ["sf_mu_id", ...]}},
                "shape":   {"common": {"inclusive": ["jet_calibration", "muons_rochester", ...]}}},
    variables={"jet1_pt": HistConf([Axis(coll="jet1", field="pt", bins=100, start=0, stop=1000, label="...")]),
               "mT_vs_pt": HistConf([Axis(...), Axis(...)], storage="weight")},   # 2D
)
```

- **Histograms are absolute yields** (`genWeight × lumi × XS / sum_genweights` applied). Don't renormalise
  downstream.
- **Cost scales with categories × variables × variations.** About 150 variations × 17 categories is why a
  pass here takes hours. For a light setup, start with `variations={}` or only the variations you need.
- **Weight names must match the registered class `name`.** For example, importing `SF_ele_trigger` from
  `weights_run2_UL` registers it as **`sf_ele_trigger_run2`**, while `sf_ele_trigger` is a different Run-3
  weight. A wrong name fails on the first chunk.

---

## 6. Columnar output (`config_semilep.py`)

The same `Configurator`, plus a `columns` block. Selected per-event values are saved, per category, into the
`.coffea` output, and are typically used for ML training:

```python
from pocket_coffea.lib.columns_manager import ColOut
columns={"common": {"bycategory": {
    "sr_mu": [ColOut("events", ["nJetGood", "mt_w_leptonic"]),
              ColOut("jet1", ["pt", "eta", "phi", "btagDeepFlavB"]),
              ColOut("dR", ["jet1_jet2", "jet1_lepton1"])],
}}}
```

- **Columns are keyed by SAMPLE** in the output, while everything else is keyed by DATASET. Mind this when
  reading them back.
- **Size is the binding constraint.** The full block here costs **2.23 KB per event**: 45 datasets came to
  77.6 GiB, and an uncapped 2017 MC set projected to about 476 GB. Size the disk *first* (`/work` has a
  100 GB quota), cap files per sample, and restrict columns to the categories you train on.
- Keep `variations` minimal in a columnar pass. Columns are saved for the nominal only, but shape variations
  still cost a full recompute.
- Merging columnar outputs needs a small `tree-reduction` (4). The default fan-in OOM'd workers at 99 %.
- The training script used here is `outputs/bdt_kfold.py`. Its models feed back through
  `params/classifiers.yaml` (§4).

---

## 7. Calibrations and custom weight classes

**Calibrators** apply object corrections and provide shape variations.
- `default_calibrators_sequence` = `JetsCalibrator` (`jet_calibration`), `ElectronsScaleCalibrator`
  (`electron_scale_and_smearing`), `MuonsCalibrator` (`muons_scale_and_resolution`) and `METCalibrator`.
- Run-2 UL here appends the legacy `MuonsRochesterCalibrator` (`muons_rochester`) and
  `ElectronsScaleSmearingLegacyCalibrator` (`electron_scale_smearing_legacy`).
- If `calibrators=` is omitted, the default sequence is used with a warning.
- The names in `variations["shape"]` must match the calibrator `name`s that are actually active. Note that
  `config_validation.py` and `config_semilep.py` currently request different lepton-calibrator names; align
  them in your own setup.
- JEC/JER tags are set in `params/jets_calibration.yaml` (`factory_config_clib`, per year, JSON from cvmfs
  `jsonpog`).
- **Validate before every campaign** that each tag exists in the cvmfs JSON. A mismatch fails the first chunk
  of every dataset and looks like a total outage (runbook §7).

To write your own calibrator, subclass `pocket_coffea.lib.calibrators.calibrator.Calibrator`:

```python
class MyCalibrator(Calibrator):
    name = "my_calib"; has_variations = True; isMC_only = True
    calibrated_collections = ["Jet.pt"]
    def initialize(self, events):           # once per chunk: load corrections, set self._variations
        self._variations = ["my_calibUp", "my_calibDown"]
    def calibrate(self, events, orig, variation, already_applied_calibrators=None):
        return {"Jet.pt": new_pt}            # never modify events in place
```

**Custom weights.** There are two forms, both in `config_validation.py`:
- **One-liners** with `WeightLambda.wrap_func(name=..., function=lambda params, metadata, events, size,
  shape_variations: ..., has_variations=...)`. Examples: `PileupWeight` (custom pileup JSON) and
  `wjet_reweight`.
- **Classes**: `WeightWrapper` subclasses with `name`, `has_variations`, `isMC_only`, and
  `compute(events, size, shape_variation)` returning `WeightData(name, nominal, up, down)`. Use
  `WeightDataMultiVariation` for several named variations. Examples:
  - `LHEScaleWeightWrapper` and `LHEPdfWeightWrapper`: theory envelopes
  - `QGTaggingWeight`, `FatJetTau21Weight`, `FatJetWvsQCDWeight`: correctionlib SFs with many sources
  - `TopPtReweight`: a yield-preserving shape weight

Lessons:
- **Load correction files lazily, once per worker.** Cache them in a module-level dict keyed by year
  (`_qgtagging_csets`). Don't load inside `compute`.
- **Shape-only SFs need renormalisation.** Compute the constant on the nominal pass and reuse it for
  shape-variation passes, which see a different event set (see `_renorm_constant`).
- **correctionlib category keys are not weight names.** The source lists (for example `_FJ_SOURCES`) must
  match keys baked into the JSON. Renaming a weight must not leak into them (`IndexError: Index not
  available in Category`).
- **Remove debug `print`s from `compute`** (`MuonGoodLeadWeight` still has some). They run on every chunk
  and every variation.

---

## 8. Running: interactive, Dask, Condor

Always go through the stages in this order. Each one catches a different class of error cheaply.

| Stage | Command | What it catches | Typical time |
|---|---|---|---|
| **0. Import check** | `python -c "import config_hists"` from the config directory | syntax, imports, `${config_dir:}` resolution, missing param keys | seconds |
| **1. Interactive test** | `pocket-coffea run --cfg config_hists.py -o test_out -e iterative --limit-files 1 --limit-chunks 1 --filter-years 2018 --filter-datasets <one_mc_key>` | every runtime bug: missing fields, weight-name or correction-tag mismatches, histogram axes | minutes. One chunk pays all the one-time setup (corrections, model loads), so **don't extrapolate throughput from it**: it understated the real rate about 5.5× here. |
| 1b. Same, on one **data** dataset | as above with a data key | data-only paths (golden JSON, per-era JEC, `isMC_only` weights) | minutes |
| 1c. `-e futures -s 4` | local process pool on the session's cores | pickling problems (an unregistered module fails here, not under `iterative`) | minutes |
| **2. Small distributed run** | `-e dask@purdue-af --custom-run-options <opts>.yaml -s 5 --limit-files 2 -ps` | worker environment, proxy visibility, gateway config, env-variable allowlist | 10–20 min including cluster start (2–7 min) |
| **3. Full campaign** | as the runbook describes | | hours to days (below) |

Notes on flags:
- `--test` / `-t` forces the `iterative` executor with `limit-files=2` and `limit-chunks=2` by default. Pass
  `--limit-files 1 --limit-chunks 1` explicitly for the smallest test.
- `--limit-files N` takes the first N files of *each* dataset. If N is at least the file count, it is
  silently a no-op. It doesn't bias normalisation (`sum_genweights` follows the processed events); it only
  costs statistics.
- `-ps` / `--process-separately` gives one output per dataset, failures are logged and the run continues,
  and resume is trivial. Without it you get one merged output, and any failure loses the whole pass.
- `--overwrite` is accepted but silently ignored.

### Which executor

| Executor | Where | Use for | Caveats |
|---|---|---|---|
| `iterative` | the session itself, one process | debugging, stage 1 | no parallelism; uses your session's memory |
| `futures` | the session itself, process pool (`-s` = processes) | small productions that fit in a few hours locally; pickling check | bounded by the cores reserved at session start |
| `dask@purdue-af` | Purdue AF Dask Gateway | production on the AF | see the next subsection |
| `condor@lxplus` / `condor@lpc` | HTCondor at CERN or FNAL | production close to EOS data | Not available from the Purdue AF (no condor there; Slurm instead). Jobs run the `custom-setup-commands` from the run options (env activation, `PYTHONPATH`) and need the proxy and code on a shared filesystem. |
| `dask@lxplus` / `dask@lpc` | Dask on HTCondor | interactive-style scale-out at CERN or FNAL | site-specific options in `executor_options_defaults.yaml` |

For `dask@purdue-af` specifically:
- It does **not** read `custom-setup-commands`. Workers get the env through `conda_env` and a fixed
  env-variable allowlist.
- Override `gateway-address` and `gateway-proxy-address` in the run options, because the built-in `-k8s`
  hostnames no longer resolve.
- You get one cluster per user.
- Put the proxy on `/work`, never `/tmp`.

### Run options that matter (production values used here)

| Option | Value | Why |
|---|---|---|
| `scaleout` | 50 | Throughput tracks **worker count**, not cores (GIL). 50 × 1 core reached 99 % utilisation; 12 × 8 cores reached 14 %. |
| `cores-per-worker` | 1–2 | More cores per worker means one bad chunk strands more tasks (`KilledWorker`). |
| `mem-per-worker` | 6 GB (histogram pass), 24–32 GB (columnar pass) | Budget at least 4 GB per task. Very large requests sit PENDING forever. |
| `chunksize` | 20000–50000 | Below about 5000, per-chunk setup dominates and runs take hours. 20000 was stable under concurrency. |
| `tree-reduction` | 4 | The default 20 OOM'd the merge with columnar output. |
| `retries` | 1–3 | The default 10 silently resubmits *everything* after a scheduler drop: the bar sits at 99 % for hours. |
| `OMP_NUM_THREADS` etc. | 1 | Otherwise BLAS takes about 22 threads per task. These need the executor allowlist patch. |

### Estimated run times (measured on this workspace, 50 workers × 1 core)

| Workload | Rate | Wall-clock |
|---|---|---|
| Histogram pass (`config_validation`, ~150 variations, 17 categories) | ~27 events/s per worker, ~1300 events/s total | one 2.6 M-event dataset ≈ 32 min |
| Columnar pass (`config_semilep`, 5 categories + BDT) | ~34 events/s per worker, ~1700 events/s total | one 4.7 M-event dataset ≈ 46 min |
| Per dataset with `-ps` | | ~10 min (small) to ~2 h (large ttbar, capped) |
| One era, histogram pass + inverted pass (~51–55 datasets each) | | ~20–24 h |
| Full Run 2, both passes (428 dataset outputs) | | ~3–4 days |
| Interactive single chunk | | minutes; dominated by setup |

These rates are for a heavy configuration. A light config (few variations, no ML scoring) runs several
times faster. Re-measure with stage 2 rather than scaling from stage 1. Also, the first 15 minutes of a
campaign (preprocessing, small datasets finishing first) overstate the rate 3–10×.

**Output size:** a histogram pass is about 100 MB per dataset (5–6 GB per era). A columnar pass is about
2.2 KB per event.

---

## 9. Common issues when running

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: workflow` on workers | local module not shipped | `cloudpickle.register_pickle_by_value(module)` for every local import |
| `ConfigKeyError: Missing key 2016_PostVFP` | a params YAML lacks a year block | add the year to that YAML |
| `AttributeError: no field named 'jetcentral1'` | a consumer (BDT, histogram, column) expects a field the workflow doesn't build | create it in `apply_object_preselection` |
| `IndexError: Index not available in Category … val: X_up` | correctionlib source list contains a key the SF JSON lacks | fix the source list; don't rename the weight |
| Every dataset fails on its first chunk | correction tag or weight-name mismatch | run the stage-1 test; validate JEC/JER tags against cvmfs |
| `Auth failed: No protocols left to try` | stale or invisible proxy | renew; copy to `/work/users/<user>/`; export `X509_USER_PROXY` |
| `OSError: XRootD … Operation expired` on open | too many concurrent opens against one origin | keep workers × cores around 50 against FNAL EOS |
| `ClientConnectorDNSError … dask-gateway-k8s` | stale gateway hostnames in the executor defaults | override `gateway-address` and `gateway-proxy-address` in the run options |
| `KilledWorker` on a processor task | per-task OOM, or one bad chunk with many neighbours | lower `cores-per-worker` and/or `chunksize`, raise `mem-per-worker` |
| `KilledWorker` on a `reduce-*` task | merge OOM | `tree-reduction: 4` |
| `scheduler-connection-lost`, then many `ClosedClientError` | client died; `-ps` keeps iterating against it | re-run: per-dataset resume picks it up |
| Bar pinned at 99 % for hours | `retries: 10` resubmitted the whole graph, or a bar not scaled to task count | `retries: 1–3`; judge progress by output files and scheduler task counts |
| Log silent, cluster RUNNING, 0 tasks | launch hang | kill, stop the cluster, relaunch; set `PYTHONUNBUFFERED=1` so stalls are visible |
| Cluster stuck PENDING | request too large, or facility-side | shrink workers or memory; check facility health |
| Empty electron channels in some years | wrong primary dataset for the year | `SingleElectron` for 2016/2017, `EGamma` for 2018 |
| Merged outputs unreadable downstream | merged with numpy 2 | merge in the same numpy 1.26 env that produced them |
| A tiny dataset runs as one chunk on one core | upstream `adapt_chunksize` bug | apply `handoff/patches/pocketcoffea_purdue_af.patch` |

Deeper write-ups: `GRID_WORKFLOW_NOTES.txt` §7 (failure modes) and `PIPELINE_RUNBOOK.md` §5 (detection
and remedies).
