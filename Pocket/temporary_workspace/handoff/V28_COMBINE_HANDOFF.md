# v28 Output Handoff — VBS Semileptonic → CMS Higgs Combine

Everything a downstream agent needs to locate, open and correctly interpret the
v28 PocketCoffea outputs when building Combine datacards.

Workspace: `/work/users/vsloken-fnal/semilep_vbs_base/Pocket/temporary_workspace`

---

## 1. Why the outputs are split across many files

The analysis was **not** run as one job. It is split along two axes:

1. **By data-taking period** — one pass per year, enforced with `--filter-years`.
   Years are never mixed inside a file. This was a deliberate requirement so that
   year combination is explicit and controlled downstream.
2. **By sample group** — samples are binned into 10 groups (`g01`…`g10`) by event
   count, run with `--filter-samples`. Heavy samples run alone; light samples are
   packed together.

Grouping exists for fault isolation, not physics. Running all samples in one job
meant a single bad chunk (`KilledWorker`) discarded the entire year's work. With
groups, a failure costs one group-year.

**There is no physics meaning to a group.** `g01` is not a process category — it
is simply "the samples that were run together". Do not treat group boundaries as
analysis boundaries.

`--process-separately` was deliberately NOT used, so each pass writes a single
combined `output_all.coffea` containing all datasets of that group-year.

---

## 2. File layout

```
<stage-prefix>_<YEAR>_<GROUP>/output_all.coffea
```

| component | values |
|---|---|
| stage-prefix | `cr_v28` = nominal selection (`config_validation.py`) — **use these**<br>`nonp_v28_<YEAR>_<GROUP>/` = inverted-lepton **intermediates — do NOT use directly** |
| YEAR | `2018`, `2017`, `2016_PostVFP`, `2016_PreVFP` |
| GROUP | `g01` … `g10` |

Examples:
```
cr_v28_2017_g03/output_all.coffea      # nominal,  2017, TTTo2L2Nu
nonp_v28_2018_g09/output_all.coffea    # inverted, 2018, 19 light samples
```

### Nonprompt: use the merged per-year files, NOT the inverted passes

The nonprompt background is delivered as **one merged file per year** at the
workspace root:

```
nonp_v28_2018_oct12.coffea
nonp_v28_2017_oct12.coffea
nonp_v28_2016_PostVFP_oct12.coffea
nonp_v28_2016_PreVFP_oct12.coffea
```

| | |
|---|---|
| sample | `nonprompt` |
| dataset | `nonprompt_<YEAR>` |
| histograms | 197 (same set as validation) |
| `cat` axis | 17 (same categories) |
| `variation` axis | **5**: `nominal`, `muon_inverttight_to_fake{Up,Down}`, `electron_inverttight_to_fake{Up,Down}` |

These carry the fake-rate-weighted estimate with its own systematics. The 37
`nonp_v28_<YEAR>_<GROUP>/output_all.coffea` files are the raw inverted-selection
passes that feed the merge — they are intermediates and must not be treated as
the nonprompt background.

Both interpreters can read the merged files, but note that *writing* merged
nonprompt files requires the `my-env` python (numpy 1.26).

Each directory also holds `config.json`, `configurator.pkl`,
`parameters_dump.yaml` (the exact configuration used) and `error/*.err` if the
pass failed.

**Target: 37 passes per stage = 74 total.** Some group-year combinations are
intentionally absent — e.g. `g04` (EGamma) exists only for 2018, because EGamma
is the 2018 electron primary dataset while 2016/2017 use SingleElectron.

---

## 3. The authoritative index: `v28_manifest.csv`

Do not infer contents from directory names. Regenerate and read the manifest:

```bash
/work/pixi/global/.pixi/envs/default/bin/python make_v28_manifest.py
# writes v28_manifest.csv (machine-readable) and v28_manifest.txt (human-readable)
```

One row **per dataset** (`sample_year[_Era]`), with columns:

| column | meaning |
|---|---|
| `dataset` | `sample_year[_Era]` — the precise unit |
| `sample` | sample name (what `variables` is keyed by) |
| `year`, `stage`, `group` | provenance |
| `file` | the `output_all.coffea` containing it |
| `n_variations` | size of the `variation` axis (1 for data, ~153 for MC) |
| `limit_files` | **files actually read** — see §6 |
| `worker_config` | cluster shape used |
| `periods_in_file` | independent check that no file mixes years |

`v28_manifest.txt` additionally lists passes with **no usable output** and the
parsed failure cause.

---

## 4. Coffea file substructure

```python
from coffea.util import load
o = load("cr_v28_2017_g03/output_all.coffea")
sorted(o.keys())
# ['columns', 'cutflow', 'datasets_metadata', 'processing_metadata',
#  'sum_genweights', 'sum_signOf_genweights', 'sumw', 'sumw2', 'variables']
```

### 4.1 `variables` — the histograms

**Two levels of nesting, and the keys differ at each level:**

```
o["variables"][<histogram>][<SAMPLE>][<DATASET>]  ->  hist.Hist
                            ^^^^^^^^  ^^^^^^^^^
                            sample    sample_year[_Era]
```

~197 histograms (`bdt_resolved_mu`, `HT_sum`, `DeepMETResponseTune_pt`, …).

Each `Hist` has three axes:

| axis | size | contents |
|---|---|---|
| `cat` | 17 | analysis category (§5) |
| `variation` | 153 (MC) / 1 (data) | `nominal` + systematic up/down (§7) |
| observable | varies | the quantity being histogrammed |

Extract a single template:

```python
h = o["variables"]["bdt_resolved_mu"][sample][dataset]
nominal = h[{"cat": "w_cr_mu", "variation": "nominal"}]
up      = h[{"cat": "w_cr_mu", "variation": "sf_btag_hfUp"}]
```

### 4.2 Keys that are NOT sample-keyed

| object | keyed by | notes |
|---|---|---|
| `variables[hist]` | **sample**, then dataset | two levels |
| `cutflow[category]` | **dataset** | 20 entries incl. `initial`, `skim`, `presel` |
| `sumw[category]`, `sumw2[category]` | **dataset** | 17 categories (no `initial`/`skim`/`presel`) |
| `sum_genweights` | **dataset** | normalization denominator |
| `sum_signOf_genweights` | **dataset** | used instead for amcatnlo-type samples |
| `datasets_metadata.by_dataset` | **dataset** | xsec, nevents, skim factors |
| `datasets_metadata.by_datataking_period` | **year** → list of samples |

> **Pitfall.** `variables` is keyed by *sample*, so the year appears only in the
> inner dataset key, the directory name, and `by_datataking_period`. Naively
> merging `variables` dicts across year-directories will silently stack different
> years under one sample key. Always carry the year explicitly.

### 4.3 `datasets_metadata.by_dataset[dataset]`

```
das_names, isMC, isSkim, nevents, sample, size,
skim_efficiency, skim_rescale_genweights, xsec, year
```

---

## 5. Categories (`cat` axis, 17)

```
baseline
resolved_mu            resolved_e            boosted_mu            boosted_e
w_cr_mu                w_cr_e                w_cr_boosted_mu       w_cr_boosted_e
ttbar_cr_resolved_mu   ttbar_cr_resolved_e   ttbar_cr_boosted_mu   ttbar_cr_boosted_e
vr_mu                  vr_e                  vr_boosted_mu         vr_boosted_e
```

`cutflow` additionally carries `initial`, `skim`, `presel` (not in `sumw`/`cat`).

---

## 6. Normalization — VERIFIED, read this before building datacards

**Histograms are absolute expected yields.** The weight list applied in
`config_validation.py` includes `genWeight`, `lumi` and `XS`, so
luminosity × cross-section normalization is already in the histogram contents.
Do **not** re-apply lumi or xsec.

**`--limit-files` does NOT bias yields.** Every MC pass was run with
`--limit-files 50` (some retries used 30 or 20 — see `limit_files` in the
manifest). This reads only a subset of each dataset's files. It is safe because
`sum_genweights` is **accumulated from the events actually processed**
(`PocketCoffea/pocket_coffea/workflows/base.py:806`), not taken from
full-dataset metadata, so the `lumi·XS/sum_genweights` normalization is
self-consistent with whatever subset was read.

Verified empirically on `TTTo2L2Nu_..._2017`:

```
output  nevents   19,202,224
json    nevents   47,237,472     ratio = 0.4065
--limit-files 50 / 123 files     ratio = 0.4065     exact match
```

**Consequence:** `--limit-files` costs *statistical precision*, not accuracy.
MC statistical uncertainties are larger than full-dataset running would give,
and the effective MC stats differ between datasets capped at 50 vs 30 vs 20.
For Combine this matters for MC-stat treatment (e.g. `autoMCStats`) — consult
`limit_files` per dataset rather than assuming a uniform cap.

**Skim rescaling** is handled internally: these are skimmed inputs
(`isSkim: True`), and `skimRescaleGenWeight` (`base.py:223`) corrects
`sum_genweights` for skim efficiency. `skim_efficiency` and
`skim_rescale_genweights` in metadata are for cross-checks.

**Data files** have no `sum_genweights` and a single `nominal` variation.

---

## 7. Systematic variations (`variation` axis)

Axis size depends on the sample, **verified across all 201 validation datasets**:

| variations | datasets | what |
|---|---|---|
| **153** | 8 | `TTTo2L2Nu`, `TTToSemiLeptonic` (all 4 years) — includes `top_pt_reweight` Up/Down |
| **151** | 172 | all other MC (43 samples) |
| **1** | 21 | data (`SingleMuon`, `EGamma`) — `nominal` only |

The 153 vs 151 difference is **exactly** `top_pt_reweightUp` / `top_pt_reweightDown`,
a `bysample` weight applied only to the TT samples (diff verified empirically).

The nonprompt background is separate: the merged `nonp_v28_<YEAR>_*.coffea` files
carry **5** variations (fake-rate systematics), and the raw inverted passes carry 9.
Neither should be compared against validation axis sizes.

153 = `nominal` + 76 families × (Up, Down).

| bucket | families | examples |
|---|---|---|
| JES/JER AK4 | 12 | `AK4PFchs_JER`, `AK4PFchs_JES_Regrouped_Absolute_2017`, `…_FlavorQCD`, `…_RelativeBal` |
| JES/JER AK8 | 12 | `AK8PFPuppi_JER`, `AK8PFPuppi_JES_Regrouped_BBEC1_2017`, … |
| lepton / other SF | 21 | `sf_mu_id`, `sf_ele_reco`, `sf_ele_trigger_run2_sf`, `sf_jet_puId`, `sf_L1prefiring`, `sf_partonshower_isr/fsr`, `sf_qgtagging_*` |
| fat-jet tagger SF | 14 | `sf_fj_WvsQCD_sf_fj_WvsQCD_{stat,pileup,sf_btag,JES,JER,…}`, `sf_fj_tau21_*` |
| b-tag SF | 8 | `sf_btag_{hf,lf,cferr1,cferr2,hfstats1,hfstats2,lfstats1,lfstats2}` |
| LHE / theory | 4 | `LHEScaleWeight_{renorm,fact}_scale`, `LHEPdfWeight_{pdf,alpha_S}` |
| detector / other | 4 | `PileupWeight`, `muon_roccor`, `ele_scale_legacy`, `ele_smear_legacy` |
| top pT | 1 | `top_pt_reweight` — **TT samples only** |

Axis labels are `<family>Up` / `<family>Down`.

**Correlation guidance for datacards:**
- Families ending `_<YEAR>` (e.g. `AK4PFchs_JES_Regrouped_Absolute_2017`) are
  **year-decorrelated** by construction — keep them uncorrelated across years.
- Families without a year suffix are intended to be **correlated** across years.
- The doubled prefixes (`sf_fj_WvsQCD_sf_fj_WvsQCD_stat`) are a PocketCoffea
  naming artifact — the weight name is prepended to its own variation list.
  Cosmetic only; the same convention appears in `config_validation.py`.
- `top_pt_reweight` exists only on TT samples, so the `variation` axis is not
  identical across all datasets. **Do not blindly add histograms from different
  samples** — select the variation subset explicitly first. This caused an
  `axes not mergable` failure earlier in this analysis.

---

## 8. Known gaps and caveats

| item | status |
|---|---|
| Nonprompt / inverted set | **complete, 37/37** |
| Nominal / validation set | 30/37 at last check; failures being retried with reduced `--limit-files` |
| `g04` (EGamma) | 2018 only — correct, not a gap |
| single-top, ttW/ttZ in 2016/2017 | present via `datasets/skimmed_dataset_definition_top1617.json`, added mid-run. Passes `g08`/`g09`/`g10` for 2016/2017 include them. |
| TTToSemiLeptonic 2016_PostVFP | repeatedly `KilledWorker`; may end up at reduced `limit_files` or missing — **check the manifest** |

Before building datacards, confirm from `v28_manifest.txt` that every
(process, year, category) you need is actually present. The manifest's trailing
"PASSES WITH NO USABLE OUTPUT" section is the definitive list of what is missing.

---

## 9. Minimal extraction recipe

```python
import glob, re, os
from coffea.util import load

def iter_v28(stage="cr_v28"):
    """Yield (year, group, sample, dataset, file, output)."""
    for d in sorted(glob.glob(f"{stage}_*_g*")):
        f = os.path.join(d, "output_all.coffea")
        if not os.path.exists(f):
            continue                      # failed pass — see manifest
        year, group = re.match(rf"{stage}_(.+)_(g\d+)$", d).groups()
        o = load(f)
        for hist in o["variables"]:
            for sample, by_ds in o["variables"][hist].items():
                for dataset in by_ds:
                    yield year, group, sample, dataset, f, o

def template(o, hist, sample, dataset, category, variation="nominal"):
    h = o["variables"][hist][sample][dataset]
    return h[{"cat": category, "variation": variation}]
```

Environment: use `/work/pixi/global/.pixi/envs/default/bin/python` (has coffea),
not bare `python`.

---

## 10. Provenance

| | |
|---|---|
| configs | `config_validation.py` (nominal), `config_inverted.py` (nonprompt) |
| workflow | `workflow_v9.py` (`VBSSemileptonicProcessor`) |
| dataset jsons | `rare_skim.json`, `skimmed_rescale.json`, `skimmed_dataset_definition_rare1617_fixed.json`, `skimmed_dataset_definition_top1617.json`, `TTToSemiLeptonic_2016_PreVFP_skim.json` |
| group definitions | `v28_groups.txt` (one comma-separated group per line) |
| pass queue | `v28_passes.txt` (`group \t year \t samples`) |
| runner | `run_v28_hybrid2.sh` (retry ladder 50 → 30 → 20 files on worker death) |
| cluster opts | `custom_50w2c_heavy.yaml` (heavy groups), `custom_50w8c_mt.yaml` (light) |
| fake-lepton weights | `params/fakelepton_weights_noiso_3j.yaml` (all 4 years) |
| inputs | FNAL skims, `root://cmseos.fnal.gov//eos/uscms/store/group/lnujj/semilep_vbs_skim/` |
