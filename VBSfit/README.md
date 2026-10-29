# VBSfit — histograms, datacards and CMS Combine fits

Everything downstream of the PocketCoffea processing: it turns the `.coffea` outputs of
[`../Pocket/temporary_workspace`](../Pocket/temporary_workspace/README.md) into ROOT histograms, builds CMS
**Combine** datacards for a simultaneous signal-region + control-region fit, runs the expected-sensitivity and
data-constrained fits, produces the diagnostic plots, and combines the four Run-2 year-eras.

## The current Combine workflow — which script do I run?

| I want to… | Run | Notes |
|---|---|---|
| Process **one year** end to end (histograms → datacards → fits → plots) | **`./run_all_each_year.sh --year=<YEAR>`** | year-tagged outputs: one datacard per category per year |
| Fit **all Run-2 year-eras together** (per category and combined) | **`./run_run2_fits.sh`** | regenerates per-year input cards with the cross-year correlation scheme, combines, checks, fits |
| Study the **BDT binning** (flat-signal bins, N = 4…8) | **`./run_binning_scan.sh`** (+ `python/make_binning_scan.py`) | runs the full per-year workflow once per binning variant, in parallel |
| Split the signal into WW / WZ strengths | `./run_multipoi.sh` | multi-POI sensitivity |
| Check the QvG-tagger systematic against data | `./run_qvg_postfit_check.sh` | postfit QvG score distributions with nested bands |

`run_all.sh` is the same per-year pipeline **plus** the options the binning scan needs
(`--binning`, `--tag`, a cross-process lock for the memory-heavy steps, and EOS flush barriers) — `run_binning_scan.sh`
calls it. `run_all_cr_data.sh`, `run_all_minos_hesse_patched.sh` and `run_minos_cr.sh` are earlier snapshots of
the same driver kept for history ([§13](#13-historical-driver-snapshots)).

## Table of contents

1. [Design principles](#1-design-principles)
2. [Environments and conventions](#2-environments-and-conventions)
3. [Directory layout](#3-directory-layout)
4. [Inputs: manifest → per-year config](#4-inputs-manifest--per-year-config)
5. [The per-year pipeline](#5-the-per-year-pipeline-run_all_each_yearsh)
6. [Datacard anatomy and decorrelation policy](#6-datacard-anatomy-and-decorrelation-policy)
7. [Fit methodology](#7-fit-methodology)
8. [The Run-2 combination](#8-the-run-2-combination-run_run2_fitssh)
9. [BDT binning studies](#9-bdt-binning-studies)
10. [Multi-POI (WW / WZ) sensitivity](#10-multi-poi-ww--wz-sensitivity)
11. [Script reference](#11-script-reference)
12. [Troubleshooting](#12-troubleshooting)
13. [Historical driver snapshots](#13-historical-driver-snapshots)
14. [Status notes (dated)](#14-status-notes-dated)

---

## 1. Design principles

* **Per year, then combine.** Each year-era is processed and fitted independently; years are combined only in
  the last step, by name-based correlation (§8).
* **Free-floating background normalizations, constrained by control regions.** No scale factors are
  pre-applied. `norm_top_<topology>` and `norm_wjet_<topology>` are `rateParam`s (range `[0.2, 5]`) tied to the
  `TTbar` and `WJets` processes and constrained by the single-bin t̄t and W+jets control regions, so the CR
  statistical uncertainty propagates into the signal strength.
* **The signal region stays blind.** Fits on data use a **hybrid dataset**: real data in the CRs, MC
  pseudo-data in the SRs (§7.3). Results quoted from it are sensitivity estimates under the CR-measured
  normalizations, not measurements.
* **Fail loudly.** Hard aborts instead of silent fallbacks: duplicate nuisance rows or clobbered ROOT keys abort,
  a card `rateParam` with no matching norm aborts the hybrid build, missing masks abort the CR-only fits, and
  every combine call keeps its complete log (a `| tail` once hid for months the one warning that explained
  zero-error postfit bands).
* **Judge fits by values and errors, never by status words.** With `--robustHesse` the saved status reads
  `300`/`-1` even for good fits.

---

## 2. Environments and conventions

Two incompatible environments, switched by the scripts themselves — run the drivers from a plain interactive
shell **inside `tmux`**; do not pre-source either:

| Environment | Provides | Lacks | Used for |
|---|---|---|---|
| **CMSSW_14_1_0_pre4** (`source /cvmfs/cms.cern.ch/cmsset_default.sh; cd CMSSW_14_1_0_pre4/src; eval $(scramv1 runtime -sh)`) | `combine`, `combineTool.py`, `combineCards.py`, `text2workspace.py`, PyROOT, `plotImpacts.py` | `uproot`, `coffea`, `mplhep` | all fits, `extract_postfit_params.py`, `read_cr_norms.py` |
| **PocketCoffea Apptainer image** (`…/pocketcoffea:lxplus-el9-latest`) via the `pyrun` wrapper | `uproot`, `coffea`, `numpy`, `matplotlib`, `mplhep` | `combine` | `make_histograms*.py`, `make_datacards.py`, `make_hybrid_data.py`, all plotters |

`pyrun` binds `/eos /afs /cvmfs /tmp`, forwards Kerberos (`KRB5CCNAME` is inherited — **never override it with a
path that does not exist**, which caused `Permission denied` on EOS), and `cd`s into `VBSFIT_DIR`.

Conventions every driver follows:

* **Full logs for every combine call** (`run_combine_logged`: `stdbuf -oL` line-buffered, complete log, `tail -20`
  on failure). Never pipe a fit through `| tail` under `set -e`: it hides both the exit status and mid-log warnings.
* **EOS flush barriers.** ROOT silently redirects `/eos` FUSE paths to xrootd (`TFile.CrossProtocolRedirects`).
  A file just written through FUSE can be opened by combine through xrootd before EOS has committed the whole
  file → `TNetXNGFile … is truncated at N bytes: should be M`. `wait_eos_xrootd_sync` polls `xrdfs stat` (via the
  file's `eos.url.xroot` xattr) until the sizes match; it warns and continues if `xrdfs` cannot measure at all.
* **Absolute paths everywhere** for card inputs: `combineCards.py` rewrites each `shapes` line by prefixing the
  input card's directory *as given*, so relative inputs produce combined cards that resolve only from one cwd.
  `check_shapes_resolve` verifies every shapes file exists right after cards are built.
* **Run labels.** `README.txt` files dropped into result directories say whether a result is the **MC-only
  Asimov reference** (`mconly`) or **CR-data** (`crdata`) — the two sets of plots look alike and must not be
  confused.
* Memory-heavy coffea/uproot steps take a cross-process lock (`PYRUN_LOCK`) when the binning scan runs
  variants in parallel; the lxplus per-user memory cgroup fits one of them at a time.

---

## 3. Directory layout

```
VBSfit/
├── run_all_each_year.sh        per-year pipeline (current)
├── run_all.sh                  per-year pipeline + --binning/--tag/locking (driven by the binning scan)
├── run_run2_fits.sh            full Run-2 combination fits
├── run_binning_scan.sh         BDT binning variants, in parallel
├── run_multipoi.sh             WW / WZ multi-POI sensitivity
├── run_qvg_postfit_check.sh    QvG tagger postfit check
├── run_impacts_one.sh          impacts for one card (MC Asimov)
├── run_impacts_one_cr.sh       impacts for one card, optional dataset/params (used for hybrid cards)
├── run_all_cr_data.sh, run_all_minos_hesse_patched.sh, run_minos_cr.sh   historical snapshots
├── config/
│   ├── inputs_v28_<YEAR>.yaml  per-year inputs (generated from the v28 manifest)
│   ├── binning_example.yaml    default SR BDT bin edges
│   ├── binning_scan_2018/      flat{4..8}bins.yaml (generated by the scan)
│   └── inputs_2018.yaml, inputs_2022postEE.yaml   pre-v28 hand-written configs (reference)
└── python/                     the library (see §11)
```

Generated trees (all tagged with the run label `<RUNTAG>` = `<YEAR>` or `<YEAR>_<tag>` for scan variants):

| Path | Content |
|---|---|
| `histograms/histograms_cr_<RUNTAG>.root` | 12 channels = 4 SRs × {SR, t̄t CR, W CR}; plus `histograms_kinematics_cr_<RUNTAG>.root`, `histograms_mjj_<RUNTAG>.root`, `histograms_native_<YEAR>.root` (40-bin BDT, for the scan) |
| `cards_cr_<RUNTAG>/{combined,resolved_mu,resolved_e,boosted_mu,boosted_e}/datacard.txt` | the per-category cards (+ `histograms_decorrelated.root` alias file, `cr_only/` CR-fit results) |
| `cards_crdata_srasimov_<RUNTAG>/<target>/` | **hybrid** cards + `histograms_hybrid.root` (targets: `combined resolved boosted resolved_mu resolved_e boosted_mu boosted_e`) |
| `plots_cr_<RUNTAG>/` | stacked/prefit plots, syst-ratio plots, fitdiagnostics, postfit plots, impacts, breakdowns, `significances_m125.txt`, and `crdata_srasimov/` (hybrid-fit results, closure plots) |
| `cards_mjj_<RUNTAG>/` | mjj sanity-check cards (`--with-mjj`) |
| `cards_run2_input_<YEAR>/`, `cards_run2/`, `plots_run2/`, `logs_run2/` | Run-2 combination (§8) |
| `logs_binning_scan_<YEAR>/`, `plots_binning_scan_<YEAR>/` | binning scan logs and s/√b plots (§9) |

---

## 4. Inputs: manifest → per-year config

The v28 PocketCoffea production is split per **year × sample group** (`<stage>_<YEAR>_<GROUP>/output_all.coffea`,
several samples per file, no physics meaning to the groups). The only reliable sample→file mapping is the
production's `v28_manifest.csv`, so the per-year configs are **generated, not hand-edited**:

```bash
python3 python/make_inputs_from_manifest.py \
    --manifest ../v28/v28_manifest.csv --outdir config --base-dir ../v28
# → config/inputs_v28_{2018,2017,2016_PreVFP,2016_PostVFP}.yaml
```

Options: `--stage-nominal` (default `validation`; the error message lists the stage values actually present),
`--stage-nonprompt` (default `nonprompt`), a glob **fallback** for nonprompt years the manifest does not cover
(`--nonprompt-dir`, `--nonprompt-pattern 'nonp_v*_{year}_*.coffea'`, `--nonprompt-name`), and `--check-files`.
Classification: `dipoleRecoil` in the sample name → signal, `n_variations == 1` → data, else background.
Entries carry `files`, `datasets`, `group` and `limit_files`; an optional `year_guard: false` disables the dataset-year
filter for one entry. Regenerate after any change to the v28 outputs or when paths move.

How the loaders treat these files (`make_histograms.py`):

* multi-GB group files are loaded through an LRU cache (`V28_CACHE_SIZE`, default 4) so each is deserialized once;
* the inner dataset key (`<sample>_<year>[_Era]`) is matched against the target year (eras are summed); an
  all-filtered result produces a warning;
* absolute normalization is already in the histograms (no lumi/xsec re-application);
* `variation` axes are never merged across samples (153 / 151 / 5 / 1 — see the workspace README).

Merged process names in the ROOT file: `TTbar` (TTTo2L2Nu + TTToSemiLeptonic), `WJets` (all HT bins), `DY`
(both mass bins); single-top, diboson and signal processes keep their own names; `nonprompt` is the merged
data-driven estimate; data comes from `SingleMuon` and `EGamma` (plus `SingleElectron` in 2016/2017).

---

## 5. The per-year pipeline (`run_all_each_year.sh`)

```bash
./run_all_each_year.sh --year=2018            # everything (default year 2018)
./run_all_each_year.sh --year=2017 --cr-only  # steps 5c-5e + breakdowns (cards must exist)
./run_all_each_year.sh --skip-cr              # everything except 5c-5e
./run_all_each_year.sh --breakdown-only       # ONLY the uncertainty breakdowns (5b + 5f)
./run_all_each_year.sh --skip-breakdown       # everything except 5b + 5f
./run_all_each_year.sh --with-mjj             # also the mjj sanity fits (step 9)
./run_all_each_year.sh --mjj-only             # ONLY the mjj sanity fits
```

`run_all.sh` accepts the same flags and in addition `--binning=<yaml>` (SR BDT bin edges; default
`config/binning_example.yaml`) and `--tag=<name>` (suffixes **every** output path with `_<name>` after the year, so
variants of one year can run in parallel without colliding). The working copy of `run_all.sh` also has
`--targets=boosted,boosted_mu,…` to restrict steps 5c–5f to a subset of fit targets (the closure needs `combined`
in the list); check with `grep -- --targets run_all.sh`.

| Step | What it does | Main outputs |
|---|---|---|
| 1 | `make_histograms.py`: coffea → ROOT for 12 channels. CRs collapse to **one counting bin**; SR BDT rebinned to `--binning` edges (must be a subset of the native 40-bin edges), else uniform. | `histograms/histograms_cr_<RUNTAG>.root` |
| 2 | `make_datacards.py`: one card per category (SR + t̄t CR + W CR, `imax 3`) and the `combined` card (`imax 12`) with `rateParam`s. A post-build check confirms the rateParams reached the workspace (`check_rateparams.py`). | `cards_cr_<RUNTAG>/…/datacard.txt` |
| 3 | `make_histograms_kinematics.py` + `plot_syst_ratios.py`: kinematic histograms and systematic ratio plots (signal and background shown separately). | `plots_cr_<RUNTAG>/syst_ratios/` |
| 4 | **Expected significance, MC-only Asimov** (`-M Significance -t -1 --expectSignal 1`, strategy-2 minimizer with fallbacks) for combined and each category. | `significances_m125.txt` |
| 5 | **FitDiagnostics, MC Asimov**, combined, with `--saveShapes --saveWithUncertainties --numToysForShapes 200`, strategy 0 + `--robustHesse 1 --robustFit 1`, `--rMin -2 --rMax 2`. | `plots_cr_<RUNTAG>/fitdiagnostics/` |
| **5c** | **CR-only fits on real data with the SRs masked** (`--channel-masks`, `--setParameters mask_<bin>=1`, `r` frozen, constrained nuisances frozen, strategy 0 + `--minos all` + `--robustHesse 1`); one fit per `CR_SOURCE` case: per category, per topology, and `shared`. Gives the central `norm_*` values. | `cards_cr_<RUNTAG>/cr_only/cr_only_norms.txt` |
| **5d** | **Build the hybrid dataset** (`make_hybrid_data.py`): CR `data_obs` untouched, SR `data_obs` replaced by `r·signal + Σ scale_p·nominal_p` at the 5c norms (`INJECT_SIGNAL=1.0` → S+B). Verifies after every build that CR data is bit-identical and SR data was replaced; hard-aborts if a card `rateParam` has no `--norms` value. | `cards_crdata_srasimov_<RUNTAG>/<target>/` |
| **5e** | On the hybrid cards: significance, FitDiagnostics, impacts, closure inputs (§7). | `plots_cr_<RUNTAG>/crdata_srasimov/` |
| 5b / **5f** | **Uncertainty breakdown per category** (`uncertainty_breakdown.py`); 5b = MC-only (off by default), 5f = with CR data (default, `BREAKDOWN_RUNS=crdata`). ~7000 likelihood-scan minimizations in total — trim `BREAKDOWN_TARGETS` while iterating. | `uncertainty_breakdown_<target>/` |
| 6 | Stacked prefit plots (`plot_stacked.py`, data blinded). | `plots_cr_<RUNTAG>/stacked/` |
| 7 | Postfit plots (`plot_postfit.py`) for the MC-Asimov fit and for the hybrid fit (`--dataset crdata --blind-above 0`). | `plots_cr_<RUNTAG>/postfit/` |
| **5g** | **CR closure**: `extract_postfit_params.py` (CMSSW) dumps pulls/rateParams/covariance to JSON, `plot_cr_closure.py` (apptainer) rebuilds each CR at full BDT binning and compares real CR data with the pre- and post-fit prediction (the fit only sees the CR as one bin, so its *shape* is an independent check). | `crdata_srasimov/cr_closure/` |
| 8 | Impact plots, MC-Asimov (`run_impacts_one_cr.sh`). | `plots_cr_<RUNTAG>/impacts_*` |
| 9 | `--with-mjj`: rebuild histograms/cards on `mjj_vbs` (20 equal-width SR bins) as a no-BDT sanity check, Asimov significance only. | `significances_mjj_m125.txt` |

Cost notes: the sampling stage of shape-saving FitDiagnostics fits dominates and is linear in
`--numToysForShapes` (1000 toys ≈ 12 h on the combined card; the default of 200 keeps it manageable — raise only for
final plots). Steps use `run_impacts_one_cr.sh` with an **empty** 4th argument for hybrid cards, which fits the
card's stored dataset (adding `-t -1` would overwrite the CR data with Asimov).

---

## 6. Datacard anatomy and decorrelation policy

**Channels and processes.** Per category: `{sr}` (multi-bin BDT), `ttbar_cr_{sr}`, `w_cr_{sr}` (one bin each).
Signal processes are listed first (process indices `0, -1, -2, …`); all share the signal-strength POI `r`.
Backgrounds follow in order, dropped per channel if their yield is below `1e-3`.

**RateParams** (the normalization model): `norm_top_<topology>` → `TTbar` and `norm_wjet_<topology>` → `WJets`,
`rateParam <bin-pattern> <process> 1 [0.2,5]`, with topology split `resolved` / `boosted`
(`--decorrelate-norms topology`, the default; `none` gives one shared pair). Bin patterns are wildcards
(`*resolved*`, `*boosted*`) on purpose: `combineCards.py` prefixes bins (`ch1_`, `Y2017_`) without rewriting rateParam
bins. The explicit range matters: an unbounded norm breaks `--robustFit` (silently returning the central value for
both crossings and dropping the rate parameters from impact plots).

**Statistics and luminosity.** `autoMCStats 10` (Barlow–Beeston → `prop_bin*`); `lumi_<year> lnN 1.073` per year-era
(`LUMI_UNC_BY_YEAR`; `--require-lumi` aborts instead of falling back when an era has no value).

**Shape nuisances.** Every systematic with Up/Down templates becomes a `shape 1` row. Families: `AK4PFchs_*`/`AK8PFPuppi_*`
(JES/JER), `sf_btag_*`, `sf_qgtagging_*`, `sf_fj_*`, `sf_jet_puId`, lepton ID/iso/reco/trigger, `PileupWeight`,
`sf_L1prefiring`, `sf_partonshower_{isr,fsr}`, `LHEScaleWeight_*`, `LHEPdfWeight_*` (signal and diboson).

**Decorrelation rules** (`make_datacards.py`):

* **Per-MC-group (theory families).** `sf_partonshower_isr/fsr` and `LHEScaleWeight_fact_scale/renorm_scale` are split
  into one nuisance per MC group — `signal`, `TTbar`, `WJets`, `DY`, `SingleTop`, `Nonprompt`, `Diboson` — because a
  shower variation of t̄t says nothing about the signal's. The renamed templates live in an **augmented copy of the
  histogram file** (`cards_*/histograms_decorrelated.root`) that the cards point at; `--decorrelate-shower none` keeps one
  correlated nuisance each.
* **Per-year (`--decorrelate-years`, default `year`).** Every nuisance name gets a `_<year>` suffix **except** names that
  already end in `_<year>` and names matching `YEAR_CORRELATED_PATTERNS` — the allowlist, currently the JME families
  (`AK4PFchs_JES_*`, `AK8PFPuppi_JES_*`, `AK?_JER*`), whose native names already encode the intended correlation (base
  components correlated across years, `*_<year>` components not). `--decorrelate-years none` writes bare names — exactly
  the scheme the Run-2 combination needs (§8). A collision-avoiding `_yr<year>` fallback exists for the case where the
  tagged name is already taken by a different nuisance.
* **Symmetrization and renormalization** happen in `make_histograms.py` (`symmetrize_syst`: per category and process,
  never on CRs; `nom ± max(|up−nom|,|down−nom|)`; year-aware targets; dead entries abort). Scale/shower rate components on
  `TTbar`/`WJets` are removed per topology since the free norms already absorb them.

**Guards that protect the fit** (each exists because the failure was silent or confusing):

| Guard | Why |
|---|---|
| duplicate nuisance names abort (`guard_unique`); alias targets clashing with existing ROOT keys abort | duplicate ROOT keys/rows are read as "last cycle wins" and corrupt cards silently |
| a variation is written only if its **signed** Up/Down integrals exceed `1e-4 ×` the signed nominal; processes with signed nominal ≤ 0 are dropped | combine's kappas use the raw `TH1::Integral()` and aborts with `Bogus norm`/`Null norm` on negative-weight cancellations; a printed note lists every drop |
| `--require-lumi`, per-era lumi table | no silent placeholder lumi in combined fits |
| `check_shapes_resolve` after every card build | relative-path trap of `combineCards.py` |

---

## 7. Fit methodology

### 7.1 Minimizer settings (and why)

* FitDiagnostics: `--cminDefaultMinimizerStrategy 0 --robustHesse 1 --robustFit 1 --rMin -2 --rMax 2`.
  Plain HESSE cannot produce a covariance that `--saveWithUncertainties` accepts on these cards: combine then prints
  *"The option --saveWithUncertainties will be ignored"* mid-log and writes every shape with **zero bin errors**
  (`check_band_saved` aborts on that line). `--robustHesse` (finite-difference covariance) fixes the covariance;
  `--robustFit` supplies the NLL-crossing error on `r`. The `[−2, 2]` range keeps the minimum away from combine's default
  `r ≥ 0` boundary.
* `-M Significance` uses strategy 2 with fallbacks (2 → 1, never 0) — it is the one place where the r = 0 fit must reach
  the true minimum.
* CR-only fits (5c): `r` frozen, only the norms float → `--minos all` is cheap; `--robustHesse` symmetric errors are the
  fallback.
* **On the hybrid *combined* card, `-M Significance` is untrustworthy** (its internal r = 0 fit strands ~60 NLL units high →
  Z = 11.7 instead of 4.16). Hybrid significances are therefore taken from a shapeless FitDiagnostics via
  `Z = sqrt(2·(nll_min(fit_b) − nll_min(fit_s)))` per target. Sanity guard: a combined Z above the quadrature sum of the
  per-topology Z values means a broken fit. Any σ(r) not read from a scan should not be quoted on the resolved-containing
  cards (the breakdown's grid scan is the trusted estimator).

### 7.2 Uncertainty breakdown

`uncertainty_breakdown.py`: group contribution = `sqrt(σ_total² − σ_with_group_frozen²)`, iterating
`MultiDimFit --algo grid` (41 points by default) with different freeze lists per named group (regex/list), including a
dedicated *Rate params* group (norms). All scans load a **post-fit snapshot** (`--saveWorkspace` once, then
`--snapshotName MultiDimFit`) — without it `--freezeParameters` pins at *prefit* values and the table clamps to exact zeros.
`rgx{…}` patterns are validated against the workspace; a frozen scan wider than the total (beyond 2 %) is flagged `(!)`
instead of silently clamping. Interpretation: the "Data statistical" row overlaps the "Rate params" row by design (norms
float in the stat-only fit). `--freezeParameters` wins over `--floatParameters` for overlapping names, so the MC-stat
group uses an explicit complementary freeze list.

### 7.3 Hybrid dataset and blinding

`make_hybrid_data.py` writes a copy of the shapes file whose SR `data_obs` is the MC expectation at the CR-fit norms
(+ injected signal) and a card pointing at it. The hybrid fits then run with **no `-t` and no `--expectSignal`**; the
norms stay floating so the CR data constrains them and its uncertainty reaches `r`. Because the SR pseudo-data is built
*at* the CR norms, the SR adds no independent norm information by construction. `check_hybrid_norms.py` audits existing
hybrids by recovering the injected norms algebraically from the files and comparing with the 5c table (it flags SRs
accidentally built at 1.0); `read_cr_norms.py` emits the fitted `norm_*=value` string (names verbatim, errors to stderr,
nonzero exit on unusable values — never defaults to 1.0).

### 7.4 Postfit uncertainty bands and closure

`plot_postfit.py` reads `shapes_fit_s` with `total_background`/`total_signal` bin errors, which combine fills by sampling
the full postfit covariance (all constrained nuisances, rateParams, autoMCStats). A zero-error band is detected, stamped
on the figure, and warned about. The closure plotter (`plot_cr_closure.py`) propagates the **covariance** through a linear
shape model (`Var = J·Cov·Jᵀ`) at the full CR BDT binning; fitted nuisance names (year/`_yr`/MC-group tags) are mapped
back to raw template names, MC-group-tagged pulls are applied only to their group's processes, and the pre-fit panel uses
the post-fit norms so the two panels differ **only** by nuisance pulls.

---

## 8. The Run-2 combination (`run_run2_fits.sh`)

The standard per-year cards are year-tagged on (almost) every nuisance, so combining them directly would decorrelate
too much. `run_run2_fits.sh` therefore **regenerates lightweight per-year input cards** from the existing histogram files with
`make_datacards.py --decorrelate-years none` (uproot only, minutes, no coffea, per-year trees untouched) and then combines:

| Class | Correlation across years |
|---|---|
| experimental nuisances **with a year label** (`AK4PFchs_JES_Regrouped_EC2_2017`, `AK4PFchs_JER_2018`, `lumi_2016_PreVFP`, …) | **decorrelated** — distinct names never merge |
| experimental nuisances **without** a year label (`sf_btag_*`, triggers, pileup, …) | **correlated** — one shared nuisance |
| **theory** (LHE scales, PDF, parton shower incl. the per-MC-group split) | **correlated** |
| **normalizations** `norm_top_<topo>`, `norm_wjet_<topo>` | **correlated** — one parameter per topology floats that background in every year; the `*resolved*`/`*boosted*` wildcard bins now deliberately span all years |
| luminosity | `lumi_<era>` per era (decorrelated), each 1.073 |

The two 2016 halves share the bare `*_2016` JME components as **one** nuisance.

```bash
./run_run2_fits.sh                          # all 4 year-eras, categories + combined
./run_run2_fits.sh --years="2018 2017"      # a subset (≥ 2 eras with histograms + inputs config)
./run_run2_fits.sh --categories="combined"  # re-run just one category
./run_run2_fits.sh --jobs=2 --sig-only      # significances only, no FitDiagnostics
./run_run2_fits.sh --cards-only             # stop after building + checking the cards
./run_run2_fits.sh --allow-lumi-fallback    # structure/timing tests only
```

Steps: **(1)** per-year input cards → `cards_run2_input_<y>/` (old directory removed first, `--year` passed explicitly,
`--require-lumi`); **(2)** `combineCards.py Y<era>=…` per category (absolute inputs, bins become `Y2017_resolved_mu`, …)
into `cards_run2/<cat>/datacard.txt`, then `check_shapes_resolve`; **(3)**
`check_run2_correlations.py` verifies the scheme mechanically and aborts on a violation (labeled nuisance acting outside its
year, theory nuisance carrying a year label, rateParam with a year suffix; bare `_2016` labels are allowed on both 2016
halves only); **(4)** per-category `text2workspace` + MC-Asimov `-M Significance` (+ FitDiagnostics unless `--sig-only`,
with the band guard, printing the correlated `norm_*` and `r`), `--jobs` categories at a time — **the large `combined`
card runs alone** and with an unlimited stack (`ulimit -s unlimited`), since its 48 channels otherwise overflow the default
stack in `text2workspace`'s recursive model building (segfault mid-autoMCStats); **(5)** summary →
`plots_run2/significances_run2_m125.txt`. Success is judged by per-category `.ok` marker files (the `wait -n` throttle
reaps children, so per-pid `wait` would report successful categories as failed). Outputs: `cards_run2/<cat>/`,
`logs_run2/fit_<cat>.log`.

Caveats: the first time you run it for non-2018 eras, make sure each has a `histograms/histograms_cr_<era>.root` and
`config/inputs_v28_<era>.yaml`; the checker classifies a nuisance with an unrecognised era spelling (e.g. `…_2016APV`)
as unlabeled single-year, which is harmless.

---

## 9. BDT binning studies

Goal: pick the merged-bin structure of the SR BDT. Variants have **equal signal per bin, at most 8 bins** (so coarse bins
at low score, fine bins at high score), with the bin count varied.

```bash
./run_binning_scan.sh --year=2018 --jobs=3                 # N = 4..8
./run_binning_scan.sh --nbins="5 6 8" --jobs=2             # chosen variants
./run_binning_scan.sh --summary-only                       # reprint the comparison table
```

| Step | What happens |
|---|---|
| 0 | `make_histograms.py` **without** `--binning` → `histograms/histograms_native_<YEAR>.root` (native 40 bins), reused while newer than the config |
| 1 | `make_binning_scan.py`: per region, the summed `dipoleRecoil` signal distribution → **exact dynamic-programming** partition of the 40 native bins into N groups minimizing Σ(sᵢ − total/N)² (edges always on native edges; fewer bins than requested are reported when the signal is too concentrated). Writes `config/binning_scan_<YEAR>/flat<N>bins.yaml` |
| 2 | one **full per-year workflow per variant** (`run_all.sh --year --binning=… --tag=flat<N> --skip-breakdown`), `--jobs` at a time, each with its own `histograms_cr_<YEAR>_flat<N>.root`, `cards_cr_<YEAR>_flat<N>/`, `plots_cr_<YEAR>_flat<N>/` |
| 3 | summary table of MC-Asimov and CR-data-hybrid significances per variant |

Details that matter:

* **Memory.** Each variant would load tens of GB in step 1 of its pipeline; `PYRUN_LOCK` serializes the coffea/python steps
  across variants (one variant builds histograms/plots while the others run `combine`). Three simultaneous loaders got every
  variant OOM-killed.
* **Breakdowns are skipped** (`--extra="--skip-breakdown"`): ~7000 scan minimizations per variant do not inform the binning
  choice. Run the winner alone: `./run_all.sh --year=<Y> --binning=… --tag=flat<N> --breakdown-only`.
* **Per-variant diagnostics from the generator** (`make_binning_scan.py --plotdir plots_binning_scan_<YEAR>`): per-bin
  signal, background and `s/√b` with MC-stat errors, a 2×2 figure per variant (`ssqrtb_flat<N>bins.png/.pdf`, bar width =
  true bin width, `b = 0` bins flagged), and **χ²/ndof of weighted fits to `s/√b` vs bin centre** — linear, exponential and a
  Gaussian *pinned at the peak bin and fitted on the rising side only* — printed per region and as a cross-variant table.
  Use the χ² comparatively across variants (ndof ≤ 6); the quadrature `√Σz²` shown in each panel is the stat-only
  significance the binning retains.
* Success is tracked by `.ok` marker files, as in §8.

---

## 10. Multi-POI (WW / WZ) sensitivity

`run_multipoi.sh` produces, from the *same* cards `run_all*.sh` built: the single `r`, and `r_WW`, `r_WZ` from a
`multiSignalModel` workspace — for each POI a significance, a 1D scan, `MultiDimFit --algo singles` intervals, impacts, and a
2D `r_WW`–`r_WZ` scan, on both card sets (`mconly`, `crdata`). `OTHER_POI_MODE` computes both **profiled** (the other strength
floats — the honest "measure WW without assuming WZ"; quote this) and **fixed** (other at SM). The four WW and four WZ signal
processes are classified from the card by `make_multipoi_args.py`; the ZZ process is folded into `r_WZ` (`--zz wz`, default),
`ww`, or gets its own POI (`own`). `read_multipoi_signals.py` reads best fit and ±1σ from the `limit` tree. Flags:
`--mconly-only`, `--crdata-only`, `--targets "combined"`, `--zz own`, `--no-impacts`, `--no-scan`. Copy the script to a new name
before editing it during a run (bash reads scripts incrementally).

---

## 11. Script reference

**Pipeline / building**

| Script | Purpose |
|---|---|
| `make_inputs_from_manifest.py` | v28 manifest → `inputs_v28_<YEAR>.yaml` (§4) |
| `make_histograms.py` | coffea → ROOT histograms for 12 channels; `--config --output --year --scale-systs {selected,all,none} --variable --sr-rebin --binning` |
| `make_histograms_kinematics.py` | kinematic-variable histograms (`--config --sffile --output --year`) |
| `make_datacards.py` | cards with rateParams (`--histfile --config --outdir --year --decorrelate-shower {group,none} --decorrelate-norms {topology,none} --decorrelate-years {year,none} --require-lumi`) |
| `make_hybrid_data.py` | hybrid CR-data / SR-pseudo-data dataset + card (`--datacard --norms --inject-signal --outdir --mass --verify-only`) |
| `make_binning_scan.py` | flat-signal binnings, s/√b plots and shape-fit χ² (§9) |
| `make_cutflow_table.py` | cutflow/efficiency tables from coffea outputs (`--config --lumi --outdir`) |
| `make_multipoi_args.py`, `read_multipoi_signals.py` | multi-POI helpers (§10) |
| `run_cr_only_v8.py` | condor builder for CR-only fits with native channel masking (`--collect`) |
| `uncertainty_breakdown.py` | grouped uncertainty breakdown (§7.2; `--datacard --outdir --label --tag --mass --algo --points --rmin --rmax --robust-fit`) |

**Reading fit results**

| Script | Purpose |
|---|---|
| `read_cr_norms.py` | extract CR-fit norms as `name=value,…` (`--params auto`, `--fit`, `--strict-status`, `--max-deviation`) |
| `extract_postfit_params.py` | pulls, rateParams and covariance → JSON for the closure/QvG plotters (CMSSW/PyROOT; year-tolerant rateParam names) |

**Plots**

| Script | Purpose |
|---|---|
| `plot_stacked.py` | stacked prefit plots with systematic + stat band, signal ×10 |
| `plot_syst_ratios.py` | prefit stacks with systematic ratio panels (signal and background each divided by their own nominal) |
| `plot_postfit.py` | postfit stacks from FitDiagnostics (`--fitfile --histfile --fitdir --outdir --blind-above --dataset --channels --tag`) |
| `plot_cr_closure.py` | CR shape closure (§7.4) |
| `plot_qvg_postfit.py` | postfit `jet1..4_btagDeepFlavQG` with nested stat / stat⊕QvG / stat⊕all bands and an artificially inflated QvG band (`--qvg-scale`, default 2), plus prefit panels |
| `quick_syst_prepost.py` | prefit vs postfit effect of chosen *fitted* nuisance names |
| `plot_postfit_jinw.py` | earlier Asimov-only postfit plotter (reference) |

**Checks**

| Script | Purpose |
|---|---|
| `check_rateparams.py` | confirm every declared rateParam reached the workspace (wildcard bins) |
| `check_hybrid_norms.py` | audit existing hybrids for the norms actually injected |
| `check_run2_correlations.py` | verify the cross-year correlation scheme on a combined card (§8) |
| `diagnose_significance.py` | stat-only Asimov bound from the card's templates, bin by bin, to locate where a combined significance comes from (`--datacard --fitfile --reported --norms --top --min-b`) |

---

## 12. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Postfit plots have **zero uncertainty bands**; log says *"saveWithUncertainties will be ignored"* | covariance not accepted → use strategy 0 + `--robustHesse 1`; never pipe the fit through `tail`; `check_band_saved` aborts on it |
| `text2workspace`: `Bogus norm 0.0 … systematic X Down` | (a) signed-integral cancellation → guarded; (b) **template clobbering from a nuisance name collision** (`EC2` tagged to `EC2_2017` overwrote the real `EC2_2017` templates) → guarded by duplicate-key aborts |
| `TNetXNGFile … is truncated at N bytes: should be M` | EOS FUSE→xrootd flush race → `wait_eos_xrootd_sync`; or `TFile.CrossProtocolRedirects: 0` in `~/.rootrc` as a blunt global fix |
| `text2workspace` **segfault** on the Run-2 `combined` card | stack overflow in recursive model building → `ulimit -s unlimited`, run it alone |
| `Killed` on several variants of the binning scan | OOM from parallel coffea loaders → `PYRUN_LOCK` (already exported by the scan) |
| Successful variant/category reported `[FAIL]` | per-pid `wait` after `wait -n` reaped it → success must come from `.ok` markers |
| Script dies silently after the first card log | `set -eo pipefail` + a failing command inside a `tee` pipe or `$(…)` → check the log tail; drivers now print it |
| `rateParam` uncertainties missing from impact plots | norms without explicit range → `[0.2,5]` in the card |
| MC-stat contribution reads exactly zero in the breakdown | `--freezeParameters allConstrainedNuisances` beats `--floatParameters` → explicit complementary freeze list (already used) |
| Rate-params row `NaN` / one-sided clamped rows in the breakdown | missing post-fit snapshot anchoring or stale name patterns → snapshot + year-tolerant `rgx{…}` (already used) |
| CR-only fits `status 300`, NaN MINOS errors | floating constrained NPs vs 2 CR points, `r` frozen without POI redefinition → freeze constrained NPs, `--redefineSignalPOIs norm_top…,norm_wjet…` |
| Hybrid SR pseudo-data silently built at norm 1.0 | rateParam names not covered by `--norms` → `make_hybrid_data.py` now aborts; audit with `check_hybrid_norms.py` |
| `--channel-masks` mask names | combine creates variables literally named `mask_<bin>`; mismatching names silently match nothing (SR unmasked!) → `check_masks_exist` |
| Hybrid combined significance ≫ quadrature sum | `-M Significance` internal r = 0 fit stranded → use the NLL-difference significance |
| FitDiagnostics "sitting" for hours | the shapes-sampling stage, not a hang; linear in `--numToysForShapes` |
| EOS `Permission denied` inside apptainer | `KRB5CCNAME` overridden to a nonexistent file → inherit it |

---

## 13. Historical driver snapshots

`run_all_cr_data.sh` (no year tagging), `run_all_minos_hesse_patched.sh` and `run_minos_cr.sh` (no year tagging; later
minimizer settings) are earlier iterations of the same driver and are **not** the current pipeline. `run_all_each_year.sh`
is the current year-tagged per-year pipeline; `run_all.sh` is its extended copy used by the binning scan. The pre-v28 configs
`config/inputs_2018.yaml` and `config/inputs_2022postEE.yaml` (hand-written, one process list per year) are kept for
reference; `config/binning_example.yaml~`/`inputs_2018.yaml~` are editor backups.

---

## 14. Status notes (dated)

*State around 2026-10-29; check the commit log for newer information.*

* Per-year chains are validated end to end (CR-only norms year-tagged with sane errors, hybrids audited, postfit CR
  closure reasonable); the Run-2 per-category combination fits run.
* **2017/2018 results made before the coffea re-production are provisional.** An upstream PocketCoffea JER-smearing bug
  (unmatched jets received the full JER SF instead of stochastic smearing) distorted nominal templates and inflated the
  JES/JER variations, which — through the degeneracy of jet-nuisance and background-rate information — eroded the
  normalization impacts in 2017/2018 (2016 is essentially unaffected). Fit-level fingerprint: AK4 nuisance variance
  shares of 19–24 % / 10–19 % (2018 / 2017, resolved / boosted) versus 3–7 % for 2016.
* The 2018 boosted AK8 prominence (≈15 % variance share vs 0.6–5 % elsewhere) is not bug-correlated and is to be revisited
  after the re-production.
* Open items: the closure/QvG plotters morph with un-renormalized templates while the fit used renormalized ones for
  scale/shower on `TTbar`/`WJets`; `--scale-systs none` aborts (its renorm-pattern check matches nothing); the merged
  nonprompt estimate has no separate boosted fake-rate split; QCD-VV and multiboson backgrounds still enter as ~16
  individual processes with no shared normalization.
