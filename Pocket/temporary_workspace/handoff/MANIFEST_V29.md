# v29 — config_semilep, 2017, Signal Region only, MC only, columnar output

45 datasets, 77.6 GiB. One `.coffea` per dataset; there is **no** `output_all.coffea` because the campaign ran with
`--process-separately` (no cross-dataset merge).

Per-dataset keys inside each file: `sum_genweights`, `sum_signOf_genweights`, `sumw`/`sumw2` (`[category][dataset][sample][variation]`), `cutflow`, `variables`, `datasets_metadata`.
`columns` is keyed by **sample**, everything else by **dataset**.

## Capped datasets — reduced statistics

Normalization is unaffected (`sum_genweights` accumulates from processed events); statistical precision is reduced.

| dataset | files used | files total |
|---|---|---|
| `TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2017` | 25 | 219 |
| `TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_2017` | 30 | 123 |
| `WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_2017` | 12 | 24 |

## All outputs

| dir | dataset | size | files used/total |
|---|---|---|---|
| `v29_semilep_2017_g01` | `TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2017` | 9.07 GiB | 25/219 |
| `v29_semilep_2017_g02` | `TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_2017` | 7.27 GiB | 30/123 |
| `v29_semilep_2017_g03` | `ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8_2017` | 5.87 GiB | 17/17 |
| `v29_semilep_2017_g03` | `WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_2017` | 4.37 GiB | 12/24 |
| `v29_semilep_2017_g03` | `ttWJets_TuneCP5_13TeV_madgraphMLM_pythia8_2017` | 5.25 GiB | 18/18 |
| `v29_semilep_2017_g03` | `ttZJets_TuneCP5_13TeV_madgraphMLM_pythia8_2017` | 4.95 GiB | 18/18 |
| `v29_semilep_2017_g04` | `DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_2017` | 2.42 GiB | 11/11 |
| `v29_semilep_2017_g04` | `GluGluWWToLNuQQ_TuneCP5_13TeV_madgraph-pythia8_2017` | 2.12 GiB | 5/5 |
| `v29_semilep_2017_g04` | `ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8_2017` | 1.45 GiB | 5/5 |
| `v29_semilep_2017_g04` | `ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8_2017` | 3.24 GiB | 10/10 |
| `v29_semilep_2017_g04` | `WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8_2017` | 4.34 GiB | 14/14 |
| `v29_semilep_2017_g04` | `WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8_2017` | 2.09 GiB | 4/4 |
| `v29_semilep_2017_g04` | `WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8_2017` | 1.53 GiB | 5/5 |
| `v29_semilep_2017_g04` | `WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8_2017` | 1.87 GiB | 4/4 |
| `v29_semilep_2017_g04` | `WZTo3LNu_mllmin01_NNPDF31_TuneCP5_13TeV_powheg_pythia8_2017` | 2.66 GiB | 17/17 |
| `v29_semilep_2017_g05` | `DYJetsToLL_M-10to50_TuneCP5_13TeV-amcatnloFXFX-pythia8_2017` | 0.05 GiB | 2/2 |
| `v29_semilep_2017_g05` | `ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8_2017` | 0.71 GiB | 3/3 |
| `v29_semilep_2017_g05` | `ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8_2017` | 0.71 GiB | 3/3 |
| `v29_semilep_2017_g05` | `WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8_2017` | 1.52 GiB | 4/4 |
| `v29_semilep_2017_g05` | `WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8_2017` | 0.33 GiB | 2/2 |
| `v29_semilep_2017_g05` | `WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8_2017` | 1.84 GiB | 4/4 |
| `v29_semilep_2017_g05` | `WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8_2017` | 0.11 GiB | 2/2 |
| `v29_semilep_2017_g05` | `WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8_2017` | 0.07 GiB | 1/1 |
| `v29_semilep_2017_g05` | `WWZ_4F_TuneCP5_13TeV-amcatnlo-pythia8_2017` | 0.07 GiB | 1/1 |
| `v29_semilep_2017_g05` | `WZZ_TuneCP5_13TeV-amcatnlo-pythia8_2017` | 0.07 GiB | 1/1 |
| `v29_semilep_2017_g05` | `WminusTo2JZTo2LJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8_2017` | 0.44 GiB | 2/2 |
| `v29_semilep_2017_g05` | `WminusTo2JZTo2LJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8_2017` | 0.43 GiB | 2/2 |
| `v29_semilep_2017_g05` | `WminusToLNuWminusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8_2017` | 0.71 GiB | 2/2 |
| `v29_semilep_2017_g05` | `WminusToLNuWminusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8_2017` | 1.46 GiB | 2/2 |
| `v29_semilep_2017_g05` | `WminusToLNuZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8_2017` | 0.79 GiB | 2/2 |
| `v29_semilep_2017_g05` | `WminusToLNuZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8_2017` | 0.81 GiB | 2/2 |
| `v29_semilep_2017_g05` | `WplusTo2JWminusToLNuJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8_2017` | 0.75 GiB | 2/2 |
| `v29_semilep_2017_g05` | `WplusTo2JWminusToLNuJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8_2017` | 0.64 GiB | 2/2 |
| `v29_semilep_2017_g05` | `WplusTo2JZTo2LJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8_2017` | 0.43 GiB | 2/2 |
| `v29_semilep_2017_g05` | `WplusTo2JZTo2LJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8_2017` | 0.44 GiB | 2/2 |
| `v29_semilep_2017_g05` | `WplusToLNuWminusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8_2017` | 0.66 GiB | 2/2 |
| `v29_semilep_2017_g05` | `WplusToLNuWminusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8_2017` | 0.61 GiB | 2/2 |
| `v29_semilep_2017_g05` | `WplusToLNuWplusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8_2017` | 0.73 GiB | 2/2 |
| `v29_semilep_2017_g05` | `WplusToLNuWplusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8_2017` | 1.65 GiB | 2/2 |
| `v29_semilep_2017_g05` | `WplusToLNuZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8_2017` | 0.74 GiB | 2/2 |
| `v29_semilep_2017_g05` | `WplusToLNuZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8_2017` | 0.79 GiB | 2/2 |
| `v29_semilep_2017_g05` | `ZGToLLG_01J_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8_2017` | 0.54 GiB | 3/3 |
| `v29_semilep_2017_g05` | `ZTo2LZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8_2017` | 0.40 GiB | 2/2 |
| `v29_semilep_2017_g05` | `ZTo2LZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8_2017` | 0.55 GiB | 2/2 |
| `v29_semilep_2017_g05` | `ZZZ_TuneCP5_13TeV-amcatnlo-pythia8_2017` | 0.02 GiB | 1/1 |
