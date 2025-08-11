# VBS W±W± Semileptonic Analysis with PocketCoffea

## Overview

This hands-on tutorial will guide you through performing a semileptonic **Vector Boson Scattering (VBS)** W±W± analysis using the [PocketCoffea](https://github.com/PocketCoffea) framework.

You will learn how to:
- Configure PocketCoffea for a physics analysis.
- Select physics objects (jets, leptons, MET).
- Apply VBS-like preselections for the semileptonic channel.
- Reconstruct the hadronic W boson and compute its kinematic variables.
- Make histograms and explore distributions.
- Understand how this connects to **jet tagging** (e.g., ParticleNet).

---

## Physics Motivation

Vector Boson Scattering is a rare electroweak process where two vector bosons (here W±W±) are produced with two forward tagging jets. The semileptonic channel has:
- One **leptonic W** (W → lν)  
- One **hadronic W** (W → jj)  
- Two forward **VBS jets**  

Studying this process is essential for probing the Electroweak Symmetry Breaking (EWSB) mechanism and searching for deviations from the Standard Model.

---

## Setup Instructions

### 1. Environment
You can run this tutorial on **lxplus** or any system with Docker/Singularity support and PocketCoffea installed.

```bash
# Clone the tutorial repository
git clone https://github.com/<your-github-user>/Tutorial_Pocket_VBS_semilep.git
cd Tutorial_Pocket_VBS_semilep

# (Optional) If using Singularity
singularity shell /cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask:latest
 ´´´bash

## 2. Datasets 

For this tutorial, we use two signal MC datasets:
* /WpWpJJ-EWK_TuneCP5_13p6TeV-powheg-pythia8/.../NANOAODSIM
* /WmWmJJ-EWK_TuneCP5_13p6TeV-powheg-pythia8/.../NANOAODSIM
* JSON files describing the dataset locations are in the datasets/ folder.

### Tutorial Structure:

We will work with three main scripts:

example_config_semileptonic.py
Configures PocketCoffea:
* Loads datasets and parameters
* Sets up skimming and preselections
* Defines histograms

custom_cut_functions.py
Defines:
* nLepton_skim_cut: requires ≥1 good lepton
* vbs_semileptonic_presel: VBS-like preselections

workflow.py
Implements VBSWWBaseProcessor:
* Applies object preselection
* Reconstructs VBS jets and hadronic W boson
* Computes helper variables for plotting

Hands-On plan

We will proceed step-by-step:

1. Load the configuration and inspect available datasets.
2. Run with a small subset (--limit-files 1 --limit-chunks 2) to check everything works.
3. Inspect histograms for:
* Number of jets
* MET
* mjj(VBS jets)
* mjj(W hadronic)
* Modify preselections:
* Change MET cut
* Change mjj(VBS) cut
* See how distributions change
4. Add new plots:
* dR between hadronic W jets
* eta distribution of leptons

Exercises:

Q1. In custom_cut_functions.py, we cut on mjj_vbs > 500. What happens if you lower it to 300?
Q2. The hadronic W mass window is set to ±15 GeV around 80.4. What happens if you remove this window?
Q3. Add a histogram for ΔR between the two hadronic W jets.
Q4. Change the lepton centrality requirement (require_lep_central) to True. What is the effect?
Q5. Change the b-jet veto to allow events with one b-jet. How does mjj(W had) change?

Hint: Running the Code
Example run:

'''bash
pocket-coffea run --cfg example_config_semileptonic.py \
                  -o output_test \
                  -e futures \
                  --scaleout 4 \
                  --limit-files 1 \
                  --limit-chunks 2
'''bash
