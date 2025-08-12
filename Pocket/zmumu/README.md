# PocketCoffea Tutorial: Z $\rightarrow$ $\mu$ $\mu$

Welcome to this hands-on tutorial on pocket-coffea. This guide is designed to take you through a complete particle physics analysis, from defining the datasets to creating the final plots. The goal is to learn the workflow and philosophy behind this framework.

The analysis we will perform as an example is the reconstruction of the Z boson from its decay into two muons (Z $\rightarrow$ $\mu$ $\mu$), using data and simulation of the CMS detector.

## General Workflow

The analysis process in pocket-coffea is divided into four major steps that we will follow in this tutorial:

1. Dataset Preparation: we will define the datasets (Monte Carlo samples and real data) and use the pocket-coffea tools to find the paths to the corresponding files in the Grid.

2. Analysis Configuration: We will create the configuration files (.py and .yaml) where all the physics logic is specified: object preselection, event selection, histograms, weights, etc.

3. Running the Processor: We will launch the coffea processor to iterate over all the events, apply our selections and save the results (histograms) in an output file.

4. Plot Creation: We will use pocket-coffea's plotting tools to read the results and generate the final plots of our analysis.

### 1. Preparation of the Datasets (.json)

The first step is to tell the framework where to find the data we want to analyze.

a) Definition of the Datasets

In pocket-coffea, this is done through a .json definitions file. This file acts as a master map that tells the framework what to look for in the CMS DAS.

For our analysis of Z $\rightarrow$ $\mu +$ $\mu -$, we will use a sample simulation (Monte Carlo) of DYJetsToLL (the signal process) and real SingleMuon collision data.

The file datasets/datasets_definitions.json contains these definitions. It is a dictionary where each entry defines a sample:

```bash

{
    "DYJetsToLL_M-50": {
        "sample": "DYJetsToLL",
        "json_output": "datasets/DYJetsToLL_M-50.json",
        "files": [
            {
                "das_names": ["/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM"],
                "metadata": {
                    "year": "2018",
                    "isMC": true,
                    "xsec": 6077.22
                }
            }
        ]
    },
    "DATA_SingleMuon": {
        "sample": "DATA_SingleMuon",
        "json_output": "datasets/DATA_SingleMuon.json",
        "files": [
            {
                "das_names": ["/SingleMuon/Run2018C-UL2018_MiniAODv2_NanoAODv9-v2/NANOAOD"],
                "metadata": {
                    "year": "2018",
                    "isMC": false,
                    "primaryDataset": "SingleMuon",
                    "era": "C"
                }
            }
        ]
    }
}

```

Important keys:

* sample: Internal label to group events (very useful for plotting).

* json_output: Path where the final file with the list of .root files will be saved.

* das_names: The official name of the dataset in DAS.

* metadata: Crucial information such as the year, if it is Monte Carlo (isMC), the effective section (xsec), and for data, the era.

b) Creation of the "Filesets

Once the .json is defined, we use a pocket-coffea command to fetch these datasets from the Grid and generate the final files (called “datasets”) with the complete lists of files.

First, make sure you have a valid Grid proxy:

```bash

voms-proxy-init -voms cms -rfc --valid 168:0

```

Now, run the command to build the datasets:

```bash

# This command reads the .json, queries the Grid and creates the filesets
pocket-coffea build-datasets --cfg datasets/datasets_definitions.json

# If you are at CERN, it is useful to restrict the search to nearby Tiers
pocket-coffea build-datasets --cfg datasets/datasets_definitions.json -rs 'T[123]_(FR|IT|BE|CH|DE)_\w+'

```

This will create two files for each dataset in the datasets/ folder: one with direct paths to a Grid site and one with the _redirector.json suffix, which uses a global redirector and is more portable. We will use the redirector ones in our configuration.

### 2. Analysis configuration (config.py and .yaml)

Here we define the physics of our analysis. pocket-coffea distinguishes between Parameters (general definitions like triggers, lepton IDs) and Configuration (a specific analysis with the cuts and histograms).

a) Parameters (.yaml)

Parameters are defined in .yaml files to keep the configuration clean and modular.

Object Preselection: In params/object_preselection.yaml, we specify the criteria to define what is a “good” muon or a “good” jet.

```bash

# params/object_preselection.yaml
object_preselection:
  Muon:
    pt: 15
    eta: 2.4
    iso: 0.25 # PFIsoLoose
    id: tightId
  Jet:
    dr_lepton: 0.4
    pt: 30
    eta: 2.4
    jetId: 2

```

#### Triggers: In params/triggers.yaml we define the HLT triggers we want to apply.

b) The Main Configuration File (config.py)

This is the brain of our analysis. Here the Configurator class is instantiated, which groups all the information.

Parameters Load: At the beginning of the file, we load the parameters from the .yaml.

```bash

# config.py
from pocket_coffea.parameters import defaults
# ...
default_parameters = defaults.get_default_parameters()
parameters = defaults.merge_parameters_from_files(
 default_parameters,
 "params/object_preselection.yaml",
 "params/triggers.yaml",
 update=True
)

```

### 3. Configurator Instance: 

We create the cfg object that the framework will look for.

```bash

# config.py
from pocket_coffea.utils.configurator import Configurator
# ...

cfg = Configurator(
 parameters = parámetros,

    datasets = {
 "jsons": [
 "datasets/DATA_SingleMuon_redirector.json",
 "datasets/DYJetsToLL_M-50_redirector.json"
 ],
 "filter": {
 "samples": ["DATA_SingleMuon", "DYJetsToLL_M-50"],
 "year": ['2018']
 }
 },

    workflow = ZmumuBaseProcessor, # Nuestra clase de análisis

    # Aquí se definen los cortes, categorías, pesos e histogramas
    # ... ver abajo ...
)

```

### 4. Definition of Selections (Cuts):

Cuts are defined in a list. For Z $\rightarrow$ $\mu$ $\mu$ analysis:

Skim: A first filter to reduce data. We ask for at least 1 muon and apply the SingleMuon trigger.

Preselection: The main selection. We ask for exactly 2 muons of opposite charge, with a cut at the leader pt and at the invariant mass.

Categorization: We divide the events that pass the preselection into categories. Here, only a baseline category that does not apply additional cuts.

### 5. Definition of Weights and Variations:
Here we activate the weights for the simulation (lumi, xsec, pileup, etc.) and the systematic variations we want to propagate.

```bash

# config.py
# ... dentro de Configurator ...
 weights = {
 "common": {
 "inclusive": ["genWeight", "lumi", "XS", "pileup", "sf_mu_id", "sf_mu_iso"],
 }
    },
 variations = {
 "pesos": {
 "común": {"inclusive": ["pileup", "sf_mu_id", "sf_mu_iso"] },
 }
    },

```

### 6. Histograms definition:

Finally, we tell the framework which histograms we want to create.

```bash

# config.py
# ... inside Configurator ...
    variables = {
 "mll": HistConf(
 [Axis(coll="ll", field="mass", bins=100, start=50, stop=150, label=r"$M_{"GeV]")]]
        ),
        # ... other histograms such as pt, eta, nJets, etc.
 }

```

## 3. Running the Processor

With everything set up, it's time to process the data. pocket-coffea offers several “runners”.

* iterative: Run locally on a single thread - ideal for debugging!

* futures: Runs locally using multiple threads. Good for quick tests.

* dask@lxplus: Scales analysis by submitting jobs to HTCondor in LXPLUS. For production.

For testing code, always start with iterative and the --test option:

```bash

# Limit processing to 1 file and 2 chunks for quick testing
pocket-coffea run --cfg config.py --test -o output_test

```

Once everything works, you can launch it on the whole dataset:

```bash

pocket-coffea run --cfg config.py --executor dask@lxplus --scaleout 10 -o output_dask

```

This will generate an output folder (e.g. output_dask) with the results in a .coffea file and copies of your settings to ensure reproducibility.

## 4. Creating the Plots 

The last step is to take the .coffea file with the histograms and create the plots.

Use the make-plots command:

```bash

pocket-coffea make-plots -i output_dask/output_all.coffea --cfg output_dask/parameters_dump.yaml -o plots_dask

```

Key arguments:

* -i: The input .coffea file.

* --cfg: The parameter file that was saved along with the results.

* -o: The output folder for the graphics.


# Exercise: Strengthen Z $\rightarrow$ $\mu$ $\mu$ selection and plot $\DeltaR$  in PocketCoffea  

Starting point
Original files:

workflow.py
custom_cut_functions.py
example_config.py
params/object_preselection.yaml, params/triggers.yaml, params/plotting.yaml
Do not change object_preselection.yaml or triggers.yaml for this exercise.

1) Preselection Z $\rightarrow$ $\mu$ $\mu$ within the cut 

Objective: that the Z $\rightarrow$ $\mu$ $\mu$ cut does not depend on events.ll and is robust.

In custom_cut_functions.py:

A. In the dimuon preselection function:
* Construct muon pairs per event using combinations of MuonGood.
* Define and apply minimum pT for leading and subleading muons (values from parameters).
* Require opposite charge between the two muons in the pair.
* Compute dR(mu,mu) per pair and apply a minimum.
* Compute mll of the pair and apply window [low, high] (values from parameters).

The final mask at the event level must accept if any pair meets all criteria.
Be sure to handle None values by returning False in those cases.

B. In the Cut corresponding to dimuon, add the new keys needed for the following to params:
* pT of the leading and subleading muon
* Minimum dR
* Mass window (low, high)
* 
Keep everything parameterized (no hardcoding in the code).
Verification:
Run a smoke test. The cut should not crash and should filter events reasonably.

```bash

pocket-coffea run --cfg example_config.py --test -o outputs/test

```

2) Plot dR(mu,mu)
   
Objective: produce a histogram of dR between the two selected muons.

A — Define the variable (workflow):
In workflow.py, without changing the construction of ll, calculate a scalar dR_mumu per event:
* Construct μμ pairs (MuonGood combinations).
* Obtain dR per pair.
* To have one value per event, select the one from the corresponding pair (if there are exactly two muons, there will be only one pair).
* Fill in events without a pair with a sentinel value (e.g., -1).
* Save the result as self.events[“dR_mumu”].
  
B — Histogram (config):
In example_config.py, add an entry in variables to histogram dR_mumu:
* coll="events"
* Typical range: 0–6, ~60 bins
* Axis label: dR(mu,mu)
Verification:
Run a short run and generate plots.
Verify that the dR figure exists in plots/.../dr_mumu_*.png.
If you use zpeak, compare baseline vs zpeak and comment on differences.

```bash

pocket-coffea run --cfg example_config.py --test -o outputs/test

```

