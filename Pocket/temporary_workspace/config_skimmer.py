import os, cloudpickle
from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_functions import get_HLTsel, get_nPVgood, goldenJson, eventFlags
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.parameters import defaults

import workflow_v9, custom_cut_functions
from workflow_v9 import VBSSemileptonicProcessor
from custom_cut_functions import (
    nLepton_skim_cut,   # >=1 e (pT>35) or mu (pT>30)
    nJet_skim_cut,      # N(AK4, pT>30) + N(AK8, pT>30) >= 3
    met_skim_cut,       # DeepMETResolutionTune pT > 30 GeV
)

cloudpickle.register_pickle_by_value(workflow_v9)
cloudpickle.register_pickle_by_value(custom_cut_functions)

localdir = os.path.dirname(os.path.abspath(__file__))

default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir + "/params")
parameters = defaults.merge_parameters_from_files(
    default_parameters,
    f"{localdir}/params/object_preselection_run2_v9.yaml",
    f"{localdir}/params/triggers.yaml",
    f"{localdir}/params/pileup.yaml",
    f"{localdir}/params/jets_calibration.yaml",
    f"{localdir}/params/lepton_scale_factors.yaml",
    update=True,
)

cfg = Configurator(
    parameters=parameters,
    datasets={
        "jsons": [
            # ---------- W+jets ----------
            f"{localdir}/datasets/WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            f"{localdir}/datasets/WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            f"{localdir}/datasets/WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            f"{localdir}/datasets/WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            f"{localdir}/datasets/WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            f"{localdir}/datasets/WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            f"{localdir}/datasets/WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            f"{localdir}/datasets/WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            f"{localdir}/datasets/WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            f"{localdir}/datasets/WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8.json",
            # ---------- DY ----------
            f"{localdir}/datasets/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_fast.json",
            f"{localdir}/datasets/DYJetsToLL_M-10to50_TuneCP5_13TeV-amcatnloFXFX-pythia8_fast.json",
            # ---------- TTbar ----------
            f"{localdir}/datasets/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8.json",
            f"{localdir}/datasets/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_fast.json",
            # ---------- EWK VBS signal ----------
            f"{localdir}/datasets/WplusTo2JWminusToLNuJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            f"{localdir}/datasets/WplusToLNuWminusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            f"{localdir}/datasets/WminusToLNuWminusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            f"{localdir}/datasets/WplusToLNuWplusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            f"{localdir}/datasets/WminusToLNuZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            f"{localdir}/datasets/WplusToLNuZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            # ---------- Data ----------
            f"{localdir}/datasets/SingleMuon.json",
            f"{localdir}/datasets/SingleElectron.json",
            f"{localdir}/datasets/EGamma.json",
        ],
        "filter": {
            "samples": [
                "WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8",
                "WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8",
                "WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8",
                "WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8",
                "WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8",
                "WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8",
                "WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8",
                "WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8",
                "WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8",
                "WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8",
                "DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8",
                "DYJetsToLL_M-10to50_TuneCP5_13TeV-amcatnloFXFX-pythia8",
                "TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8",
                "TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8",
                # EWK VBS signal — uncomment when signal JSONs are added above
                "WplusTo2JWminusToLNuJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
                "WplusToLNuWminusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
                "WminusToLNuWminusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
                "WplusToLNuWplusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
                "WminusToLNuZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
                "WplusToLNuZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
                "SingleMuon",
                "EGamma",
                "SingleElectron"
            ],
            "year": ["2018"],
            "nfiles": {
                "TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8": 50,
                "TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8": 50,
            },
        },
    },

    workflow=VBSSemileptonicProcessor,

    # Skim: loose cuts applied before NanoAOD files are written out.
    # All cuts combined with AND logic. Designed to be inclusive enough
    # to support any downstream CR or SR selection without re-skimming.
    skim=[
        get_nPVgood(1),
        eventFlags,
        goldenJson,
        get_HLTsel(primaryDatasets=["SingleMuon", "EGamma"]),
        nLepton_skim_cut,   # >=1 lepton (e pT>35 or mu pT>30)
        met_skim_cut,       # DeepMETResolutionTune pT > 30 GeV
        nJet_skim_cut,      # N(AK4, pT>30) + N(AK8, pT>30) >= 3
    ],

    preselections=[passthrough],

    categories={
        "baseline": [passthrough],
    },

    # No weights / variations / variables: processing halts after the skim
    # and one skimmed NanoAOD file per chunk is written to save_skimmed_files.
    weights_classes=[],
    weights={"common": {"inclusive": []}},
    variations={},
    variables={},

    # Output path on T3_US_FNALLPC shared EOS space (xrootd endpoint).
    save_skimmed_files="root://cmseos.fnal.gov//eos/uscms/store/group/lnujj/semilep_vbs_skim/",
)
