# example_config_semileptonic.py
import os, cloudpickle
from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_functions import get_HLTsel, get_nPVgood, goldenJson, eventFlags
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.parameters.histograms import *
from pocket_coffea.lib.weights.common import common_weights
from pocket_coffea.lib.weights.common.common import SF_L1prefiring
from pocket_coffea.lib.weights.common.weights_run2_UL import SF_ele_trigger
from pocket_coffea.parameters import defaults
from pocket_coffea.lib.columns_manager import ColOut

import numpy as np
import awkward as ak
from pocket_coffea.lib.weights import WeightWrapper, WeightData, WeightDataMultiVariation, WeightLambda
from pocket_coffea.lib.scale_factors import sf_pileup_reweight


import workflow_v9, custom_cut_functions, reweighting_st
from workflow_v9 import VBSSemileptonicProcessor
from reweighting_st import ratio_function
from custom_cut_functions import (
    nLepton_skim_cut,
    nJet_skim_cut,
    vbs_semileptonic_presel,
    whad_window_cut_e,
    met_skim_cut,
    whad_window_cut_bveto_e,
    msd_window_cut_e,
    whad_window_cut_mu,
    whad_window_cut_bveto_mu,
    msd_window_cut_mu,
    # whad_window_cut_no_fj0_e,
    # whad_window_cut_no_jet4_e,
    # whad_window_cut_no_loose_e,
    # Muon_good1,
    # Muon_good2,
    # Muon_good3,
    # Ele_good1,
    # Ele_good2,
    # Ele_good3
)

cloudpickle.register_pickle_by_value(workflow_v9)
cloudpickle.register_pickle_by_value(reweighting_st)
cloudpickle.register_pickle_by_value(custom_cut_functions)

localdir = os.path.dirname(os.path.abspath(__file__))


default_parameters = defaults.get_default_parameters( 
    # group_tags={"EGM": 
    #     {   "Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15":"latest", 
    #         "Run3-23DSep23-Summer23BPix-NanoAODv12": "latest",
    #         "Run3-23CSep23-Summer23-NanoAODv12": "latest",
    #         "Run3-22EFGSep23-Summer22EE-NanoAODv12": "latest",
    #         "Run3-22CDSep23-Summer22-NanoAODv12": "latest",
    #         "Run2-2018-UL-NanoAODv9": "latest",
    #         "Run2-2017-UL-NanoAODv9": "latest",
    #         "Run2-2016preVFP-UL-NanoAODv9": "latest",
    #         "Run2-2016postVFP-UL-NanoAODv9": "latest"
    #     }
    # }
)
defaults.register_configuration_dir("config_dir", localdir + "/params")
parameters = defaults.merge_parameters_from_files(
    default_parameters,
    f"{localdir}/params/object_preselection_run2_v9.yaml",
    f"{localdir}/params/triggers.yaml",
    f"{localdir}/params/plotting.yaml",
    f"{localdir}/params/pileup.yaml",
    f"{localdir}/params/jets_calibration.yaml",
    f"{localdir}/params/lepton_scale_factors.yaml",
    f"{localdir}/params/classifiers.yaml",
    f"{localdir}/params/variations.yaml",
    update=True,
)

PileupWeight = WeightLambda.wrap_func(
    name="PileupWeight",
    function=lambda params, metadata, events, size, shape_variations:
        sf_pileup_reweight(params, events, metadata["year"]),
    has_variations=True  # no list of variations it means only up and down
    )

wjet_reweight = WeightLambda.wrap_func(
    name="wjet_reweight",
    function=lambda params, metadata, events, size, shape_variations:
       ratio_function(ak.sum(events.GenJet[events.GenJet.pt > 20].pt, axis=1)),
    has_variations=False
    )

cfg = Configurator(
    parameters=parameters,
    datasets={
        "jsons": [
            f"{localdir}/datasets/skimmed_rescale.json",
            f"{localdir}/datasets/rare_skim.json",
            # # f"{localdir}/datasets/DYJetsToLL_M-10to50_TuneCP5_13TeV-amcatnloFXFX-pythia8.json",
            # # f"{localdir}/datasets/DYJetsToLL_M-10to50_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            # # f"{localdir}/datasets/DYJetsToLL_M-4to50_HT-100to200_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            # # f"{localdir}/datasets/DYJetsToLL_M-4to50_HT-200to400_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            # # f"{localdir}/datasets/DYJetsToLL_M-4to50_HT-400to600_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            # # f"{localdir}/datasets/DYJetsToLL_M-4to50_HT-600toInf_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            # # f"{localdir}/datasets/DYJetsToLL_M-4to50_HT-70to100_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            # # f"{localdir}/datasets/DYJetsToLL_M-50_HT-100to200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8.json",
            # # f"{localdir}/datasets/DYJetsToLL_M-50_HT-1200to2500_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8.json",
            # # f"{localdir}/datasets/DYJetsToLL_M-50_HT-200to400_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8.json",
            # # f"{localdir}/datasets/DYJetsToLL_M-50_HT-2500toInf_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8.json",
            # # f"{localdir}/datasets/DYJetsToLL_M-50_HT-400to600_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8.json",
            # # f"{localdir}/datasets/DYJetsToLL_M-50_HT-600to800_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8.json",
            # # f"{localdir}/datasets/DYJetsToLL_M-50_HT-70to100_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8.json",
            # # f"{localdir}/datasets/DYJetsToLL_M-50_HT-800to1200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8.json",
            # # f"{localdir}/datasets/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8.json",
            # # f"{localdir}/datasets/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            # f"{localdir}/datasets/GluGluWWToLNuQQ_TuneCP5_13TeV_madgraph-pythia8.json",
            # f"{localdir}/datasets/ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8.json",
            # f"{localdir}/datasets/ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8.json",
            # f"{localdir}/datasets/ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8.json",
            # f"{localdir}/datasets/ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8.json",
            # f"{localdir}/datasets/ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8.json",
            # # f"{localdir}/datasets/SingleMuon.json",
            # # f"{localdir}/datasets/SingleElectron.json",
            # # f"{localdir}/datasets/EGamma.json",
            # # f"{localdir}/datasets/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8.json",
            # # f"{localdir}/datasets/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8.json",
            # f"{localdir}/datasets/WGToLNuG_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            # # f"{localdir}/datasets/WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            # # f"{localdir}/datasets/WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            # # f"{localdir}/datasets/WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            # # f"{localdir}/datasets/WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            # # f"{localdir}/datasets/WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            # # f"{localdir}/datasets/WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            # # f"{localdir}/datasets/WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            # # f"{localdir}/datasets/WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8.json",

            # # f"{localdir}/datasets/WJetsToLNu_Pt-100To250_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8.json",
            # # f"{localdir}/datasets/WJetsToLNu_Pt-250To400_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8.json",
            # # f"{localdir}/datasets/WJetsToLNu_Pt-400To600_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8.json",
            # # f"{localdir}/datasets/WJetsToLNu_Pt-600ToInf_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8.json",

            # # f"{localdir}/datasets/WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8.json",
            # # f"{localdir}/datasets/WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            # f"{localdir}/datasets/WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8.json",
            # f"{localdir}/datasets/WWZ_4F_TuneCP5_13TeV-amcatnlo-pythia8.json",
            # f"{localdir}/datasets/WZTo3LNu_mllmin01_NNPDF31_TuneCP5_13TeV_powheg_pythia8.json",
            # f"{localdir}/datasets/WZZ_TuneCP5_13TeV-amcatnlo-pythia8.json",
            # f"{localdir}/datasets/WminusTo2JZTo2LJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            # f"{localdir}/datasets/WminusTo2JZTo2LJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            # f"{localdir}/datasets/WminusToLNuWminusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            # f"{localdir}/datasets/WminusToLNuWminusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            # f"{localdir}/datasets/WminusToLNuZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            # f"{localdir}/datasets/WminusToLNuZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            # f"{localdir}/datasets/WplusTo2JWminusToLNuJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            # f"{localdir}/datasets/WplusTo2JZTo2LJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            # f"{localdir}/datasets/WplusTo2JZTo2LJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            # f"{localdir}/datasets/WplusToLNuWminusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            # f"{localdir}/datasets/WplusToLNuWminusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV.json",
            # f"{localdir}/datasets/WplusToLNuWminusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            # f"{localdir}/datasets/WplusToLNuWplusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            # f"{localdir}/datasets/WplusToLNuWplusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            # f"{localdir}/datasets/WplusToLNuZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            # f"{localdir}/datasets/WplusToLNuZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            # f"{localdir}/datasets/ZGToLLG_01J_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8.json",
            # f"{localdir}/datasets/ZTo2LZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            # f"{localdir}/datasets/ZTo2LZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            # f"{localdir}/datasets/ZZZ_TuneCP5_13TeV-amcatnlo-pythia8.json",
            # f"{localdir}/datasets/ttWJets_TuneCP5_13TeV_madgraphMLM_pythia8.json",
            # f"{localdir}/datasets/ttZJets_TuneCP5_13TeV_madgraphMLM_pythia8.json",
        ],
        "filter": {
            "samples": [
                
            #########
            ## RUN 2 BKG
            #########

            # "WJetsToLNu_Pt-100To250_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8",
            #"WJetsToLNu_Pt-250To400_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8",
            #"WJetsToLNu_Pt-400To600_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8",
            #"WJetsToLNu_Pt-600ToInf_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8",

            # "DYJetsToLL_M-10to50_TuneCP5_13TeV-amcatnloFXFX-pythia8",
            # # # "DYJetsToLL_M-10to50_TuneCP5_13TeV-madgraphMLM-pythia8",

            #"DYJetsToLL_M-4to50_HT-100to200_TuneCP5_13TeV-madgraphMLM-pythia8",
            #"DYJetsToLL_M-4to50_HT-200to400_TuneCP5_13TeV-madgraphMLM-pythia8",
            #"DYJetsToLL_M-4to50_HT-400to600_TuneCP5_13TeV-madgraphMLM-pythia8",
            #"DYJetsToLL_M-4to50_HT-600toInf_TuneCP5_13TeV-madgraphMLM-pythia8",
            # # # "DYJetsToLL_M-4to50_HT-70to100_TuneCP5_13TeV-madgraphMLM-pythia8",

            
            #"DYJetsToLL_M-50_HT-100to200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
            #"DYJetsToLL_M-50_HT-1200to2500_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
            #"DYJetsToLL_M-50_HT-200to400_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
            #"DYJetsToLL_M-50_HT-2500toInf_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
            #"DYJetsToLL_M-50_HT-400to600_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
            #"DYJetsToLL_M-50_HT-600to800_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
            #"DYJetsToLL_M-50_HT-70to100_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
            #"DYJetsToLL_M-50_HT-800to1200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",

            # "DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8",
            # # # # # #"DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8",

            
            
            "ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8",
            "ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8",
            "ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8",
            "ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8",
            "ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8",

            "ttWJets_TuneCP5_13TeV_madgraphMLM_pythia8",
            "ttZJets_TuneCP5_13TeV_madgraphMLM_pythia8",

            "TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8",
            "TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8",

            

            "WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8",
            "WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8",
            "WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8",
            "WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8",
            "WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8",
            "WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8",
            "WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8",
            "WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8",

            # "WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8",
            # # "WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8",

            "GluGluWWToLNuQQ_TuneCP5_13TeV_madgraph-pythia8",
            "WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8",
            "WWZ_4F_TuneCP5_13TeV-amcatnlo-pythia8",
            "WZTo3LNu_mllmin01_NNPDF31_TuneCP5_13TeV_powheg_pythia8",
            "WZZ_TuneCP5_13TeV-amcatnlo-pythia8",
            "ZGToLLG_01J_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8",
            "ZZZ_TuneCP5_13TeV-amcatnlo-pythia8",

            "WminusTo2JZTo2LJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WminusToLNuWminusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WminusToLNuZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WplusTo2JZTo2LJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WplusTo2JWminusToLNuJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WplusToLNuWminusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WplusToLNuWplusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WplusToLNuZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "ZTo2LZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",

            
            
            
            
           
            # ###### SIGNAL #########
            "WminusTo2JZTo2LJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WminusToLNuWminusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WminusToLNuZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WplusTo2JWminusToLNuJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8", #WplusTo2JWminusToLNuJJ missing in QCD
            "WplusTo2JZTo2LJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WplusToLNuWminusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WplusToLNuWplusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "WplusToLNuZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            "ZTo2LZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",


            # # # ##### DATA #######
            # "SingleMuon", #2017 or 2018 data
            # "SingleElectron",
            # "EGamma",
            ],
            "year": ["2018"],#"2018"],
        },
    },
    workflow=VBSSemileptonicProcessor,

    
    skim=[
        get_nPVgood(1),    # nPV>0
        eventFlags,        # PileupID
        goldenJson,        
        nLepton_skim_cut,
        nJet_skim_cut,  
        met_skim_cut,
        get_HLTsel(primaryDatasets=["SingleMuon", "EGamma"]),
    ],

    # 2) preselections 
    preselections=[vbs_semileptonic_presel],

   
    categories={
        "baseline": [passthrough],
        # "mT_validation_mu": [Muon_good1],
        # "mll_validation_mu": [Muon_good2],
        # "mT_noMET_mu": [Muon_good3],
        # "mT_validation_e": [Ele_good1],
        # "mll_validation_e": [Ele_good2],
        # "mT_noMET_e": [Ele_good3],
        # # "whad_baseline_e":  [whad_window_cut_no_jet4_e],
        # # "whad_4jets_e":  [whad_window_cut_no_loose_e],
        # # "whad_4jets_looselep_e":  [whad_window_cut_no_fj0_e],
        # # "whad_4jets_looselep_fj_e": [whad_window_cut_e],  # |mjj^W - 80.4| < window
        "boosted_e": [msd_window_cut_e],
        # # "whad_peak_mu": [whad_window_cut_mu],  # |mjj^W - 80.4| < window
        "boosted_mu": [msd_window_cut_mu],
        "resolved_mu":  [whad_window_cut_bveto_mu],
        "resolved_e": [whad_window_cut_bveto_e],
    },

   
    weights_classes=common_weights+[PileupWeight]+[SF_L1prefiring]+[wjet_reweight]+[SF_ele_trigger],
       weights={"common": {"inclusive": ["genWeight", "lumi", "XS","PileupWeight", "sf_mu_id","sf_mu_iso","sf_ele_id","sf_ele_reco","sf_mu_trigger","sf_ele_trigger","sf_L1prefiring","sf_jet_puId","sf_partonshower_isr", "sf_partonshower_fsr"]},
        # "bysample": {
        #        "WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8": {
        #             "inclusive": ["wjet_reweight"],},
        #         "WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8": {
        #             "inclusive": ["wjet_reweight"],},
        #         "WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8": {
        #             "inclusive": ["wjet_reweight"],},
        #         "WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8": {
        #             "inclusive": ["wjet_reweight"],},
        #         "WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8": {
        #             "inclusive": ["wjet_reweight"],},
        #         "WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8": {
        #             "inclusive": ["wjet_reweight"],},
        #         "WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8": {
        #             "inclusive": ["wjet_reweight"],},
        #         "WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8": {
        #             "inclusive": ["wjet_reweight"],},
        #         "WJetsToLNu_13TeV-madgraphMLM-pythia8":{
        #              "inclusive": ["wjet_reweight"],},
        #     }    
        },
        variations={"weights": {"common": {"inclusive": ["PileupWeight", "sf_mu_id","sf_mu_iso","sf_ele_id","sf_ele_reco","sf_mu_trigger", "sf_ele_trigger","sf_L1prefiring","sf_jet_puId", "sf_partonshower_isr", "sf_partonshower_fsr"]}}}, #"pileup"

   
    variables={
        "nJets":          HistConf([Axis(coll="events", field="nJetGood",           bins=12, start=0,    stop=12,   label="N(jets)")]),
        "nBJets":         HistConf([Axis(coll="events", field="nBJetGood",          bins=8,  start=0,    stop=8,    label="N(bjets)")]),
        "nCentralJets":   HistConf([Axis(coll="events", field="nCentralJetsGood",   bins=12, start=0,    stop=12,   label="N(Central Jets)")]),
        "nFatJets":       HistConf([Axis(coll="events", field="nFatJetGood",        bins=4,  start=0,    stop=4,    label="N(Fat Jets)")]),
        "nFatJetCentral": HistConf([Axis(coll="events", field="nFatJetCentral",     bins=4,  start=0,    stop=4,    label="N(Central Fat Jets)")]),
        "nLeptonLoose":   HistConf([Axis(coll="events", field="nLeptonLoose",       bins=4,  start=0,    stop=4,    label="N(Lepton Loose)")]),
        "nFatJetCandidate": HistConf([Axis(coll="events", field="nFatJetCandidate", bins=4,  start=0,    stop=4,    label="N(Candidate Fat Jets)")]),
        "nFarAK4Jets":    HistConf([Axis(coll="events", field="nFarAK4Jets",        bins=8,  start=0,    stop=8,    label="N(nFarAK4Jets)")]),
        "nMuonGood":      HistConf([Axis(coll="events", field="nMuonGood",          bins=6,  start=0,    stop=6,    label="N(muon good)")]),
        "nElectronGood":  HistConf([Axis(coll="events", field="nElectronGood",      bins=6,  start=0,    stop=6,    label="N(electron good)")]),
        "nLeptonGood":    HistConf([Axis(coll="events", field="nLeptonGood",        bins=6,  start=0,    stop=6,    label="N(lepton good)")]),
        "btagDeepFlavB":  HistConf([Axis(coll="JetGood", field="btagDeepFlavB",     bins=20, start=0,    stop=1,    label="deepFlavB discrim score")]),
        "btagDeepB":      HistConf([Axis(coll="JetGood", field="btagDeepB",         bins=20, start=0,    stop=1,    label="deepCSV discrim score")]),
        "leading_bscore": HistConf([Axis(coll="events", field="leading_bscore",     bins=33, start=0,    stop=1,    label="max(deepFlavB discrim score)")]),
        "met":            HistConf([Axis(coll="MET",    field="pt",                  bins=50, start=0,    stop=250,  label=r"$p_T^{miss}$ [GeV]")]),
        "met_phi":        HistConf([Axis(coll="MET",    field="phi",                 bins=50, start=-4,   stop=4,    label=r"$\phi^{miss}$ [GeV]")]),
        "puppimet":       HistConf([Axis(coll="PuppiMET", field="pt",               bins=50, start=0,    stop=250,  label=r"$p_T^{miss}$ [GeV]")]),
        "puppimet_phi":   HistConf([Axis(coll="PuppiMET", field="phi",              bins=50, start=-4,   stop=4,    label=r"$\phi^{miss}$ [GeV]")]),
        "DeepMETResponseTune_pt":  HistConf([Axis(coll="DeepMETResponseTune",  field="pt",  bins=50, start=0,  stop=250, label=r"$p_T^{miss}$ [GeV]")]),
        "DeepMETResponseTune_phi": HistConf([Axis(coll="DeepMETResponseTune",  field="phi", bins=50, start=-4, stop=4,   label=r"$\phi^{miss}$ [GeV]")]),
        "DeepMETResolutionTune_pt":  HistConf([Axis(coll="DeepMETResolutionTune", field="pt",  bins=50, start=0,  stop=250, label=r"$p_T^{miss}$ [GeV]")]),
        "DeepMETResolutionTune_phi": HistConf([Axis(coll="DeepMETResolutionTune", field="phi", bins=50, start=-4, stop=4,   label=r"$\phi^{miss}$ [GeV]")]),
        "mt_w_lep":       HistConf([Axis(coll="events", field="mt_w_leptonic", bins=30, start=0, stop=200, label=r"$m_T(W_{lep})$ [GeV]")]),
        "neutrino_pz":    HistConf([Axis(coll="events", field="neutrino_pz",               bins=50, start=0,    stop=250, label=r"$p_z^{\nu}$ [GeV]")]),
        "neutrino_eta":   HistConf([Axis(coll="events", field="neutrino_eta",              bins=32, start=-4.0, stop=4.0, label=r"$\eta^{\nu}$")]),
        "neutrino_deta":  HistConf([Axis(coll="events", field="lead_wlep_neutrino_deta",   bins=32, start=0,    stop=4.0, label=r"$\delta\eta^{l,\nu}$")]),
        "neutrino_dR":    HistConf([Axis(coll="events", field="lead_wlep_neutrino_dR",     bins=32, start=0.0,  stop=4,   label=r"$\delta R^{l,\nu}$")]),
        "mjj_vbs":        HistConf([Axis(coll="vbsjets", field="mass",        bins=50, start=300, stop=4000, label=r"$M_{jj}^{forward}$ [GeV]")]),
        "deta_vbs":       HistConf([Axis(coll="vbsjets", field="delta_eta",   bins=36, start=0,   stop=9.0,  label=r"$|\Delta\eta_{jj}^{forward}|$")]),
        "dR_vbs":         HistConf([Axis(coll="events", field="vbs_dR",       bins=40, start=0.0, stop=7.0,  label=r"$\Delta R(jj)^{forward}$")]),
        "dR_fj_vbs1":     HistConf([Axis(coll="events", field="vbs1_fj_dR",   bins=40, start=0.0, stop=7.0,  label=r"$\Delta R(AK8, j_{forward_1})$")]),
        "dR_fj_vbs2":     HistConf([Axis(coll="events", field="vbs2_fj_dR",   bins=40, start=0.0, stop=7.0,  label=r"$\Delta R(AK8, j_{forward_2})$")]),
        "jet_eta":        HistConf([Axis(coll="JetGood", field="eta",          bins=48, start=-4.8, stop=4.8, label="JetGood eta")]),
        "jet_id":         HistConf([Axis(coll="JetGood", field="jetId",        bins=10, start=0,   stop=10,   label="Jet id")]),
        "jet_rel_iso":    HistConf([Axis(coll="LeptonGood", field="jetRelIso", bins=50, start=0,   stop=2,    label="Jet iso in lep")]),
        "dxy_mu":         HistConf([Axis(coll="LeptonGood", field="dxy",       bins=50, start=0,   stop=0.5,  label="dxy mu")]),
        "dxy_ele":        HistConf([Axis(coll="LeptonGood", field="dxy",       bins=50, start=0,   stop=0.2,  label="dxy ele")]),
        "dz_mu":          HistConf([Axis(coll="LeptonGood", field="dz",        bins=50, start=0,   stop=1,    label="dz mu")]),
        "dz_ele":         HistConf([Axis(coll="LeptonGood", field="dz",        bins=50, start=0,   stop=0.5,  label="dz ele")]),
        "m_jj_w":         HistConf([Axis(coll="w_had_jets", field="mass",      bins=40, start=65,  stop=105,  label=r"$M_{jj}^{had. \, V}$ [GeV]")]),
        "pt_jj_w":        HistConf([Axis(coll="w_had_jets", field="pt",        bins=40, start=40,  stop=210,  label=r"$p_T(jj)^{had. \, V}$ [GeV]")]),
        "dR_w_had":       HistConf([Axis(coll="events", field="w_had_dR",      bins=40, start=0.0, stop=4.0,  label=r"$\Delta R(jj)^{had. \, V}$")]),
        "eta_w_had1":     HistConf([Axis(coll="events", field="w_had_jet1_eta", bins=48, start=-4.0, stop=4.0, label=r"$\eta(j1^{had. \, V})$")]),
        "eta_w_had2":     HistConf([Axis(coll="events", field="w_had_jet2_eta", bins=48, start=-4.0, stop=4.0, label=r"$\eta(j2^{had. \, V})$")]),
        "pt_w_had1":      HistConf([Axis(coll="events", field="w_had_jet1_pt",  bins=60, start=0.0, stop=300.0, label=r"$p_T(j1^{had. \, V})$ [GeV]")]),
        "pt_w_had2":      HistConf([Axis(coll="events", field="w_had_jet2_pt",  bins=60, start=0.0, stop=300.0, label=r"$p_T(j2^{had. \, V})$ [GeV]")]),
        "pt_tag1":        HistConf([Axis(coll="events", field="jet1_pt",  bins=60, start=0, stop=300, label=r"$p_T(j_1)$ [GeV]")]),
        "pt_tag2":        HistConf([Axis(coll="events", field="jet2_pt",  bins=60, start=0, stop=300, label=r"$p_T(j_2)$ [GeV]")]),
        "eta_tag1":       HistConf([Axis(coll="events", field="jet1_eta", bins=48, start=-4.8, stop=4.8, label=r"$\eta(j_1)$")]),
        "eta_tag2":       HistConf([Axis(coll="events", field="jet2_eta", bins=48, start=-4.8, stop=4.8, label=r"$\eta(j_2)$")]),
        "eta_w_lep":      HistConf([Axis(coll="events", field="w_lep_eta", bins=32, start=-4.0, stop=4.0, label=r"$\eta^{lep_1}$")]),
        "pt_w_lep":       HistConf([Axis(coll="events", field="w_lep_pt",  bins=40, start=0.0,  stop=300.0, label=r"$p_T^{lep_1}$ [GeV]")]),
        "phi_w_lep":      HistConf([Axis(coll="events", field="w_lep_phi", bins=32, start=-4.0, stop=4.0,   label=r"$\phi^{lep_1}$")]),
        "m_ll":           HistConf([Axis(coll="ll", field="m_ll", bins=50, start=50, stop=125.0, label=r"$m_{ll}$ [GeV]")]),
        "lead_wlep_wjet1_dR":       HistConf([Axis(coll="events", field="lead_wlep_wjet1_dR",       bins=40, start=0.0, stop=4.0, label=r"$\Delta R(l,j_1)^{had. \, V}$")]),
        "lead_wlep_wjet2_dR":       HistConf([Axis(coll="events", field="lead_wlep_wjet2_dR",       bins=40, start=0.0, stop=4.0, label=r"$\Delta R(l,j_2)^{had. \, V}$")]),
        "lead_wlep_wfatjet1_dR":    HistConf([Axis(coll="events", field="lead_wlep_wfatjet1_dR",    bins=40, start=0.0, stop=4.0, label=r"$\Delta R(l,AK8)$")]),
        "lead_wlep_w_resolved_dR":  HistConf([Axis(coll="events", field="lead_wlep_w_resolved_dR",  bins=40, start=0.0, stop=4.0, label=r"$\Delta R(l,W_{resolved})$")]),
        "lead_wlep_vbsjet1_dR":     HistConf([Axis(coll="events", field="lead_wlep_vbsjet1_dR",     bins=40, start=0.0, stop=4.0, label=r"$\Delta R(l,j_1)^{forward}$")]),
        "lead_wlep_vbsjet2_dR":     HistConf([Axis(coll="events", field="lead_wlep_vbsjet2_dR",     bins=40, start=0.0, stop=4.0, label=r"$\Delta R(l,j_2)^{forward}$")]),
        "lead_wlep_wfatjet1_deta":  HistConf([Axis(coll="events", field="lead_wlep_wfatjet1_deta",  bins=24, start=0.0, stop=9.0, label=r"$|\Delta\eta_{l,J}^{W}|$")]),
        "lead_wlep_wjet1_deta":     HistConf([Axis(coll="events", field="lead_wlep_wjet1_deta",     bins=24, start=0.0, stop=9.0, label=r"$|\Delta\eta_{l,j_1^{had. \, V}}|$")]),
        "lead_wlep_wjet2_deta":     HistConf([Axis(coll="events", field="lead_wlep_wjet2_deta",     bins=24, start=0.0, stop=9.0, label=r"$|\Delta\eta_{l,j_2^{had. \, V}}|$")]),
        "lead_wlep_w_resolved_deta": HistConf([Axis(coll="events", field="lead_wlep_w_resolved_deta", bins=24, start=2.0, stop=9.0, label=r"$|\Delta\eta_{l,W_{resolved}}|$")]),
        "lead_wlep_MET_dphi":       HistConf([Axis(coll="events", field="lead_wlep_MET_dphi",       bins=32, start=-4, stop=4.0, label=r"$|\Delta\phi_{l,MET}|$")]),
        "lead_wlep_wfatjet1_dphi":  HistConf([Axis(coll="events", field="lead_wlep_wfatjet1_dphi",  bins=32, start=-4, stop=4.0, label=r"$|\Delta\phi_{l,AK8}|$")]),
        "lead_wlep_wjet1_dphi":     HistConf([Axis(coll="events", field="lead_wlep_wjet1_dphi",     bins=32, start=-4, stop=4.0, label=r"$|\Delta\phi_{l,j_1^{had. \, V}}|$")]),
        "lead_wlep_wjet2_dphi":     HistConf([Axis(coll="events", field="lead_wlep_wjet2_dphi",     bins=32, start=-4, stop=4.0, label=r"$|\Delta\phi_{l,j_2^{had. \, V}}|$")]),
        "w_lep_w_resolved_dphi":    HistConf([Axis(coll="events", field="w_lep_w_resolved_dphi",    bins=32, start=-4, stop=4.0, label=r"$|\Delta\phi_{W_{lep},W_{res}}|$")]),
        "w_lep_w_boost_dphi":       HistConf([Axis(coll="events", field="w_lep_w_boost_dphi",       bins=32, start=-4, stop=4.0, label=r"$|\Delta\phi_{W_{lep},W_{boost}}|$")]),
        "wleptonic_eta":  HistConf([Axis(coll="events", field="wleptonic_eta", bins=48, start=-2.4, stop=2.4, label=r"$\eta(W_{leptonic})$")]),
        "wleptonic_pt":   HistConf([Axis(coll="events", field="wleptonic_pt",  bins=48, start=0,    stop=500, label=r"$p_T(W_{leptonic})$")]),
        "fj_pt":          HistConf([Axis(coll="candidate_boost", field="pt",           bins=60, start=150, stop=1000, label=r"$p_T(AK8)$ [GeV]")]),
        "fj_eta":         HistConf([Axis(coll="candidate_boost", field="eta",          bins=48, start=-2.4, stop=2.4,  label=r"$\eta(AK8)$")]),
        "fj_msd":         HistConf([Axis(coll="candidate_boost", field="msoftdrop",    bins=40, start=0,    stop=200,  label=r"$m_{SD}(AK8)$ [GeV]")]),
        "fj_t21":         HistConf([Axis(coll="candidate_boost", field="tau21",        bins=32, start=0,    stop=1.1,  label=r"$\tau_{21}$")]),
        "fj_W_vs_QCD":    HistConf([Axis(coll="candidate_boost", field="particleNet_WvsQCD", bins=32, start=0, stop=1.1, label=r"particleNet_WvsQCD")]),
        "fj_Z_vs_QCD":    HistConf([Axis(coll="candidate_boost", field="particleNet_ZvsQCD", bins=32, start=0, stop=1.1, label=r"particleNet_ZvsQCD")]),
        "fj_pn_mass":     HistConf([Axis(coll="candidate_boost", field="particleNet_mass",   bins=15, start=40, stop=115, label=r"particleNet_mass")]),
        "fj_W_vs_QCD_deeptag":   HistConf([Axis(coll="candidate_boost", field="deepTag_WvsQCD",   bins=32, start=0, stop=1.1, label=r"deepTag_WvsQCD")]),
        "fj_Z_vs_QCD_deeptag":   HistConf([Axis(coll="candidate_boost", field="deepTag_ZvsQCD",   bins=32, start=0, stop=1.1, label=r"deepTag_ZvsQCD")]),
        "fj_W_vs_QCD_deeptagMD": HistConf([Axis(coll="candidate_boost", field="deepTagMD_WvsQCD", bins=32, start=0, stop=1.1, label=r"deepTagMD_WvsQCD")]),
        "fj_Z_vs_QCD_deeptagMD": HistConf([Axis(coll="candidate_boost", field="deepTagMD_ZvsQCD", bins=32, start=0, stop=1.1, label=r"deepTagMD_ZvsQCD")]),
        "z_lep":          HistConf([Axis(coll="events", field="z_lep", bins=40, start=-1.0, stop=1.0, label=r"$Zepp. lepton$")]),
        "z_fat":          HistConf([Axis(coll="events", field="z_fat", bins=40, start=-1.0, stop=1.0, label=r"$Zepp. boosted jet$")]),
        "centrality_resolved": HistConf([Axis(coll="w_had_jets", field="centrality_resolved", bins=40, start=-5.0, stop=5.0, label=r"$Centrality_{resolved}$")]),
        "centrality_boosted":  HistConf([Axis(coll="events",    field="centrality_boosted",  bins=40, start=-5.0, stop=5.0, label=r"$Centrality_{boosted}$")]),
        "qgl_vbs1_resolved": HistConf([Axis(coll="events", field="qgl_vbs1_resolved", bins=40, start=0, stop=1.0, label=r"QGL VBS jet1 (resolved)")]),
        "qgl_vbs2_resolved": HistConf([Axis(coll="events", field="qgl_vbs2_resolved", bins=40, start=0, stop=1.0, label=r"QGL VBS jet2 (resolved)")]),
        "qgl_wjet1_resolved": HistConf([Axis(coll="w_had_jets", field="qgl_wjet1_resolved", bins=40, start=0, stop=1.0, label=r"QGL had. W jet 1")]),
        "qgl_wjet2_resolved": HistConf([Axis(coll="w_had_jets", field="qgl_wjet2_resolved", bins=40, start=0, stop=1.0, label=r"QGL had. W jet 2")]),
        "pt_vbsjet1":     HistConf([Axis(coll="events", field="vbsjet1_pt",  bins=60, start=0,    stop=300, label=r"$p_T(j_1)^{forward}$ [GeV]")]),
        "pt_vbsjet2":     HistConf([Axis(coll="events", field="vbsjet2_pt",  bins=60, start=0,    stop=300, label=r"$p_T(j_2)^{forward}$ [GeV]")]),
        "eta_vbsjet1":    HistConf([Axis(coll="events", field="vbsjet1_eta", bins=48, start=-4.8, stop=4.8, label=r"$\eta(j_1)^{forward}$")]),
        "eta_vbsjet2":    HistConf([Axis(coll="events", field="vbsjet2_eta", bins=48, start=-4.8, stop=4.8, label=r"$\eta(j_2)^{forward}$")]),
        "bjet_lepton_separation": HistConf([Axis(coll="events", field="lep_bjet_dR", bins=32, start=0, stop=9.0, label=r"$\Delta R_{lep,b}$")]),
        "LeadJetIdx":     HistConf([Axis(coll="events", field="jet1_idx", bins=10, start=-1, stop=10, label="tag 1 idx")]),
        "SecondJetIdx":   HistConf([Axis(coll="events", field="jet2_idx", bins=10, start=-1, stop=10, label="tag 2 idx")]),
        "jet_new_neHEF":  HistConf([Axis(coll="JetGood", field="neHEF", bins=10,  start=0, stop=1.2, label="neHEF (good jets)")]),
        "jet_new_chHEF":  HistConf([Axis(coll="JetGood", field="chHEF", bins=100, start=0, stop=1.2, label="chHEF (good jets)")]),
        "jet_new_muEF":   HistConf([Axis(coll="JetGood", field="muEF",  bins=10,  start=0, stop=1.2, label="muEF (good jets)")]),
        "jet_new_neEmEF": HistConf([Axis(coll="JetGood", field="neEmEF",bins=10,  start=0, stop=1.2, label="neEmEF (good jets)")]),
        "jet_new_eta":    HistConf([Axis(coll="JetGood", field="eta",   bins=10,  start=-4.8, stop=4.8, label=r"$\eta(jet_{good})$")]),
        "HT_sum":         HistConf([Axis(coll="events", field="ht_sum", bins=35, start=0, stop=3500, label="reco HT [GeV]")]),
        "bdt_boosted_mu":   HistConf([Axis(coll="events", field="bdt_boosted_mu",   bins=40, start=0, stop=1, label="BDT mu boosted")]),
        "bdt_resolved_mu":  HistConf([Axis(coll="events", field="bdt_resolved_mu",  bins=40, start=0, stop=1, label="BDT mu resolved")]),
        "bdt_boosted_e":    HistConf([Axis(coll="events", field="bdt_boosted_e",    bins=40, start=0, stop=1, label="BDT e boosted")]),
        "bdt_resolved_e":   HistConf([Axis(coll="events", field="bdt_resolved_e",   bins=40, start=0, stop=1, label="BDT e resolved")]),
        "mass_jet1_jet2":   HistConf([Axis(coll="mass", field="jet1_jet2",  bins=100, start=0, stop=2000, label="mass jet1 jet2")]),
        "mass_jet1_jet3":   HistConf([Axis(coll="mass", field="jet1_jet3",  bins=100, start=0, stop=2000, label="mass jet1 jet3")]),
        "mass_jet1_jet4":   HistConf([Axis(coll="mass", field="jet1_jet4",  bins=100, start=0, stop=2000, label="mass jet1 jet4")]),
        "mass_jet1_jet5":   HistConf([Axis(coll="mass", field="jet1_jet5",  bins=100, start=0, stop=2000, label="mass jet1 jet5")]),
        "mass_jet1_jet6":   HistConf([Axis(coll="mass", field="jet1_jet6",  bins=100, start=0, stop=2000, label="mass jet1 jet6")]),
        "mass_jet1_lepton1": HistConf([Axis(coll="mass", field="jet1_lepton1", bins=100, start=0, stop=2000, label="mass jet1 lepton1")]),
        "mass_jet2_jet3":   HistConf([Axis(coll="mass", field="jet2_jet3",  bins=100, start=0, stop=2000, label="mass jet2 jet3")]),
        "mass_jet2_jet4":   HistConf([Axis(coll="mass", field="jet2_jet4",  bins=100, start=0, stop=2000, label="mass jet2 jet4")]),
        "mass_jet2_jet5":   HistConf([Axis(coll="mass", field="jet2_jet5",  bins=100, start=0, stop=2000, label="mass jet2 jet5")]),
        "mass_jet2_jet6":   HistConf([Axis(coll="mass", field="jet2_jet6",  bins=100, start=0, stop=2000, label="mass jet2 jet6")]),
        "mass_jet2_lepton1": HistConf([Axis(coll="mass", field="jet2_lepton1", bins=100, start=0, stop=2000, label="mass jet2 lepton1")]),
        "mass_jet3_jet4":   HistConf([Axis(coll="mass", field="jet3_jet4",  bins=100, start=0, stop=2000, label="mass jet3 jet4")]),
        "mass_jet3_jet5":   HistConf([Axis(coll="mass", field="jet3_jet5",  bins=100, start=0, stop=2000, label="mass jet3 jet5")]),
        "mass_jet3_jet6":   HistConf([Axis(coll="mass", field="jet3_jet6",  bins=100, start=0, stop=2000, label="mass jet3 jet6")]),
        "mass_jet3_lepton1": HistConf([Axis(coll="mass", field="jet3_lepton1", bins=100, start=0, stop=2000, label="mass jet3 lepton1")]),
        "mass_jet4_jet5":   HistConf([Axis(coll="mass", field="jet4_jet5",  bins=100, start=0, stop=2000, label="mass jet4 jet5")]),
        "mass_jet4_jet6":   HistConf([Axis(coll="mass", field="jet4_jet6",  bins=100, start=0, stop=2000, label="mass jet4 jet6")]),
        "mass_jet4_lepton1": HistConf([Axis(coll="mass", field="jet4_lepton1", bins=100, start=0, stop=2000, label="mass jet4 lepton1")]),
        "mass_jet5_jet6":   HistConf([Axis(coll="mass", field="jet5_jet6",  bins=100, start=0, stop=2000, label="mass jet5 jet6")]),
        "mass_jet5_lepton1": HistConf([Axis(coll="mass", field="jet5_lepton1", bins=100, start=0, stop=2000, label="mass jet5 lepton1")]),
        "mass_jet6_lepton1": HistConf([Axis(coll="mass", field="jet6_lepton1", bins=100, start=0, stop=2000, label="mass jet6 lepton1")]),
        "mass_jet1_DeepMETResolutionTune": HistConf([Axis(coll="mass", field="jet1_DeepMETResolutionTune", bins=100, start=0, stop=2000, label="mass jet1 met")]),
        "mass_jet2_DeepMETResolutionTune": HistConf([Axis(coll="mass", field="jet2_DeepMETResolutionTune", bins=100, start=0, stop=2000, label="mass jet2 met")]),
        "mass_jet3_DeepMETResolutionTune": HistConf([Axis(coll="mass", field="jet3_DeepMETResolutionTune", bins=100, start=0, stop=2000, label="mass jet3 met")]),
        "mass_jet4_DeepMETResolutionTune": HistConf([Axis(coll="mass", field="jet4_DeepMETResolutionTune", bins=100, start=0, stop=2000, label="mass jet4 met")]),
        "mass_jet5_DeepMETResolutionTune": HistConf([Axis(coll="mass", field="jet5_DeepMETResolutionTune", bins=100, start=0, stop=2000, label="mass jet5 met")]),
        "mass_jet6_DeepMETResolutionTune": HistConf([Axis(coll="mass", field="jet6_DeepMETResolutionTune", bins=100, start=0, stop=2000, label="mass jet6 met")]),
        "mass_lepton1_DeepMETResolutionTune": HistConf([Axis(coll="mass", field="lepton1_DeepMETResolutionTune", bins=100, start=0, stop=2000, label="mass lepton1 met")]),
        "dR_jet1_jet2":    HistConf([Axis(coll="dR", field="jet1_jet2",    bins=50, start=0, stop=6, label="dR jet1 jet2")]),
        "dR_jet1_jet3":    HistConf([Axis(coll="dR", field="jet1_jet3",    bins=50, start=0, stop=6, label="dR jet1 jet3")]),
        "dR_jet1_jet4":    HistConf([Axis(coll="dR", field="jet1_jet4",    bins=50, start=0, stop=6, label="dR jet1 jet4")]),
        "dR_jet1_jet5":    HistConf([Axis(coll="dR", field="jet1_jet5",    bins=50, start=0, stop=6, label="dR jet1 jet5")]),
        "dR_jet1_jet6":    HistConf([Axis(coll="dR", field="jet1_jet6",    bins=50, start=0, stop=6, label="dR jet1 jet6")]),
        "dR_jet1_lepton1": HistConf([Axis(coll="dR", field="jet1_lepton1", bins=50, start=0, stop=6, label="dR jet1 lepton1")]),
        "dR_jet2_jet3":    HistConf([Axis(coll="dR", field="jet2_jet3",    bins=50, start=0, stop=6, label="dR jet2 jet3")]),
        "dR_jet2_jet4":    HistConf([Axis(coll="dR", field="jet2_jet4",    bins=50, start=0, stop=6, label="dR jet2 jet4")]),
        "dR_jet2_jet5":    HistConf([Axis(coll="dR", field="jet2_jet5",    bins=50, start=0, stop=6, label="dR jet2 jet5")]),
        "dR_jet2_jet6":    HistConf([Axis(coll="dR", field="jet2_jet6",    bins=50, start=0, stop=6, label="dR jet2 jet6")]),
        "dR_jet2_lepton1": HistConf([Axis(coll="dR", field="jet2_lepton1", bins=50, start=0, stop=6, label="dR jet2 lepton1")]),
        "dR_jet3_jet4":    HistConf([Axis(coll="dR", field="jet3_jet4",    bins=50, start=0, stop=6, label="dR jet3 jet4")]),
        "dR_jet3_jet5":    HistConf([Axis(coll="dR", field="jet3_jet5",    bins=50, start=0, stop=6, label="dR jet3 jet5")]),
        "dR_jet3_jet6":    HistConf([Axis(coll="dR", field="jet3_jet6",    bins=50, start=0, stop=6, label="dR jet3 jet6")]),
        "dR_jet3_lepton1": HistConf([Axis(coll="dR", field="jet3_lepton1", bins=50, start=0, stop=6, label="dR jet3 lepton1")]),
        "dR_jet4_jet5":    HistConf([Axis(coll="dR", field="jet4_jet5",    bins=50, start=0, stop=6, label="dR jet4 jet5")]),
        "dR_jet4_jet6":    HistConf([Axis(coll="dR", field="jet4_jet6",    bins=50, start=0, stop=6, label="dR jet4 jet6")]),
        "dR_jet4_lepton1": HistConf([Axis(coll="dR", field="jet4_lepton1", bins=50, start=0, stop=6, label="dR jet4 lepton1")]),
        "dR_jet5_jet6":    HistConf([Axis(coll="dR", field="jet5_jet6",    bins=50, start=0, stop=6, label="dR jet5 jet6")]),
        "dR_jet5_lepton1": HistConf([Axis(coll="dR", field="jet5_lepton1", bins=50, start=0, stop=6, label="dR jet5 lepton1")]),
        "dR_jet6_lepton1": HistConf([Axis(coll="dR", field="jet6_lepton1", bins=50, start=0, stop=6, label="dR jet6 lepton1")]),
        "dR_jet1_DeepMETResolutionTune": HistConf([Axis(coll="dR", field="jet1_DeepMETResolutionTune", bins=50, start=0, stop=6, label="dR jet1 met")]),
        "dR_jet2_DeepMETResolutionTune": HistConf([Axis(coll="dR", field="jet2_DeepMETResolutionTune", bins=50, start=0, stop=6, label="dR jet2 met")]),
        "dR_jet3_DeepMETResolutionTune": HistConf([Axis(coll="dR", field="jet3_DeepMETResolutionTune", bins=50, start=0, stop=6, label="dR jet3 met")]),
        "dR_jet4_DeepMETResolutionTune": HistConf([Axis(coll="dR", field="jet4_DeepMETResolutionTune", bins=50, start=0, stop=6, label="dR jet4 met")]),
        "dR_jet5_DeepMETResolutionTune": HistConf([Axis(coll="dR", field="jet5_DeepMETResolutionTune", bins=50, start=0, stop=6, label="dR jet5 met")]),
        "dR_jet6_DeepMETResolutionTune": HistConf([Axis(coll="dR", field="jet6_DeepMETResolutionTune", bins=50, start=0, stop=6, label="dR jet6 met")]),
        "dR_lepton1_DeepMETResolutionTune": HistConf([Axis(coll="dR", field="lepton1_DeepMETResolutionTune", bins=50, start=0, stop=6, label="dR lepton1 met")]),
        "dphi_jet1_jet2":    HistConf([Axis(coll="dphi", field="jet1_jet2",    bins=64, start=-3.2, stop=3.2, label="dphi jet1 jet2")]),
        "dphi_jet1_jet3":    HistConf([Axis(coll="dphi", field="jet1_jet3",    bins=64, start=-3.2, stop=3.2, label="dphi jet1 jet3")]),
        "dphi_jet1_jet4":    HistConf([Axis(coll="dphi", field="jet1_jet4",    bins=64, start=-3.2, stop=3.2, label="dphi jet1 jet4")]),
        "dphi_jet1_lepton1": HistConf([Axis(coll="dphi", field="jet1_lepton1", bins=64, start=-3.2, stop=3.2, label="dphi jet1 lepton1")]),
        "dphi_jet2_jet3":    HistConf([Axis(coll="dphi", field="jet2_jet3",    bins=64, start=-3.2, stop=3.2, label="dphi jet2 jet3")]),
        "dphi_jet2_jet4":    HistConf([Axis(coll="dphi", field="jet2_jet4",    bins=64, start=-3.2, stop=3.2, label="dphi jet2 jet4")]),
        "dphi_jet2_lepton1": HistConf([Axis(coll="dphi", field="jet2_lepton1", bins=64, start=-3.2, stop=3.2, label="dphi jet2 lepton1")]),
        "dphi_jet3_jet4":    HistConf([Axis(coll="dphi", field="jet3_jet4",    bins=64, start=-3.2, stop=3.2, label="dphi jet3 jet4")]),
        "dphi_jet3_lepton1": HistConf([Axis(coll="dphi", field="jet3_lepton1", bins=64, start=-3.2, stop=3.2, label="dphi jet3 lepton1")]),
        "dphi_jet4_lepton1": HistConf([Axis(coll="dphi", field="jet4_lepton1", bins=64, start=-3.2, stop=3.2, label="dphi jet4 lepton1")]),
        "dphi_jet1_DeepMETResolutionTune": HistConf([Axis(coll="dphi", field="jet1_DeepMETResolutionTune", bins=64, start=-3.2, stop=3.2, label="dphi jet1 met")]),
        "dphi_jet2_DeepMETResolutionTune": HistConf([Axis(coll="dphi", field="jet2_DeepMETResolutionTune", bins=64, start=-3.2, stop=3.2, label="dphi jet2 met")]),
        "dphi_jet3_DeepMETResolutionTune": HistConf([Axis(coll="dphi", field="jet3_DeepMETResolutionTune", bins=64, start=-3.2, stop=3.2, label="dphi jet3 met")]),
        "dphi_jet4_DeepMETResolutionTune": HistConf([Axis(coll="dphi", field="jet4_DeepMETResolutionTune", bins=64, start=-3.2, stop=3.2, label="dphi jet4 met")]),
        "dphi_lepton1_DeepMETResolutionTune": HistConf([Axis(coll="dphi", field="lepton1_DeepMETResolutionTune", bins=64, start=-3.2, stop=3.2, label="dphi lepton1 met")]),
        "deta_jet1_jet2":    HistConf([Axis(coll="deta", field="jet1_jet2",    bins=50, start=-5, stop=5, label="deta jet1 jet2")]),
        "deta_jet1_jet3":    HistConf([Axis(coll="deta", field="jet1_jet3",    bins=50, start=-5, stop=5, label="deta jet1 jet3")]),
        "deta_jet1_jet4":    HistConf([Axis(coll="deta", field="jet1_jet4",    bins=50, start=-5, stop=5, label="deta jet1 jet4")]),
        "deta_jet1_lepton1": HistConf([Axis(coll="deta", field="jet1_lepton1", bins=50, start=-5, stop=5, label="deta jet1 lepton1")]),
        "deta_jet2_jet3":    HistConf([Axis(coll="deta", field="jet2_jet3",    bins=50, start=-5, stop=5, label="deta jet2 jet3")]),
        "deta_jet2_jet4":    HistConf([Axis(coll="deta", field="jet2_jet4",    bins=50, start=-5, stop=5, label="deta jet2 jet4")]),
        "deta_jet2_lepton1": HistConf([Axis(coll="deta", field="jet2_lepton1", bins=50, start=-5, stop=5, label="deta jet2 lepton1")]),
        "deta_jet3_jet4":    HistConf([Axis(coll="deta", field="jet3_jet4",    bins=50, start=-5, stop=5, label="deta jet3 jet4")]),
        "deta_jet3_lepton1": HistConf([Axis(coll="deta", field="jet3_lepton1", bins=50, start=-5, stop=5, label="deta jet3 lepton1")]),
        "deta_jet4_lepton1": HistConf([Axis(coll="deta", field="jet4_lepton1", bins=50, start=-5, stop=5, label="deta jet4 lepton1")]),
        "deta_jet1_DeepMETResolutionTune": HistConf([Axis(coll="deta", field="jet1_DeepMETResolutionTune", bins=50, start=-5, stop=5, label="deta jet1 met")]),
        "deta_jet2_DeepMETResolutionTune": HistConf([Axis(coll="deta", field="jet2_DeepMETResolutionTune", bins=50, start=-5, stop=5, label="deta jet2 met")]),
        "deta_jet3_DeepMETResolutionTune": HistConf([Axis(coll="deta", field="jet3_DeepMETResolutionTune", bins=50, start=-5, stop=5, label="deta jet3 met")]),
        "deta_jet4_DeepMETResolutionTune": HistConf([Axis(coll="deta", field="jet4_DeepMETResolutionTune", bins=50, start=-5, stop=5, label="deta jet4 met")]),
        "deta_lepton1_DeepMETResolutionTune": HistConf([Axis(coll="deta", field="lepton1_DeepMETResolutionTune", bins=50, start=-5, stop=5, label="deta lepton1 met")]),
        "jet1_eta":  HistConf([Axis(coll="jet1", field="eta",           bins=50,  start=-5,   stop=5,   label="jet1 eta")]),
        "jet1_phi":  HistConf([Axis(coll="jet1", field="phi",           bins=64,  start=-3.2, stop=3.2, label="jet1 phi")]),
        "jet1_pt":   HistConf([Axis(coll="jet1", field="pt",            bins=100, start=0,    stop=1000, label="jet1 pt")]),
        "jet1_qgl":  HistConf([Axis(coll="jet1", field="qgl",           bins=50,  start=0,    stop=1,   label="jet1 qgl")]),
        "jet1_btagDeepFlavB": HistConf([Axis(coll="jet1", field="btagDeepFlavB", bins=50, start=0, stop=1, label="jet1 btagDeepFlavB")]),
        "jet2_eta":  HistConf([Axis(coll="jet2", field="eta",           bins=50,  start=-5,   stop=5,   label="jet2 eta")]),
        "jet2_phi":  HistConf([Axis(coll="jet2", field="phi",           bins=64,  start=-3.2, stop=3.2, label="jet2 phi")]),
        "jet2_pt":   HistConf([Axis(coll="jet2", field="pt",            bins=100, start=0,    stop=1000, label="jet2 pt")]),
        "jet2_qgl":  HistConf([Axis(coll="jet2", field="qgl",           bins=50,  start=0,    stop=1,   label="jet2 qgl")]),
        "jet2_btagDeepFlavB": HistConf([Axis(coll="jet2", field="btagDeepFlavB", bins=50, start=0, stop=1, label="jet2 btagDeepFlavB")]),
        "jet3_eta":  HistConf([Axis(coll="jet3", field="eta",           bins=50,  start=-5,   stop=5,   label="jet3 eta")]),
        "jet3_phi":  HistConf([Axis(coll="jet3", field="phi",           bins=64,  start=-3.2, stop=3.2, label="jet3 phi")]),
        "jet3_pt":   HistConf([Axis(coll="jet3", field="pt",            bins=100, start=0,    stop=1000, label="jet3 pt")]),
        "jet3_qgl":  HistConf([Axis(coll="jet3", field="qgl",           bins=50,  start=0,    stop=1,   label="jet3 qgl")]),
        "jet3_btagDeepFlavB": HistConf([Axis(coll="jet3", field="btagDeepFlavB", bins=50, start=0, stop=1, label="jet3 btagDeepFlavB")]),
        "jet4_eta":  HistConf([Axis(coll="jet4", field="eta",           bins=50,  start=-5,   stop=5,   label="jet4 eta")]),
        "jet4_phi":  HistConf([Axis(coll="jet4", field="phi",           bins=64,  start=-3.2, stop=3.2, label="jet4 phi")]),
        "jet4_pt":   HistConf([Axis(coll="jet4", field="pt",            bins=100, start=0,    stop=1000, label="jet4 pt")]),
        "jet4_qgl":  HistConf([Axis(coll="jet4", field="qgl",           bins=50,  start=0,    stop=1,   label="jet4 qgl")]),
        "jet4_btagDeepFlavB": HistConf([Axis(coll="jet4", field="btagDeepFlavB", bins=50, start=0, stop=1, label="jet4 btagDeepFlavB")]),
        "jet5_eta":  HistConf([Axis(coll="jet5", field="eta",           bins=50,  start=-5,   stop=5,   label="jet5 eta")]),
        "jet5_phi":  HistConf([Axis(coll="jet5", field="phi",           bins=64,  start=-3.2, stop=3.2, label="jet5 phi")]),
        "jet5_pt":   HistConf([Axis(coll="jet5", field="pt",            bins=100, start=0,    stop=1000, label="jet5 pt")]),
        "jet5_qgl":  HistConf([Axis(coll="jet5", field="qgl",           bins=50,  start=0,    stop=1,   label="jet5 qgl")]),
        "jet5_btagDeepFlavB": HistConf([Axis(coll="jet5", field="btagDeepFlavB", bins=50, start=0, stop=1, label="jet5 btagDeepFlavB")]),
        "jet6_eta":  HistConf([Axis(coll="jet6", field="eta",           bins=50,  start=-5,   stop=5,   label="jet6 eta")]),
        "jet6_phi":  HistConf([Axis(coll="jet6", field="phi",           bins=64,  start=-3.2, stop=3.2, label="jet6 phi")]),
        "jet6_pt":   HistConf([Axis(coll="jet6", field="pt",            bins=100, start=0,    stop=1000, label="jet6 pt")]),
        "jet6_qgl":  HistConf([Axis(coll="jet6", field="qgl",           bins=50,  start=0,    stop=1,   label="jet6 qgl")]),
        "jet6_btagDeepFlavB": HistConf([Axis(coll="jet6", field="btagDeepFlavB", bins=50, start=0, stop=1, label="jet6 btagDeepFlavB")]),
        "lepton1_eta": HistConf([Axis(coll="lepton1", field="eta", bins=50,  start=-3,   stop=3,   label="lepton1 eta")]),
        "lepton1_phi": HistConf([Axis(coll="lepton1", field="phi", bins=64,  start=-3.2, stop=3.2, label="lepton1 phi")]),
        "lepton1_pt":  HistConf([Axis(coll="lepton1", field="pt",  bins=100, start=0,    stop=500, label="lepton1 pt")]),
    },
    #THIS IS WHERE YOU ADD COLUMN VARIABLES TO SAVE FOR EVENT INFORMATION PROCESSING
    columns = {
        "common": {
           "bycategory": {
                "resolved_e": [ColOut("events",["nJetGood","nCentralJetsGood","mt_w_leptonic", "z_lep",]), ColOut("DeepMETResolutionTune", ["pt","phi"]),ColOut("w_had_jets", ["centrality_resolved","mass"]), ColOut("vbsjets", ["delta_eta","mass", "delta_phi"]), ColOut("jet1",["eta","phi","pt", "qgl", "btagDeepFlavB"]), ColOut("jet2",["eta","phi","pt", "qgl", "btagDeepFlavB"]),ColOut("jet3",["eta","phi","pt", "qgl", "btagDeepFlavB"]), ColOut("jet4",["eta","phi","pt", "qgl", "btagDeepFlavB"]), ColOut("jet5",["eta","phi","pt", "qgl", "btagDeepFlavB"]), ColOut("jet6",["eta","phi","pt", "qgl", "btagDeepFlavB"]), ColOut("lepton1",["eta","phi","pt"]), ColOut("mass",["jet1_jet2", "jet1_jet3", "jet1_jet4", "jet1_jet5", "jet1_jet6", "jet1_lepton1", "jet2_jet3", "jet2_jet4", "jet2_jet5", "jet2_jet6", "jet2_lepton1", "jet3_jet4", "jet3_jet5", "jet3_jet6", "jet3_lepton1", "jet4_jet5", "jet4_jet6", "jet4_lepton1", "jet5_jet6", "jet5_lepton1", "jet6_lepton1","jet1_DeepMETResolutionTune","jet2_DeepMETResolutionTune","jet3_DeepMETResolutionTune","jet4_DeepMETResolutionTune","jet5_DeepMETResolutionTune","jet6_DeepMETResolutionTune","lepton1_DeepMETResolutionTune"]), ColOut("dR",["jet1_jet2", "jet1_jet3", "jet1_jet4", "jet1_jet5", "jet1_jet6", "jet1_lepton1", "jet2_jet3", "jet2_jet4", "jet2_jet5", "jet2_jet6", "jet2_lepton1", "jet3_jet4", "jet3_jet5", "jet3_jet6", "jet3_lepton1", "jet4_jet5", "jet4_jet6", "jet4_lepton1", "jet5_jet6", "jet5_lepton1", "jet6_lepton1","jet1_DeepMETResolutionTune","jet2_DeepMETResolutionTune","jet3_DeepMETResolutionTune","jet4_DeepMETResolutionTune","jet5_DeepMETResolutionTune","jet6_DeepMETResolutionTune","lepton1_DeepMETResolutionTune"]), ColOut("dphi",["jet1_jet2", "jet1_jet3", "jet1_jet4", "jet1_jet5", "jet1_jet6", "jet1_lepton1", "jet2_jet3", "jet2_jet4", "jet2_jet5", "jet2_jet6", "jet2_lepton1", "jet3_jet4", "jet3_jet5", "jet3_jet6", "jet3_lepton1", "jet4_jet5", "jet4_jet6", "jet4_lepton1", "jet5_jet6", "jet5_lepton1", "jet6_lepton1","jet1_DeepMETResolutionTune","jet2_DeepMETResolutionTune","jet3_DeepMETResolutionTune","jet4_DeepMETResolutionTune","jet5_DeepMETResolutionTune","jet6_DeepMETResolutionTune","lepton1_DeepMETResolutionTune"]), ColOut("deta",["jet1_jet2", "jet1_jet3", "jet1_jet4", "jet1_jet5", "jet1_jet6", "jet1_lepton1", "jet2_jet3", "jet2_jet4", "jet2_jet5", "jet2_jet6", "jet2_lepton1", "jet3_jet4", "jet3_jet5", "jet3_jet6", "jet3_lepton1", "jet4_jet5", "jet4_jet6", "jet4_lepton1", "jet5_jet6", "jet5_lepton1", "jet6_lepton1", "jet1_DeepMETResolutionTune","jet2_DeepMETResolutionTune","jet3_DeepMETResolutionTune","jet4_DeepMETResolutionTune","jet5_DeepMETResolutionTune","jet6_DeepMETResolutionTune","lepton1_DeepMETResolutionTune"])],
                "resolved_mu": [ColOut("events",["nJetGood","nCentralJetsGood","mt_w_leptonic", "z_lep",]), ColOut("DeepMETResolutionTune", ["pt","phi"]),ColOut("w_had_jets", ["centrality_resolved","mass"]), ColOut("vbsjets", ["delta_eta","mass", "delta_phi"]), ColOut("jet1",["eta","phi","pt", "qgl", "btagDeepFlavB"]), ColOut("jet2",["eta","phi","pt", "qgl", "btagDeepFlavB"]),ColOut("jet3",["eta","phi","pt", "qgl", "btagDeepFlavB"]), ColOut("jet4",["eta","phi","pt", "qgl", "btagDeepFlavB"]), ColOut("jet5",["eta","phi","pt", "qgl", "btagDeepFlavB"]), ColOut("jet6",["eta","phi","pt", "qgl","btagDeepFlavB"]), ColOut("lepton1",["eta","phi","pt"]), ColOut("mass",["jet1_jet2", "jet1_jet3", "jet1_jet4", "jet1_jet5", "jet1_jet6", "jet1_lepton1", "jet2_jet3", "jet2_jet4", "jet2_jet5", "jet2_jet6", "jet2_lepton1", "jet3_jet4", "jet3_jet5", "jet3_jet6", "jet3_lepton1", "jet4_jet5", "jet4_jet6", "jet4_lepton1", "jet5_jet6", "jet5_lepton1", "jet6_lepton1","jet1_DeepMETResolutionTune","jet2_DeepMETResolutionTune","jet3_DeepMETResolutionTune","jet4_DeepMETResolutionTune","jet5_DeepMETResolutionTune","jet6_DeepMETResolutionTune","lepton1_DeepMETResolutionTune"]), ColOut("dR",["jet1_jet2", "jet1_jet3", "jet1_jet4", "jet1_jet5", "jet1_jet6", "jet1_lepton1", "jet2_jet3", "jet2_jet4", "jet2_jet5", "jet2_jet6", "jet2_lepton1", "jet3_jet4", "jet3_jet5", "jet3_jet6", "jet3_lepton1", "jet4_jet5", "jet4_jet6", "jet4_lepton1", "jet5_jet6", "jet5_lepton1", "jet6_lepton1","jet1_DeepMETResolutionTune","jet2_DeepMETResolutionTune","jet3_DeepMETResolutionTune","jet4_DeepMETResolutionTune","jet5_DeepMETResolutionTune","jet6_DeepMETResolutionTune","lepton1_DeepMETResolutionTune"]), ColOut("dphi",["jet1_jet2", "jet1_jet3", "jet1_jet4", "jet1_jet5", "jet1_jet6", "jet1_lepton1", "jet2_jet3", "jet2_jet4", "jet2_jet5", "jet2_jet6", "jet2_lepton1", "jet3_jet4", "jet3_jet5", "jet3_jet6", "jet3_lepton1", "jet4_jet5", "jet4_jet6", "jet4_lepton1", "jet5_jet6", "jet5_lepton1", "jet6_lepton1","jet1_DeepMETResolutionTune","jet2_DeepMETResolutionTune","jet3_DeepMETResolutionTune","jet4_DeepMETResolutionTune","jet5_DeepMETResolutionTune","jet6_DeepMETResolutionTune","lepton1_DeepMETResolutionTune"]), ColOut("deta",["jet1_jet2", "jet1_jet3", "jet1_jet4", "jet1_jet5", "jet1_jet6", "jet1_lepton1", "jet2_jet3", "jet2_jet4", "jet2_jet5", "jet2_jet6", "jet2_lepton1", "jet3_jet4", "jet3_jet5", "jet3_jet6", "jet3_lepton1", "jet4_jet5", "jet4_jet6", "jet4_lepton1", "jet5_jet6", "jet5_lepton1", "jet6_lepton1", "jet1_DeepMETResolutionTune","jet2_DeepMETResolutionTune","jet3_DeepMETResolutionTune","jet4_DeepMETResolutionTune","jet5_DeepMETResolutionTune","jet6_DeepMETResolutionTune","lepton1_DeepMETResolutionTune"])],
                "boosted_e": [ColOut("events",["nJetGood", "nCentralJetsGood", "nFatJetGood","mt_w_leptonic","z_lep","z_fat","centrality_boosted"]), ColOut("DeepMETResolutionTune", ["pt","phi"]), ColOut("vbsjets", ["mass", "delta_eta", "delta_phi"]), ColOut("fatjet1", ["msoftdrop","btagDeepB","pt","eta","phi","particleNet_WvsQCD","particleNet_ZvsQCD","tau1","tau2","tau3","tau4"]), ColOut("jet1",["eta","phi","pt", "qgl", "btagDeepFlavB"]), ColOut("jet2",["eta","phi","pt", "qgl", "btagDeepFlavB"]),ColOut("jet3",["eta","phi","pt", "qgl", "btagDeepFlavB"]), ColOut("jet4",["eta","phi","pt", "qgl", "btagDeepFlavB"]), ColOut("mass",["jet1_jet2","jet1_jet3","jet1_jet4","jet1_lepton1","jet2_jet3","jet2_jet4","jet2_lepton1","jet3_jet4","jet3_lepton1","jet4_lepton1","jet1_DeepMETResolutionTune","jet2_DeepMETResolutionTune","jet3_DeepMETResolutionTune","jet4_DeepMETResolutionTune","lepton1_DeepMETResolutionTune","jet1_fatjet1","jet2_fatjet1","jet3_fatjet1","jet4_fatjet1","lepton1_fatjet1","DeepMETResolutionTune_fatjet1"]), ColOut("dR",["jet1_jet2","jet1_jet3","jet1_jet4","jet1_lepton1","jet2_jet3","jet2_jet4","jet2_lepton1","jet3_jet4","jet3_lepton1","jet4_lepton1","jet1_DeepMETResolutionTune","jet2_DeepMETResolutionTune","jet3_DeepMETResolutionTune","jet4_DeepMETResolutionTune","lepton1_DeepMETResolutionTune","jet1_fatjet1","jet2_fatjet1","jet3_fatjet1","jet4_fatjet1","lepton1_fatjet1","DeepMETResolutionTune_fatjet1"]), ColOut("dphi",["jet1_jet2","jet1_jet3","jet1_jet4","jet1_lepton1","jet2_jet3","jet2_jet4","jet2_lepton1","jet3_jet4","jet3_lepton1","jet4_lepton1","jet1_DeepMETResolutionTune","jet2_DeepMETResolutionTune","jet3_DeepMETResolutionTune","jet4_DeepMETResolutionTune","lepton1_DeepMETResolutionTune","jet1_fatjet1","jet2_fatjet1","jet3_fatjet1","jet4_fatjet1","lepton1_fatjet1","DeepMETResolutionTune_fatjet1"]), ColOut("deta",["jet1_jet2","jet1_jet3","jet1_jet4","jet1_lepton1","jet2_jet3","jet2_jet4","jet2_lepton1","jet3_jet4","jet3_lepton1","jet4_lepton1","jet1_DeepMETResolutionTune","jet2_DeepMETResolutionTune","jet3_DeepMETResolutionTune","jet4_DeepMETResolutionTune","lepton1_DeepMETResolutionTune","jet1_fatjet1","jet2_fatjet1","jet3_fatjet1","jet4_fatjet1","lepton1_fatjet1","DeepMETResolutionTune_fatjet1"])],
                "boosted_mu": [ColOut("events",["nJetGood", "nCentralJetsGood", "nFatJetGood","mt_w_leptonic","z_lep","z_fat","centrality_boosted"]), ColOut("DeepMETResolutionTune", ["pt","phi"]), ColOut("vbsjets", ["mass", "delta_eta", "delta_phi"]), ColOut("fatjet1", ["msoftdrop","btagDeepB","pt","eta","phi","particleNet_WvsQCD","particleNet_ZvsQCD","tau1","tau2","tau3","tau4"]), ColOut("jet1",["eta","phi","pt", "qgl", "btagDeepFlavB"]), ColOut("jet2",["eta","phi","pt", "qgl", "btagDeepFlavB"]),ColOut("jet3",["eta","phi","pt", "qgl", "btagDeepFlavB"]), ColOut("jet4",["eta","phi","pt", "qgl", "btagDeepFlavB"]), ColOut("mass",["jet1_jet2","jet1_jet3","jet1_jet4","jet1_lepton1","jet2_jet3","jet2_jet4","jet2_lepton1","jet3_jet4","jet3_lepton1","jet4_lepton1","jet1_DeepMETResolutionTune","jet2_DeepMETResolutionTune","jet3_DeepMETResolutionTune","jet4_DeepMETResolutionTune","lepton1_DeepMETResolutionTune","jet1_fatjet1","jet2_fatjet1","jet3_fatjet1","jet4_fatjet1","lepton1_fatjet1","DeepMETResolutionTune_fatjet1"]), ColOut("dR",["jet1_jet2","jet1_jet3","jet1_jet4","jet1_lepton1","jet2_jet3","jet2_jet4","jet2_lepton1","jet3_jet4","jet3_lepton1","jet4_lepton1","jet1_DeepMETResolutionTune","jet2_DeepMETResolutionTune","jet3_DeepMETResolutionTune","jet4_DeepMETResolutionTune","lepton1_DeepMETResolutionTune","jet1_fatjet1","jet2_fatjet1","jet3_fatjet1","jet4_fatjet1","lepton1_fatjet1","DeepMETResolutionTune_fatjet1"]), ColOut("dphi",["jet1_jet2","jet1_jet3","jet1_jet4","jet1_lepton1","jet2_jet3","jet2_jet4","jet2_lepton1","jet3_jet4","jet3_lepton1","jet4_lepton1","jet1_DeepMETResolutionTune","jet2_DeepMETResolutionTune","jet3_DeepMETResolutionTune","jet4_DeepMETResolutionTune","lepton1_DeepMETResolutionTune","jet1_fatjet1","jet2_fatjet1","jet3_fatjet1","jet4_fatjet1","lepton1_fatjet1","DeepMETResolutionTune_fatjet1"]), ColOut("deta",["jet1_jet2","jet1_jet3","jet1_jet4","jet1_lepton1","jet2_jet3","jet2_jet4","jet2_lepton1","jet3_jet4","jet3_lepton1","jet4_lepton1","jet1_DeepMETResolutionTune","jet2_DeepMETResolutionTune","jet3_DeepMETResolutionTune","jet4_DeepMETResolutionTune","lepton1_DeepMETResolutionTune","jet1_fatjet1","jet2_fatjet1","jet3_fatjet1","jet4_fatjet1","lepton1_fatjet1","DeepMETResolutionTune_fatjet1"])],
                }
        }
    },
)
