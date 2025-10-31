# example_config_semileptonic.py
import os, cloudpickle
from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_functions import get_HLTsel, get_nPVgood, goldenJson, eventFlags
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.parameters.histograms import *
from pocket_coffea.lib.weights.common import common_weights
from pocket_coffea.parameters import defaults
from pocket_coffea.lib.columns_manager import ColOut

import numpy as np
import awkward as ak
from pocket_coffea.lib.weights import WeightWrapper, WeightData, WeightDataMultiVariation, WeightLambda
from pocket_coffea.lib.scale_factors import sf_pileup_reweight


import workflow, custom_cut_functions
from workflow import VBSSemileptonicProcessor
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
    msd_window_cut_mu
)


# class PileupWeight(WeightWrapper):
#     name = "PileupWeight"
#     has_variations = True

#     def __init__(self, parameters, metadata):
#         super().__init__(parameters, metadata)
#         self.year = metadata["year"]
#         self._variations = parameters.pileupJSONfiles[self.year]["variations"]
#         self.params = parameters

#     def compute(self, events, size, shape_variation):
#         if shape_variation == "nominal":
#             sf, sfup, sfdown = sf_pileup_reweight(self.params, events, self.year)
#             sf_data = {
#                 "nominal": sf,
#                 "up": sfup,
#                 "down": sfdown
#             }
#             return WeightDataMultiVariation(
#                 name=self.name,
#                 nominal=sf_data["nominal"],
#                 variations=self._variations["up"] + self._variations["down"],
#                 up=[sf_data[var] for var in self._variations["up"]],
#                 down=[sf_data[var] for var in self._variations["down"]]
#             )
#         else:
#             return WeightData(
#                 name=self.name,
#                 nominal=np.ones(size),
#             )


cloudpickle.register_pickle_by_value(workflow)
cloudpickle.register_pickle_by_value(custom_cut_functions)

localdir = os.path.dirname(os.path.abspath(__file__))


default_parameters = defaults.get_default_parameters( 
    group_tags={"EGM": 
        {   "Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15":"latest", 
            "Run3-23DSep23-Summer23BPix-NanoAODv12": "latest",
            "Run3-23CSep23-Summer23-NanoAODv12": "latest",
            "Run3-22EFGSep23-Summer22EE-NanoAODv12": "latest",
            "Run3-22CDSep23-Summer22-NanoAODv12": "latest",
            "Run2-2018-UL-NanoAODv9": "latest",
            "Run2-2017-UL-NanoAODv9": "latest",
            "Run2-2016preVFP-UL-NanoAODv9": "latest",
            "Run2-2016postVFP-UL-NanoAODv9": "latest"
        }
    }
)
defaults.register_configuration_dir("config_dir", localdir + "/params")
parameters = defaults.merge_parameters_from_files(
    default_parameters,
    f"{localdir}/params/object_preselection_run2.yaml",
    f"{localdir}/params/triggers.yaml",
    f"{localdir}/params/plotting.yaml",
    f"{localdir}/params/pileup.yaml",
    
    update=True,
)

PileupWeight = WeightLambda.wrap_func(
    name="PileupWeight",
    function=lambda params, metadata, events, size, shape_variations:
        sf_pileup_reweight(params, events, metadata["year"]),
    has_variations=True  # no list of variations it means only up and down
    )

cfg = Configurator(
    parameters=parameters,
    datasets={
        "jsons": [
            f"{localdir}/datasets/DYJetsToLL_M-10to50_TuneCP5_13TeV-amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/DYJetsToLL_M-10to50_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            f"{localdir}/datasets/DYJetsToLL_M-4to50_HT-100to200_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            f"{localdir}/datasets/DYJetsToLL_M-4to50_HT-200to400_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            f"{localdir}/datasets/DYJetsToLL_M-4to50_HT-400to600_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            f"{localdir}/datasets/DYJetsToLL_M-4to50_HT-600toInf_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            f"{localdir}/datasets/DYJetsToLL_M-4to50_HT-70to100_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            f"{localdir}/datasets/DYJetsToLL_M-50_HT-100to200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8.json",
            f"{localdir}/datasets/DYJetsToLL_M-50_HT-1200to2500_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8.json",
            f"{localdir}/datasets/DYJetsToLL_M-50_HT-200to400_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8.json",
            f"{localdir}/datasets/DYJetsToLL_M-50_HT-2500toInf_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8.json",
            f"{localdir}/datasets/DYJetsToLL_M-50_HT-400to600_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8.json",
            f"{localdir}/datasets/DYJetsToLL_M-50_HT-600to800_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8.json",
            f"{localdir}/datasets/DYJetsToLL_M-50_HT-70to100_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8.json",
            f"{localdir}/datasets/DYJetsToLL_M-50_HT-800to1200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8.json",
            f"{localdir}/datasets/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            f"{localdir}/datasets/GluGluWWToLNuQQ_TuneCP5_13TeV_madgraph-pythia8.json",
            f"{localdir}/datasets/ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8.json",
            f"{localdir}/datasets/ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8.json",
            f"{localdir}/datasets/ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8.json",
            f"{localdir}/datasets/ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8.json",
            f"{localdir}/datasets/ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8.json",
            f"{localdir}/datasets/SingleMuon.json",
            f"{localdir}/datasets/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8.json",
            f"{localdir}/datasets/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8.json",
            f"{localdir}/datasets/WGToLNuG_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            f"{localdir}/datasets/WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            f"{localdir}/datasets/WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            f"{localdir}/datasets/WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            f"{localdir}/datasets/WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            f"{localdir}/datasets/WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            f"{localdir}/datasets/WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            f"{localdir}/datasets/WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            f"{localdir}/datasets/WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            f"{localdir}/datasets/WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            f"{localdir}/datasets/WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8.json",
            f"{localdir}/datasets/WWZ_4F_TuneCP5_13TeV-amcatnlo-pythia8.json",
            f"{localdir}/datasets/WZTo3LNu_mllmin01_NNPDF31_TuneCP5_13TeV_powheg_pythia8.json",
            f"{localdir}/datasets/WZZ_TuneCP5_13TeV-amcatnlo-pythia8.json",
            f"{localdir}/datasets/WminusTo2JZTo2LJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            f"{localdir}/datasets/WminusTo2JZTo2LJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            f"{localdir}/datasets/WminusToLNuWminusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            f"{localdir}/datasets/WminusToLNuWminusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            f"{localdir}/datasets/WminusToLNuZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            f"{localdir}/datasets/WminusToLNuZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            f"{localdir}/datasets/WplusTo2JWminusToLNuJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            f"{localdir}/datasets/WplusTo2JZTo2LJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            f"{localdir}/datasets/WplusTo2JZTo2LJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            f"{localdir}/datasets/WplusToLNuWminusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            f"{localdir}/datasets/WplusToLNuWminusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV.json",
            f"{localdir}/datasets/WplusToLNuWminusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            f"{localdir}/datasets/WplusToLNuWplusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            f"{localdir}/datasets/WplusToLNuWplusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            f"{localdir}/datasets/WplusToLNuZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            f"{localdir}/datasets/WplusToLNuZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            f"{localdir}/datasets/ZGToLLG_01J_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/ZTo2LZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            f"{localdir}/datasets/ZTo2LZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            f"{localdir}/datasets/ZZZ_TuneCP5_13TeV-amcatnlo-pythia8.json",
            f"{localdir}/datasets/ttWJets_TuneCP5_13TeV_madgraphMLM_pythia8.json",
            f"{localdir}/datasets/ttZJets_TuneCP5_13TeV_madgraphMLM_pythia8.json",
        ],
        "filter": {
            "samples": [
                
            #########
            ## RUN 2 BKG
            #########
            # "DYJetsToLL_M-10to50_TuneCP5_13TeV-amcatnloFXFX-pythia8",
            # # "DYJetsToLL_M-10to50_TuneCP5_13TeV-madgraphMLM-pythia8",

            # #"DYJetsToLL_M-4to50_HT-100to200_TuneCP5_13TeV-madgraphMLM-pythia8",
            # "DYJetsToLL_M-4to50_HT-200to400_TuneCP5_13TeV-madgraphMLM-pythia8",
            # "DYJetsToLL_M-4to50_HT-400to600_TuneCP5_13TeV-madgraphMLM-pythia8",
            # "DYJetsToLL_M-4to50_HT-600toInf_TuneCP5_13TeV-madgraphMLM-pythia8",
            # # "DYJetsToLL_M-4to50_HT-70to100_TuneCP5_13TeV-madgraphMLM-pythia8",

            
            # "DYJetsToLL_M-50_HT-100to200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
            # "DYJetsToLL_M-50_HT-1200to2500_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
            # "DYJetsToLL_M-50_HT-200to400_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
            # "DYJetsToLL_M-50_HT-2500toInf_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
            # "DYJetsToLL_M-50_HT-400to600_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
            # "DYJetsToLL_M-50_HT-600to800_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
            # "DYJetsToLL_M-50_HT-70to100_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
            # "DYJetsToLL_M-50_HT-800to1200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",

            # "DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8",
            # # #"DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8",

            
            
            # "ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8",
            # "ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8",
            # "ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8",
            # "ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8",
            # "ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8",

            # "ttWJets_TuneCP5_13TeV_madgraphMLM_pythia8",
            # "ttZJets_TuneCP5_13TeV_madgraphMLM_pythia8",

            # "TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8",
            # "TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8",

            

            # "WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8",
            # "WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8",
            # "WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8",
            # "WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8",
            # "WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8",
            # "WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8",
            # "WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8",
            # "WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8",

            # "WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8",
            # # # "WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8",

            # "GluGluWWToLNuQQ_TuneCP5_13TeV_madgraph-pythia8",
            # "WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8",
            # "WWZ_4F_TuneCP5_13TeV-amcatnlo-pythia8",
            # "WZTo3LNu_mllmin01_NNPDF31_TuneCP5_13TeV_powheg_pythia8",
            # "WZZ_TuneCP5_13TeV-amcatnlo-pythia8",
            # "ZGToLLG_01J_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8",
            # "ZZZ_TuneCP5_13TeV-amcatnlo-pythia8",

            # "WminusTo2JZTo2LJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            # "WminusToLNuWminusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            # "WminusToLNuZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            # "WplusTo2JZTo2LJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            # "WplusTo2JWminusToLNuJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            # "WplusToLNuWminusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            # "WplusToLNuWplusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            # "WplusToLNuZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            # "ZTo2LZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",

            
            
            
            
           
            # ###### SIGNAL #########
            "WminusTo2JZTo2LJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            # "WminusToLNuWminusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            # "WminusToLNuZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            # "WplusTo2JWminusToLNuJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8", #WplusTo2JWminusToLNuJJ missing in QCD
            # "WplusTo2JZTo2LJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            # "WplusToLNuWminusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            # "WplusToLNuWplusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            # "WplusToLNuZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            # "ZTo2LZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",


            # ##### DATA #######
            # "SingleMuon", #2017B data
            ],
            "year": ["2017"],
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
        get_HLTsel(primaryDatasets=["SingleMuon", "SingleEle"]),
    ],

    # 2) preselections 
    preselections=[vbs_semileptonic_presel],

   
    categories={
        "baseline": [passthrough],
        # # "whad_peak_e": [whad_window_cut_e],  # |mjj^W - 80.4| < window
        "boosted_jet_in_window_e": [msd_window_cut_e],
        "whad_withbveto_e":  [whad_window_cut_bveto_e],
        #"whad_peak_mu": [whad_window_cut_mu],  # |mjj^W - 80.4| < window
        "boosted_jet_in_window_mu": [msd_window_cut_mu],
        "whad_withbveto_mu":  [whad_window_cut_bveto_mu],

    },

   
    weights_classes=common_weights+[PileupWeight],
    weights={"common": {"inclusive": ["genWeight", "lumi", "XS", "PileupWeight", "sf_mu_id", "sf_mu_iso", "sf_ele_id", "sf_ele_reco"]}},
    variations={"weights": {"common": {"inclusive": ["PileupWeight", "sf_mu_id","sf_mu_iso","sf_ele_id","sf_ele_reco"]}}}, #"pileup"

   
    variables={
        # "psweight":     HistConf([Axis(coll="events", field="PSWeight", bins=30, start=-1.0, stop=12, label="PSWeight")]),
        # "lheweight":        HistConf([Axis(coll="events", field="LHEWeight", bins=30, start=-1.0, stop=12, label="LHEWeight")]),
        # "LHEReweightingWeight":     HistConf([Axis(coll="events", field="LHEReweightingWeight", bins=50, start=-1.0, stop=40, label="LHEReweightingWeight")]),
        # "LHEPdfWeight":     HistConf([Axis(coll="events", field="LHEPdfWeight", bins=50, start=-1, stop=40, label="LHEpdfweight")]),
        # "LHEScaleWeight":       HistConf([Axis(coll="events", field="LHEScaleWeight", bins=50, start=-1, stop=40, label="(LHEScaleWeight)")]),
        "nJets":      HistConf([Axis(coll="events", field="nJetGood", bins=12, start=0, stop=12, label="N(jets)")]),
        "nBJets":     HistConf([Axis(coll="events", field="nBJetGood", bins=8, start=0, stop=8, label="N(bjets)")]),
        "nBJet_csv":    HistConf([Axis(coll="events", field="nBJet_csv", bins=8, start=0, stop=8, label="N(bjets_csv)")]),
        "nCentralJets": HistConf([Axis(coll="events", field="nCentralJetsGood", bins=12, start=0, stop=12, label="N(Central Jets)")]),
        "nFatJets": HistConf([Axis(coll="events", field="nFatJetGood", bins=4, start=0, stop=4, label="N(Fat Jets)")]),
        "nFatJetCentral": HistConf([Axis(coll="events", field="nFatJetCentral", bins=4, start=0, stop=4, label="N(Central Fat Jets)")]),
        "nLeptonLoose":      HistConf([Axis(coll="events", field="nLeptonLoose", bins=4, start=0, stop=4, label="N(Lepton Loose)")]),
        "nFatJetCandidate": HistConf([Axis(coll="events", field="nFatJetCandidate", bins=4, start=0, stop=4, label="N(Candidate Fat Jets)")]),
        "nFarAK4Jets":  HistConf([Axis(coll="events", field="nFarAK4Jets", bins=8, start=0, stop=8, label="N(nFarAK4Jets)")]),
        "nMuonGood":     HistConf([Axis(coll="events", field="nMuonGood", bins=6, start=0, stop=6, label="N(muon good)")]),
        "nElectronGood":      HistConf([Axis(coll="events", field="nElectronGood", bins=6, start=0, stop=6, label="N(electron good)")]),
        "nLeptonGood":    HistConf([Axis(coll="events", field="nLeptonGood", bins=6, start=0, stop=6, label="N(lepton good)")]),
        #"nGenJet":      HistConf([Axis(coll="events", field="nGenJet", bins=6, start=0, stop=6, label="N(gen jet)")]),
        # "flav_genjet_hadron":       HistConf([Axis(coll="GenJet", field="hadronFlavour", bins=60, start=-30, stop=30, label="gen jet flav (hadron))")]),
        # "flav_jet_hadron":       HistConf([Axis(coll="Jet", field="hadronFlavour", bins=60, start=-30, stop=30, label="jet flav (hadron))")]),
        # "flav_genjet_parton":       HistConf([Axis(coll="GenJet", field="partonFlavour", bins=60, start=-30, stop=30, label="gen jet flav (parton))")]),
        # "flav_jet_parton":       HistConf([Axis(coll="Jet", field="partonFlavour", bins=60, start=-30, stop=30, label="jet flav (parton))")]),
        "btagDeepFlavB":    HistConf([Axis(coll="JetGood", field="btagDeepFlavB", bins=20, start=0, stop=1, label="deepFlavB discrim score")]),
        "btagDeepB":        HistConf([Axis(coll="JetGood", field="btagDeepB", bins=20, start=0, stop=1, label="deepCSV discrim score")]),

        #"nCleanJet_30":     HistConf([Axis(coll="events", field="nCleanJet_30", bins=12, start=0, stop=12, label="N(jet pt >= 30 GeV)")]),

        #"genmatch_btagDeepFlavB":    HistConf([Axis(coll="BJet_genmatch", field="btagDeepFlavB", bins=20, start=0, stop=1, label="Gen matched deepFlavB discrim score")]),
        #"genmatch_btagDeepB":        HistConf([Axis(coll="BJet_genmatch", field="btagDeepB", bins=20, start=0, stop=1, label="Gen matched deepCSV discrim score")]),
        #"cand_btagDeepFlavB":    HistConf([Axis(coll="BJet_csv", field="btagDeepFlavB", bins=20, start=0, stop=1, label="deepFlavB discrim score (b jet)")]),
        #"cand_btagDeepB":        HistConf([Axis(coll="BJet_csv", field="btagDeepB", bins=20, start=0, stop=1, label="deepCSV discrim score (b jet)")]),
        #"JetGood_tagger_check_deepFlavB":    HistConf([Axis(coll="JetGood_tagger_check", field="btagDeepFlavB", bins=20, start=0, stop=1, label="deepFlavB discrim score (all jet)")]),
        #"JetGood_tagger_check_deepB":        HistConf([Axis(coll="JetGood_tagger_check", field="btagDeepB", bins=20, start=0, stop=1, label="deepCSV discrim score (all jet)")]),
        "leading_bscore":      HistConf([Axis(coll="events", field="leading_bscore", bins=33, start=0, stop=1, label="max(deepCSV discrim score)")]),
         

        #pileup check
        'nTrueInt':      HistConf([Axis(coll="Pileup", field="nTrueInt", bins=50, start=0, stop=100, label="nTrueInt")]),
        'pudensity':       HistConf([Axis(coll="Pileup", field="pudensity", bins=100, start=0, stop=8, label="pudensity")]),
        'gpudensity':      HistConf([Axis(coll="Pileup", field="gpudensity", bins=50, start=0, stop=1, label="gpudensity")]),
        'nPU':      HistConf([Axis(coll="Pileup", field="nPU", bins=50, start=0, stop=100, label="nPU")]),
        'sumEOOT':      HistConf([Axis(coll="Pileup", field="nTrueInt", bins=50, start=0, stop=400, label="sumEOOT")]),
        'sumLOOT':      HistConf([Axis(coll="Pileup", field="nTrueInt", bins=20, start=0, stop=120, label="sumLOOT")]),


        #PV check
        'npvs':   HistConf([Axis(coll="PV", field="npvs", bins=20, start=0, stop=100, label="nPV")]),
        'npvsGood':   HistConf([Axis(coll="PV", field="npvsGood", bins=20, start=0, stop=100, label=r"$nPV_{good}$")]),
        # MET and mT
        "met":        HistConf([Axis(coll="MET", field="pt", bins=50, start=0, stop=250, label=r"$p_T^{miss}$ [GeV]")]),
        "met_phi":    HistConf([Axis(coll="MET", field="phi", bins=50, start=-4, stop=4, label=r"$\phi^{miss}$ [GeV]")]),
        "puppimet_phi":    HistConf([Axis(coll="PuppiMET", field="phi", bins=50, start=-4, stop=4, label=r"$\phi^{miss}$ [GeV]")]),
        "puppimet":        HistConf([Axis(coll="PuppiMET", field="pt", bins=50, start=0, stop=250, label=r"$p_T^{miss}$ [GeV]")]),
        "mt_w_lep":   HistConf([Axis(coll="events", field="mt_w_leptonic", bins=30, start=0, stop=200, label=r"$m_T(W_{lep})$ [GeV]")]),
        "neutrino_pz":  HistConf([Axis(coll="events", field="neutrino_pz", bins=50, start=0, stop=250, label=r"$p_z^{\nu}$ [GeV]")]),
        "neutrino_eta":  HistConf([Axis(coll="events", field="neutrino_eta", bins=32, start=-4.0, stop=4.0, label=r"$\eta^{\nu}$ [GeV]")]),
        "neutrino_deta": HistConf([Axis(coll="events", field="lead_wlep_neutrino_deta", bins=32, start=0, stop=4.0, label=r"$\delta\eta^{l,\nu}$")]),
        "neutrino_dR": HistConf([Axis(coll="events", field="lead_wlep_neutrino_dR", bins=32, start=0.0, stop=4, label=r"$\delta R^{l,\nu}$")]),
       

        # Tagging jets (VBS)
        "mjj_vbs":    HistConf([Axis(coll="vbsjets", field="mass", bins=50, start=300, stop=4000, label=r"$M_{jj}^{forward}$ [GeV]")]),
        "deta_vbs":   HistConf([Axis(coll="vbsjets", field="delta_eta", bins=36, start=0, stop=9.0, label=r"$|\Delta\eta_{jj}^{forward}|$")]),
        "dR_vbs":     HistConf([Axis(coll="events", field="vbs_dR", bins=40, start=0.0, stop=7.0, label=r"$\Delta R(jj)^{forward}$")]),
        "dR_fj_vbs1":     HistConf([Axis(coll="events", field="vbs1_fj_dR", bins=40, start=0.0, stop=7.0, label=r"$\Delta R(J^W \,j_{forward_1})$")]),
        "dR_fj_vbs2":     HistConf([Axis(coll="events", field="vbs2_fj_dR", bins=40, start=0.0, stop=7.0, label=r"$\Delta R(J^W\,j_{forward_1})$")]),
        "mjj_vbs_boost":    HistConf([Axis(coll="vbsjets_boost", field="mass", bins=50, start=300, stop=4000, label=r"$M_{jj}^{forward_{boost}}$ [GeV]")]),
        "deta_vbs_boost":   HistConf([Axis(coll="vbsjets_boost", field="delta_eta", bins=36, start=0, stop=9.0, label=r"$|\Delta\eta_{jj}^{forward_{boost}}|$")]),
        "dR_vbs_boost":     HistConf([Axis(coll="events", field="vbs_boost_dR", bins=40, start=0.0, stop=7.0, label=r"$\Delta R(jj)^{forward_{boost}}$")]),
       
        "jet_id":   HistConf([Axis(coll="JetGood", field="jetId", bins=10, start=0, stop=10, label="Jet id")]),
        "jet_rel_iso":  HistConf([Axis(coll="LeptonGood", field="jetRelIso", bins=50, start=0, stop=2, label="Jet iso in lep")]),
        #"lepton pdg":  HistConf([Axis(coll="LeptonGood", field="pdgId", bins=50, start=-15, stop=15, label="lepton id")]),
        "dxy_mu":   HistConf([Axis(coll="LeptonGood", field="dxy", bins=50, start=0, stop=0.5, label="dxy mu")]),
        "dxy_ele":  HistConf([Axis(coll="LeptonGood", field="dxy", bins=50, start=0, stop=0.2, label="dxy ele")]),
        "dz_mu":    HistConf([Axis(coll="LeptonGood", field="dz", bins=50, start=0, stop=1, label="dz mu")]),
        "dz_ele":   HistConf([Axis(coll="LeptonGood", field="dz", bins=50, start=0, stop=0.5, label="dz ele")]),

        # W hadronic
        "m_jj_w":     HistConf([Axis(coll="w_had_jets", field="mass", bins=40, start=65, stop=105, label=r"$M_{jj}^{W\,had}$ [GeV]")]),
        "pt_jj_w":     HistConf([Axis(coll="w_had_jets", field="pt", bins=40, start=40, stop=210, label=r"$p_T(jj^{W\,had})$ [GeV]")]),
        "dR_w_had":   HistConf([Axis(coll="events", field="w_had_dR", bins=40, start=0.0, stop=4.0, label=r"$\Delta R(jj)^{W\,had}$")]),
        "eta_w_had1":   HistConf([Axis(coll="events", field="w_had_jet1_eta", bins=48, start=-4.0, stop=4.0, label=r"$\eta(j2_{W\,had})$ [GeV]")]),
        "eta_w_had2":   HistConf([Axis(coll="events", field="w_had_jet2_eta", bins=48, start=-4.0, stop=4.0, label=r"$\eta(j2_{W\,had})$ [GeV]")]),
        "pt_w_had1":   HistConf([Axis(coll="events", field="w_had_jet1_pt", bins=60, start=0.0, stop=300.0, label=r"$p_T(j1_{W\,had})$ [GeV]")]),
        "pt_w_had2":   HistConf([Axis(coll="events", field="w_had_jet2_pt", bins=60, start=0.0, stop=300.0, label=r"$p_T(j2_{W\,had})$ [GeV]")]),
        "phi_w_had1":   HistConf([Axis(coll="events", field="w_had_jet1_phi", bins=48, start=-4.0, stop=4.0, label=r"$\phi(j2_{W\,had})$ [GeV]")]),
        "phi_w_had2":   HistConf([Axis(coll="events", field="w_had_jet2_phi", bins=48, start=-4.0, stop=4.0, label=r"$\phi(j2_{W\,had})$ [GeV]")]),
        # jets leading
        "pt_tag1":    HistConf([Axis(coll="events", field="jet1_pt", bins=60, start=0, stop=300, label=r"$p_T(j_1)$ [GeV]")]),
        "pt_tag2":    HistConf([Axis(coll="events", field="jet2_pt", bins=60, start=0, stop=300, label=r"$p_T(j_2)$ [GeV]")]),
        "eta_tag1":   HistConf([Axis(coll="events", field="jet1_eta", bins=48, start=-4.8, stop=4.8, label=r"$\eta(j_1)$")]),
        "eta_tag2":   HistConf([Axis(coll="events", field="jet2_eta", bins=48, start=-4.8, stop=4.8, label=r"$\eta(j_2)$")]),
        "phi_tag1":   HistConf([Axis(coll="events", field="jet1_phi", bins=48, start=-4., stop=4., label=r"$\phi(j_1)$")]),
        "phi_tag2":   HistConf([Axis(coll="events", field="jet2_phi", bins=48, start=-4., stop=4., label=r"$\phi(j_2)$")]),
        # lead lepton
        "eta_w_lep":   HistConf([Axis(coll="events", field="w_lep_eta", bins=32, start=-4.0, stop=4.0, label=r"$\eta^{lep_1}$ ")]),
        "pt_w_lep":   HistConf([Axis(coll="events", field="w_lep_pt", bins=40, start=0.0, stop=300.0, label=r"$p_T^{lep_1}$ [GeV]")]),
        "phi_w_lep":   HistConf([Axis(coll="events", field="w_lep_phi", bins=32, start=-4.0, stop=4.0, label=r"$\phi^{lep_1}$ ")]),
        "m_ll":   HistConf([Axis(coll="ll", field="m_ll", bins=50, start=0, stop=200.0, label=r"$m_{ll}$ [GeV]")]),
        # lead lepton dR
        "lead_wlep_wjet1_dR": HistConf([Axis(coll="events", field="lead_wlep_wjet1_dR", bins=40, start=0.0, stop=4.0, label=r"$\Delta R(lj_1)^{W}$")]),
        "lead_wlep_wjet2_dR": HistConf([Axis(coll="events", field="lead_wlep_wjet2_dR", bins=40, start=0.0, stop=4.0, label=r"$\Delta R(lj_2)^{W}$")]),
        "lead_wlep_wfatjet1_dR": HistConf([Axis(coll="events", field="lead_wlep_wfatjet1_dR", bins=40, start=0.0, stop=4.0, label=r"$\Delta R(lJ_1)^{W}$")]),
        "lead_wlep_w_resolved_dR": HistConf([Axis(coll="events", field="lead_wlep_w_resolved_dR", bins=40, start=0.0, stop=4.0, label=r"$\Delta R(lW_{resolved})$")]),
        "lead_wlep_vbsjet1_dR": HistConf([Axis(coll="events", field="lead_wlep_vbsjet1_dR", bins=40, start=0.0, stop=4.0, label=r"$\Delta R(lj_1)^{forward}$")]),
        "lead_wlep_vbsjet2_dR": HistConf([Axis(coll="events", field="lead_wlep_vbsjet2_dR", bins=40, start=0.0, stop=4.0, label=r"$\Delta R(lj_2)^{forward}$")]),
       
        "lead_wlep_vbsjet1_dR_boost": HistConf([Axis(coll="events", field="lead_wlep_vbsjet1_dR_boost", bins=40, start=0.0, stop=4.0, label=r"$\Delta R(lj_1)^{forward,boost}$")]),
        "lead_wlep_vbsjet2_dR_boost": HistConf([Axis(coll="events", field="lead_wlep_vbsjet2_dR_boost", bins=40, start=0.0, stop=4.0, label=r"$\Delta R(lj_2)^{forward,boost}$")]),
       
        # lead lepton dEta
        "lead_wlep_wfatjet1_deta":   HistConf([Axis(coll="events", field="lead_wlep_wfatjet1_deta", bins=24, start=0.0, stop=9.0, label=r"$|\Delta\eta_{lJ}^{W}|$")]),
        "lead_wlep_wjet1_deta":   HistConf([Axis(coll="events", field="lead_wlep_wjet1_deta", bins=24, start=0.0, stop=9.0, label=r"$|\Delta\eta_{lj_1}^{W}|$")]),
        "lead_wlep_wjet2_deta":   HistConf([Axis(coll="events", field="lead_wlep_wjet2_deta", bins=24, start=0.0, stop=9.0, label=r"$|\Delta\eta_{lj_2}^{W}|$")]),
        "lead_wlep_w_resolved_deta":   HistConf([Axis(coll="events", field="lead_wlep_w_resolved_deta", bins=24, start=2.0, stop=9.0, label=r"$|\Delta\eta_{lW}^{resolved}|$")]),

        # dphi plots
        "lead_wlep_MET_dphi":   HistConf([Axis(coll="events", field="lead_wlep_MET_dphi", bins=32, start=-4, stop=4.0, label=r"$|\Delta\phi_{l,MET}|$")]),
        "lead_wlep_wfatjet1_dphi":   HistConf([Axis(coll="events", field="lead_wlep_wfatjet1_dphi", bins=32, start=-4, stop=4.0, label=r"$|\Delta\phi_{l,J^W}|$")]),
        "lead_wlep_wjet1_dphi":   HistConf([Axis(coll="events", field="lead_wlep_wjet1_dphi", bins=32, start=-4, stop=4.0, label=r"$|\Delta\phi_{l,j_1^WT}|$")]),
        "lead_wlep_wjet2_dphi":   HistConf([Axis(coll="events", field="lead_wlep_wjet2_dphi", bins=32, start=-4, stop=4.0, label=r"$|\Delta\phi_{l,j_1^WTT}|$")]),
        "w_lep_w_resolved_dphi":   HistConf([Axis(coll="events", field="w_lep_w_resolved_dphi", bins=32, start=-4, stop=4.0, label=r"$|\Delta\phi_{W_{lep}W_{resolved}}|$")]),
        "w_lep_w_boost_dphi":   HistConf([Axis(coll="events", field="w_lep_w_boost_dphi", bins=32, start=-4, stop=4.0, label=r"$|\Delta\phi_{W_{lep}W_{boost}}|$")]),

        # leptonic W
        "wleptonic_eta":    HistConf([Axis(coll="events", field="wleptonic_eta", bins=48, start=-2.4, stop=2.4,   label=r"$\eta(W_{lep})$")]),

        # # W fat jet
        "fj_pt":    HistConf([Axis(coll="candidate_boost", field="pt",  bins=60, start=150, stop=1000, label=r"$p_T(J^{W})$ [GeV]")]),
        "fj_eta":   HistConf([Axis(coll="candidate_boost", field="eta", bins=48, start=-2.4, stop=2.4,   label=r"$\eta(J^{W})$")]),
        "fj_msd":   HistConf([Axis(coll="candidate_boost", field="msoftdrop", bins=40, start=0,   stop=200,   label=r"$m_{SD}(J^{W})$ [GeV]")]),
        "fj_t21":   HistConf([Axis(coll="candidate_boost", field="tau21", bins=32, start=0, stop=1.1,   label=r"$\tau_{21}$")]),

        "fj_W_vs_QCD":  HistConf([Axis(coll="candidate_boost", field="particleNet_WvsQCD", bins=32, start=0, stop=1.1,   label=r"particleNet_WvsQCD$")]),
        "fj_Z_vs_QCD":  HistConf([Axis(coll="candidate_boost", field="particleNet_ZvsQCD", bins=32, start=0, stop=1.1,   label=r"particleNet_ZvsQCD$")]),
        "fj_pn_mass":  HistConf([Axis(coll="candidate_boost", field="particleNet_mass", bins=15, start=40, stop=115,   label=r"particleNet_mass$")]),


        "fj_W_vs_QCD_deeptag":  HistConf([Axis(coll="candidate_boost", field="deepTag_WvsQCD", bins=32, start=0, stop=1.1,   label=r"deepTag_WvsQCD$")]),
        "fj_Z_vs_QCD_deeptag":  HistConf([Axis(coll="candidate_boost", field="deepTag_ZvsQCD", bins=32, start=0, stop=1.1,   label=r"deepTag_ZvsQCD$")]),
        
        "fj_W_vs_QCD_deeptagMD":  HistConf([Axis(coll="candidate_boost", field="deepTagMD_WvsQCD", bins=32, start=0, stop=1.1,   label=r"deepTagMD_WvsQCD$")]),
        "fj_Z_vs_QCD_deeptagMD":  HistConf([Axis(coll="candidate_boost", field="deepTagMD_ZvsQCD", bins=32, start=0, stop=1.1,   label=r"deepTagMD_ZvsQCD$")]),
        
        #"ak8_ak4_separation":       HistConf([Axis(coll="events", field="separation", bins=40, start=0.0, stop=4.0, label=r"$\Delta R(AK8 to AK4)$")]),
    
        "z_lep":   HistConf([Axis(coll="events", field="z_lep", bins=40, start=-1.0, stop=1.0, label=r"$Zepp. lepton$")]),
        "z_fat":      HistConf([Axis(coll="events", field="z_fat", bins=40, start=-1.0, stop=1.0, label=r"$Zepp. boosted jet$")]),
        
        "centrality_resolved":  HistConf([Axis(coll="events", field="centrality_resolved", bins=40, start=-5.0, stop=5.0, label=r"$Centrality_{resolved}$")]),
        "centrality_boosted":   HistConf([Axis(coll="events", field="centrality_boosted", bins=40, start=-5.0, stop=5.0, label=r"$Centrality_{boosted}$")]),

        "qgl_vbs1_resolved":  HistConf([Axis(coll="events", field="qgl_vbs1_resolved", bins=40, start=0, stop=1.0, label=r"$QGL VBS jet1 (resolved)$")]),
        "qgl_vbs2_resolved":  HistConf([Axis(coll="events", field="qgl_vbs2_resolved", bins=40, start=0, stop=1.0, label=r"$QGL VBS jet2 (resolved)$")]),
       
        "qgl_vbs1_boost":  HistConf([Axis(coll="events", field="qgl_vbs1_boost", bins=40, start=0, stop=1.0, label=r"$QGL VBS jet1 (boosted) $")]),
        "qgl_vbs2_boost":  HistConf([Axis(coll="events", field="qgl_vbs2_boost", bins=40, start=0, stop=1.0, label=r"$QGL VBS jet2 (boosted)$")]),
       
        "qgl_wjet1_resolved":  HistConf([Axis(coll="events", field="qgl_wjet1_resolved", bins=40, start=0, stop=1.0, label=r"$QGL had. W jet 1 $")]),
        "qgl_wjet2_resolved":  HistConf([Axis(coll="events", field="qgl_wjet2_resolved", bins=40, start=0, stop=1.0, label=r"$QGL had. W jet 2 $")]),
        #"qgl_fatjet":  HistConf([Axis(coll="events", field="qgl_fatjet", bins=40, start=0, stop=1.0, label=r"$QGL AK8 W jet $")]),

        
        # VBS jet kinematics
        "pt_vbsjet1":    HistConf([Axis(coll="events", field="vbsjet1_pt", bins=60, start=0, stop=300, label=r"$p_T(j_1)^{forward}$ [GeV]")]),
        "pt_vbsjet2":    HistConf([Axis(coll="events", field="vbsjet2_pt", bins=60, start=0, stop=300, label=r"$p_T(j_2)^{forward}$ [GeV]")]),
        "eta_vbsjet1":   HistConf([Axis(coll="events", field="vbsjet1_eta", bins=48, start=-4.8, stop=4.8, label=r"$\eta(j_1)^{forward}$")]),
        "eta_vbsjet2":   HistConf([Axis(coll="events", field="vbsjet2_eta", bins=48, start=-4.8, stop=4.8, label=r"$\eta(j_2)^{forward}$")]),
        "phi_vbsjet1":   HistConf([Axis(coll="events", field="vbsjet1_phi", bins=48, start=-4., stop=4., label=r"$\phi(j_1)^{forward}$")]),
        "phi_vbsjet2":   HistConf([Axis(coll="events", field="vbsjet2_phi", bins=48, start=-4., stop=4., label=r"$\phi(j_2)^{forward}$")]),
        
        # STUPID B JETS CAUSING PROBLEMS
        
        "bjet_pt":    HistConf([Axis(coll="BJet_csv", field="pt", bins=60, start=0, stop=300, label=r"$p_T(b)$ [GeV]")]),
        "bjet_eta":   HistConf([Axis(coll="BJet_csv", field="eta", bins=48, start=-4.8, stop=4.8, label=r"$\eta(b)$")]),
        "bjet_phi":   HistConf([Axis(coll="BJet_csv", field="phi", bins=48, start=-4., stop=4., label=r"$\phi(b)$")]),
        "bjet_lepton_separation":      HistConf([Axis(coll="events", field="lep_bjet_dR", bins=32, start=0, stop=9.0, label=r"$\Delta R_{lep,b} $")]),
        # "genJetIdx":       HistConf([Axis(coll="JetGood", field="genJetIdx", bins=10, start=-1, stop=10, label="genJet idx")]),
        # "genBJetIdx":       HistConf([Axis(coll="BJet_csv", field="genJetIdx", bins=10, start=-1, stop=10, label="genBJet idx")]),
        "BJetIdx":       HistConf([Axis(coll="BJet_csv", field="idx", bins=10, start=-1, stop=10, label="BJet idx")]),

        "LeadJetIdx":       HistConf([Axis(coll="events", field="jet1_idx", bins=10, start=-1, stop=10, label="tag 1 idx")]),
        "SecondJetIdx":       HistConf([Axis(coll="events", field="jet2_idx", bins=10, start=-1, stop=10, label="tag 2 idx")]),


        #"flav_genjet_hadron":       HistConf([Axis(coll="matched_gen_to_b", field="hadronFlavour", bins=20, start=-10, stop=10, label="gen jet matched flav (hadron))")]),
        #"flav_jet_hadron":       HistConf([Axis(coll="BJet_csv", field="hadronFlavour", bins=20, start=-10, stop=10, label="b jet flav (hadron))")]),
        # #"flav_genjet_parton":       HistConf([Axis(coll="matched_gen_to_b", field="partonFlavour", bins=20, start=-10, stop=10, label="gen jet matchedflav (parton))")]),
        #"flav_jet_parton":       HistConf([Axis(coll="BJet_csv", field="partonFlavour", bins=20, start=-10, stop=10, label="b jet flav (parton))")]),

        "HT_check":     HistConf([Axis(coll="LHE", field="HT", label="gen HT", type="variable", bins=[0,70,100,200,400,600,800,1200,2500,3500])]),
        "HT_sum":       HistConf([Axis(coll="events", field="ht_sum", bins=35, start=0, stop=3500, label="reco HT")]),
    },
    #THIS IS WHERE YOU ADD COLUMN VARIABLES TO SAVE FOR BDT THINGS
    columns = {
        "common": {
           "bycategory": {
                "whad_withbveto_e": [ColOut("events",["nJetGood","mt_w_leptonic", "w_had_dR","z_lep","centrality_resolved","vbs_dR","jet1_pt","qgl_vbs1_resolved","qgl_vbs2_resolved","wleptonic_eta","w_lep_eta","lead_wlep_vbsjet1_dR","lead_wlep_vbsjet2_dR"]), ColOut("PuppiMET", ["pt"]), ColOut("vbsjets", ["mass", "delta_eta"]), ColOut("w_had_jets", ["pt"])],
                "whad_withbveto_mu": [ColOut("events",["nJetGood","mt_w_leptonic", "w_had_dR","z_lep","centrality_resolved","vbs_dR","jet1_pt","qgl_vbs1_resolved","qgl_vbs2_resolved","wleptonic_eta","w_lep_eta","lead_wlep_vbsjet1_dR","lead_wlep_vbsjet2_dR"]), ColOut("PuppiMET", ["pt"]), ColOut("vbsjets", ["mass", "delta_eta"]), ColOut("w_had_jets", ["pt"])],
                # "boosted_jet_in_window_e": [ColOut("events",["nJetGood","nFatJetGood","mt_w_leptonic","z_lep","z_fat","centrality_boosted","vbs_dR","jet1_pt","qgl_vbs1_resolved","qgl_vbs2_resolved","wleptonic_eta","w_lep_eta","lead_wlep_vbsjet1_dR"]), ColOut("PuppiMET", ["pt"]), ColOut("vbsjets", ["mass", "delta_eta"]), ColOut("candidate_boost", ["particleNet_WvsQCD","particleNet_ZvsQCD","tau21"])],
                # "boosted_jet_in_window_mu": [ColOut("events",["nJetGood","nFatJetGood","mt_w_leptonic","z_lep","z_fat","centrality_boosted","vbs_dR","jet1_pt","qgl_vbs1_resolved","qgl_vbs2_resolved","wleptonic_eta","w_lep_eta","lead_wlep_vbsjet1_dR"]), ColOut("PuppiMET", ["pt"]), ColOut("vbsjets", ["mass", "delta_eta"]), ColOut("candidate_boost", ["particleNet_WvsQCD","particleNet_ZvsQCD","tau21"])],
           }
        }
    },
)
