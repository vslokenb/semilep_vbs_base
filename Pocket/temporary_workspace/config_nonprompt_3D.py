# example_config_semileptonic.py
import os, cloudpickle
from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_functions import get_HLTsel, get_nPVgood, goldenJson, eventFlags, get_JetVetoMap
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.parameters.histograms import HistConf, Axis
from pocket_coffea.lib.weights.common import common_weights
from pocket_coffea.lib.weights.common.common import SF_L1prefiring
from pocket_coffea.lib.weights.common.weights_run2_UL import SF_ele_trigger
from pocket_coffea.parameters import defaults
import numpy as np
import awkward as ak
from pocket_coffea.lib.weights import WeightWrapper, WeightData, WeightDataMultiVariation, WeightLambda
from pocket_coffea.lib.scale_factors import sf_pileup_reweight


import workflow_nonprompt, custom_cut_functions_nonprompt, reweighting_st
from workflow_nonprompt import VBSSemileptonicProcessor
from custom_cut_functions_nonprompt import (
    nLepton_skim_cut,
    nJet_skim_cut,
    met_skim_cut,
    wjets_mu_sel,
    wjets_e_sel,
    qcd_enriched_cut_30,
    qcd_enriched_cut_35,
    qcd_enriched_cut_40,
    qcd_enriched_cut_45,
    qcd_enriched_cut_30_2j,
    qcd_enriched_cut_35_2j,
    qcd_enriched_cut_40_2j,
    qcd_enriched_cut_45_2j,
    qcd_enriched_cut_30_3j,
    qcd_enriched_cut_35_3j,
    qcd_enriched_cut_40_3j,
    qcd_enriched_cut_45_3j,
    qcd_enriched_cut_30_4j,
    qcd_enriched_cut_35_4j,
    qcd_enriched_cut_40_4j,
    qcd_enriched_cut_45_4j,
    qcd_enriched_cut_35_deepmet_response,
    qcd_enriched_cut_30_deepmet_response,
    qcd_enriched_cut_40_deepmet_response,
    qcd_enriched_cut_45_deepmet_response,
    qcd_enriched_cut_35_deepmet_resolution,
    qcd_enriched_cut_30_deepmet_resolution,
    qcd_enriched_cut_40_deepmet_resolution,
    qcd_enriched_cut_45_deepmet_resolution,
    qcd_enriched_cut_35_deepmet_response_2j,
    qcd_enriched_cut_30_deepmet_response_2j,
    qcd_enriched_cut_40_deepmet_response_2j,
    qcd_enriched_cut_45_deepmet_response_2j,
    qcd_enriched_cut_35_deepmet_response_3j,
    qcd_enriched_cut_30_deepmet_response_3j,
    qcd_enriched_cut_40_deepmet_response_3j,
    qcd_enriched_cut_45_deepmet_response_3j,
    qcd_enriched_cut_35_deepmet_resolution_2j,
    qcd_enriched_cut_30_deepmet_resolution_2j,
    qcd_enriched_cut_40_deepmet_resolution_2j,
    qcd_enriched_cut_45_deepmet_resolution_2j,
    qcd_enriched_cut_35_deepmet_resolution_3j,
    qcd_enriched_cut_30_deepmet_resolution_3j,
    qcd_enriched_cut_40_deepmet_resolution_3j,
    qcd_enriched_cut_35_deepmet20_resolution_3j,
    qcd_enriched_cut_30_deepmet20_resolution_3j,
    qcd_enriched_cut_40_deepmet20_resolution_3j,
    qcd_enriched_cut_45_deepmet_resolution_3j,
    qcd_enriched_cut_35_deepmet_resolution_3j60,
    qcd_enriched_cut_35_deepmet_resolution_3j45,
    qcd_enriched_cut_35_deepmet_resolution_3j_mT,
    qcd_enriched_cut_35_deepmet_resolution_3j60_mT,
    qcd_enriched_cut_20_deepmet_resolution_3j,
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


cloudpickle.register_pickle_by_value(workflow_nonprompt)
cloudpickle.register_pickle_by_value(reweighting_st)
cloudpickle.register_pickle_by_value(custom_cut_functions_nonprompt)
from reweighting_st import ratio_function
localdir = os.path.dirname(os.path.abspath(__file__))


default_parameters = defaults.get_default_parameters()
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
            #######
            ## RUN 2 BKG
            # #########
            
            # f"{localdir}/datasets/WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8_17.json",
            #
            f"{localdir}/datasets/WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            f"{localdir}/datasets/WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            f"{localdir}/datasets/WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            f"{localdir}/datasets/WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8.json",

            # XSEC STUDIES
            # f"{localdir}/datasets/WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8_17.json",
            # f"{localdir}/datasets/WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_17.json",
            # f"{localdir}/datasets/WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8_17.json",
            
            # f"{localdir}/datasets/WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8_17_2.json",
            # f"{localdir}/datasets/WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_17_2.json",
            # f"{localdir}/datasets/WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8_17_2.json",
            

            # f"{localdir}/datasets/WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8_fix.json",
            # f"{localdir}/datasets/WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_fix.json",
            # f"{localdir}/datasets/WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8_fix.json",
            
            # f"{localdir}/datasets/WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8_fix2.json",
            # f"{localdir}/datasets/WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_fix2.json",
            # f"{localdir}/datasets/WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8_fix2.json",
            
            # END XSEC STUDIES

            f"{localdir}/datasets/WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            f"{localdir}/datasets/WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            f"{localdir}/datasets/WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            f"{localdir}/datasets/WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            f"{localdir}/datasets/WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8.json",

            f"{localdir}/datasets/WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8.json",
            # f"{localdir}/datasets/WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8_17.json",
            

            # f"{localdir}/datasets/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8.json",
            #f"{localdir}/datasets/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_17.json",

            #f"{localdir}/datasets/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            #f"{localdir}/datasets/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8_17.json",

            # f"{localdir}/datasets/DYJetsToLL_M-10to50_TuneCP5_13TeV-amcatnloFXFX-pythia8.json",

            #f"{localdir}/datasets/DYJetsToLL_M-50_HT-70to100_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8.json",
            #f"{localdir}/datasets/DYJetsToLL_M-50_HT-100to200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8.json",
            #f"{localdir}/datasets/DYJetsToLL_M-50_HT-200to400_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8.json",
            #f"{localdir}/datasets/DYJetsToLL_M-50_HT-400to600_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8.json",
            #f"{localdir}/datasets/DYJetsToLL_M-50_HT-600to800_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8.json",
            #f"{localdir}/datasets/DYJetsToLL_M-50_HT-800to1200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8.json",
            #f"{localdir}/datasets/DYJetsToLL_M-50_HT-1200to2500_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8.json",
            #f"{localdir}/datasets/DYJetsToLL_M-50_HT-2500toInf_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8.json",

            f"{localdir}/datasets/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8.json",
            # f"{localdir}/datasets/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8.json",
            # f"{localdir}/datasets/ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8.json",
            # f"{localdir}/datasets/ST_t-channel_top_4f_inclusiveDecays_TuneCP5_13TeV-powhegV2-madspin-pythia8.json",
            # f"{localdir}/datasets/ST_t-channel_antitop_4f_inclusiveDecays_TuneCP5_13TeV-powhegV2-madspin-pythia8.json",
            # f"{localdir}/datasets/ttZJets_TuneCP5_13TeV_madgraphMLM_pythia8.json",
            # f"{localdir}/datasets/ttWJets_TuneCP5_13TeV_madgraphMLM_pythia8.json",
            # f"{localdir}/datasets/WplusTo2JZTo2LJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            # f"{localdir}/datasets/WplusToLNuWminusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV.json",
            # f"{localdir}/datasets/WplusToLNuWplusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            # f"{localdir}/datasets/WminusTo2JZTo2LJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            # f"{localdir}/datasets/WminusToLNuWminusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            # f"{localdir}/datasets/WminusToLNuZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            # f"{localdir}/datasets/WplusTo2JWminusToLNuJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV.json",
            # f"{localdir}/datasets/WplusToLNuZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            # f"{localdir}/datasets/ZTo2LZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            # f"{localdir}/datasets/WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8.json",
            # f"{localdir}/datasets/WZZ_TuneCP5_13TeV-amcatnlo-pythia8.json",
            # f"{localdir}/datasets/ZZZ_TuneCP5_13TeV-amcatnlo-pythia8.json",
            # f"{localdir}/datasets/WGToLNuG_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            # f"{localdir}/datasets/ZGToLLG_01J_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8.json",
            # f"{localdir}/datasets/WZTo3LNu_mllmin01_NNPDF31_TuneCP5_13TeV_powheg_pythia8.json",
            
            # #########
            # ## RUN 2 SIGNAL
            # ########
            # f"{localdir}/datasets/WplusTo2JWminusToLNuJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            # f"{localdir}/datasets/WplusToLNuWminusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            # f"{localdir}/datasets/WminusToLNuWminusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            # f"{localdir}/datasets/WplusToLNuWplusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            # f"{localdir}/datasets/WminusToLNuZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            # f"{localdir}/datasets/WplusToLNuZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
 
            ########
            ## RUN 3 BKG
            ########
            #f"{localdir}/datasets/WWtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8.json",
            #f"{localdir}/datasets/WtoLNu-2Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            #f"{localdir}/datasets/ZZto2L2Q_TuneCP5_13p6TeV_powheg-pythia8.json",
            #f"{localdir}/datasets/TTto2L2Nu_TuneCP5_ERDOn_13p6TeV_powheg-pythia8.json",
            #f"{localdir}/datasets/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8.json",
            #f"{localdir}/datasets/DYto2L-2Jets_MLL-10to50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            #f"{localdir}/datasets/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",'''
            #f"{localdir}/datasets/WtoLNu-2Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            #f"{localdir}/datasets/WtoLNu-2Jets_0J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            #f"{localdir}/datasets/WtoLNu-2Jets_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            #f"{localdir}/datasets/WtoLNu-2Jets_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            # f"{localdir}/datasets/DYto2L-2Jets_MLL-10to50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            # f"{localdir}/datasets/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            #f"{localdir}/datasets/DYto2L-2Jets_MLL-50_0J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            #f"{localdir}/datasets/DYto2L-2Jets_MLL-50_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            #f"{localdir}/datasets/DYto2L-2Jets_MLL-50_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            #f"{localdir}/datasets/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8.json",

            # #########
            # ## RUN 3 SIGNAL
            # ########
            #f"{localdir}/datasets/WpWpJJ-EWK_TuneCP5_13p6TeV-powheg-pythia8.json",
            #f"{localdir}/datasets/WmWmJJ-EWK_TuneCP5_13p6TeV-powheg-pythia8.json",
            #########
            ## SOME DATA
            #########
            f"{localdir}/datasets/SingleMuon.json", ## 2017B Single Muon dataset
            # f"{localdir}/datasets/SingleElectron.json", ## 2017B Single Muon dataset
            f"{localdir}/datasets/EGamma.json", # 2022_postEE EGamma
            # #f"{localdir}/datasets/EGamma_G.json"
            # #f"{localdir}/datasets/Muon.json",
            # f"{localdir}/datasets/Muon_2022E.json"

                        #### only 1 era below
            
            # f"{localdir}/datasets/SingleMuon_fast.json",
            # f"{localdir}/datasets/SingleElectron_fast.json",
            f"{localdir}/datasets/DYJetsToLL_M-10to50_TuneCP5_13TeV-amcatnloFXFX-pythia8_fast.json",
            f"{localdir}/datasets/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_fast.json",
            # f"{localdir}/datasets/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_fast.json",
            f"{localdir}/datasets/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_fast.json",
            # f"{localdir}/datasets/WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8_fast.json",
            # f"{localdir}/datasets/WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8_fast.json",
            # f"{localdir}/datasets/WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_fast.json",
            # f"{localdir}/datasets/WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8_fast.json",
            # f"{localdir}/datasets/WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8_fast.json",
            # f"{localdir}/datasets/WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8_fast.json",
            # f"{localdir}/datasets/WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8_fast.json",
            # f"{localdir}/datasets/WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8_fast.json",
            # f"{localdir}/datasets/WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8_fast.json",
            f"{localdir}/datasets/WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8_fast.json",  
        ],
        
        "filter": {
            "samples": [
                
            #########
            ## RUN 2 BKG
            #########
            # "WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8",

            # # "WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8",
            # "WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8",
            # "WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8",
            # # "WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8",
            # "WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8",
            # "WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8",
            # "WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8",
            # "WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8",

            "WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8", 
            #"WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8_17", 
            #"DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8_17",
            # "DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8",
            # "DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8", 
            # # #"DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_17", 
            # "DYJetsToLL_M-10to50_TuneCP5_13TeV-amcatnloFXFX-pythia8", 

            # "DYJetsToLL_M-50_HT-70to100_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
            # "DYJetsToLL_M-50_HT-100to200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
            # "DYJetsToLL_M-50_HT-200to400_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
            # "DYJetsToLL_M-50_HT-400to600_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
            # "DYJetsToLL_M-50_HT-600to800_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
            # "DYJetsToLL_M-50_HT-800to1200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
            # "DYJetsToLL_M-50_HT-1200to2500_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
            # "DYJetsToLL_M-50_HT-2500toInf_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",

            # "TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8", 
            # "TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8", 
            # "ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8", 
            # "ST_t-channel_top_4f_inclusiveDecays_TuneCP5_13TeV-powhegV2-madspin-pythia8",
            # "ST_t-channel_antitop_4f_inclusiveDecays_TuneCP5_13TeV-powhegV2-madspin-pythia8", 
            # "ttZJets_TuneCP5_13TeV_madgraphMLM_pythia8", 
            # "ttWJets_TuneCP5_13TeV_madgraphMLM_pythia8", 
            # # "WplusTo2JZTo2LJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8", 
            # # "WplusToLNuWminusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV", 
            # # "WplusToLNuWplusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8", 
            # # "WminusTo2JZTo2LJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8", 
            # # "WminusToLNuWminusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8", 
            # # "WminusToLNuZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8", 
            # # "WplusTo2JWminusToLNuJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV", 
            # # "WplusToLNuZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8", 
            # # "ZTo2LZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8", 
            # # "WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8", 
            # # "WZZ_TuneCP5_13TeV-amcatnlo-pythia8", 
            # # "ZZZ_TuneCP5_13TeV-amcatnlo-pythia8", 
            # # "WGToLNuG_TuneCP5_13TeV-madgraphMLM-pythia8", 
            # # "ZGToLLG_01J_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8", 
            # # "WZTo3LNu_mllmin01_NNPDF31_TuneCP5_13TeV_powheg_pythia8", 
        
                        
            # #########
            # ## RUN 2 SIGNAL
            # ########
            #"WminusToLNuWminusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            #"WplusTo2JWminusToLNuJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            #"WplusToLNuWminusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            #"WplusToLNuWplusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            #"WminusToLNuZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            #"WplusToLNuZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            
            ########
            ## RUN 3 BKG
            ########
            #"WWtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8",
            #"WtoLNu-2Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
            #"ZZto2L2Q_TuneCP5_13p6TeV_powheg-pythia8",
            #"TTto2L2Nu_TuneCP5_ERDOn_13p6TeV_powheg-pythia8",
            #"TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8",
            #"DYto2L-2Jets_MLL-10to50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
            #"DYto2L-2Jets_MLL-50_0J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
            #"DYto2L-2Jets_MLL-50_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
            #"DYto2L-2Jets_MLL-50_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
            #"DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
            #"WtoLNu-2Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
            #"WtoLNu-2Jets_0J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
            #"WtoLNu-2Jets_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
            #"WtoLNu-2Jets_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
            #"DYto2L-2Jets_MLL-10to50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
            #"DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
            #"TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8",
            # #######
            # # RUN 3 SIGNAL
            # #######
            #"WpWpJJ-EWK_TuneCP5_13p6TeV-powheg-pythia8",
            #"WmWmJJ-EWK_TuneCP5_13p6TeV-powheg-pythia8",

            #########
            ## SOME DATA
            #########
            # "SingleMuon", ## 2017B Single Muon dataset
            # "SingleElectron",
            # "EGamma",
            # "Muon"
            ],
            "year": ["2018"],
        },
    },
    workflow=VBSSemileptonicProcessor,

    
    skim=[
        get_nPVgood(1),    # nPV>0
        eventFlags,        # PileupID
        goldenJson,        
        nLepton_skim_cut,
        get_HLTsel(primaryDatasets=["SingleMuon","EGamma"]),#, "SingleElectron"]),
        # get_JetVetoMap()
    ],

    # 2) preselections 
    preselections=[passthrough],

   
    categories={
        "baseline": [passthrough],
        "wjet_mu": [wjets_mu_sel],
        "wjet_e": [wjets_e_sel],
        # "qcd_enriched_30": [qcd_enriched_cut_30],
        # "qcd_enriched_35": [qcd_enriched_cut_35],
        # "qcd_enriched_40": [qcd_enriched_cut_40],
        # "qcd_enriched_45": [qcd_enriched_cut_45],
        # "qcd_enriched_30_2j": [qcd_enriched_cut_30_2j],
        # "qcd_enriched_35_2j": [qcd_enriched_cut_35_2j],
        # "qcd_enriched_40_2j": [qcd_enriched_cut_40_2j],
        # "qcd_enriched_45_2j": [qcd_enriched_cut_45_2j],
        # "qcd_enriched_30_3j": [qcd_enriched_cut_30_3j],
        # "qcd_enriched_35_3j": [qcd_enriched_cut_35_3j],
        # "qcd_enriched_40_3j": [qcd_enriched_cut_40_3j],
        # "qcd_enriched_45_3j": [qcd_enriched_cut_45_3j],
        # "qcd_enriched_30_4j": [qcd_enriched_cut_30_4j],
        # "qcd_enriched_35_4j": [qcd_enriched_cut_35_4j],
        # "qcd_enriched_40_4j": [qcd_enriched_cut_40_4j],
        # "qcd_enriched_45_4j": [qcd_enriched_cut_45_4j],
        # "qcd_enriched_30_deepmet_response": [qcd_enriched_cut_30_deepmet_response],
        # "qcd_enriched_35_deepmet_response": [qcd_enriched_cut_35_deepmet_response],
        # "qcd_enriched_40_deepmet_response": [qcd_enriched_cut_40_deepmet_response],
        # "qcd_enriched_45_deepmet_response": [qcd_enriched_cut_45_deepmet_response],
        "qcd_enriched_30_deepmet_resolution": [qcd_enriched_cut_30_deepmet_resolution],
        "qcd_enriched_35_deepmet_resolution": [qcd_enriched_cut_35_deepmet_resolution],
        "qcd_enriched_40_deepmet_resolution": [qcd_enriched_cut_40_deepmet_resolution],
        "qcd_enriched_45_deepmet_resolution": [qcd_enriched_cut_45_deepmet_resolution],
        # "qcd_enriched_30_deepmet_response_2j": [qcd_enriched_cut_30_deepmet_response_2j],
        # "qcd_enriched_35_deepmet_response_2j": [qcd_enriched_cut_35_deepmet_response_2j],
        # "qcd_enriched_40_deepmet_response_2j": [qcd_enriched_cut_40_deepmet_response_2j],
        # "qcd_enriched_45_deepmet_response_2j": [qcd_enriched_cut_45_deepmet_response_2j],
        # "qcd_enriched_30_deepmet_resolution_2j": [qcd_enriched_cut_30_deepmet_resolution_2j],
        "qcd_enriched_35_deepmet_resolution_2j": [qcd_enriched_cut_35_deepmet_resolution_2j],
        "qcd_enriched_40_deepmet_resolution_2j": [qcd_enriched_cut_40_deepmet_resolution_2j],
        "qcd_enriched_45_deepmet_resolution_2j": [qcd_enriched_cut_45_deepmet_resolution_2j],
        # "qcd_enriched_30_deepmet_response_3j": [qcd_enriched_cut_30_deepmet_response_3j],
        # "qcd_enriched_35_deepmet_response_3j": [qcd_enriched_cut_35_deepmet_response_3j],
        # "qcd_enriched_40_deepmet_response_3j": [qcd_enriched_cut_40_deepmet_response_3j],
        # "qcd_enriched_45_deepmet_response_3j": [qcd_enriched_cut_45_deepmet_response_3j],
        "qcd_enriched_30_deepmet_resolution_3j": [qcd_enriched_cut_30_deepmet_resolution_3j],
        "qcd_enriched_35_deepmet_resolution_3j": [qcd_enriched_cut_35_deepmet_resolution_3j],
        "qcd_enriched_40_deepmet_resolution_3j": [qcd_enriched_cut_40_deepmet_resolution_3j],
        "qcd_enriched_45_deepmet_resolution_3j": [qcd_enriched_cut_45_deepmet_resolution_3j],
        "qcd_enriched_35_deepmet_resolution_3j60": [qcd_enriched_cut_35_deepmet_resolution_3j60],
        "qcd_enriched_35_deepmet_resolution_3j45": [qcd_enriched_cut_35_deepmet_resolution_3j45],
        "qcd_enriched_35_deepmet_resolution_3j_mT": [qcd_enriched_cut_35_deepmet_resolution_3j_mT],
        "qcd_enriched_35_deepmet_resolution_3j60_mT": [qcd_enriched_cut_35_deepmet_resolution_3j60_mT],
        "qcd_enriched_30_deepmet20_resolution_3j": [qcd_enriched_cut_30_deepmet20_resolution_3j],
        "qcd_enriched_35_deepmet20_resolution_3j": [qcd_enriched_cut_35_deepmet20_resolution_3j],
        "qcd_enriched_40_deepmet20_resolution_3j": [qcd_enriched_cut_40_deepmet20_resolution_3j],
        "qcd_enriched_20_deepmet_resolution_3j": [qcd_enriched_cut_20_deepmet_resolution_3j],
  
    },

   
    weights_classes=common_weights+[SF_ele_trigger]+[PileupWeight]+[SF_L1prefiring]+[wjet_reweight],
    #weights={"common": {"inclusive": ["genWeight", "lumi", "XS", "PileupWeight", "sf_mu_id", "sf_mu_iso", "sf_ele_id", "sf_ele_reco"]}},
    #variations={"weights": {"common": {"inclusive": ["PileupWeight", "sf_mu_id","sf_mu_iso","sf_ele_id","sf_ele_reco"]}}}, #"pileup"
    #weights={"common": {"inclusive": ["genWeight", "lumi", "XS", "pileup", "sf_mu_id", "sf_mu_iso", "sf_ele_id", "sf_ele_reco"]}},
    #variations={"weights": {"common": {"inclusive": ["pileup", "sf_mu_id","sf_mu_iso","sf_ele_id","sf_ele_reco"]}}}, #"pileup"
    weights={"common": {"inclusive": ["lumi", "XS","PileupWeight", "genWeight","sf_mu_id", "sf_mu_iso", "sf_ele_id", "sf_ele_reco","sf_mu_trigger","sf_ele_trigger","sf_jet_puId","sf_L1prefiring", "sf_partonshower_isr", "sf_partonshower_fsr"]},
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
        #         "WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8":{
        #              "inclusive": ["wjet_reweight"],},
        #     }    
        },
    variations={"weights": {"common": {"inclusive": []}}},#"PileupWeight","sf_mu_id","sf_mu_iso","sf_ele_id","sf_ele_reco","sf_mu_trigger","sf_jet_puId","sf_L1prefiring", "sf_partonshower_isr", "sf_partonshower_fsr"]}}},

    variables={
        "electron_loose_pt" : HistConf(
            [
                Axis(coll="ElectronLoose", field="pt",bins=[35,40,45,50,55,60,70,100], label="loose electrons $p_T$")
            ] ),
        "electron_loose_eta" : HistConf(
            [
                Axis(coll="ElectronLoose", field="eta", bins=[-2.4,-2.15,-1.8,-1.479,-1.,-0.5,0.,0.5,1.,1.479,1.8,2.15,2.4], label="loose electrons $\eta$")
            ] ),
        "electron_loose_dphi": HistConf(
            [
                Axis(coll="ElectronLoose", field="dphi_met", bins=[0.,0.1,0.4,0.5,1.0,3.14], label="loose electrons $\Delta \phi$")
            ] ),
        "electron_loose_pt_eta_dphi" : HistConf(
            [
                Axis(coll="ElectronLoose", field="pt",bins=[35,40,45,50,55,60,70,100], label="loose electrons $p_T$"),
                Axis(coll="ElectronLoose", field="eta", bins=[-2.4,-2.15,-1.8,-1.479,-1.,-0.5,0.,0.5,1.,1.479,1.8,2.15,2.4], label="loose electrons $\eta$"),
                Axis(coll="ElectronLoose", field="dphi_met", bins=[0.,0.5,1.0,3.14], label="loose electrons $\Delta \phi$")
            ] ),
        "electron_tight_pt" : HistConf(
            [
                Axis(coll="ElectronGood", field="pt",bins=[35,40,45,50,55,60,70,100], label="tight electrons $p_T$")
            ] ),
        "electron_tight_eta" : HistConf(
            [
                Axis(coll="ElectronGood", field="eta", bins=[-2.4,-2.15,-1.8,-1.479,-1.,-0.5,0.,0.5,1.,1.479,1.8,2.15,2.4], label="tight electrons $\eta$")
            ] ),
        "electron_tight_dphi": HistConf(
            [
                Axis(coll="ElectronGood", field="dphi_met", bins=[0.,0.1,0.4,0.5,1.0,3.14], label="tight electrons $\Delta \phi$")
            ] ),
        "electron_tight_pt_eta_dphi" : HistConf(
            [
                Axis(coll="ElectronGood", field="pt",bins=[35,40,45,50,55,60,70,100], label="tight electrons $p_T$"),
                Axis(coll="ElectronGood", field="eta", bins=[-2.4,-2.15,-1.8,-1.479,-1.,-0.5,0.,0.5,1.,1.479,1.8,2.15,2.4], label="tight electrons $\eta$"),
                Axis(coll="ElectronGood", field="dphi_met", bins=[0.,0.5,1.0,3.14], label="tight electrons $\Delta \phi$")
            ] ),
        
        "muon_loose_pt" : HistConf(
            [
                Axis(coll="MuonLoose", field="pt",bins=[26,28,30,32,35,40,45,55,100], label="loose muons $p_T$")
            ] ),
        "muon_loose_eta" : HistConf(
            [
                Axis(coll="MuonLoose", field="eta", bins=[-2.4,-2.15,-1.8,-1.479,-1.,-0.5,0.,0.5,1.,1.479,1.8,2.15,2.4], label="loose muons $\eta$")
            ] ),
        "muon_loose_dphi": HistConf(
            [
                Axis(coll="MuonLoose", field="dphi_met", bins=[0.,0.5,1.0,3.14], label="loose muon $\Delta \phi$")
            ] ),
        "muon_loose_pt_eta_dphi" : HistConf(
            [
                Axis(coll="MuonLoose", field="pt",bins=[26,28,30,32,35,40,45,55,100], label="loose muons $p_T$"),
                Axis(coll="MuonLoose", field="eta", bins=[-2.4,-2.15,-1.8,-1.479,-1.,-0.5,0.,0.5,1.,1.479,1.8,2.15,2.4], label="loose muons $\eta$"),
                Axis(coll="MuonLoose", field="dphi_met", bins=[0.,0.1,0.4,0.5,1.0,3.14], label="loose muon $\Delta \phi$")
            ] ),
        "muon_tight_pt" : HistConf(
            [
                Axis(coll="MuonGood", field="pt",bins=[26,28,30,32,35,40,45,55,100], label="tight muons $p_T$")
            ] ),
        "muon_tight_eta" : HistConf(
            [
                Axis(coll="MuonGood", field="eta", bins=[-2.4,-2.15,-1.8,-1.479,-1.,-0.5,0.,0.5,1.,1.479,1.8,2.15,2.4], label="tight muons $\eta$")
            ] ),
        "muon_tight_dphi": HistConf(
            [
                Axis(coll="MuonGood", field="dphi_met", bins=[0.,0.1,0.4,0.5,1.0,3.14], label="tight muons $\Delta \phi$")
            ] ),
        "muon_tight_pt_eta_dphi" : HistConf(
            [
                Axis(coll="MuonGood", field="pt",bins=[26,28,30,32,35,40,45,55,100], label="tight muons $p_T$"),
                Axis(coll="MuonGood", field="eta", bins=[-2.4,-2.15,-1.8,-1.479,-1.,-0.5,0.,0.5,1.,1.479,1.8,2.15,2.4], label="tight muons $\eta$"),
                Axis(coll="MuonGood", field="dphi_met", bins=[0.,0.5,1.0,3.14], label="tight muons $\Delta \phi$")
            ] ),
        "mt_w_leptonic": HistConf([Axis(coll="events", field="mt_w_leptonic",bins=20, start=0, stop=100, label=r"$m_T(W_{lep})$ [GeV]")]),
        "pt_miss": HistConf([Axis(coll="DeepMETResolutionTune", field="pt",bins=20, start=0, stop=100, label=r"$pT_{miss})$ [GeV]")]),
        
    },
)
