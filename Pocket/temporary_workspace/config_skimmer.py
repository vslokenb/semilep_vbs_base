# example_config_semileptonic.py
import os, cloudpickle
from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_functions import get_HLTsel, get_nPVgood, goldenJson, eventFlags
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.parameters.histograms import HistConf, Axis
from pocket_coffea.lib.weights.common import common_weights
from pocket_coffea.parameters import defaults

import workflow, custom_cut_functions
from workflow import VBSSemileptonicProcessor
from custom_cut_functions import (
    nLepton_skim_cut,
    vbs_semileptonic_presel,
    nJet_skim_cut,
    met_skim_cut
)

cloudpickle.register_pickle_by_value(workflow)
cloudpickle.register_pickle_by_value(custom_cut_functions)

localdir = os.path.dirname(os.path.abspath(__file__))


default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir + "/params")
parameters = defaults.merge_parameters_from_files(
    default_parameters,
    f"{localdir}/params/object_preselection_run2.yaml",
    f"{localdir}/params/triggers.yaml",
    f"{localdir}/params/plotting.yaml",
    update=True,
)


cfg = Configurator(
    parameters=parameters,
    datasets={
        "jsons": [
            f"{localdir}/datasets/WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/DYJetsToLL_M-10to50_TuneCP5_13TeV-amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8.json",
            f"{localdir}/datasets/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8.json",
            f"{localdir}/datasets/ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8.json",
            f"{localdir}/datasets/ST_t-channel_top_4f_inclusiveDecays_TuneCP5_13TeV-powhegV2-madspin-pythia8.json",
            f"{localdir}/datasets/ST_t-channel_antitop_4f_inclusiveDecays_TuneCP5_13TeV-powhegV2-madspin-pythia8.json",
            f"{localdir}/datasets/ttZJets_TuneCP5_13TeV_madgraphMLM_pythia8.json",
            f"{localdir}/datasets/ttWJets_TuneCP5_13TeV_madgraphMLM_pythia8.json",
            f"{localdir}/datasets/WplusTo2JZTo2LJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            f"{localdir}/datasets/WplusToLNuWminusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV.json",
            f"{localdir}/datasets/WplusToLNuWplusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            f"{localdir}/datasets/WminusTo2JZTo2LJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            f"{localdir}/datasets/WminusToLNuWminusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            f"{localdir}/datasets/WminusToLNuZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            f"{localdir}/datasets/WplusTo2JWminusToLNuJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV.json",
            f"{localdir}/datasets/WplusToLNuZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            f"{localdir}/datasets/ZTo2LZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            f"{localdir}/datasets/WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8.json",
            f"{localdir}/datasets/WZZ_TuneCP5_13TeV-amcatnlo-pythia8.json",
            f"{localdir}/datasets/ZZZ_TuneCP5_13TeV-amcatnlo-pythia8.json",
            f"{localdir}/datasets/WGToLNuG_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            f"{localdir}/datasets/ZGToLLG_01J_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8.json",
            f"{localdir}/datasets/WZTo3LNu_mllmin01_NNPDF31_TuneCP5_13TeV_powheg_pythia8.json",
            
            #########
            ## RUN 2 SIGNAL
            ########
            f"{localdir}/datasets/WplusTo2JWminusToLNuJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            f"{localdir}/datasets/WplusToLNuWminusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            f"{localdir}/datasets/WminusToLNuWminusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            f"{localdir}/datasets/WplusToLNuWplusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            f"{localdir}/datasets/WminusToLNuZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            f"{localdir}/datasets/WplusToLNuZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            #########
            #### SOME DATA
            #########
            #f"{localdir}/datasets/SingleMuon.json", ## 2017B Single Muon dataset
            
        ],
        "filter": {
            "samples": [
                
            "WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8",
            # "DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8",
            # "DYJetsToLL_M-10to50_TuneCP5_13TeV-amcatnloFXFX-pythia8",
            # "TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8",
            # "TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8",
            # "ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8",
            # "ST_t-channel_top_4f_inclusiveDecays_TuneCP5_13TeV-powhegV2-madspin-pythia8",
            # "ST_t-channel_antitop_4f_inclusiveDecays_TuneCP5_13TeV-powhegV2-madspin-pythia8",
            # "ttZJets_TuneCP5_13TeV_madgraphMLM_pythia8",
            # "ttWJets_TuneCP5_13TeV_madgraphMLM_pythia8",
            # "WplusTo2JZTo2LJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            # "WplusToLNuWminusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV",
            # "WplusToLNuWplusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            # "WminusTo2JZTo2LJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            # "WminusToLNuWminusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            # "WminusToLNuZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            # "WplusTo2JWminusToLNuJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV",
            # "WplusToLNuZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            # "ZTo2LZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            # "WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8",
            # "WZZ_TuneCP5_13TeV-amcatnlo-pythia8",
            # "ZZZ_TuneCP5_13TeV-amcatnlo-pythia8",
            # "WGToLNuG_TuneCP5_13TeV-madgraphMLM-pythia8",
            # "ZGToLLG_01J_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8",
            # "WZTo3LNu_mllmin01_NNPDF31_TuneCP5_13TeV_powheg_pythia8",
            
            # "WplusTo2JWminusToLNuJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8", 
            # "WplusToLNuWminusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8", 
            # "WminusToLNuWminusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8", 
            # "WplusToLNuWplusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8", 
            # "WminusToLNuZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8", 
            # "WplusToLNuZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            #########
            ## SOME DATA
            #########
            #"SingleMuon", ## 2017B Single Muon dataset
            ],
            "year": ["2017"],
        },
    },
    workflow=VBSSemileptonicProcessor,

    save_skimmed_files ="root://eosuser.cern.ch://eos/user/v/vslokenb/semilep_vbs_test/2017_UL_skim/",
    skim=[
        get_nPVgood(1),    # nPV>0
        eventFlags,        # PileupID
        goldenJson,        
        nLepton_skim_cut,  #nLepton >= 1
        nJet_skim_cut,      #nJet + nFatJet >= 3
        met_skim_cut,       # PuppiMET > 30 GeV
        get_HLTsel(primaryDatasets=["SingleMuon", "SingleEle"]),
    ],

    # 2) preselections 
    preselections=[vbs_semileptonic_presel],

   
    categories={
        "baseline": [passthrough],
        #"whad_peak": [whad_window_cut],  # |mjj^W - 80.4| < window
        #"central_jet_cuts": [central_fj_cut],
        #"boosted_jet_in_window": [msd_window_cut],
    },

   
    weights= {
        "common": {
            "inclusive": [
            ],
            "bycategory": {}
        },
    },
    variations = {
        "weights": {
            "common": {
                "inclusive": [
                              ],
                "bycategory": {}
            },
        },
    },
   
    variables={},
)
#         "psweight":     HistConf([Axis(coll="events", field="PSWeight", bins=30, start=-1.0, stop=12, label="PSWeight")]),
#         "lheweight":        HistConf([Axis(coll="events", field="LHEWeight", bins=30, start=-1.0, stop=12, label="LHEWeight")]),
#         "LHEReweightingWeight":     HistConf([Axis(coll="events", field="LHEReweightingWeight", bins=30, start=-1.0, stop=12, label="LHEReweightingWeight")]),
#         "LHEPdfWeight":     HistConf([Axis(coll="events", field="LHEPdfWeight", bins=30, start=-1, stop=12, label="LHEpdfweight")]),
#         "LHEScaleWeight":       HistConf([Axis(coll="events", field="LHEScaleWeight", bins=12, start=-1, stop=12, label="(LHEScaleWeight)")]),
#         "nJets":      HistConf([Axis(coll="events", field="nJetGood", bins=12, start=0, stop=12, label="N(jets)")]),
#         "nBJets":     HistConf([Axis(coll="events", field="nBJetGood", bins=6, start=0, stop=6, label="N(bjets)")]),
#         "nCentralJets": HistConf([Axis(coll="events", field="nCentralJetsGood", bins=12, start=0, stop=12, label="N(Central Jets)")]),
#         "nFatJets": HistConf([Axis(coll="events", field="nFatJetGood", bins=4, start=0, stop=4, label="N(Fat Jets)")]),
#         "nFatJetCentral": HistConf([Axis(coll="events", field="nFatJetCentral", bins=4, start=0, stop=4, label="N(Central Fat Jets)")]),

#         #pileup check
#         'nTrueInt':      HistConf([Axis(coll="Pileup", field="nTrueInt", bins=100, start=0, stop=100, label="nTrueInt")]),
#         'pudensity':       HistConf([Axis(coll="Pileup", field="pudensity", bins=100, start=0, stop=8, label="pudensity")]),
#         'gpudensity':      HistConf([Axis(coll="Pileup", field="gpudensity", bins=50, start=0, stop=1, label="gpudensity")]),
#         'nPU':      HistConf([Axis(coll="Pileup", field="nPU", bins=100, start=0, stop=100, label="nPU")]),
#         'sumEOOT':      HistConf([Axis(coll="Pileup", field="nTrueInt", bins=50, start=0, stop=400, label="sumEOOT")]),
#         'sumLOOT':      HistConf([Axis(coll="Pileup", field="nTrueInt", bins=20, start=0, stop=120, label="sumLOOT")]),


#         #PV check
#         'npvs':   HistConf([Axis(coll="PV", field="npvs", bins=20, start=0, stop=100, label="nPV")]),
#         'npvsGood':   HistConf([Axis(coll="PV", field="npvsGood", bins=20, start=0, stop=100, label=r"$nPV_{good}$")]),
#         # MET and mT
#         "met":        HistConf([Axis(coll="MET", field="pt", bins=50, start=0, stop=250, label=r"$p_T^{miss}$ [GeV]")]),
#         "met_phi":    HistConf([Axis(coll="MET", field="phi", bins=50, start=-4, stop=4, label=r"$\phi^{miss}$ [GeV]")]),
#         "puppimet_phi":    HistConf([Axis(coll="PuppiMET", field="phi", bins=50, start=-4, stop=4, label=r"$\phi^{miss}$ [GeV]")]),
#         "puppimet":        HistConf([Axis(coll="PuppiMET", field="pt", bins=50, start=0, stop=250, label=r"$p_T^{miss}$ [GeV]")]),
#         "mt_w_lep":   HistConf([Axis(coll="events", field="mt_w_leptonic", bins=30, start=0, stop=200, label=r"$m_T(W_{lep})$ [GeV]")]),
#         # Tagging jets (VBS)
#         "mjj_vbs":    HistConf([Axis(coll="vbsjets", field="mass", bins=50, start=300, stop=4000, label=r"$M_{jj}^{VBS}$ [GeV]")]),
#         "deta_vbs":   HistConf([Axis(coll="vbsjets", field="delta_eta", bins=24, start=2.0, stop=9.0, label=r"$|\Delta\eta_{jj}^{VBS}|$")]),
#         "dR_vbs":     HistConf([Axis(coll="events", field="vbs_dR", bins=40, start=0.0, stop=7.0, label=r"$\Delta R(jj)^{VBS}$")]),
#         # W hadronic
#         "m_jj_w":     HistConf([Axis(coll="w_had_jets", field="mass", bins=40, start=40, stop=120, label=r"$M_{jj}^{W\,had}$ [GeV]")]),
#         "pt_jj_w":     HistConf([Axis(coll="w_had_jets", field="pt", bins=40, start=40, stop=210, label=r"$p_T(jj^{W\,had})$ [GeV]")]),
#         "dR_w_had":   HistConf([Axis(coll="events", field="w_had_dR", bins=40, start=0.0, stop=4.0, label=r"$\Delta R(jj)^{W\,had}$")]),
#         "eta_w_had1":   HistConf([Axis(coll="events", field="w_had_jet1_eta", bins=48, start=-4.0, stop=4.0, label=r"$\eta(j2_{W\,had})$ [GeV]")]),
#         "eta_w_had2":   HistConf([Axis(coll="events", field="w_had_jet2_eta", bins=48, start=-4.0, stop=4.0, label=r"$\eta(j2_{W\,had})$ [GeV]")]),
#         "pt_w_had1":   HistConf([Axis(coll="events", field="w_had_jet1_pt", bins=60, start=0.0, stop=300.0, label=r"$p_T(j1_{W\,had})$ [GeV]")]),
#         "pt_w_had2":   HistConf([Axis(coll="events", field="w_had_jet2_pt", bins=60, start=0.0, stop=300.0, label=r"$p_T(j2_{W\,had})$ [GeV]")]),
#         "phi_w_had1":   HistConf([Axis(coll="events", field="w_had_jet1_phi", bins=48, start=-4.0, stop=4.0, label=r"$\phi(j2_{W\,had})$ [GeV]")]),
#         "phi_w_had2":   HistConf([Axis(coll="events", field="w_had_jet2_phi", bins=48, start=-4.0, stop=4.0, label=r"$\phi(j2_{W\,had})$ [GeV]")]),
#         # jets leading
#         "pt_tag1":    HistConf([Axis(coll="events", field="jet1_pt", bins=60, start=0, stop=300, label=r"$p_T(j_1)$ [GeV]")]),
#         "pt_tag2":    HistConf([Axis(coll="events", field="jet2_pt", bins=60, start=0, stop=300, label=r"$p_T(j_2)$ [GeV]")]),
#         "eta_tag1":   HistConf([Axis(coll="events", field="jet1_eta", bins=48, start=-4.8, stop=4.8, label=r"$\eta(j_1)$")]),
#         "eta_tag2":   HistConf([Axis(coll="events", field="jet2_eta", bins=48, start=-4.8, stop=4.8, label=r"$\eta(j_2)$")]),
#         "phi_tag1":   HistConf([Axis(coll="events", field="jet1_phi", bins=48, start=-4., stop=4., label=r"$\phi(j_1)$")]),
#         "phi_tag2":   HistConf([Axis(coll="events", field="jet2_phi", bins=48, start=-4., stop=4., label=r"$\phi(j_2)$")]),
#         # W lepton
#         "eta_w_lep":   HistConf([Axis(coll="events", field="w_lep_eta", bins=32, start=-4.0, stop=4.0, label=r"$\eta^{W\,lep}$ ")]),
#         "pt_w_lep":   HistConf([Axis(coll="events", field="w_lep_pt", bins=40, start=0.0, stop=300.0, label=r"$p_T^{W\,lep}$ [GeV]")]),
#         "phi_w_lep":   HistConf([Axis(coll="events", field="w_lep_phi", bins=32, start=-4.0, stop=4.0, label=r"$\phi^{W\,lep}$ ")]),
#         "m_ll":   HistConf([Axis(coll="ll", field="m_ll", bins=50, start=0, stop=200.0, label=r"$m_{ll}$ [GeV]")]),
#         # lead lepton dR
#         "lead_wlep_wjet1_dR": HistConf([Axis(coll="events", field="lead_wlep_wjet1_dR", bins=40, start=0.0, stop=4.0, label=r"$\Delta R(lj_1)^{W}$")]),
#         "lead_wlep_wjet2_dR": HistConf([Axis(coll="events", field="lead_wlep_wjet2_dR", bins=40, start=0.0, stop=4.0, label=r"$\Delta R(lj_2)^{W}$")]),
#         "lead_wlep_wfatjet1_dR": HistConf([Axis(coll="events", field="lead_wlep_wfatjet1_dR", bins=40, start=0.0, stop=4.0, label=r"$\Delta R(lJ_1)^{W}$")]),
#         "lead_wlep_w_resolved_dR": HistConf([Axis(coll="events", field="lead_wlep_w_resolved_dR", bins=40, start=0.0, stop=4.0, label=r"$\Delta R(lW_{resolved})$")]),
        
#         # lead lepton dEta
#         "lead_wlep_wfatjet1_deta":   HistConf([Axis(coll="events", field="lead_wlep_wfatjet1_deta", bins=24, start=0.0, stop=9.0, label=r"$|\Delta\eta_{lJ}^{W}|$")]),
#         "lead_wlep_wjet1_deta":   HistConf([Axis(coll="events", field="lead_wlep_wjet1_deta", bins=24, start=0.0, stop=9.0, label=r"$|\Delta\eta_{lj_1}^{W}|$")]),
#         "lead_wlep_wjet2_deta":   HistConf([Axis(coll="events", field="lead_wlep_wjet2_deta", bins=24, start=0.0, stop=9.0, label=r"$|\Delta\eta_{lj_2}^{W}|$")]),
#         "lead_wlep_w_resolved_deta":   HistConf([Axis(coll="events", field="lead_wlep_w_resolved_deta", bins=24, start=2.0, stop=9.0, label=r"$|\Delta\eta_{lW}^{resolved}|$")]),

#         # dphi plots
#         "lead_wlep_MET_dphi":   HistConf([Axis(coll="events", field="lead_wlep_MET_dphi", bins=32, start=-4, stop=4.0, label=r"$|\Delta\phi_{l,MET}|$")]),
#         "lead_wlep_wfatjet1_dphi":   HistConf([Axis(coll="events", field="lead_wlep_wfatjet1_dphi", bins=32, start=-4, stop=4.0, label=r"$|\Delta\phi_{l,J^W}|$")]),
#         "lead_wlep_wjet1_dphi":   HistConf([Axis(coll="events", field="lead_wlep_wjet1_dphi", bins=32, start=-4, stop=4.0, label=r"$|\Delta\phi_{l,j_1^WT}|$")]),
#         "lead_wlep_wjet2_dphi":   HistConf([Axis(coll="events", field="lead_wlep_wjet2_dphi", bins=32, start=-4, stop=4.0, label=r"$|\Delta\phi_{l,j_1^WTT}|$")]),
#         "w_lep_w_resolved_dphi":   HistConf([Axis(coll="events", field="w_lep_w_resolved_dphi", bins=32, start=-4, stop=4.0, label=r"$|\Delta\phi_{W_{lep}W_{resolved}}|$")]),
#         "w_lep_w_boost_dphi":   HistConf([Axis(coll="events", field="w_lep_w_boost_dphi", bins=32, start=-4, stop=4.0, label=r"$|\Delta\phi_{W_{lep}W_{boost}}|$")]),


#         # # W fat jet
#         "fj_pt":    HistConf([Axis(coll="w_fatjet", field="pt",  bins=60, start=150, stop=1000, label=r"$p_T(J^{W})$ [GeV]")]),
#         "fj_eta":   HistConf([Axis(coll="w_fatjet", field="eta", bins=48, start=-2.4, stop=2.4,   label=r"$\eta(J^{W})$")]),
#         "fj_msd":   HistConf([Axis(coll="w_fatjet", field="msd", bins=40, start=0,   stop=200,   label=r"$m_{SD}(J^{W})$ [GeV]")]),
#         "fj_t21":   HistConf([Axis(coll="w_fatjet", field="tau21", bins=32, start=0, stop=1.1,   label=r"$\tau_{21}$")]),

#         # angles between leptonic, hadronic W
#         #"w_lep_w_resolved_dR": HistConf([Axis(coll="events", field="w_lep_w_resolved_dR", bins=40, start=0.0, stop=4.0, label=r"$\Delta R(W_{lep}W_{resolved})$")]),
#         #"w_lep_w_boost_dR": HistConf([Axis(coll="events", field="w_lep_w_boost_dR", bins=40, start=0.0, stop=4.0, label=r"$\Delta R(W_{lep}W_{boost})$")]),
#         #"w_lep_w_resolved_deta":   HistConf([Axis(coll="events", field="w_lep_w_resolved_deta", bins=24, start=2.0, stop=9.0, label=r"$|\Delta\eta_{W_{lep}W^{resolved}}|$")]),
#         #"w_lep_w_boost_deta":   HistConf([Axis(coll="events", field="w_lep_w_boost_deta", bins=24, start=2.0, stop=9.0, label=r"$|\Delta\eta_{W_{lep}W^{boost}}|$")]),
#     },
# )
