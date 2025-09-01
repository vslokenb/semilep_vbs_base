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
    whad_window_cut,
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
            #f"{localdir}/datasets/WpWpJJ-EWK_TuneCP5_13p6TeV-powheg-pythia8.json",
            #f"{localdir}/datasets/WmWmJJ-EWK_TuneCP5_13p6TeV-powheg-pythia8.json",
            #f"{localdir}/datasets/TT_TuneCP5_13p6TeV_powheg-pythia8.json",
            #f"{localdir}/datasets/WminusToLNuWminusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            #f"{localdir}/datasets/WplusTo2JWminusToLNuJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            #f"{localdir}/datasets/WplusToLNuWminusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            #f"{localdir}/datasets/WplusToLNuWplusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            #f"{localdir}/datasets/WminusToLNuZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            #f"{localdir}/datasets/WplusToLNuZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",

            #f"{localdir}/datasets/WWtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8.json",
            #f"{localdir}/datasets/WtoLNu-2Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            #f"{localdir}/datasets/ZZto2L2Q_TuneCP5_13p6TeV_powheg-pythia8.json",
            #f"{localdir}/datasets/TTto2L2Nu_TuneCP5_ERDOn_13p6TeV_powheg-pythia8.json",
            #f"{localdir}/datasets/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8.json",
            #f"{localdir}/datasets/DYto2L-2Jets_MLL-10to50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            #f"{localdir}/datasets/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",

            #f"{localdir}/datasets/SingleMuon.json",
            f"{localdir}/datasets/WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            #f"{localdir}/datasets/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8.json"
        ],
        "filter": {
            "samples": [
                #"WpWpJJ-EWK_TuneCP5_13p6TeV-powheg-pythia8",
                #"WplusTo2JWminusToLNuJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
                #"WplusToLNuWminusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
                #"WminusToLNuWminusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
                #"WplusToLNuWplusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
                #"WminusToLNuZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
                #"WplusToLNuZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",

                #"WtoLNu-2Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                #"ZZto2L2Q_TuneCP5_13p6TeV_powheg-pythia8",
                #"WWtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8",
                #"TTto2L2Nu_TuneCP5_ERDOn_13p6TeV_powheg-pythia8",
                #"TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8",
                #"DYto2L-2Jets_MLL-10to50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                #"DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
                
                "WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8",
                #"DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8",
                #"SingleMuon",
            ],
            "year": ["2022_preEE", "2022_postEE","2017","2018"],
        },
    },
    workflow=VBSSemileptonicProcessor,

    
    skim=[
        get_nPVgood(1),    # nPV>0
        eventFlags,        # PileupID
        goldenJson,        
        nLepton_skim_cut,  
        get_HLTsel(primaryDatasets=["SingleMuon"]),#, "SingleEle"]),
    ],

    # 2) preselections 
    preselections=[vbs_semileptonic_presel],

   
    categories={
        "baseline": [passthrough],
        "whad_peak": [whad_window_cut],  # |mjj^W - 80.4| < window
    },

   
    weights_classes=common_weights,
    weights={"common": {"inclusive": ["genWeight", "lumi", "XS", "pileup",
                                      "sf_mu_id", "sf_mu_iso", "sf_ele_id", "sf_ele_reco"]}},
    variations={"weights": {"common": {"inclusive": ["pileup","sf_mu_id","sf_mu_iso","sf_ele_id","sf_ele_reco"]}}},

   
    variables={
     
        "nJets":      HistConf([Axis(coll="events", field="nJetGood", bins=12, start=0, stop=12, label="N(jets)")]),
        "nBJets":     HistConf([Axis(coll="events", field="nBJetGood", bins=6, start=0, stop=6, label="N(bjets)")]),
        "nCentralJets": HistConf([Axis(coll="events", field="nCentralJetsGood", bins=12, start=0, stop=12, label="N(Central Jets)")]),
        # MET and mT
        "met":        HistConf([Axis(coll="MET", field="pt", bins=50, start=0, stop=250, label=r"$p_T^{miss}$ [GeV]")]),
        "met_phi":    HistConf([Axis(coll="MET", field="phi", bins=50, start=-4, stop=4, label=r"$\phi^{miss}$ [GeV]")]),
        "mt_w_lep":   HistConf([Axis(coll="events", field="mt_w_leptonic", bins=30, start=0, stop=200, label=r"$m_T(W_{lep})$ [GeV]")]),
        # Tagging jets (VBS)
        "mjj_vbs":    HistConf([Axis(coll="vbsjets", field="mass", bins=50, start=300, stop=4000, label=r"$M_{jj}^{VBS}$ [GeV]")]),
        "deta_vbs":   HistConf([Axis(coll="vbsjets", field="delta_eta", bins=24, start=2.0, stop=9.0, label=r"$|\Delta\eta_{jj}^{VBS}|$")]),
        "dR_vbs":     HistConf([Axis(coll="events", field="vbs_dR", bins=40, start=0.0, stop=7.0, label=r"$\Delta R(jj)^{VBS}$")]),
        # W hadronic
        "m_jj_w":     HistConf([Axis(coll="w_had_jets", field="mass", bins=40, start=40, stop=120, label=r"$M_{jj}^{W\,had}$ [GeV]")]),
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
        # W lepton
        "eta_w_lep":   HistConf([Axis(coll="events", field="w_lep_eta", bins=32, start=-4.0, stop=4.0, label=r"$\eta^{W\,lep}$ ")]),
        "pt_w_lep":   HistConf([Axis(coll="events", field="w_lep_pt", bins=40, start=0.0, stop=300.0, label=r"$p_T^{W\,lep}$ [GeV]")]),
        "phi_w_lep":   HistConf([Axis(coll="events", field="w_lep_phi", bins=32, start=-4.0, stop=4.0, label=r"$\phi^{W\,lep}$ ")]),
        "m_ll":   HistConf([Axis(coll="ll", field="m_ll", bins=50, start=0, stop=200.0, label=r"$m_{ll}$ [GeV]")]),
    },
)
