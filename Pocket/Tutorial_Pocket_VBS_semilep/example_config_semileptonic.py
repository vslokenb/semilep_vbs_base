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

# Para que el ejecutor multiproceso pueda serializar tus módulos locales
cloudpickle.register_pickle_by_value(workflow)
cloudpickle.register_pickle_by_value(custom_cut_functions)

localdir = os.path.dirname(os.path.abspath(__file__))

# --- parámetros por defecto + tus YAMLs ---
default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir + "/params")
parameters = defaults.merge_parameters_from_files(
    default_parameters,
    f"{localdir}/params/object_preselection.yaml",
    f"{localdir}/params/triggers.yaml",
    f"{localdir}/params/plotting.yaml",
    update=True,
)

# --- configuración ---
cfg = Configurator(
    parameters=parameters,
    datasets={
        "jsons": [
            f"{localdir}/datasets/WpWpJJ-EWK_TuneCP5_13p6TeV-powheg-pythia8.json",
            f"{localdir}/datasets/WmWmJJ-EWK_TuneCP5_13p6TeV-powheg-pythia8.json",
        ],
        "filter": {
            "samples": [
                "WpWpJJ-EWK_TuneCP5_13p6TeV-powheg-pythia8",
                "WmWmJJ-EWK_TuneCP5_13p6TeV-powheg-pythia8",
            ],
            "year": ["2022_preEE", "2022_postEE"],
        },
    },
    workflow=VBSSemileptonicProcessor,

    # 1) skim rápido (barato) antes de cargar todo
    skim=[
        get_nPVgood(1),    # nPV>0
        eventFlags,        # PileupID, HBHE, etc.
        goldenJson,        # seguro para DATA, no molesta en MC
        nLepton_skim_cut,  # ≥1 leptón cualquiera (mu/e)
        get_HLTsel(primaryDatasets=["SingleMuon", "EGamma"]),
    ],

    # 2) preselections “de física”
    preselections=[vbs_semileptonic_presel],

    # 3) categorías para los plots
    categories={
        "baseline": [passthrough],
        "whad_peak": [whad_window_cut],  # |mjj^W - 80.4| < window
    },

    # pesos y variaciones típicas (puedes podar si quieres)
    weights_classes=common_weights,
    weights={"common": {"inclusive": ["genWeight", "lumi", "XS", "pileup",
                                      "sf_mu_id", "sf_mu_iso", "sf_ele_id", "sf_ele_reco"]}},
    variations={"weights": {"common": {"inclusive": ["pileup","sf_mu_id","sf_mu_iso","sf_ele_id","sf_ele_reco"]}}},

    # 4) variables / histogramas
    variables={
        # conteos sencillos
        "nJets":      HistConf([Axis(coll="events", field="nJetGood", bins=12, start=0, stop=12, label="N(jets)")]),
        "nBJets":     HistConf([Axis(coll="events", field="nBJetGood", bins=6, start=0, stop=6, label="N(bjets)")]),
        # MET y mT
        "met":        HistConf([Axis(coll="MET", field="pt", bins=50, start=0, stop=250, label=r"$p_T^{miss}$ [GeV]")]),
        "mt_w_lep":   HistConf([Axis(coll="events", field="mt_w_leptonic", bins=50, start=0, stop=200, label=r"$m_T(W_{lep})$ [GeV]")]),
        # Tagging jets (VBS)
        "mjj_vbs":    HistConf([Axis(coll="vbsjets", field="mass", bins=50, start=300, stop=4000, label=r"$M_{jj}^{VBS}$ [GeV]")]),
        "deta_vbs":   HistConf([Axis(coll="vbsjets", field="delta_eta", bins=24, start=2.0, stop=8.0, label=r"$|\Delta\eta_{jj}^{VBS}|$")]),
        "dR_vbs":     HistConf([Axis(coll="events", field="vbs_dR", bins=40, start=0.0, stop=6.0, label=r"$\Delta R(jj)^{VBS}$")]),
        # W hadrónico (resuelto)
        "m_jj_w":     HistConf([Axis(coll="w_had_jets", field="mass", bins=40, start=40, stop=120, label=r"$M_{jj}^{W\,had}$ [GeV]")]),
        "dR_w_had":   HistConf([Axis(coll="events", field="w_had_dR", bins=40, start=0.0, stop=4.0, label=r"$\Delta R(jj)^{W\,had}$")]),
        # jets líderes (para ver cinemática general)
        "pt_tag1":    HistConf([Axis(coll="events", field="jet1_pt", bins=60, start=0, stop=300, label=r"$p_T(j_1)$ [GeV]")]),
        "pt_tag2":    HistConf([Axis(coll="events", field="jet2_pt", bins=60, start=0, stop=300, label=r"$p_T(j_2)$ [GeV]")]),
        "eta_tag1":   HistConf([Axis(coll="events", field="jet1_eta", bins=48, start=-4.8, stop=4.8, label=r"$\eta(j_1)$")]),
        "eta_tag2":   HistConf([Axis(coll="events", field="jet2_eta", bins=48, start=-4.8, stop=4.8, label=r"$\eta(j_2)$")]),
    },
)
