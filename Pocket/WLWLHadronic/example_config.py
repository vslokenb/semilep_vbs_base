from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_definition import Cut
from pocket_coffea.lib.cut_functions import get_nObj_min, get_HLTsel, get_nPVgood, goldenJson, eventFlags
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.parameters.histograms import *
import workflow
from workflow import VBSWWBaseProcessor
from pocket_coffea.lib.weights.common import common_weights

# Register custom modules in cloudpickle to propagate them to dask workers
import cloudpickle
import custom_cut_functions 
cloudpickle.register_pickle_by_value(workflow)
cloudpickle.register_pickle_by_value(custom_cut_functions)

from custom_cut_functions import *
import os
localdir = os.path.dirname(os.path.abspath(__file__))

from pocket_coffea.parameters import defaults
default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir+"/params")

parameters = defaults.merge_parameters_from_files(default_parameters,
                                                  f"{localdir}/params/object_preselection.yaml",
                                                  f"{localdir}/params/triggers.yaml",
                                                  f"{localdir}/params/plotting.yaml",
                                                  update=True)



cfg = Configurator(
    parameters = parameters,
    datasets = {
        "jsons": [f"{localdir}/datasets/DATA_DoubleMuon.json",
                  f"{localdir}/datasets/DATA_EGamma.json",
                  f"{localdir}/datasets/DATA_MuonEG.json",
                  f"{localdir}/datasets/VBS-SSWW_PolarizationLL_TuneCP5_13p6TeV_madgraph-pythia8.json"
                    ],
        "filter" : {
            "samples": ["DATA_DoubleMuon",
                        "DATA_EGamma",
                        "DATA_MuonEG",
                        "VBS-SSWW_PolarizationLL_TuneCP5_13p6TeV_madgraph-pythia8"],
            "samples_exclude" : [],
            "year": ['2022_preEE', '2022_postEE']
        }
    },

    workflow = VBSWWBaseProcessor,

    skim = [get_nPVgood(1), eventFlags, goldenJson,
            nLepton_skim_cut,
            
            get_HLTsel(primaryDatasets=["DoubleMuon", "EGamma", "MuonEG"])], 
    
    preselections = [vbs_ss_dilepton_presel],
    categories = {
        "baseline": [passthrough],
    },

    weights_classes = common_weights,
    
    weights = {
        "common": {
            "inclusive": ["genWeight","lumi","XS",
                          "pileup",
                          "sf_mu_id","sf_mu_iso", "sf_ele_id", "sf_ele_reco"
                          ],
            "bycategory" : {
            }
        },
        "bysample": {
        }
    },

    variations = {
        "weights": {
            "common": {
                "inclusive": [  "pileup",
                                "sf_mu_id", "sf_mu_iso", "sf_ele_id", "sf_ele_reco"
                              ],
                "bycategory" : {
                }
            },
        "bysample": {
        }    
        },
    },

    
   variables = {
        **muon_hists(coll="LeptonGood", pos=0, name="leading_lepton"),
        **count_hist(name="nLeptonGood", coll="LeptonGood",bins=4, start=0, stop=4),
        **count_hist(name="nJets", coll="JetGood",bins=10, start=0, stop=10),
        **count_hist(name="nBJets", coll="BJetGood",bins=5, start=0, stop=5),
        **jet_hists(coll="JetGood", pos=0, name='leading_jet'),
        "mll" : HistConf( [Axis(coll="ll", field="mass", bins=100, start=0, stop=200, label=r"$M_{\ell\ell}$ [GeV]")] ),
        "mjj_vbs": HistConf([Axis(coll="vbsjets", field="mass", bins=50, start=200, stop=2500, label=r"$M_{jj}^{VBS}$ [GeV]")]),
        "delta_eta_vbs": HistConf([Axis(coll="vbsjets", field="delta_eta", bins=20, start=2.5, stop=8, label=r"$|\Delta\eta_{jj}^{VBS}|$")]),
        "met": HistConf([Axis(coll="MET", field="pt", bins=50, start=0, stop=400, label=r"$p_{T}^{miss}$ [GeV]")] ),
    }
)

