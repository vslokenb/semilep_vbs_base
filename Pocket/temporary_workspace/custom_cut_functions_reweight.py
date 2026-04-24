# custom_cut_functions.py
import awkward as ak
import numpy as np
from pocket_coffea.lib.cut_definition import Cut

# ---------- Skim: ≥1 leptón  (mu/e) ----------
def nLepton_skim(events, params, **kwargs):
    good_elec = events.Electron[events.Electron.pt > 38]
    good_muon = events.Muon[events.Muon.pt > 30]
   
    return (ak.num(good_elec) + ak.num(good_muon) >= 1)

nLepton_skim_cut = Cut(name="nLepton_skim", params={}, function=nLepton_skim)

def nJet_skim(events, params, **kwargs):
    good_jet =events.Jet[events.Jet.pt > 30]
    good_fatjet =events.FatJet[events.FatJet.pt > 30]
    return (ak.num(good_jet) + ak.num(good_fatjet) >= 3)

nJet_skim_cut = Cut(name="nJet_skim", params={}, function=nJet_skim)

def met_skim(events, params, **kwargs):
    met_cut = (events.DeepMETResolutionTune.pt > 30)
    return met_cut

met_skim_cut = Cut(name="met_skim", params={}, function=met_skim)

# ---------- Preselection semileptonic VBS ----------
def select_vbs_semileptonic(events, params, **kwargs):
    
    one_loosest_mu = (ak.num(events.Muon) >= 1) #& (ak.num(events.Electron) == 0)
    one_loosest_ele = (ak.num(events.Electron) >= 1)
    if "WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8" in events.metadata["dataset"]:
        ht_mask = (events.LHE.HT <= 70.)
    else:
        ht_mask = True
    
    # w_pt_stitch = (events.gen_w_pt_by_pdg < 100)
    mask = (one_loosest_mu | one_loosest_ele) & ht_mask#& w_pt_stitch#& ht_mask#& met_cut & two_j & cut_mt_w & b_veto #& ht_mask#&  loose_lep_veto #(lep.pt > 35.0) &
    return ak.values_astype(mask, np.bool_)

vbs_semileptonic_presel = Cut(
    name="vbs_semileptonic",
    params={
        "met_pt": 30.0,
        #"mjj_vbs": 500.0,
        #"delta_eta_vbs": 2.5,
        "apply_b_veto": False,
        "require_lep_central": True,
    },
    function=select_vbs_semileptonic,
)


def w_check_mu(events, params, **kwargs):
    muon_ch = (events.nMuonGood == 1) #& (events.nMuonGood30 == 1)
    met_cut = (events.DeepMETResolutionTune.pt      > 30)

    mask =  muon_ch & met_cut 
    return ak.values_astype(mask, np.bool_)

w_check_mu = Cut(
    name="w_check_mu",
    params={"mjj_w_window": 20.0,
            "mjj_vbs": 500.0,
            "delta_eta_vbs": 2.5},  
    function=w_check_mu,
)

def w_check_e(events, params, **kwargs):
    muon_ch = (events.nElectronGood == 1) #& (events.nMuonGood30 == 1)
    met_cut = (events.DeepMETResolutionTune.pt  > 30)

    mask =  muon_ch & met_cut 
    return ak.values_astype(mask, np.bool_)

w_check_e = Cut(
    name="w_check_e",
    params={"mjj_w_window": 20.0,
            "mjj_vbs": 500.0,
            "delta_eta_vbs": 2.5},  
    function=w_check_e,
)

def z_check_mu(events, params, **kwargs):
    muon_ch = (events.nMuonGood == 2)
    zmass = ak.fill_none(ak.firsts(getattr(events.ll, "m_ll", None)), np.nan)
    within = np.where(np.isnan(zmass), False, np.abs(zmass - 91) < params["z_window"])
    
    mask = within & muon_ch
    return ak.values_astype(mask, np.bool_)


z_check_mu = Cut(
    name="z_check_mu",
    params={"z_window": 10,
            "mjj_vbs": 500.0,
            "delta_eta_vbs": 2.5}, 
    function=z_check_mu,
    )

def z_check_e(events, params, **kwargs):
    muon_ch = (events.nElectronGood == 2)
    zmass = ak.fill_none(ak.firsts(getattr(events.ll, "m_ll", None)), np.nan)
    within = np.where(np.isnan(zmass), False, np.abs(zmass - 91) < params["z_window"])
    
    mask = within & muon_ch
    return ak.values_astype(mask, np.bool_)


z_check_e = Cut(
    name="z_check_e",
    params={"z_window": 10,
            "mjj_vbs": 500.0,
            "delta_eta_vbs": 2.5}, 
    function=z_check_e,
    )



def w_check_mu_2j(events, params, **kwargs):
    two_j  = (events.nJetGood30    >= 2)
    muon_ch = (events.nMuonGood == 1) #& (events.nMuonGood30 == 1)
    met_cut = (events.DeepMETResolutionTune.pt      > 30)

    mask =  muon_ch & met_cut & two_j
    return ak.values_astype(mask, np.bool_)

w_check_mu_2j = Cut(
    name="w_check_mu_2j",
    params={"mjj_w_window": 20.0,
            "mjj_vbs": 500.0,
            "delta_eta_vbs": 2.5},  
    function=w_check_mu_2j,
)

def w_check_e_2j(events, params, **kwargs):
    two_j  = (events.nJetGood30    >= 2)
    muon_ch = (events.nElectronGood == 1) #& (events.nMuonGood30 == 1)
    met_cut = (events.DeepMETResolutionTune.pt  > 30)

    mask =  muon_ch & met_cut & two_j
    return ak.values_astype(mask, np.bool_)

w_check_e_2j = Cut(
    name="w_check_e_2j",
    params={"mjj_w_window": 20.0,
            "mjj_vbs": 500.0,
            "delta_eta_vbs": 2.5},  
    function=w_check_e_2j,
)

def z_check_mu_2j(events, params, **kwargs):
    two_j  = (events.nJetGood30    >= 2)
    muon_ch = (events.nMuonGood == 2)
    zmass = ak.fill_none(ak.firsts(getattr(events.ll, "m_ll", None)), np.nan)
    within = np.where(np.isnan(zmass), False, np.abs(zmass - 91) < params["z_window"])
    
    mask = within & muon_ch & two_j
    return ak.values_astype(mask, np.bool_)


z_check_mu_2j = Cut(
    name="z_check_mu_2j",
    params={"z_window": 10,
            "mjj_vbs": 500.0,
            "delta_eta_vbs": 2.5}, 
    function=z_check_mu_2j,
    )

def z_check_e_2j(events, params, **kwargs):
    two_j  = (events.nJetGood30    >= 2)
    muon_ch = (events.nElectronGood == 2)
    zmass = ak.fill_none(ak.firsts(getattr(events.ll, "m_ll", None)), np.nan)
    within = np.where(np.isnan(zmass), False, np.abs(zmass - 91) < params["z_window"])
    
    mask = within & muon_ch & two_j
    return ak.values_astype(mask, np.bool_)


z_check_e_2j = Cut(
    name="z_check_e_2j",
    params={"z_window": 10,
            "mjj_vbs": 500.0,
            "delta_eta_vbs": 2.5}, 
    function=z_check_e_2j,
    )
