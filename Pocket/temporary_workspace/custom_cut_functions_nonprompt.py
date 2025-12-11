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
    return (ak.num(good_jet) + ak.num(good_fatjet) >= 1)

nJet_skim_cut = Cut(name="nJet_skim", params={}, function=nJet_skim)

def met_skim(events, params, **kwargs):
    met_cut = (events.PuppiMET.pt < 20)
    return met_cut

met_skim_cut = Cut(name="met_skim", params={}, function=met_skim)

# ---------- Preselection semileptonic VBS ----------
def select_vbs_semileptonic(events, params, **kwargs):
    
    one_lep = (events.nLeptonLoose >=1)
    
    met_cut = (events.PuppiMET.pt      <  params["met_pt"]) #USE PUPPIMET NOT PF MET
    recoil_jet = ((events.nFatJetGood + events.nJetGood_recoil) >= 1)
    cut_mt_w = (events.mt_w_leptonic < 20.0)

    # ht_mask = (events.LHE.HT <= 70.)
    # w_pt_stitch = (events.gen_w_pt_by_pdg < 100)
    mask = one_lep & met_cut & recoil_jet & cut_mt_w #& w_pt_stitch#& ht_mask#& b_veto #& ht_mask#&  loose_lep_veto #(lep.pt > 35.0) &
    return ak.values_astype(mask, np.bool_)

vbs_semileptonic_presel = Cut(
    name="vbs_semileptonic",
    params={
        "met_pt": 20.0,
        #"mjj_vbs": 500.0,
        #"delta_eta_vbs": 2.5,
        "apply_b_veto": False,
        "require_lep_central": True,
    },
    function=select_vbs_semileptonic,
)

def lele_ch(events, params, **kwargs):
    ele_ch = (events.nElectronGood38 >= 1) & (events.nMuonGood30 == 0)
    

    mask = ele_ch
    return ak.values_astype(mask, np.bool_)

lele_ch_cut = Cut(
    name="lele_ch",
    params={"mjj_w_window": 20.0,
            "mjj_vbs": 500.0,
            "delta_eta_vbs": 2.5},  
    function=lele_ch,
)

def lmu_ch(events, params, **kwargs):
    mu_ch = (events.nElectronLoose == 0) & (events.nMuonLoose >= 1)
    

    mask = mu_ch
    return ak.values_astype(mask, np.bool_)

lmu_ch_cut = Cut(
    name="lmu_ch",
    params={"mjj_w_window": 20.0,
            "mjj_vbs": 500.0,
            "delta_eta_vbs": 2.5},  
    function=lmu_ch,
)


def ele_ch(events, params, **kwargs):
    ele_ch = (events.nElectronLoose >= 1) & (events.nMuonLoose == 0)
    

    mask = ele_ch
    return ak.values_astype(mask, np.bool_)

ele_ch_cut = Cut(
    name="ele_ch",
    params={"mjj_w_window": 20.0,
            "mjj_vbs": 500.0,
            "delta_eta_vbs": 2.5},  
    function=ele_ch,
)

def mu_ch(events, params, **kwargs):
    mu_ch = (events.nElectronGood38 >= 0) & (events.nMuonGood30 >= 1)
    

    mask = mu_ch
    return ak.values_astype(mask, np.bool_)

mu_ch_cut = Cut(
    name="mu_ch",
    params={"mjj_w_window": 20.0,
            "mjj_vbs": 500.0,
            "delta_eta_vbs": 2.5},  
    function=mu_ch,
)