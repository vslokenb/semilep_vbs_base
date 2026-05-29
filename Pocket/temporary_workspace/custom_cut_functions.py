# custom_cut_functions.py
import awkward as ak
import numpy as np
from pocket_coffea.lib.cut_definition import Cut

# ---------- Skim: ≥1 leptón  (mu/e) ----------
def nLepton_skim(events, params, **kwargs):
    good_elec = events.Electron[events.Electron.pt > 35]
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
    
    #pu_pv_corrections = (events.PV.npvsGood < 55) | (events.PV.npvsGood > 60) 
    one_lep = (events.nLeptonGood == 1)
    
    two_j  = (events.nJetGood30    >= 2)
    met_cut = (events.DeepMETResolutionTune.pt      >  params["met_pt"]) #USE DeepMETResolutionTune NOT PF MET

    #good_bjet =events.JetGood[(np.abs(events.JetGood.eta) < 2.5) & (np.abs(events.JetGood.partonFlavour) == 5)]
    
    #dR_investigation = (events.Jet.jetId >= 6)
    # wjj_pt = ak.fill_none(ak.firsts(getattr(events.w_had_jets, "pt", None)), np.nan)
    # wjj_pt_cut = np.where(np.isnan(wjj_pt),  False, wjj_pt  < params["wjj_pt"])

    cut_mt_w = (events.mt_w_leptonic < 185.0)

    # veto b optional

    b_veto = (events.nBJetGood == 0) if params.get("apply_b_veto", True) else True
    #b_veto_gen = (ak.num(good_bjet) == 0)
    # if params.get("require_lep_central", False):
    lep = ak.firsts(events.LeptonGood)
    #     j1  = ak.firsts(getattr(events.vbsjets, "jet1", None))
    #     j2  = ak.firsts(getattr(events.vbsjets, "jet2", None))

    #     j1_eta = ak.fill_none(getattr(j1, "eta", None), np.nan)
    #     j2_eta = ak.fill_none(getattr(j2, "eta", None), np.nan)
    #     lep_eta = ak.fill_none(getattr(lep, "eta", None), np.nan)

    #     j1_pt_min = (j1.pt > 50)
        #j2_pt_min = (j2.pt > 30)

    #     eta_min = np.minimum(j1_eta, j2_eta)
    #     eta_max = np.maximum(j1_eta, j2_eta)
    #     #lep_central = 
    #     lep_central = (np.isnan(lep_eta)) & (np.isnan(eta_min)) & (np.isnan(eta_max)) & (lep_eta > eta_min) & (lep_eta < eta_max) & (lep.pt > 35.0) & j1_pt_min
    # else:
    #     lep_central = True

    if "WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8" in events.metadata["dataset"]:
        ht_mask = (events.LHE.HT <= 70.)
    else:
        ht_mask = True
    mask = one_lep & met_cut & two_j & cut_mt_w & b_veto & ht_mask#&  loose_lep_veto #(lep.pt > 35.0) &
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


def in_whad_window_mu(events, params, **kwargs):
    muon_ch = (events.nElectronGood38 == 0) & (events.nMuonGood30 == 1)
    four_j  = (events.nJetGood30 >= 4)
    no_fat = (events.nFatJetCandidate180 == 0)
    loose_lep_veto = (events.nLeptonVeto < 2)
    wjj_pt = ak.fill_none(ak.firsts(getattr(events.w_had_jets, "pt", None)), np.nan)
    wjj_pt_cut = np.where(np.isnan(wjj_pt),  False, wjj_pt  < 200.)
    wmass = ak.fill_none(ak.firsts(getattr(events.w_had_jets, "mass", None)), np.nan)
    within = np.where(np.isnan(wmass), False, np.abs(wmass - 85) < params["mjj_w_window"])
    # lead_lep_dR_cut1 = (events.lead_wlep_wjet1_dR > 0.8)
    # lead_lep_dR_cut2 = (events.lead_wlep_wjet2_dR > 0.8)
    lep = ak.firsts(events.MuonGood30)
    j1  = ak.firsts(getattr(events.vbsjets, "jet1", None))
    j2  = ak.firsts(getattr(events.vbsjets, "jet2", None))

    j1_eta = ak.fill_none(getattr(j1, "eta", None), np.nan)
    j2_eta = ak.fill_none(getattr(j2, "eta", None), np.nan)
    lep_eta = ak.fill_none(getattr(lep, "eta", None), np.nan)

    j1_pt_min = (j1.pt > 50)
    j2_pt_min = (j2.pt > 30)

    

    eta_min = np.minimum(j1_eta, j2_eta)
    eta_max = np.maximum(j1_eta, j2_eta)
     
    lep_central = j1_pt_min #& (np.isnan(lep_eta)) & (np.isnan(eta_min)) & (np.isnan(eta_max)) & (lep_eta > eta_min) & (lep_eta < eta_max) 
    
    mjj_vbs   = ak.fill_none(ak.firsts(getattr(events.vbsjets, "mass", None)), np.nan)
    deta_vbs  = ak.fill_none(ak.firsts(getattr(events.vbsjets, "delta_eta", None)), np.nan)

    cut_mjj   = np.where(np.isnan(mjj_vbs),  False, mjj_vbs  > params["mjj_vbs"])
    cut_deta  = np.where(np.isnan(deta_vbs), False, deta_vbs > params["delta_eta_vbs"])

    mask = four_j & within & wjj_pt_cut & cut_mjj & cut_deta & lep_central & loose_lep_veto & no_fat & muon_ch & j2_pt_min 
    return ak.values_astype(mask, np.bool_)

whad_window_cut_mu = Cut(
    name="whad_window_mu",
    params={"mjj_w_window": 20.0,
            "mjj_vbs": 500.0,
            "delta_eta_vbs": 2.5},  
    function=in_whad_window_mu,
)

def in_whad_window_bveto_mu(events, params, **kwargs):
    muon_ch = (events.nElectronGood38 == 0) & (events.nMuonGood30 == 1)
    four_j  = (events.nJetGood30 >= 4)
    b_veto = (events.nBJetGood == 0) 
    no_fat = (events.nFatJetCandidate180 == 0)
    loose_lep_veto = (events.nLeptonVeto < 2)
    wjj_pt = ak.fill_none(ak.firsts(getattr(events.w_had_jets, "pt", None)), np.nan)
    wjj_pt_cut = np.where(np.isnan(wjj_pt),  False, wjj_pt  < 200.)
    wmass = ak.fill_none(ak.firsts(getattr(events.w_had_jets, "mass", None)), np.nan)
    within = np.where(np.isnan(wmass), False, np.abs(wmass - 85) < params["mjj_w_window"])
    # lead_lep_dR_cut1 = (events.lead_wlep_wjet1_dR > 0.8)
    # lead_lep_dR_cut2 = (events.lead_wlep_wjet2_dR > 0.8)
    lep = ak.firsts(events.MuonGood30)
    j1  = ak.firsts(getattr(events.vbsjets, "jet1", None))
    j2  = ak.firsts(getattr(events.vbsjets, "jet2", None))

    j1_eta = ak.fill_none(getattr(j1, "eta", None), np.nan)
    j2_eta = ak.fill_none(getattr(j2, "eta", None), np.nan)
    lep_eta = ak.fill_none(getattr(lep, "eta", None), np.nan)

    j1_pt_min = (j1.pt > 50)
    j2_pt_min = (j2.pt > 30)


    

    eta_min = np.minimum(j1_eta, j2_eta)
    eta_max = np.maximum(j1_eta, j2_eta)
     
    lep_central = j1_pt_min #& (np.isnan(lep_eta)) & (np.isnan(eta_min)) & (np.isnan(eta_max)) & (lep_eta > eta_min) & (lep_eta < eta_max) 
    
    mjj_vbs   = ak.fill_none(ak.firsts(getattr(events.vbsjets, "mass", None)), np.nan)
    deta_vbs  = ak.fill_none(ak.firsts(getattr(events.vbsjets, "delta_eta", None)), np.nan)

    cut_mjj   = np.where(np.isnan(mjj_vbs),  False, mjj_vbs  > params["mjj_vbs"])
    cut_deta  = np.where(np.isnan(deta_vbs), False, deta_vbs > params["delta_eta_vbs"])

    mask = four_j & b_veto & within & wjj_pt_cut & cut_mjj & cut_deta & lep_central & loose_lep_veto & no_fat & b_veto & muon_ch & j2_pt_min 
    return ak.values_astype(mask, np.bool_)

whad_window_cut_bveto_mu = Cut(
    name="whad_window_bveto_mu",
    params={"mjj_w_window": 20.0,
            "mjj_vbs": 500.0,
            "delta_eta_vbs": 2.5},  
    function=in_whad_window_bveto_mu,
)


def in_msd_window_fatjet_mu(events, params, **kwargs):
    muon_ch = (events.nElectronGood38 == 0) & (events.nMuonGood30 == 1)
    #yes_fat = (events.nFatJetCentral >= 1)
    b_veto = (events.nBJetGood == 0) & (events.nBJet_ak8 == 0)
    loose_lep_veto = (events.nLeptonVeto < 2)
    yes_fat = (events.nFatJetCandidate == 1)
    fj1_pt = ak.fill_none(ak.firsts(getattr(events.candidate_boost, "pt", None)), np.nan)
    fj1_msd = ak.fill_none(ak.firsts(getattr(events.candidate_boost, "msoftdrop", None)), np.nan)
    pt_cut = np.where(np.isnan(fj1_pt), False, fj1_pt > 200.)
    within = np.where(np.isnan(fj1_msd), False, np.abs(fj1_msd - 92.5) < params["msd_w_window"])
    #sel = (np.isnan(fj1_msd)) & (fj1_msd > wlo) & (fj1_msd < whi)
    #lead_lep_dR_cut = (events.lead_wlep_wfatjet1_dR > 0.8)
    # jet1_dR_cut = (events.vbs1_fj_dR > 0.8)
    # jet2_dR_cut = (events.vbs2_fj_dR > 0.8)

    lep = ak.firsts(events.MuonGood30)
    j1  = ak.firsts(getattr(events.vbsjets, "jet1", None))
    j2  = ak.firsts(getattr(events.vbsjets, "jet2", None))

    j1_eta = ak.fill_none(getattr(j1, "eta", None), np.nan)
    j2_eta = ak.fill_none(getattr(j2, "eta", None), np.nan)
    lep_eta = ak.fill_none(getattr(lep, "eta", None), np.nan)

    j1_pt_min = (j1.pt > 50)
    j2_pt_min = (j2.pt > 30)

    eta_min = np.minimum(j1_eta, j2_eta)
    eta_max = np.maximum(j1_eta, j2_eta)
     
    lep_central = j1_pt_min #& (np.isnan(lep_eta)) & (np.isnan(eta_min)) & (np.isnan(eta_max)) & (lep_eta > eta_min) & (lep_eta < eta_max) 
    
    

    mjj_vbs   = ak.fill_none(ak.firsts(getattr(events.vbsjets, "mass", None)), np.nan)
    deta_vbs  = ak.fill_none(ak.firsts(getattr(events.vbsjets, "delta_eta", None)), np.nan)

    cut_mjj   = np.where(np.isnan(mjj_vbs),  False, mjj_vbs  > params["mjj_vbs"])
    cut_deta  = np.where(np.isnan(deta_vbs), False, deta_vbs > params["delta_eta_vbs"])

    mask = yes_fat & within & pt_cut & cut_mjj & cut_deta & lep_central & loose_lep_veto & muon_ch & j2_pt_min & b_veto #& jet_dR_cut 
    return ak.values_astype(mask, np.bool_)


msd_window_cut_mu = Cut(
    name="msd_window_mu",
    params={"msd_w_window": 22.5,
            "mjj_vbs": 500.0,
            "delta_eta_vbs": 2.5}, 
    function=in_msd_window_fatjet_mu,
    )


############################################
##### MUON CHANNEL
###########################################

def in_whad_window_e(events, params, **kwargs):
    electron_ch = (events.nElectronGood38 == 1) & (events.nMuonGood30 == 0)
    four_j  = (events.nJetGood30 >= 4)
    no_fat = (events.nFatJetCandidate180 == 0)
    loose_lep_veto = (events.nLeptonVeto < 2)
    wjj_pt = ak.fill_none(ak.firsts(getattr(events.w_had_jets, "pt", None)), np.nan)
    wjj_pt_cut = np.where(np.isnan(wjj_pt),  False, wjj_pt  < 200.)
    wmass = ak.fill_none(ak.firsts(getattr(events.w_had_jets, "mass", None)), np.nan)
    within = np.where(np.isnan(wmass), False, np.abs(wmass - 85) < params["mjj_w_window"])
    # lead_lep_dR_cut1 = (events.lead_wlep_wjet1_dR > 0.8)
    # lead_lep_dR_cut2 = (events.lead_wlep_wjet2_dR > 0.8)
    lep = ak.firsts(events.ElectronGood35)
    j1  = ak.firsts(getattr(events.vbsjets, "jet1", None))
    j2  = ak.firsts(getattr(events.vbsjets, "jet2", None))

    j1_eta = ak.fill_none(getattr(j1, "eta", None), np.nan)
    j2_eta = ak.fill_none(getattr(j2, "eta", None), np.nan)
    lep_eta = ak.fill_none(getattr(lep, "eta", None), np.nan)

    j1_pt_min = (j1.pt > 50)
    j2_pt_min = (j2.pt > 30)


    

    eta_min = np.minimum(j1_eta, j2_eta)
    eta_max = np.maximum(j1_eta, j2_eta)
     
    lep_central = j1_pt_min #& (np.isnan(lep_eta)) & (np.isnan(eta_min)) & (np.isnan(eta_max)) & (lep_eta > eta_min) & (lep_eta < eta_max) 
    
    mjj_vbs   = ak.fill_none(ak.firsts(getattr(events.vbsjets, "mass", None)), np.nan)
    deta_vbs  = ak.fill_none(ak.firsts(getattr(events.vbsjets, "delta_eta", None)), np.nan)

    cut_mjj   = np.where(np.isnan(mjj_vbs),  False, mjj_vbs  > params["mjj_vbs"])
    cut_deta  = np.where(np.isnan(deta_vbs), False, deta_vbs > params["delta_eta_vbs"])

    mask = four_j & within & wjj_pt_cut & cut_mjj & cut_deta & lep_central & loose_lep_veto & no_fat & electron_ch & j2_pt_min 
    return ak.values_astype(mask, np.bool_)

whad_window_cut_e = Cut(
    name="whad_window_e",
    params={"mjj_w_window": 20.0,
            "mjj_vbs": 500.0,
            "delta_eta_vbs": 2.5},  
    function=in_whad_window_e,
)

def in_whad_window_bveto_e(events, params, **kwargs):
    electron_ch = (events.nElectronGood38 == 1) & (events.nMuonGood30 == 0)
    four_j  = (events.nJetGood30 >= 4)
    b_veto = (events.nBJetGood == 0) 
    no_fat = (events.nFatJetCandidate180 == 0)
    loose_lep_veto = (events.nLeptonVeto < 2)
    wjj_pt = ak.fill_none(ak.firsts(getattr(events.w_had_jets, "pt", None)), np.nan)
    wjj_pt_cut = np.where(np.isnan(wjj_pt),  False, wjj_pt  < 200.)
    wmass = ak.fill_none(ak.firsts(getattr(events.w_had_jets, "mass", None)), np.nan)
    within = np.where(np.isnan(wmass), False, np.abs(wmass - 85) < params["mjj_w_window"])
    # lead_lep_dR_cut1 = (events.lead_wlep_wjet1_dR > 0.8)
    # lead_lep_dR_cut2 = (events.lead_wlep_wjet2_dR > 0.8)
    lep = ak.firsts(events.ElectronGood35)
    j1  = ak.firsts(getattr(events.vbsjets, "jet1", None))
    j2  = ak.firsts(getattr(events.vbsjets, "jet2", None))

    j1_eta = ak.fill_none(getattr(j1, "eta", None), np.nan)
    j2_eta = ak.fill_none(getattr(j2, "eta", None), np.nan)
    lep_eta = ak.fill_none(getattr(lep, "eta", None), np.nan)

    j1_pt_min = (j1.pt > 50)
    j2_pt_min = (j2.pt > 30)


    
    eta_min = np.minimum(j1_eta, j2_eta)
    eta_max = np.maximum(j1_eta, j2_eta)
     
    lep_central = j1_pt_min #& (np.isnan(lep_eta)) & (np.isnan(eta_min)) & (np.isnan(eta_max)) & (lep_eta > eta_min) & (lep_eta < eta_max) 
    
    mjj_vbs   = ak.fill_none(ak.firsts(getattr(events.vbsjets, "mass", None)), np.nan)
    deta_vbs  = ak.fill_none(ak.firsts(getattr(events.vbsjets, "delta_eta", None)), np.nan)

    cut_mjj   = np.where(np.isnan(mjj_vbs),  False, mjj_vbs  > params["mjj_vbs"])
    cut_deta  = np.where(np.isnan(deta_vbs), False, deta_vbs > params["delta_eta_vbs"])

    mask = four_j & b_veto & within & wjj_pt_cut & cut_mjj & cut_deta & lep_central & loose_lep_veto & no_fat & b_veto & electron_ch & j2_pt_min 
    return ak.values_astype(mask, np.bool_)

whad_window_cut_bveto_e = Cut(
    name="whad_window_bveto_e",
    params={"mjj_w_window": 20.0,
            "mjj_vbs": 500.0,
            "delta_eta_vbs": 2.5},  
    function=in_whad_window_bveto_e,
)


def in_msd_window_fatjet_e(events, params, **kwargs):
    electron_ch = (events.nElectronGood38 == 1) & (events.nMuonGood30 == 0)
    #yes_fat = (events.nFatJetCentral >= 1)
    b_veto = (events.nBJetGood == 0) & (events.nBJet_ak8 == 0)

    loose_lep_veto = (events.nLeptonVeto < 2)
    yes_fat = (events.nFatJetCandidate == 1)
    fj1_pt = ak.fill_none(ak.firsts(getattr(events.candidate_boost, "pt", None)), np.nan)
    fj1_msd = ak.fill_none(ak.firsts(getattr(events.candidate_boost, "msoftdrop", None)), np.nan)
    pt_cut = np.where(np.isnan(fj1_pt), False, fj1_pt > 200.)
    within = np.where(np.isnan(fj1_msd), False, np.abs(fj1_msd - 92.5) < params["msd_w_window"])
    #sel = (np.isnan(fj1_msd)) & (fj1_msd > wlo) & (fj1_msd < whi)
    lead_lep_dR_cut = (events.lead_wlep_wfatjet1_dR > 0.8)
    # jet1_dR_cut = (events.vbs1_fj_dR > 0.8)
    # jet2_dR_cut = (events.vbs2_fj_dR > 0.8)

    lep = ak.firsts(events.ElectronGood35)
    j1  = ak.firsts(getattr(events.vbsjets, "jet1", None))
    j2  = ak.firsts(getattr(events.vbsjets, "jet2", None))

    j1_eta = ak.fill_none(getattr(j1, "eta", None), np.nan)
    j2_eta = ak.fill_none(getattr(j2, "eta", None), np.nan)
    lep_eta = ak.fill_none(getattr(lep, "eta", None), np.nan)

    j1_pt_min = (j1.pt > 50)
    j2_pt_min = (j2.pt > 30)

    eta_min = np.minimum(j1_eta, j2_eta)
    eta_max = np.maximum(j1_eta, j2_eta)
     
    lep_central = j1_pt_min #& (np.isnan(lep_eta)) & (np.isnan(eta_min)) & (np.isnan(eta_max)) & (lep_eta > eta_min) & (lep_eta < eta_max) 
    
    

    mjj_vbs   = ak.fill_none(ak.firsts(getattr(events.vbsjets, "mass", None)), np.nan)
    deta_vbs  = ak.fill_none(ak.firsts(getattr(events.vbsjets, "delta_eta", None)), np.nan)

    cut_mjj   = np.where(np.isnan(mjj_vbs),  False, mjj_vbs  > params["mjj_vbs"])
    cut_deta  = np.where(np.isnan(deta_vbs), False, deta_vbs > params["delta_eta_vbs"])

    mask = yes_fat & within & pt_cut & cut_mjj & cut_deta & lep_central & loose_lep_veto & electron_ch & j2_pt_min & b_veto
    return ak.values_astype(mask, np.bool_)


msd_window_cut_e = Cut(
    name="msd_window_e",
    params={"msd_w_window": 22.5,
            "mjj_vbs": 500.0,
            "delta_eta_vbs": 2.5},
    function=in_msd_window_fatjet_e,
    )


_SR_PARAMS = {"mjj_w_window": 20.0, "mjj_vbs": 500.0, "delta_eta_vbs": 2.5}
_W_CR_PARAMS = {**_SR_PARAMS, "mt_w_lo": 30.0}
_VR_PARAMS   = {**_SR_PARAMS, "mt_w_lo": 20.0, "mt_w_hi": 30.0}

############################################
##### W CONTROL REGION
##### SR selection + inverted hadronic W mass window + mT > 30
############################################

def _sr_skeleton_mu(events, params):
    """Shared resolved SR skeleton for mu channel (no mT, no window applied yet)."""
    muon_ch        = (events.nElectronGood38 == 0) & (events.nMuonGood30 == 1)
    four_j         = (events.nJetGood30 >= 4)
    b_veto         = (events.nBJetGood == 0)
    no_fat         = (events.nFatJetCandidate180 == 0)
    loose_lep_veto = (events.nLeptonVeto < 2)
    wjj_pt  = ak.fill_none(ak.firsts(getattr(events.w_had_jets, "pt",   None)), np.nan)
    wjj_pt_cut = np.where(np.isnan(wjj_pt), False, wjj_pt < 200.)
    wmass   = ak.fill_none(ak.firsts(getattr(events.w_had_jets, "mass", None)), np.nan)
    within  = np.where(np.isnan(wmass), False, np.abs(wmass - 85) < params["mjj_w_window"])
    j1 = ak.firsts(getattr(events.vbsjets, "jet1", None))
    j2 = ak.firsts(getattr(events.vbsjets, "jet2", None))
    j1_pt_min   = (j1.pt > 50)
    j2_pt_min   = (j2.pt > 30)
    lep_central = j1_pt_min
    mjj_vbs  = ak.fill_none(ak.firsts(getattr(events.vbsjets, "mass",      None)), np.nan)
    deta_vbs = ak.fill_none(ak.firsts(getattr(events.vbsjets, "delta_eta", None)), np.nan)
    cut_mjj  = np.where(np.isnan(mjj_vbs),  False, mjj_vbs  > params["mjj_vbs"])
    cut_deta = np.where(np.isnan(deta_vbs), False, deta_vbs > params["delta_eta_vbs"])
    if "WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8" in events.metadata["dataset"]:
        ht_mask = (events.LHE.HT <= 70.)
    else:
        ht_mask = True
    base = muon_ch & four_j & b_veto & no_fat & loose_lep_veto & wjj_pt_cut & cut_mjj & cut_deta & lep_central & j2_pt_min & ht_mask
    return base, within

def _sr_skeleton_e(events, params):
    """Shared resolved SR skeleton for e channel (no mT, no window applied yet)."""
    electron_ch    = (events.nElectronGood38 == 1) & (events.nMuonGood30 == 0)
    four_j         = (events.nJetGood30 >= 4)
    b_veto         = (events.nBJetGood == 0)
    no_fat         = (events.nFatJetCandidate180 == 0)
    loose_lep_veto = (events.nLeptonVeto < 2)
    wjj_pt  = ak.fill_none(ak.firsts(getattr(events.w_had_jets, "pt",   None)), np.nan)
    wjj_pt_cut = np.where(np.isnan(wjj_pt), False, wjj_pt < 200.)
    wmass   = ak.fill_none(ak.firsts(getattr(events.w_had_jets, "mass", None)), np.nan)
    within  = np.where(np.isnan(wmass), False, np.abs(wmass - 85) < params["mjj_w_window"])
    j1 = ak.firsts(getattr(events.vbsjets, "jet1", None))
    j2 = ak.firsts(getattr(events.vbsjets, "jet2", None))
    j1_pt_min   = (j1.pt > 50)
    j2_pt_min   = (j2.pt > 30)
    lep_central = j1_pt_min
    mjj_vbs  = ak.fill_none(ak.firsts(getattr(events.vbsjets, "mass",      None)), np.nan)
    deta_vbs = ak.fill_none(ak.firsts(getattr(events.vbsjets, "delta_eta", None)), np.nan)
    cut_mjj  = np.where(np.isnan(mjj_vbs),  False, mjj_vbs  > params["mjj_vbs"])
    cut_deta = np.where(np.isnan(deta_vbs), False, deta_vbs > params["delta_eta_vbs"])
    if "WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8" in events.metadata["dataset"]:
        ht_mask = (events.LHE.HT <= 70.)
    else:
        ht_mask = True
    base = electron_ch & four_j & b_veto & no_fat & loose_lep_veto & wjj_pt_cut & cut_mjj & cut_deta & lep_central & j2_pt_min & ht_mask
    return base, within

def in_w_cr_mu(events, params, **kwargs):
    base, within = _sr_skeleton_mu(events, params)
    cut_mt = (events.mt_w_leptonic > params["mt_w_lo"])
    mask = base & (~within) & cut_mt
    return ak.values_astype(mask, np.bool_)

def in_w_cr_e(events, params, **kwargs):
    base, within = _sr_skeleton_e(events, params)
    cut_mt = (events.mt_w_leptonic > params["mt_w_lo"])
    mask = base & (~within) & cut_mt
    return ak.values_astype(mask, np.bool_)

w_cr_mu = Cut(name="w_cr_mu", params=_W_CR_PARAMS, function=in_w_cr_mu)
w_cr_e  = Cut(name="w_cr_e",  params=_W_CR_PARAMS, function=in_w_cr_e)

############################################
##### VALIDATION REGION
##### SR selection exactly (no W mass window), mT in [20, 30]
############################################

def in_vr_mu(events, params, **kwargs):
    base, _ = _sr_skeleton_mu(events, params)
    cut_mt = (
        (events.mt_w_leptonic > params["mt_w_lo"]) &
        (events.mt_w_leptonic < params["mt_w_hi"])
    )
    mask = base & cut_mt
    return ak.values_astype(mask, np.bool_)

def in_vr_e(events, params, **kwargs):
    base, _ = _sr_skeleton_e(events, params)
    cut_mt = (
        (events.mt_w_leptonic > params["mt_w_lo"]) &
        (events.mt_w_leptonic < params["mt_w_hi"])
    )
    mask = base & cut_mt
    return ak.values_astype(mask, np.bool_)

vr_mu = Cut(name="vr_mu", params=_VR_PARAMS, function=in_vr_mu)
vr_e  = Cut(name="vr_e",  params=_VR_PARAMS, function=in_vr_e)

_BOOSTED_SR_PARAMS   = {"msd_w_window": 22.5, "mjj_vbs": 500.0, "delta_eta_vbs": 2.5}
_BOOSTED_W_CR_PARAMS = {**_BOOSTED_SR_PARAMS, "mt_w_lo": 30.0}
_BOOSTED_VR_PARAMS   = {**_BOOSTED_SR_PARAMS, "mt_w_lo": 20.0, "mt_w_hi": 30.0}

############################################
##### W CONTROL REGION — BOOSTED
##### Boosted SR structure + inverted fat jet mSD window + mT > 30
############################################

def _boosted_skeleton_mu(events, params):
    """Shared boosted SR skeleton for mu channel (no mT, no mSD window applied yet)."""
    muon_ch        = (events.nElectronGood38 == 0) & (events.nMuonGood30 == 1)
    yes_fat        = (events.nFatJetCandidate == 1)
    b_veto         = (events.nBJetGood == 0) & (events.nBJet_ak8 == 0)
    loose_lep_veto = (events.nLeptonVeto < 2)
    fj1_pt  = ak.fill_none(ak.firsts(getattr(events.candidate_boost, "pt",        None)), np.nan)
    fj1_msd = ak.fill_none(ak.firsts(getattr(events.candidate_boost, "msoftdrop", None)), np.nan)
    pt_cut  = np.where(np.isnan(fj1_pt),  False, fj1_pt  > 200.)
    within  = np.where(np.isnan(fj1_msd), False, np.abs(fj1_msd - 92.5) < params["msd_w_window"])
    j1 = ak.firsts(getattr(events.vbsjets, "jet1", None))
    j2 = ak.firsts(getattr(events.vbsjets, "jet2", None))
    j1_pt_min   = (j1.pt > 50)
    j2_pt_min   = (j2.pt > 30)
    lep_central = j1_pt_min
    mjj_vbs  = ak.fill_none(ak.firsts(getattr(events.vbsjets, "mass",      None)), np.nan)
    deta_vbs = ak.fill_none(ak.firsts(getattr(events.vbsjets, "delta_eta", None)), np.nan)
    cut_mjj  = np.where(np.isnan(mjj_vbs),  False, mjj_vbs  > params["mjj_vbs"])
    cut_deta = np.where(np.isnan(deta_vbs), False, deta_vbs > params["delta_eta_vbs"])
    if "WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8" in events.metadata["dataset"]:
        ht_mask = (events.LHE.HT <= 70.)
    else:
        ht_mask = True
    base = muon_ch & yes_fat & b_veto & loose_lep_veto & pt_cut & cut_mjj & cut_deta & lep_central & j2_pt_min & ht_mask
    return base, within

def _boosted_skeleton_e(events, params):
    """Shared boosted SR skeleton for e channel (no mT, no mSD window applied yet)."""
    electron_ch    = (events.nElectronGood38 == 1) & (events.nMuonGood30 == 0)
    yes_fat        = (events.nFatJetCandidate == 1)
    b_veto         = (events.nBJetGood == 0) & (events.nBJet_ak8 == 0)
    loose_lep_veto = (events.nLeptonVeto < 2)
    fj1_pt  = ak.fill_none(ak.firsts(getattr(events.candidate_boost, "pt",        None)), np.nan)
    fj1_msd = ak.fill_none(ak.firsts(getattr(events.candidate_boost, "msoftdrop", None)), np.nan)
    pt_cut  = np.where(np.isnan(fj1_pt),  False, fj1_pt  > 200.)
    within  = np.where(np.isnan(fj1_msd), False, np.abs(fj1_msd - 92.5) < params["msd_w_window"])
    j1 = ak.firsts(getattr(events.vbsjets, "jet1", None))
    j2 = ak.firsts(getattr(events.vbsjets, "jet2", None))
    j1_pt_min   = (j1.pt > 50)
    j2_pt_min   = (j2.pt > 30)
    lep_central = j1_pt_min
    mjj_vbs  = ak.fill_none(ak.firsts(getattr(events.vbsjets, "mass",      None)), np.nan)
    deta_vbs = ak.fill_none(ak.firsts(getattr(events.vbsjets, "delta_eta", None)), np.nan)
    cut_mjj  = np.where(np.isnan(mjj_vbs),  False, mjj_vbs  > params["mjj_vbs"])
    cut_deta = np.where(np.isnan(deta_vbs), False, deta_vbs > params["delta_eta_vbs"])
    if "WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8" in events.metadata["dataset"]:
        ht_mask = (events.LHE.HT <= 70.)
    else:
        ht_mask = True
    base = electron_ch & yes_fat & b_veto & loose_lep_veto & pt_cut & cut_mjj & cut_deta & lep_central & j2_pt_min & ht_mask
    return base, within

def in_w_cr_boosted_mu(events, params, **kwargs):
    base, within = _boosted_skeleton_mu(events, params)
    cut_mt = (events.mt_w_leptonic > params["mt_w_lo"])
    return ak.values_astype(base & (~within) & cut_mt, np.bool_)

def in_w_cr_boosted_e(events, params, **kwargs):
    base, within = _boosted_skeleton_e(events, params)
    cut_mt = (events.mt_w_leptonic > params["mt_w_lo"])
    return ak.values_astype(base & (~within) & cut_mt, np.bool_)

w_cr_boosted_mu = Cut(name="w_cr_boosted_mu", params=_BOOSTED_W_CR_PARAMS, function=in_w_cr_boosted_mu)
w_cr_boosted_e  = Cut(name="w_cr_boosted_e",  params=_BOOSTED_W_CR_PARAMS, function=in_w_cr_boosted_e)

############################################
##### VALIDATION REGION — BOOSTED
##### Boosted SR structure (no mSD window), mT in [20, 30]
############################################

def in_vr_boosted_mu(events, params, **kwargs):
    base, _ = _boosted_skeleton_mu(events, params)
    cut_mt = (
        (events.mt_w_leptonic > params["mt_w_lo"]) &
        (events.mt_w_leptonic < params["mt_w_hi"])
    )
    return ak.values_astype(base & cut_mt, np.bool_)

def in_vr_boosted_e(events, params, **kwargs):
    base, _ = _boosted_skeleton_e(events, params)
    cut_mt = (
        (events.mt_w_leptonic > params["mt_w_lo"]) &
        (events.mt_w_leptonic < params["mt_w_hi"])
    )
    return ak.values_astype(base & cut_mt, np.bool_)

vr_boosted_mu = Cut(name="vr_boosted_mu", params=_BOOSTED_VR_PARAMS, function=in_vr_boosted_mu)
vr_boosted_e  = Cut(name="vr_boosted_e",  params=_BOOSTED_VR_PARAMS, function=in_vr_boosted_e)


############################################
##### QCD CONTROL REGION
##### Resolved SR selection exactly, but:
#####   - loose lepton (nMuonLoose/nElectronLoose) instead of tight
#####   - mT < 20 instead of mT < 185
##### Plus two variations:
#####   _3j  : nJetGood30 >= 3 instead of >= 4 (resolved, no boosted)
#####   _sideband: inverted hadronic W mass window (~within)
############################################

def _qcd_cr_common(events, params, lepton_mask, invert_window=False, njet_min=4):
    """Shared logic for all QCD CR variants."""
    if "WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8" in events.metadata["dataset"]:
        ht_mask = (events.LHE.HT <= 70.)
    else:
        ht_mask = True

    loose_lep_veto = (events.nLeptonVeto < 2)
    n_j     = (events.nJetGood30 >= njet_min)
    no_fat  = (events.nFatJetCandidate180 == 0)
    met_cut = (events.DeepMETResolutionTune.pt > params["met_pt"])
    b_veto  = (events.nBJetGood == 0)
    cut_mt_w = (events.mt_w_leptonic_loose < params["mt_w"])

    wjj_pt  = ak.fill_none(ak.firsts(getattr(events.w_had_jets, "pt",   None)), np.nan)
    wjj_pt_cut = np.where(np.isnan(wjj_pt), False, wjj_pt < 200.)
    wmass   = ak.fill_none(ak.firsts(getattr(events.w_had_jets, "mass", None)), np.nan)
    within  = np.where(np.isnan(wmass), False, np.abs(wmass - 85) < params["mjj_w_window"])
    w_window = (~within) if invert_window else True

    j1 = ak.firsts(getattr(events.vbsjets, "jet1", None))
    j2 = ak.firsts(getattr(events.vbsjets, "jet2", None))
    j1_pt_min = (j1.pt > 50)
    j2_pt_min = (j2.pt > 30)
    lep_central = j1_pt_min

    mjj_vbs  = ak.fill_none(ak.firsts(getattr(events.vbsjets, "mass",      None)), np.nan)
    deta_vbs = ak.fill_none(ak.firsts(getattr(events.vbsjets, "delta_eta", None)), np.nan)
    cut_mjj  = np.where(np.isnan(mjj_vbs),  False, mjj_vbs  > params["mjj_vbs"])
    cut_deta = np.where(np.isnan(deta_vbs), False, deta_vbs > params["delta_eta_vbs"])

    mask = (lepton_mask & loose_lep_veto & n_j & no_fat & met_cut & b_veto
            & w_window & wjj_pt_cut & cut_mjj & cut_deta & lep_central & j2_pt_min
            & cut_mt_w & ht_mask)
    return ak.values_astype(mask, np.bool_)

_QCD_CR_PARAMS = {
    "met_pt": 30.0,
    "mjj_vbs": 500.0,
    "delta_eta_vbs": 2.5,
    "mjj_w_window": 20.0,
    "mt_w": 20.0,
}

# ---------- Main CR (4j, W mass window) ----------

def select_QCD_CR_mu(events, params, **kwargs):
    lep = (events.nMuonLoose >= 1) & (events.nElectronLoose == 0)
    return _qcd_cr_common(events, params, lep)

def select_QCD_CR_e(events, params, **kwargs):
    lep = (events.nElectronLoose >= 1) & (events.nMuonLoose == 0)
    return _qcd_cr_common(events, params, lep)

qcd_enriched_mu = Cut(name="qcd_enriched_mu", params=_QCD_CR_PARAMS, function=select_QCD_CR_mu)
qcd_enriched_e  = Cut(name="qcd_enriched_e",  params=_QCD_CR_PARAMS, function=select_QCD_CR_e)

# ---------- Variation 1: 3-jet (nJetGood30 >= 3, resolved) ----------

def select_QCD_CR_3j_mu(events, params, **kwargs):
    lep = (events.nMuonLoose >= 1) & (events.nElectronLoose == 0)
    return _qcd_cr_common(events, params, lep, njet_min=3)

def select_QCD_CR_3j_e(events, params, **kwargs):
    lep = (events.nElectronLoose >= 1) & (events.nMuonLoose == 0)
    return _qcd_cr_common(events, params, lep, njet_min=3)

qcd_enriched_3j_mu = Cut(name="qcd_enriched_3j_mu", params=_QCD_CR_PARAMS, function=select_QCD_CR_3j_mu)
qcd_enriched_3j_e  = Cut(name="qcd_enriched_3j_e",  params=_QCD_CR_PARAMS, function=select_QCD_CR_3j_e)

# ---------- Variation 2: inverted W mass window (W sideband) ----------

def select_QCD_CR_sideband_mu(events, params, **kwargs):
    lep = (events.nMuonLoose >= 1) & (events.nElectronLoose == 0)
    return _qcd_cr_common(events, params, lep, invert_window=True)

def select_QCD_CR_sideband_e(events, params, **kwargs):
    lep = (events.nElectronLoose >= 1) & (events.nMuonLoose == 0)
    return _qcd_cr_common(events, params, lep, invert_window=True)

qcd_enriched_sideband_mu = Cut(name="qcd_enriched_sideband_mu", params=_QCD_CR_PARAMS, function=select_QCD_CR_sideband_mu)
qcd_enriched_sideband_e  = Cut(name="qcd_enriched_sideband_e",  params=_QCD_CR_PARAMS, function=select_QCD_CR_sideband_e)


############################################
##### QCD CONTROL REGION — RECOIL JET METHOD
##### Parametric function mirroring the original select_QCD_CR format.
##### nJetGood >= 3, nJetGood_recoil >= 1, lead recoil jet pT = 30/35/40.
############################################

def select_QCD_CR_recoil(events, params, **kwargs):
    if "WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8" in events.metadata["dataset"]:
        ht_mask = (events.LHE.HT <= 70.)
    else:
        ht_mask = True

    if params["lepton"] == "mu":
        one_lep = (events.nMuonLoose == 1) & (events.nElectronLoose == 0)
    else:
        one_lep = (events.nElectronLoose == 1) & (events.nMuonLoose == 0)

    veto         = (events.nLeptonVeto < 2)
    recoil_jet   = (events.nJetGood_recoil >= 1)
    recoil_jet_pt = (events.LeadJetGood_recoil.pt >= params["recoil_jet_pt"])
    cut_njet     = (events.nJetGood >= params["njet"])
    met_cut      = (events.DeepMETResolutionTune.pt > params["met_pt"])
    cut_mt_w     = (events.mt_w_leptonic_loose < 20.0)
    bveto        = (events.nBJetGood == 0)

    mask = one_lep & met_cut & recoil_jet & recoil_jet_pt & cut_njet & cut_mt_w & bveto & veto & ht_mask
    return ak.values_astype(mask, np.bool_)

def _recoil_params(lepton, recoil_pt):
    return {"lepton": lepton, "njet": 3, "met_pt": 30.0, "recoil_jet_pt": recoil_pt}

qcd_enriched_recoil_30_mu = Cut(name="qcd_enriched_recoil_30_mu", params=_recoil_params("mu", 30), function=select_QCD_CR_recoil)
qcd_enriched_recoil_35_mu = Cut(name="qcd_enriched_recoil_35_mu", params=_recoil_params("mu", 35), function=select_QCD_CR_recoil)
qcd_enriched_recoil_40_mu = Cut(name="qcd_enriched_recoil_40_mu", params=_recoil_params("mu", 40), function=select_QCD_CR_recoil)
qcd_enriched_recoil_30_e  = Cut(name="qcd_enriched_recoil_30_e",  params=_recoil_params("e",  30), function=select_QCD_CR_recoil)
qcd_enriched_recoil_35_e  = Cut(name="qcd_enriched_recoil_35_e",  params=_recoil_params("e",  35), function=select_QCD_CR_recoil)
qcd_enriched_recoil_40_e  = Cut(name="qcd_enriched_recoil_40_e",  params=_recoil_params("e",  40), function=select_QCD_CR_recoil)


############################################
##### VALIDATION REGION — VARIATIONS
#####
##### All keep: pTmiss (presel), mT in [20, 30], b-veto, ht_mask.
##### vr_mu/e and vr_boosted_mu/e kept above as reference.
#####
##### vr_qcd_enriched    : recoil jet method (mirrors QCD CR recoil), mT [20,30]
##### vr_no_fwd          : SR VR but VBS forward-jet cuts dropped
##### vr_loose_njet      : SR VR but nJetGood30 >= 3, fat jets allowed
##### vr_no_fwd_loose_njet: both relaxations combined
############################################

def _ht_mask(events):
    if "WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8" in events.metadata["dataset"]:
        return (events.LHE.HT <= 70.)
    return True

# ---------- 1. QCD enriched VR — recoil jet method, mT [20, 30] ----------

def select_VR_recoil(events, params, **kwargs):
    if params["lepton"] == "mu":
        one_lep = (events.nMuonGood == 1) & (events.nElectronGood == 0)
    else:
        one_lep = (events.nElectronGood == 1) & (events.nMuonGood == 0)
    veto          = (events.nLeptonVeto < 2)
    recoil_jet    = (events.nJetGood_recoil >= 1)
    recoil_jet_pt = (events.LeadJetGood_recoil.pt >= params["recoil_jet_pt"])
    cut_njet      = (events.nJetGood >= params["njet"])
    met_cut       = (events.DeepMETResolutionTune.pt > params["met_pt"])
    bveto         = (events.nBJetGood == 0)
    cut_mt = (
        (events.mt_w_leptonic > params["mt_w_lo"]) &
        (events.mt_w_leptonic < params["mt_w_hi"])
    )
    mask = one_lep & met_cut & recoil_jet & recoil_jet_pt & cut_njet & cut_mt & bveto & veto & _ht_mask(events)
    return ak.values_astype(mask, np.bool_)

def _recoil_vr_params(lepton):
    return {"lepton": lepton, "njet": 3, "met_pt": 30.0, "recoil_jet_pt": 35,
            "mt_w_lo": 20.0, "mt_w_hi": 30.0}

vr_qcd_enriched_mu = Cut(name="vr_qcd_enriched_mu", params=_recoil_vr_params("mu"), function=select_VR_recoil)
vr_qcd_enriched_e  = Cut(name="vr_qcd_enriched_e",  params=_recoil_vr_params("e"),  function=select_VR_recoil)

# ---------- 1b. Inclusive recoil — mT [0, 30], superset of QCD CR + VR recoil ----------

def select_recoil_inclusive(events, params, **kwargs):
    if "WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8" in events.metadata["dataset"]:
        ht_mask = (events.LHE.HT <= 70.)
    else:
        ht_mask = True

    if params["lepton"] == "mu":
        one_lep = (events.nMuonGood == 1) & (events.nElectronGood == 0)
    else:
        one_lep = (events.nElectronGood == 1) & (events.nMuonGood == 0)

    veto          = (events.nLeptonVeto < 2)
    recoil_jet    = (events.nJetGood_recoil >= 1)
    recoil_jet_pt = (events.LeadJetGood_recoil.pt >= params["recoil_jet_pt"])
    cut_njet      = (events.nJetGood >= params["njet"])
    met_cut       = (events.DeepMETResolutionTune.pt > params["met_pt"])
    cut_mt        = (events.mt_w_leptonic < params["mt_w_hi"])
    bveto         = (events.nBJetGood == 0)

    mask = one_lep & met_cut & recoil_jet & recoil_jet_pt & cut_njet & cut_mt & bveto & veto & ht_mask
    return ak.values_astype(mask, np.bool_)

def _recoil_inclusive_params(lepton, mt_hi):
    return {"lepton": lepton, "njet": 3, "met_pt": 30.0, "recoil_jet_pt": 35, "mt_w_hi": mt_hi}

recoil_inclusive_mu = Cut(name="recoil_inclusive_mu", params=_recoil_inclusive_params("mu",30.0), function=select_recoil_inclusive)
recoil_inclusive_e  = Cut(name="recoil_inclusive_e",  params=_recoil_inclusive_params("e",30.0),  function=select_recoil_inclusive)

recoil_fullinclusive_mu = Cut(name="recoil_fullinclusive_mu", params=_recoil_inclusive_params("mu",185.0), function=select_recoil_inclusive)
recoil_fullinclusive_e  = Cut(name="recoil_fullinclusive_e",  params=_recoil_inclusive_params("e",185.0),  function=select_recoil_inclusive)


# ---------- 2. VR no forward jets — drop VBS mjj/deta/j1-j2pT ----------

def _vr_no_fwd(events, lepton_ch):
    loose_lep_veto = (events.nLeptonVeto < 2)
    n_j    = (events.nJetGood30 >= 4)
    no_fat = (events.nFatJetCandidate180 == 0)
    b_veto = (events.nBJetGood == 0)
    wjj_pt = ak.fill_none(ak.firsts(getattr(events.w_had_jets, "pt", None)), np.nan)
    wjj_pt_cut = np.where(np.isnan(wjj_pt), False, wjj_pt < 200.)
    cut_mt = (events.mt_w_leptonic > 20.0) & (events.mt_w_leptonic < 30.0)
    mask = lepton_ch & loose_lep_veto & n_j & no_fat & b_veto & wjj_pt_cut & cut_mt & _ht_mask(events)
    return ak.values_astype(mask, np.bool_)

def in_vr_no_fwd_mu(events, params, **kwargs):
    return _vr_no_fwd(events, (events.nElectronGood38 == 0) & (events.nMuonGood30 == 1))

def in_vr_no_fwd_e(events, params, **kwargs):
    return _vr_no_fwd(events, (events.nElectronGood38 == 1) & (events.nMuonGood30 == 0))

vr_no_fwd_mu = Cut(name="vr_no_fwd_mu", params={}, function=in_vr_no_fwd_mu)
vr_no_fwd_e  = Cut(name="vr_no_fwd_e",  params={}, function=in_vr_no_fwd_e)

# ---------- 3. VR loose nJet — nJetGood30 >= 3, fat jets allowed, VBS kept ----------

_VR_FWD_PARAMS = {"mjj_vbs": 500.0, "delta_eta_vbs": 2.5}

def _vr_loose_njet(events, lepton_ch, params, njet_min=3):
    loose_lep_veto = (events.nLeptonVeto < 2)
    n_j    = (events.nJetGood30 >= njet_min)
    b_veto = (events.nBJetGood == 0)
    wjj_pt = ak.fill_none(ak.firsts(getattr(events.w_had_jets, "pt", None)), np.nan)
    wjj_pt_cut = np.where(np.isnan(wjj_pt), False, wjj_pt < 200.)
    j1 = ak.firsts(getattr(events.vbsjets, "jet1", None))
    j2 = ak.firsts(getattr(events.vbsjets, "jet2", None))
    j1_pt_min   = (j1.pt > 50)
    j2_pt_min   = (j2.pt > 30)
    lep_central = j1_pt_min
    mjj_vbs  = ak.fill_none(ak.firsts(getattr(events.vbsjets, "mass",      None)), np.nan)
    deta_vbs = ak.fill_none(ak.firsts(getattr(events.vbsjets, "delta_eta", None)), np.nan)
    cut_mjj  = np.where(np.isnan(mjj_vbs),  False, mjj_vbs  > params["mjj_vbs"])
    cut_deta = np.where(np.isnan(deta_vbs), False, deta_vbs > params["delta_eta_vbs"])
    cut_mt = (events.mt_w_leptonic > 20.0) & (events.mt_w_leptonic < 30.0)
    # no nFatJetCandidate180 veto — fat jets allowed
    mask = (lepton_ch & loose_lep_veto & n_j & b_veto & wjj_pt_cut
            & cut_mjj & cut_deta & lep_central & j2_pt_min & cut_mt & _ht_mask(events))
    return ak.values_astype(mask, np.bool_)

def in_vr_loose_njet_mu(events, params, **kwargs):
    return _vr_loose_njet(events, (events.nElectronGood38 == 0) & (events.nMuonGood30 == 1), params)

def in_vr_loose_njet_e(events, params, **kwargs):
    return _vr_loose_njet(events, (events.nElectronGood38 == 1) & (events.nMuonGood30 == 0), params)

vr_loose_njet_mu = Cut(name="vr_loose_njet_mu", params=_VR_FWD_PARAMS, function=in_vr_loose_njet_mu)
vr_loose_njet_e  = Cut(name="vr_loose_njet_e",  params=_VR_FWD_PARAMS, function=in_vr_loose_njet_e)

# ---------- 4. VR no forward jets + loose nJet — both relaxations ----------

def _vr_no_fwd_loose_njet(events, lepton_ch, njet_min=3):
    loose_lep_veto = (events.nLeptonVeto < 2)
    n_j    = (events.nJetGood30 >= njet_min)
    b_veto = (events.nBJetGood == 0)
    wjj_pt = ak.fill_none(ak.firsts(getattr(events.w_had_jets, "pt", None)), np.nan)
    wjj_pt_cut = np.where(np.isnan(wjj_pt), False, wjj_pt < 200.)
    cut_mt = (events.mt_w_leptonic > 20.0) & (events.mt_w_leptonic < 30.0)
    # no fat jet veto, no VBS forward cuts
    mask = lepton_ch & loose_lep_veto & n_j & b_veto & wjj_pt_cut & cut_mt & _ht_mask(events)
    return ak.values_astype(mask, np.bool_)

def in_vr_no_fwd_loose_njet_mu(events, params, **kwargs):
    return _vr_no_fwd_loose_njet(events, (events.nElectronGood38 == 0) & (events.nMuonGood30 == 1))

def in_vr_no_fwd_loose_njet_e(events, params, **kwargs):
    return _vr_no_fwd_loose_njet(events, (events.nElectronGood38 == 1) & (events.nMuonGood30 == 0))

vr_no_fwd_loose_njet_mu = Cut(name="vr_no_fwd_loose_njet_mu", params={}, function=in_vr_no_fwd_loose_njet_mu)
vr_no_fwd_loose_njet_e  = Cut(name="vr_no_fwd_loose_njet_e",  params={}, function=in_vr_no_fwd_loose_njet_e)
