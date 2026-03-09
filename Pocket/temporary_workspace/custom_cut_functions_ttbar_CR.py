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
    
    #pu_pv_corrections = (events.PV.npvsGood < 55) | (events.PV.npvsGood > 60) 
    one_lep = (events.nLeptonGood ==1)
    
    two_j  = (events.nJetGood30    >= 2)
    met_cut = (events.DeepMETResolutionTune.pt      >  params["met_pt"]) #USE DeepMETResolutionTune NOT PF MET

    #good_bjet =events.JetGood[(np.abs(events.JetGood.eta) < 2.5) & (np.abs(events.JetGood.partonFlavour) == 5)]
    
    #dR_investigation = (events.Jet.jetId >= 6)
    # wjj_pt = ak.fill_none(ak.firsts(getattr(events.w_had_jets, "pt", None)), np.nan)
    # wjj_pt_cut = np.where(np.isnan(wjj_pt),  False, wjj_pt  < params["wjj_pt"])

    cut_mt_w = (events.mt_w_leptonic < 185.0)

    # veto b optional

    #b_veto = (events.nBJetGood == 0) if params.get("apply_b_veto", True) else True
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
    #one_loosest_lep = (ak.num(events.Muon) == 1) #& (ak.num(events.Electron) == 0)

    # ht_mask = (events.LHE.HT <= 70.)
    # w_pt_stitch = (events.gen_w_pt_by_pdg < 100)
    mask = one_lep & met_cut & two_j & cut_mt_w #& ht_mask#& b_veto #& ht_mask#&  loose_lep_veto #(lep.pt > 35.0) &
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
    loose_lep_veto = (events.nLeptonLoose < 2)
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
    b_veto = (events.nBJetGood >= 1) 
    no_fat = (events.nFatJetCandidate180 == 0)
    loose_lep_veto = (events.nLeptonLoose < 2)
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
    b_veto = (events.nBJetGood >= 1) | (events.nBJet_ak8 > 0)
    loose_lep_veto = (events.nLeptonLoose < 2)
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

    W_vs_QCD_pNet_discrim = ak.fill_none(ak.firsts(getattr(events.candidate_boost, "particleNet_WvsQCD", None)), np.nan) 
    isNotQCD = np.where(np.isnan(W_vs_QCD_pNet_discrim), False, W_vs_QCD_pNet_discrim > 0.709)

    mask = yes_fat & within & pt_cut & cut_mjj & cut_deta & lep_central & loose_lep_veto & muon_ch & j2_pt_min & b_veto & isNotQCD#& jet_dR_cut 
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
    loose_lep_veto = (events.nLeptonLoose < 2)
    wjj_pt = ak.fill_none(ak.firsts(getattr(events.w_had_jets, "pt", None)), np.nan)
    wjj_pt_cut = np.where(np.isnan(wjj_pt),  False, wjj_pt  < 200.)
    wmass = ak.fill_none(ak.firsts(getattr(events.w_had_jets, "mass", None)), np.nan)
    within = np.where(np.isnan(wmass), False, np.abs(wmass - 85) < params["mjj_w_window"])
    # lead_lep_dR_cut1 = (events.lead_wlep_wjet1_dR > 0.8)
    # lead_lep_dR_cut2 = (events.lead_wlep_wjet2_dR > 0.8)
    lep = ak.firsts(events.ElectronGood38)
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
    b_veto = (events.nBJetGood >= 1) 
    no_fat = (events.nFatJetCandidate180 == 0)
    loose_lep_veto = (events.nLeptonLoose < 2)
    wjj_pt = ak.fill_none(ak.firsts(getattr(events.w_had_jets, "pt", None)), np.nan)
    wjj_pt_cut = np.where(np.isnan(wjj_pt),  False, wjj_pt  < 200.)
    wmass = ak.fill_none(ak.firsts(getattr(events.w_had_jets, "mass", None)), np.nan)
    within = np.where(np.isnan(wmass), False, np.abs(wmass - 85) < params["mjj_w_window"])
    # lead_lep_dR_cut1 = (events.lead_wlep_wjet1_dR > 0.8)
    # lead_lep_dR_cut2 = (events.lead_wlep_wjet2_dR > 0.8)
    lep = ak.firsts(events.ElectronGood38)
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
    b_veto = (events.nBJetGood >= 1) | (events.nBJet_ak8 > 0)

    loose_lep_veto = (events.nLeptonLoose < 2)
    yes_fat = (events.nFatJetCandidate == 1)
    fj1_pt = ak.fill_none(ak.firsts(getattr(events.candidate_boost, "pt", None)), np.nan)
    fj1_msd = ak.fill_none(ak.firsts(getattr(events.candidate_boost, "msoftdrop", None)), np.nan)
    pt_cut = np.where(np.isnan(fj1_pt), False, fj1_pt > 200.)
    within = np.where(np.isnan(fj1_msd), False, np.abs(fj1_msd - 92.5) < params["msd_w_window"])
    #sel = (np.isnan(fj1_msd)) & (fj1_msd > wlo) & (fj1_msd < whi)
    lead_lep_dR_cut = (events.lead_wlep_wfatjet1_dR > 0.8)
    # jet1_dR_cut = (events.vbs1_fj_dR > 0.8)
    # jet2_dR_cut = (events.vbs2_fj_dR > 0.8)

    lep = ak.firsts(events.ElectronGood38)
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

    W_vs_QCD_pNet_discrim = ak.fill_none(ak.firsts(getattr(events.candidate_boost, "particleNet_WvsQCD", None)), np.nan) 
    isNotQCD = np.where(np.isnan(W_vs_QCD_pNet_discrim), False, W_vs_QCD_pNet_discrim > 0.709)


    mask = yes_fat & within & pt_cut & cut_mjj & cut_deta & lep_central & loose_lep_veto & electron_ch & j2_pt_min & b_veto & isNotQCD
    return ak.values_astype(mask, np.bool_)


msd_window_cut_e = Cut(
    name="msd_window_e",
    params={"msd_w_window": 22.5,
            "mjj_vbs": 500.0,
            "delta_eta_vbs": 2.5}, 
    function=in_msd_window_fatjet_e,
    )


def whad_window_no_loose_e(events, params, **kwargs):
    electron_ch = (events.nElectronGood38 >= 1) & (events.nMuonGood30 == 0)
    four_j  = (events.nJetGood30 >= 4)
    b_veto = (events.nBJetGood >= 1) 
    no_fat = (events.nFatJetCandidate180 == 0)
    loose_lep_veto = (events.nLeptonLoose < 2)
    wjj_pt = ak.fill_none(ak.firsts(getattr(events.w_had_jets, "pt", None)), np.nan)
    wjj_pt_cut = np.where(np.isnan(wjj_pt),  False, wjj_pt  < 200.)
    wmass = ak.fill_none(ak.firsts(getattr(events.w_had_jets, "mass", None)), np.nan)
    within = np.where(np.isnan(wmass), False, np.abs(wmass - 85) < params["mjj_w_window"])
    # lead_lep_dR_cut1 = (events.lead_wlep_wjet1_dR > 0.8)
    # lead_lep_dR_cut2 = (events.lead_wlep_wjet2_dR > 0.8)
    lep = ak.firsts(events.ElectronGood38)
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

    mask = four_j & within & wjj_pt_cut & cut_mjj & cut_deta & lep_central & electron_ch & j2_pt_min 
    return ak.values_astype(mask, np.bool_)

whad_window_cut_no_loose_e = Cut(
    name="whad_window_bveto_e",
    params={"mjj_w_window": 20.0,
            "mjj_vbs": 500.0,
            "delta_eta_vbs": 2.5},  
    function=whad_window_no_loose_e,
)

def whad_window_cut_no_jet4_e(events, params, **kwargs):
    electron_ch = (events.nElectronGood38 == 1) & (events.nMuonGood30 == 0)
    four_j  = (events.nJetGood30 >= 4)
    b_veto = (events.nBJetGood >= 1) 
    no_fat = (events.nFatJetCandidate180 == 0)
    loose_lep_veto = (events.nLeptonLoose < 2)
    wjj_pt = ak.fill_none(ak.firsts(getattr(events.w_had_jets, "pt", None)), np.nan)
    wjj_pt_cut = np.where(np.isnan(wjj_pt),  False, wjj_pt  < 200.)
    wmass = ak.fill_none(ak.firsts(getattr(events.w_had_jets, "mass", None)), np.nan)
    within = np.where(np.isnan(wmass), False, np.abs(wmass - 85) < params["mjj_w_window"])
    # lead_lep_dR_cut1 = (events.lead_wlep_wjet1_dR > 0.8)
    # lead_lep_dR_cut2 = (events.lead_wlep_wjet2_dR > 0.8)
    lep = ak.firsts(events.ElectronGood38)
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

    mask =   cut_mjj & cut_deta & lep_central & electron_ch & j2_pt_min 
    return ak.values_astype(mask, np.bool_)

whad_window_cut_no_jet4_e = Cut(
    name="whad_window_bveto_e",
    params={"mjj_w_window": 20.0,
            "mjj_vbs": 500.0,
            "delta_eta_vbs": 2.5},  
    function=whad_window_cut_no_jet4_e,
)

def whad_window_cut_no_fj0_e(events, params, **kwargs):
    electron_ch = (events.nElectronGood38 == 1) & (events.nMuonGood30 == 0)
    four_j  = (events.nJetGood30 >= 4)
    b_veto = (events.nBJetGood >= 1) 
    no_fat = (events.nFatJetCandidate180 == 0)
    loose_lep_veto = (events.nLeptonLoose < 2)
    wjj_pt = ak.fill_none(ak.firsts(getattr(events.w_had_jets, "pt", None)), np.nan)
    wjj_pt_cut = np.where(np.isnan(wjj_pt),  False, wjj_pt  < 200.)
    wmass = ak.fill_none(ak.firsts(getattr(events.w_had_jets, "mass", None)), np.nan)
    within = np.where(np.isnan(wmass), False, np.abs(wmass - 85) < params["mjj_w_window"])
    # lead_lep_dR_cut1 = (events.lead_wlep_wjet1_dR > 0.8)
    # lead_lep_dR_cut2 = (events.lead_wlep_wjet2_dR > 0.8)
    lep = ak.firsts(events.ElectronGood38)
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

    mask =   within & wjj_pt_cut & cut_mjj & cut_deta & lep_central & electron_ch & j2_pt_min & loose_lep_veto
    return ak.values_astype(mask, np.bool_)

whad_window_cut_no_fj0_e = Cut(
    name="whad_window_bveto_e",
    params={"mjj_w_window": 20.0,
            "mjj_vbs": 500.0,
            "delta_eta_vbs": 2.5},  
    function=whad_window_cut_no_fj0_e,
)

def Muon_good1(events, params, **kwargs):
    muon_ch = (events.nElectronGood38 == 0) & (events.nMuonGood30 == 1) & (events.DeepMETResolutionTune.pt > 30)

    return ak.values_astype(muon_ch, np.bool_)

Muon_good1 = Cut(name="Muon_good1", params={}, function=Muon_good1)

def Muon_good2(events, params, **kwargs):
    muon_ch = (events.nElectronGood38 == 0) & (events.nMuonGood30 == 2)
   
    return ak.values_astype(muon_ch, np.bool_)

Muon_good2 = Cut(name="Muon_good2", params={}, function=Muon_good2)

def Muon_good3(events, params, **kwargs):
    muon_ch = (events.nElectronGood38 == 0) & (events.nMuonGood30 == 1) #& (events.DeepMETResolutionTune.pt > 30)

    return ak.values_astype(muon_ch, np.bool_)

Muon_good3 = Cut(name="Muon_good3", params={}, function=Muon_good3)

