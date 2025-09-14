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
    return (ak.num(good_jet) + ak.num(good_fatjet)) >= 3

nJet_skim_cut = Cut(name="nJet_skim", params={}, function=nJet_skim)

def met_skim(events, params, **kwargs):
    met_cut = (events.PuppiMET.pt > 30)
    return met_cut

met_skim_cut = Cut(name="met_skim", params={}, function=met_skim)

# ---------- Preselection semileptonic VBS ----------
def select_vbs_semileptonic(events, params, **kwargs):
    
    #pu_pv_corrections = (events.PV.npvsGood < 55) | (events.PV.npvsGood > 60) 
    one_lep = (events.nMuonGood == 1) & (events.nElectronGood == 0)
    
    two_j  = (events.nJetGood    >= 2)
    met_cut = (events.PuppiMET.pt      >  params["met_pt"]) #USE PUPPIMET NOT PF MET

    #dR_investigation = (events.Jet.jetId >= 6)
    # wjj_pt = ak.fill_none(ak.firsts(getattr(events.w_had_jets, "pt", None)), np.nan)
    # wjj_pt_cut = np.where(np.isnan(wjj_pt),  False, wjj_pt  < params["wjj_pt"])

    cut_mt_w = (events.mt_w_leptonic < 185.0)

    # veto b optional
    b_veto = (events.nBJetGood == 0) if params.get("apply_b_veto", True) else True

    # if params.get("require_lep_central", False):
    lep = ak.firsts(events.LeptonGood)
    #     j1  = ak.firsts(getattr(events.vbsjets, "jet1", None))
    #     j2  = ak.firsts(getattr(events.vbsjets, "jet2", None))

    #     j1_eta = ak.fill_none(getattr(j1, "eta", None), np.nan)
    #     j2_eta = ak.fill_none(getattr(j2, "eta", None), np.nan)
    #     lep_eta = ak.fill_none(getattr(lep, "eta", None), np.nan)

    #     j1_pt_min = (j1.pt > 50)

    #     eta_min = np.minimum(j1_eta, j2_eta)
    #     eta_max = np.maximum(j1_eta, j2_eta)
    #     #lep_central = 
    #     lep_central = (~np.isnan(lep_eta)) & (~np.isnan(eta_min)) & (~np.isnan(eta_max)) & (lep_eta > eta_min) & (lep_eta < eta_max) & (lep.pt > 35.0) & j1_pt_min
    # else:
    #     lep_central = True

    mask = one_lep & two_j & met_cut  & cut_mt_w & b_veto#&  loose_lep_veto #(lep.pt > 35.0) &
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


def in_whad_window(events, params, **kwargs):
    four_j  = (events.nJetGood >= 4)
    no_fat = (events.nFatJetCandidate == 0)
    loose_lep_veto = (events.nLeptonLoose < 2)
    wjj_pt = ak.fill_none(ak.firsts(getattr(events.w_had_jets, "pt", None)), np.nan)
    wjj_pt_cut = np.where(np.isnan(wjj_pt),  False, wjj_pt  < 200.)
    wmass = ak.fill_none(ak.firsts(getattr(events.w_had_jets, "mass", None)), np.nan)
    within = np.where(np.isnan(wmass), False, np.abs(wmass - 85) < params["mjj_w_window"])
    # lead_lep_dR_cut1 = (events.lead_wlep_wjet1_dR > 0.8)
    # lead_lep_dR_cut2 = (events.lead_wlep_wjet2_dR > 0.8)
    lep = ak.firsts(events.LeptonGood)
    j1  = ak.firsts(getattr(events.vbsjets, "jet1", None))
    j2  = ak.firsts(getattr(events.vbsjets, "jet2", None))

    j1_eta = ak.fill_none(getattr(j1, "eta", None), np.nan)
    j2_eta = ak.fill_none(getattr(j2, "eta", None), np.nan)
    lep_eta = ak.fill_none(getattr(lep, "eta", None), np.nan)

    j1_pt_min = (j1.pt > 50)

    eta_min = np.minimum(j1_eta, j2_eta)
    eta_max = np.maximum(j1_eta, j2_eta)
     
    lep_central = j1_pt_min #& (~np.isnan(lep_eta)) & (~np.isnan(eta_min)) & (~np.isnan(eta_max)) & (lep_eta > eta_min) & (lep_eta < eta_max) 
    
    mjj_vbs   = ak.fill_none(ak.firsts(getattr(events.vbsjets, "mass", None)), np.nan)
    deta_vbs  = ak.fill_none(ak.firsts(getattr(events.vbsjets, "delta_eta", None)), np.nan)

    cut_mjj   = np.where(np.isnan(mjj_vbs),  False, mjj_vbs  > params["mjj_vbs"])
    cut_deta  = np.where(np.isnan(deta_vbs), False, deta_vbs > params["delta_eta_vbs"])

    mask = within & four_j  & wjj_pt_cut & cut_mjj & cut_deta & lep_central & loose_lep_veto & no_fat
    return ak.values_astype(mask, np.bool_)

whad_window_cut = Cut(
    name="whad_window",
    params={"mjj_w_window": 20.0,
            "mjj_vbs": 500.0,
            "delta_eta_vbs": 2.5},  
    function=in_whad_window,
)

def in_msd_window_fatjet(events, params, **kwargs):
    #yes_fat = (events.nFatJetCentral >= 1)
    loose_lep_veto = (events.nLeptonLoose < 2)
    yes_fat = (events.nFatJetCandidate == 1)
    fj1_pt = ak.fill_none(ak.firsts(getattr(events.w_fatjet, "pt", None)), np.nan)
    fj1_msd = ak.fill_none(ak.firsts(getattr(events.w_fatjet, "msoftdrop", None)), np.nan)
    pt_cut = np.where(np.isnan(fj1_pt), False, fj1_pt > 200.)
    within = np.where(np.isnan(fj1_msd), False, np.abs(fj1_msd - 92.5) < params["msd_w_window"])
    #sel = (~np.isnan(fj1_msd)) & (fj1_msd > wlo) & (fj1_msd < whi)
    lead_lep_dR_cut = (events.lead_wlep_wfatjet1_dR > 0.8)
    # jet1_dR_cut = (events.vbs1_fj_dR > 0.8)
    # jet2_dR_cut = (events.vbs2_fj_dR > 0.8)

    lep = ak.firsts(events.LeptonGood)
    j1  = ak.firsts(getattr(events.vbsjets_boost, "jet1", None))
    j2  = ak.firsts(getattr(events.vbsjets_boost, "jet2", None))

    j1_eta = ak.fill_none(getattr(j1, "eta", None), np.nan)
    j2_eta = ak.fill_none(getattr(j2, "eta", None), np.nan)
    lep_eta = ak.fill_none(getattr(lep, "eta", None), np.nan)

    j1_pt_min = (j1.pt > 50)

    eta_min = np.minimum(j1_eta, j2_eta)
    eta_max = np.maximum(j1_eta, j2_eta)
     
    lep_central = j1_pt_min #& (~np.isnan(lep_eta)) & (~np.isnan(eta_min)) & (~np.isnan(eta_max)) & (lep_eta > eta_min) & (lep_eta < eta_max) 
    
    

    mjj_vbs   = ak.fill_none(ak.firsts(getattr(events.vbsjets_boost, "mass", None)), np.nan)
    deta_vbs  = ak.fill_none(ak.firsts(getattr(events.vbsjets_boost, "delta_eta", None)), np.nan)

    cut_mjj   = np.where(np.isnan(mjj_vbs),  False, mjj_vbs  > params["mjj_vbs"])
    cut_deta  = np.where(np.isnan(deta_vbs), False, deta_vbs > params["delta_eta_vbs"])

    mask = yes_fat & within & pt_cut & lead_lep_dR_cut & cut_mjj & cut_deta & lep_central & loose_lep_veto#& jet_dR_cut 
    return ak.values_astype(mask, np.bool_)


msd_window_cut = Cut(
    name="msd_window",
    params={"msd_w_window": 22.5,
            "mjj_vbs": 500.0,
            "delta_eta_vbs": 2.5}, 
    function=in_msd_window_fatjet,
    )

# def narrow_whad_window(events, params, **kwargs):
#     four_j  = (events.nJetGood    >= 2)
#     no_fat = (ak.num(events.w_fatjet) == 0)
#     narrow = (events.w_had_dR <= 1.5) # require whad to be AK15 jet?
#     wmass = ak.fill_none(ak.firsts(getattr(events.w_had_jets, "mass", None)), np.nan)
#     within = np.where(np.isnan(wmass), False, np.abs(wmass - 80.4) < params["mjj_w_window"])
#     mask = within & four_j & no_fat & narrow
#     return ak.values_astype(mask, np.bool_)

# narrow_whad_window_cut = Cut(
#     name="narrow_had_window",
#     params={"mjj_w_window": 15.0},  
#     function=narrow_whad_window,
# )
