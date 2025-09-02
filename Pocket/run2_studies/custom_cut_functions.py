# custom_cut_functions.py
import awkward as ak
import numpy as np
from pocket_coffea.lib.cut_definition import Cut

# ---------- Skim: ≥1 leptón  (mu/e) ----------
def nLepton_skim(events, params, **kwargs):
    return (ak.num(events.Muon) + ak.num(events.Electron)) >= 1

nLepton_skim_cut = Cut(name="nLepton_skim", params={}, function=nLepton_skim)

# ---------- Preselection semileptonic VBS ----------
def select_vbs_semileptonic(events, params, **kwargs):
    
    
    one_lep = (events.nLeptonGood == 1)
    four_j  = (events.nJetGood    >= 4)
    met_cut = (events.MET.pt      >  params["met_pt"])
    
    wjj_pt = ak.fill_none(ak.firsts(getattr(events.w_had_jets, "pt", None)), np.nan)
    wjj_pt_cut = np.where(np.isnan(wjj_pt),  False, wjj_pt  < params["wjj_pt"])
    
    
    mjj_vbs   = ak.fill_none(ak.firsts(getattr(events.vbsjets, "mass", None)), np.nan)
    deta_vbs  = ak.fill_none(ak.firsts(getattr(events.vbsjets, "delta_eta", None)), np.nan)

    cut_mjj   = np.where(np.isnan(mjj_vbs),  False, mjj_vbs  > params["mjj_vbs"])
    cut_deta  = np.where(np.isnan(deta_vbs), False, deta_vbs > params["delta_eta_vbs"])

    # veto b optional
    b_veto = (events.nBJetGood == 0) if params.get("apply_b_veto", True) else True

    if params.get("require_lep_central", False):
        lep = ak.firsts(events.LeptonGood)
        j1  = ak.firsts(getattr(events.vbsjets, "jet1", None))
        j2  = ak.firsts(getattr(events.vbsjets, "jet2", None))

        j1_eta = ak.fill_none(getattr(j1, "eta", None), np.nan)
        j2_eta = ak.fill_none(getattr(j2, "eta", None), np.nan)
        lep_eta = ak.fill_none(getattr(lep, "eta", None), np.nan)

        eta_min = np.minimum(j1_eta, j2_eta)
        eta_max = np.maximum(j1_eta, j2_eta)
        #lep_central = 
        lep_central = (~np.isnan(lep_eta)) & (~np.isnan(eta_min)) & (~np.isnan(eta_max)) & (lep_eta > eta_min) & (lep_eta < eta_max) & (lep.pt > 35.0)
    else:
        lep_central = True

    mask = one_lep & four_j & met_cut & cut_mjj & cut_deta & b_veto & lep_central & wjj_pt_cut
    return ak.values_astype(mask, np.bool_)

vbs_semileptonic_presel = Cut(
    name="vbs_semileptonic",
    params={
        "met_pt": 30.0,
        "mjj_vbs": 500.0,
        "delta_eta_vbs": 2.5,
        "apply_b_veto": True,
        "require_lep_central": True,
        "wjj_pt": 200,
    },
    function=select_vbs_semileptonic,
)


def in_whad_window(events, params, **kwargs):

    wmass = ak.fill_none(ak.firsts(getattr(events.w_had_jets, "mass", None)), np.nan)
    within = np.where(np.isnan(wmass), False, np.abs(wmass - 80.4) < params["mjj_w_window"])
    return ak.values_astype(within, np.bool_)

whad_window_cut = Cut(
    name="whad_window",
    params={"mjj_w_window": 15.0},  
    function=in_whad_window,
)
