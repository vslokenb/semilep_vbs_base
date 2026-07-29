# custom_cut_functions.py
import awkward as ak
import numpy as np
import uproot
import correctionlib
from pocket_coffea.lib.cut_definition import Cut
from pocket_coffea.lib.jets import compute_jetId

# ---------- Skim: ≥1 leptón  (mu/e) ----------
def nLepton_skim(events, params, **kwargs):
    good_elec = events.Electron[events.Electron.pt > 35]
    good_muon = events.Muon[events.Muon.pt > 25]
   
    return (ak.num(good_elec) + ak.num(good_muon) >= 1)

nLepton_skim_cut = Cut(name="nLepton_skim", params={}, function=nLepton_skim)

def nJet_skim(events, params, **kwargs):
    good_jet =events.Jet[( ( ( abs(events.Jet.eta)<2.5 ) | (abs(events.Jet.eta)>3 ) ) & (events.Jet.pt > 30)) | ( ( ( abs(events.Jet.eta)>2.5 ) & (abs(events.Jet.eta)<3 ) ) & (events.Jet.pt > 50)) ]
    good_fatjet =events.FatJet[( ( ( abs(events.FatJet.eta)<2.5 ) | (abs(events.FatJet.eta)>3 ) ) & (events.FatJet.pt > 30)) | ( ( ( abs(events.FatJet.eta)>2.5 ) & (abs(events.FatJet.eta)<3 ) ) & (events.FatJet.pt > 50)) ]
    return (ak.num(good_jet) + ak.num(good_fatjet) >= 0)

nJet_skim_cut = Cut(name="nJet_skim", params={}, function=nJet_skim)

def met_skim(events, params, **kwargs):
    if params["met_def"] == "deepmet_response":
        met_cut = (events.DeepMETResponseTune.pt      >  30.0)
    elif params["met_def"] == "deepmet_resolution":
        met_cut = (events.DeepMETResolutionTune.pt      >  30.0)
    else:
        met_cut = (events.PuppiMET.pt      >  30.0)
    return met_cut

met_skim_cut = Cut(
        name="met_skim",
        params={
            "met_def": "deepmet_resolution",
            },
        function=met_skim)

def select_wjets(events, params, **kwargs):
    one_lep = (events.nLeptonGood ==1)
    if params["met_def"] == "deepmet_response":
        met_cut = (events.DeepMETResponseTune.pt      >  params["met_pt"])
    elif params["met_def"] == "deepmet_resolution":
        met_cut = (events.DeepMETResolutionTune.pt      >  params["met_pt"])
    else:
        met_cut = (events.PuppiMET.pt      >  params["met_pt"])
    mask = one_lep & met_cut
    return ak.values_astype(mask, np.bool_)

wjets_sel = Cut(
    name="wjet",
    params={
        "met_def": "deepmet_resolution",
        "met_pt": 30.0,
    },
    function=select_wjets,
)

def select_wjets_e(events, params, **kwargs):
    one_lep = (events.nElectronGood ==1)
    if params["met_def"] == "deepmet_response":
        met_cut = (events.DeepMETResponseTune.pt      >  params["met_pt"])
    elif params["met_def"] == "deepmet_resolution":
        met_cut = (events.DeepMETResolutionTune.pt      >  params["met_pt"])
    else:
        met_cut = (events.PuppiMET.pt      >  params["met_pt"])
    mask = one_lep & met_cut
    return ak.values_astype(mask, np.bool_)

wjets_e_sel = Cut(
    name="wjet_e",
    params={
        "met_def": "deepmet_resolution",
        "met_pt": 30.0,
    },
    function=select_wjets_e,
)

def select_wjets_mu(events, params, **kwargs):
    one_lep = (events.nMuonGood ==1)
    if params["met_def"] == "deepmet_response":
        met_cut = (events.DeepMETResponseTune.pt      >  params["met_pt"])
    elif params["met_def"] == "deepmet_resolution":
        met_cut = (events.DeepMETResolutionTune.pt      >  params["met_pt"])
    else:
        met_cut = (events.PuppiMET.pt      >  params["met_pt"])
    mask = one_lep & met_cut
    return ak.values_astype(mask, np.bool_)

wjets_mu_sel = Cut(
    name="wjet_mu",
    params={
        "met_def": "deepmet_resolution",
        "met_pt": 30.0,
    },
    function=select_wjets_mu,
)

def select_wjets_boosted(events, params, **kwargs):
    one_lep = (events.nLeptonGood ==1)
    one_boost_candidate = (events.nBoostCandidate == 1)
    b_veto = (events.nBJetLoose == 0)
    loose_lep_veto = (events.nLeptonVeto < 2)
    if params["met_def"] == "deepmet_response":
        met_cut = (events.DeepMETResponseTune.pt      >  params["met_pt"])
    elif params["met_def"] == "deepmet_resolution":
        met_cut = (events.DeepMETResolutionTune.pt      >  params["met_pt"])
    else:
        met_cut = (events.PuppiMET.pt      >  params["met_pt"])
    mask = one_lep & one_boost_candidate & b_veto & loose_lep_veto & met_cut
    return ak.values_astype(mask, np.bool_)

wjets_boosted_sel = Cut(
    name="wjet_boosted",
    params={
        "met_def": "deepmet_resolution",
        "met_pt": 30.0,
    },
    function=select_wjets_boosted,
)

def select_zjets_boosted(events, params, **kwargs):
    mask_2mu = (events.nMuonGood == 2) & (ak.sum(events.MuonGood.charge,axis=-1) == 0)
    mask_2el = (events.nElectronGood == 2) & (ak.sum(events.ElectronGood.charge,axis=-1) == 0)
    mupair = ak.combinations(events.MuonGood, 2, fields=["mu1", "mu2"])
    mll_mu = ak.sum(ak.fill_none((mupair.mu1 + mupair.mu2).mass, np.nan),axis=1)
    elpair = ak.combinations(events.ElectronGood, 2, fields=["el1", "el2"])
    mll_e = ak.sum(ak.fill_none((elpair.el1 + elpair.el2).mass, np.nan),axis=1)
    mupair_mass_window = (mll_mu > params["mll_low"]) & (mll_mu < params["mll_high"])
    elpair_mass_window = (mll_e > params["mll_low"]) & (mll_e < params["mll_high"])
    one_boost_candidate = (events.nBoostCandidate == 1)

    is_hem_dataset = (
        "EGamma_2018_EraC" in events.metadata["dataset"]
        or "EGamma_2018_EraD" in events.metadata["dataset"] 
        or "SingleMuon_2018_EraC" in events.metadata["dataset"] 
        or "SingleMuon_2018_EraD" in events.metadata["dataset"] 
    )

    if is_hem_dataset:
        hem_mask = ~(
            (events.run >= 319077)
            & (ak.num(events.JetHEM) + ak.num(events.ElectronHEM) >= 1)
        )
    else:
        hem_mask = True
    mask = one_boost_candidate & hem_mask & ((mask_2mu & mupair_mass_window)  | ( mask_2el & elpair_mass_window))
    return ak.values_astype(mask, np.bool_)

zjets_boosted_sel = Cut(
    name="zjet_boosted",
    params={
        "mll_low": 60.0,
        "mll_high": 120.0
    },
    function=select_zjets_boosted,
)



def select_zjets(events, params, **kwargs):
    mask_2mu = (events.nMuonGood == 2) & (ak.sum(events.MuonGood.charge,axis=-1) == 0)
    mask_2el = (events.nElectronGood == 2) & (ak.sum(events.ElectronGood.charge,axis=-1) == 0)
    mupair = ak.combinations(events.MuonGood, 2, fields=["mu1", "mu2"])
    mll_mu = ak.sum(ak.fill_none((mupair.mu1 + mupair.mu2).mass, np.nan),axis=1)
    elpair = ak.combinations(events.ElectronGood, 2, fields=["el1", "el2"])
    mll_e = ak.sum(ak.fill_none((elpair.el1 + elpair.el2).mass, np.nan),axis=1)
    mupair_mass_window = (mll_mu > params["mll_low"]) & (mll_mu < params["mll_high"])
    elpair_mass_window = (mll_e > params["mll_low"]) & (mll_e < params["mll_high"])
    mask = (mask_2mu & mupair_mass_window)  | ( mask_2el & elpair_mass_window )

    return ak.values_astype(mask, np.bool_)

zjets_sel = Cut(
    name="zjet",
    params={
        "mll_low": 60.0,
        "mll_high": 120.0
    },
    function=select_zjets,
)

def select_zjets_e(events, params, **kwargs):
    mask_2el = (events.nElectronGood == 2) & (ak.sum(events.ElectronGood.charge,axis=-1) == 0)
    elpair = ak.combinations(events.ElectronGood, 2, fields=["el1", "el2"])
    mll_e = ak.sum(ak.fill_none((elpair.el1 + elpair.el2).mass, np.nan),axis=1)
    elpair_mass_window = (mll_e > params["mll_low"]) & (mll_e < params["mll_high"])
    mask =  ( mask_2el & elpair_mass_window )

    return ak.values_astype(mask, np.bool_)

zjets_e_sel = Cut(
    name="zjet_e",
    params={
        "mll_low": 60.0,
        "mll_high": 120.0
    },
    function=select_zjets_e,
)

def select_zjets_mu(events, params, **kwargs):
    mask_2mu = (events.nMuonGood == 2) & (ak.sum(events.MuonGood.charge,axis=-1) == 0)
    mupair = ak.combinations(events.MuonGood, 2, fields=["mu1", "mu2"])
    mll_mu = ak.sum(ak.fill_none((mupair.mu1 + mupair.mu2).mass, np.nan),axis=1)
    mupair_mass_window = (mll_mu > params["mll_low"]) & (mll_mu < params["mll_high"])
    mask = (mask_2mu & mupair_mass_window)

    return ak.values_astype(mask, np.bool_)

zjets_mu_sel = Cut(
    name="zjet_mu",
    params={
        "mll_low": 60.0,
        "mll_high": 120.0
    },
    function=select_zjets_mu,
)



def select_TT_mu(events, params, **kwargs):
    one_lep = (events.nMuonGood ==1)
    nbj = (events.nBJetTight >= params["nbjet"])
    central_j = (events.nJetGoodCentral    >= params["njet_central"])
    nj = (events.nJetGood    >= params["njet"])
    loose_lep_veto = (events.nLeptonVeto < 2)
    mask = one_lep & nbj & central_j & nj & loose_lep_veto
    return ak.values_astype(mask, np.bool_)

TT_mu_sel = Cut(
    name="TT_mu",
    params={
        "njet_central": 3,
        "njet": 3,
        "nbjet": 1
        },
    function=select_TT_mu,
)

TT_mu_2bj_sel = Cut(
    name="TT_mu",
    params={
        "njet_central": 3,
        "njet": 3,
        "nbjet": 2
        },
    function=select_TT_mu,
)

TT_mu_2bj_4j_2central_sel = Cut(
    name="TT_mu",
    params={
        "njet_central": 2,
        "njet": 4,
        "nbjet": 2
        },
    function=select_TT_mu,
)

TT_mu_1bj_4j_2central_sel = Cut(
    name="TT_mu",
    params={
        "njet_central": 2,
        "njet": 4,
        "nbjet": 1
        },
    function=select_TT_mu,
)

def select_TT_e(events, params, **kwargs):
    one_lep = (events.nElectronGood ==1)
    nbj = (events.nBJetTight >= params["nbjet"])
    central_j = (events.nJetGoodCentral    >= params["njet_central"])
    nj = (events.nJetGood    >= params["njet"])
    loose_lep_veto = (events.nLeptonVeto < 2)
    mask = one_lep & nbj & central_j & nj & loose_lep_veto
    return ak.values_astype(mask, np.bool_)

TT_e_sel = Cut(
    name="TT_e",
    params={
        "njet_central": 3,
        "njet": 3,
        "nbjet": 1
        },
    function=select_TT_e,
)

TT_e_2bj_sel = Cut(
    name="TT_e",
    params={
        "njet_central": 3,
        "njet": 3,
        "nbjet": 2
        },
    function=select_TT_e,
)

TT_e_2bj_4j_2central_sel = Cut(
    name="TT_e",
    params={
        "njet_central": 2,
        "njet": 4,
        "nbjet": 2
        },
    function=select_TT_e,
)

TT_e_1bj_4j_2central_sel = Cut(
    name="TT_e",
    params={
        "njet_central": 2,
        "njet": 4,
        "nbjet": 1
        },
    function=select_TT_e,
)


def select_TT_boosted(events, params, **kwargs):
    one_lep = (events.nLeptonGood ==1)
    is_hem_dataset = (
        "EGamma_2018_EraC" in events.metadata["dataset"]
        or "EGamma_2018_EraD" in events.metadata["dataset"] 
        or "SingleMuon_2018_EraC" in events.metadata["dataset"] 
        or "SingleMuon_2018_EraD" in events.metadata["dataset"] 
    )

    if is_hem_dataset:
        hem_mask = ~(
            (events.run >= 319077)
            & (ak.num(events.JetHEM) + ak.num(events.ElectronHEM) >= 1)
        )
    else:
        hem_mask = True
    nbj = (events.nBJetTight >= params["nbjet"])
    loose_lep_veto = (events.nLeptonVeto < 2)
    one_boost_candidate = (events.nBoostCandidate == 1)
    mask = one_lep & nbj & one_boost_candidate & loose_lep_veto & hem_mask
    return ak.values_astype(mask, np.bool_)

TT_boosted_sel = Cut(
    name="TT_boosted",
    params={
        "nbjet": 2
        },
    function=select_TT_boosted,
)



# ---------- Preselection semileptonic VBS ----------
def select_vbs_semileptonic(events, params, **kwargs):
    
    #pu_pv_corrections = (events.PV.npvsGood < 55) | (events.PV.npvsGood > 60) 
    one_lep = (events.nLeptonGood ==1)
    
    two_j  = (events.nJetGood    >= 2)
    if params["met_def"] == "deepmet_response":
        met_cut = (events.DeepMETResponseTune.pt      >  params["met_pt"])
    elif params["met_def"] == "deepmet_resolution":
        met_cut = (events.DeepMETResolutionTune.pt      >  params["met_pt"])
    else:
        met_cut = (events.PuppiMET.pt      >  params["met_pt"])

    #good_bjet =events.JetGood[(np.abs(events.JetGood.eta) < 2.5) & (np.abs(events.JetGood.partonFlavour) == 5)]
    
    #dR_investigation = (events.Jet.jetId >= 6)
    # wjj_pt = ak.fill_none(ak.firsts(getattr(events.w_had_jets, "pt", None)), np.nan)
    # wjj_pt_cut = np.where(np.isnan(wjj_pt),  False, wjj_pt  < params["wjj_pt"])

    if params["met_def"] == "deepmet_response":
        cut_mt_w = (events.mt_w_leptonic_deepMET_responsetune < 185.0)
    elif params["met_def"] == "deepmet_resolution":
        cut_mt_w = (events.mt_w_leptonic_deepMET_resolutiontune < 185.0)
    else:
        cut_mt_w = (events.mt_w_leptonic < 185.0)

    # veto b optional

    b_veto = (events.nBJetLoose == 0) if params.get("apply_b_veto", True) else True
    #b_veto_gen = (ak.num(good_bjet) == 0)
    # if params.get("require_lep_central", False):

    #     eta_min = np.minimum(j1_eta, j2_eta)
    #     eta_max = np.maximum(j1_eta, j2_eta)
    #     #lep_central = 
    #     lep_central = (np.isnan(lep_eta)) & (np.isnan(eta_min)) & (np.isnan(eta_max)) & (lep_eta > eta_min) & (lep_eta < eta_max) & (lep.pt > 35.0) & j1_pt_min
    # else:
    #     lep_central = True

    #ht_mask = (events.LHE.HT <= 70.)
    ## W PT stitching
    if params.get("apply_w_pt_stitch", False):
        w_pt_stitch = (events.gen_w_pt_by_pdg < 40)
    else:
        w_pt_stitch = True
    mask = one_lep & met_cut & two_j & cut_mt_w & b_veto & w_pt_stitch#& ht_mask#&  loose_lep_veto #(lep.pt > 35.0) &
    #mask = one_lep & met_cut & two_j & cut_mt_w & b_veto#& ht_mask#&  loose_lep_veto #(lep.pt > 35.0) &
    return ak.values_astype(mask, np.bool_)

vbs_semileptonic_presel = Cut(
    name="vbs_semileptonic",
    params={
        "met_pt": 30.0,
        "apply_b_veto": False,
        "require_lep_central": True,
        "met_def": "deepmet_resolution",
        "apply_w_pt_stitch": False
    },
    function=select_vbs_semileptonic,
)

vbs_semileptonic_w_pt_stitch_presel = Cut(
    name="vbs_semileptonic",
    params={
        "met_pt": 30.0,
        "apply_b_veto": False,
        "require_lep_central": True,
        "met_def": "deepmet_resolution",
        "apply_w_pt_stitch": True
    },
    function=select_vbs_semileptonic,
)


# ---------- Preselection at gen level ----------

def select_gen_vbs_semileptonic(events, params, **kwargs):
    
    one_lep = (events.nGenLeptonGood ==1)
    
    two_j  = (events.nGenJetGood    >= 2)

    met_cut = (events.GenMET.pt      >  params["met_pt"])


    cut_mt_w = (events.gen_mt_w_leptonic < 185.0)

    # veto b optional

    b_veto = (events.nGenBJet == 0) if params.get("apply_b_veto", True) else True

    ## W PT stitching
    if params.get("apply_w_pt_stitch", False):
        w_pt_stitch = (events.gen_w_pt_by_pdg < 40)
    else:
        w_pt_stitch = True
    mask = one_lep & met_cut & two_j & cut_mt_w & b_veto & w_pt_stitch#& ht_mask#&  loose_lep_veto #(lep.pt > 35.0) &
    #mask = one_lep & met_cut & two_j & cut_mt_w & b_veto#& ht_mask#&  loose_lep_veto #(lep.pt > 35.0) &
    return ak.values_astype(mask, np.bool_)

vbs_gen_semileptonic_presel = Cut(
    name="gen_vbs_semileptonic",
    params={
        "met_pt": 30.0,
        "apply_b_veto": False,
        "require_lep_central": True,
        "met_def": "deepmet_resolution",
        "apply_w_pt_stitch": False
    },
    function=select_gen_vbs_semileptonic,
)

############################################
##### QCD-enriched control region
###########################################

def select_QCD_CR(events, params, **kwargs):
    one_lep = (events.nLeptonLoose >=1)
    recoil_jet = (events.nJetGood_recoil >= 1) 
    recoil_jet_pt = (events.LeadJetGood_recoil.pt >= params["recoil_jet_pt"]) 
    cut_njet = (events.nJetGood >= params["njet"])
    if params["met_def"] == "deepmet_response":
        met_cut = (events.DeepMETResponseTune.pt      <  params["met_pt"])
        cut_mt_w = (events.mt_w_leptonic_deepMET_responsetune_loose < 30.0)
    elif params["met_def"] == "deepmet_resolution":
        met_cut = (events.DeepMETResolutionTune.pt      <  params["met_pt"])
        cut_mt_w = (events.mt_w_leptonic_deepMET_resolutiontune_loose < 30.0)
    else:
        met_cut = (events.PuppiMET.pt      <  params["met_pt"])
        cut_mt_w = (events.mt_w_leptonic_loose < 30.0)
    mask = one_lep & met_cut & recoil_jet & cut_mt_w & recoil_jet_pt & cut_njet
    return ak.values_astype(mask, np.bool_)

qcd_enriched_cut_35 = Cut(
    name="qcd_enriched",
    params={
        "met_pt": 30.0,
        "recoil_jet_pt": 35,
        "njet": 0,
        "met_def": "puppimet",
    },
    function=select_QCD_CR,
)
qcd_enriched_cut_30 = Cut(
    name="qcd_enriched",
    params={
        "met_pt": 30.0,
        "recoil_jet_pt": 30,
        "njet": 0,
        "met_def": "puppimet",
    },
    function=select_QCD_CR,
)

qcd_enriched_cut_40 = Cut(
    name="qcd_enriched",
    params={
        "met_pt": 30.0,
        "recoil_jet_pt": 40,
        "njet": 0,
        "met_def": "puppimet",
    },
    function=select_QCD_CR,
)

qcd_enriched_cut_45 = Cut(
    name="qcd_enriched",
    params={
        "met_pt": 30.0,
        "recoil_jet_pt": 45,
        "njet": 0,
        "met_def": "puppimet",
    },
    function=select_QCD_CR,
)

qcd_enriched_cut_35_2j = Cut(
    name="qcd_enriched",
    params={
        "met_pt": 30.0,
        "recoil_jet_pt": 35,
        "njet": 2,
        "met_def": "puppimet",
    },
    function=select_QCD_CR,
)

qcd_enriched_cut_30_2j = Cut(
    name="qcd_enriched",
    params={
        "met_pt": 30.0,
        "recoil_jet_pt": 30,
        "njet": 2,
        "met_def": "puppimet",
    },
    function=select_QCD_CR,
)

qcd_enriched_cut_40_2j = Cut(
    name="qcd_enriched",
    params={
        "met_pt": 30.0,
        "recoil_jet_pt": 40,
        "njet": 2,
        "met_def": "puppimet",
    },
    function=select_QCD_CR,
)

qcd_enriched_cut_45_2j = Cut(
    name="qcd_enriched",
    params={
        "met_pt": 30.0,
        "recoil_jet_pt": 45,
        "njet": 2,
        "met_def": "puppimet",
    },
    function=select_QCD_CR,
)

qcd_enriched_cut_35_3j = Cut(
    name="qcd_enriched",
    params={
        "met_pt": 30.0,
        "recoil_jet_pt": 35,
        "njet": 3,
        "met_def": "puppimet",
    },
    function=select_QCD_CR,
)

qcd_enriched_cut_30_3j = Cut(
    name="qcd_enriched",
    params={
        "met_pt": 30.0,
        "recoil_jet_pt": 30,
        "njet": 3,
        "met_def": "puppimet",
    },
    function=select_QCD_CR,
)

qcd_enriched_cut_40_3j = Cut(
    name="qcd_enriched",
    params={
        "met_pt": 30.0,
        "recoil_jet_pt": 40,
        "njet": 3,
        "met_def": "puppimet",
    },
    function=select_QCD_CR,
)

qcd_enriched_cut_45_3j = Cut(
    name="qcd_enriched",
    params={
        "met_pt": 30.0,
        "recoil_jet_pt": 45,
        "njet": 3,
        "met_def": "puppimet",
    },
    function=select_QCD_CR,
)

qcd_enriched_cut_35_4j = Cut(
    name="qcd_enriched",
    params={
        "met_pt": 30.0,
        "recoil_jet_pt": 35,
        "njet": 4,
        "met_def": "puppimet",
    },
    function=select_QCD_CR,
)

qcd_enriched_cut_30_4j = Cut(
    name="qcd_enriched",
    params={
        "met_pt": 30.0,
        "recoil_jet_pt": 30,
        "njet": 4,
        "met_def": "puppimet",
    },
    function=select_QCD_CR,
)

qcd_enriched_cut_40_4j = Cut(
    name="qcd_enriched",
    params={
        "met_pt": 30.0,
        "recoil_jet_pt": 40,
        "njet": 4,
        "met_def": "puppimet",
    },
    function=select_QCD_CR,
)

qcd_enriched_cut_45_4j = Cut(
    name="qcd_enriched",
    params={
        "met_pt": 30.0,
        "recoil_jet_pt": 45,
        "njet": 4,
        "met_def": "puppimet",
    },
    function=select_QCD_CR,
)

qcd_enriched_cut_35_deepmet_response = Cut(
    name="qcd_enriched",
    params={
        "met_pt": 30.0,
        "recoil_jet_pt": 35,
        "njet": 0,
        "met_def": "deepmet_response",
    },
    function=select_QCD_CR,
)
qcd_enriched_cut_30_deepmet_response = Cut(
    name="qcd_enriched",
    params={
        "met_pt": 30.0,
        "recoil_jet_pt": 30,
        "njet": 0,
        "met_def": "deepmet_response",
    },
    function=select_QCD_CR,
)

qcd_enriched_cut_40_deepmet_response = Cut(
    name="qcd_enriched",
    params={
        "met_pt": 30.0,
        "recoil_jet_pt": 40,
        "njet": 0,
        "met_def": "deepmet_response",
    },
    function=select_QCD_CR,
)

qcd_enriched_cut_45_deepmet_response = Cut(
    name="qcd_enriched",
    params={
        "met_pt": 30.0,
        "recoil_jet_pt": 45,
        "njet": 0,
        "met_def": "deepmet_response",
    },
    function=select_QCD_CR,
)

qcd_enriched_cut_35_deepmet_resolution = Cut(
    name="qcd_enriched",
    params={
        "met_pt": 30.0,
        "recoil_jet_pt": 35,
        "njet": 0,
        "met_def": "deepmet_resolution",
    },
    function=select_QCD_CR,
)
qcd_enriched_cut_30_deepmet_resolution = Cut(
    name="qcd_enriched",
    params={
        "met_pt": 30.0,
        "recoil_jet_pt": 30,
        "njet": 0,
        "met_def": "deepmet_resolution",
    },
    function=select_QCD_CR,
)

qcd_enriched_cut_40_deepmet_resolution = Cut(
    name="qcd_enriched",
    params={
        "met_pt": 30.0,
        "recoil_jet_pt": 40,
        "njet": 0,
        "met_def": "deepmet_resolution",
    },
    function=select_QCD_CR,
)

qcd_enriched_cut_45_deepmet_resolution = Cut(
    name="qcd_enriched",
    params={
        "met_pt": 30.0,
        "recoil_jet_pt": 45,
        "njet": 0,
        "met_def": "deepmet_resolution",
    },
    function=select_QCD_CR,
)


qcd_enriched_cut_35_deepmet_response_2j = Cut(
    name="qcd_enriched",
    params={
        "met_pt": 30.0,
        "recoil_jet_pt": 35,
        "njet": 2,
        "met_def": "deepmet_response",
    },
    function=select_QCD_CR,
)
qcd_enriched_cut_30_deepmet_response_2j = Cut(
    name="qcd_enriched",
    params={
        "met_pt": 30.0,
        "recoil_jet_pt": 30,
        "njet": 2,
        "met_def": "deepmet_response",
    },
    function=select_QCD_CR,
)

qcd_enriched_cut_40_deepmet_response_2j = Cut(
    name="qcd_enriched",
    params={
        "met_pt": 30.0,
        "recoil_jet_pt": 40,
        "njet": 2,
        "met_def": "deepmet_response",
    },
    function=select_QCD_CR,
)

qcd_enriched_cut_45_deepmet_response_2j = Cut(
    name="qcd_enriched",
    params={
        "met_pt": 30.0,
        "recoil_jet_pt": 45,
        "njet": 2,
        "met_def": "deepmet_response",
    },
    function=select_QCD_CR,
)


qcd_enriched_cut_35_deepmet_response_3j = Cut(
    name="qcd_enriched",
    params={
        "met_pt": 30.0,
        "recoil_jet_pt": 35,
        "njet": 3,
        "met_def": "deepmet_response",
    },
    function=select_QCD_CR,
)
qcd_enriched_cut_30_deepmet_response_3j = Cut(
    name="qcd_enriched",
    params={
        "met_pt": 30.0,
        "recoil_jet_pt": 30,
        "njet": 3,
        "met_def": "deepmet_response",
    },
    function=select_QCD_CR,
)

qcd_enriched_cut_40_deepmet_response_3j = Cut(
    name="qcd_enriched",
    params={
        "met_pt": 30.0,
        "recoil_jet_pt": 40,
        "njet": 3,
        "met_def": "deepmet_response",
    },
    function=select_QCD_CR,
)

qcd_enriched_cut_45_deepmet_response_3j = Cut(
    name="qcd_enriched",
    params={
        "met_pt": 30.0,
        "recoil_jet_pt": 45,
        "njet": 3,
        "met_def": "deepmet_response",
    },
    function=select_QCD_CR,
)


qcd_enriched_cut_35_deepmet_resolution_2j = Cut(
    name="qcd_enriched",
    params={
        "met_pt": 30.0,
        "recoil_jet_pt": 35,
        "njet": 2,
        "met_def": "deepmet_resolution",
    },
    function=select_QCD_CR,
)
qcd_enriched_cut_30_deepmet_resolution_2j = Cut(
    name="qcd_enriched",
    params={
        "met_pt": 30.0,
        "recoil_jet_pt": 30,
        "njet": 2,
        "met_def": "deepmet_resolution",
    },
    function=select_QCD_CR,
)

qcd_enriched_cut_40_deepmet_resolution_2j = Cut(
    name="qcd_enriched",
    params={
        "met_pt": 30.0,
        "recoil_jet_pt": 40,
        "njet": 2,
        "met_def": "deepmet_resolution",
    },
    function=select_QCD_CR,
)

qcd_enriched_cut_45_deepmet_resolution_2j = Cut(
    name="qcd_enriched",
    params={
        "met_pt": 30.0,
        "recoil_jet_pt": 45,
        "njet": 2,
        "met_def": "deepmet_resolution",
    },
    function=select_QCD_CR,
)

qcd_enriched_cut_35_deepmet_resolution_3j = Cut(
    name="qcd_enriched",
    params={
        "met_pt": 30.0,
        "recoil_jet_pt": 35,
        "njet": 3,
        "met_def": "deepmet_resolution",
    },
    function=select_QCD_CR,
)   
qcd_enriched_cut_30_deepmet_resolution_3j = Cut(
    name="qcd_enriched",
    params={
        "met_pt": 30.0,
        "recoil_jet_pt": 30,
        "njet": 3,
        "met_def": "deepmet_resolution",
    },
    function=select_QCD_CR,
)

qcd_enriched_cut_40_deepmet_resolution_3j = Cut(
    name="qcd_enriched",
    params={
        "met_pt": 30.0,
        "recoil_jet_pt": 40,
        "njet": 3,
        "met_def": "deepmet_resolution",
    },
    function=select_QCD_CR,
)

qcd_enriched_cut_45_deepmet_resolution_3j = Cut(
    name="qcd_enriched",
    params={
        "met_pt": 30.0,
        "recoil_jet_pt": 45,
        "njet": 3,
        "met_def": "deepmet_resolution",
    },
    function=select_QCD_CR,
)

def select_QCD_validate(events, params, **kwargs):
    one_lep = (events.nLeptonGood == 1)
    cut_jet = (events.nJetGood >= params["nJetGood"])
    loose_lep_veto = (events.nLeptonVeto < 2)
    if params["met_def"] == "deepmet_response":
        met_cut = (events.DeepMETResponseTune.pt      <  params["met_pt"])
    elif params["met_def"] == "deepmet_resolution":
        met_cut = (events.DeepMETResolutionTune.pt      <  params["met_pt"])
    else:
        met_cut = (events.PuppiMET.pt      <  params["met_pt"])
    mask = one_lep & cut_jet & loose_lep_veto & met_cut
    return ak.values_astype(mask, np.bool_)

qcd_validate = Cut(
    name="qcd_validate",
    params={
        "met_pt": 30.0,
        "nJetGood": 4,
        "met_def": "deepmet_resolution",
    },
    function=select_QCD_validate,
)

qcd_validate_0j = Cut(
    name="qcd_validate",
    params={
        "met_pt": 30.0,
        "nJetGood": 0,
        "met_def": "deepmet_resolution",
    },
    function=select_QCD_validate,
)

def select_QCD_validate_e(events, params, **kwargs):
    one_lep = ( (events.nElectronGood == 1) & (events.nMuonGood == 0) )
    cut_jet = (events.nJetGood >= params["nJetGood"])
    loose_lep_veto = (events.nLeptonVeto < 2)
    if params["met_def"] == "deepmet_response":
        met_cut = (events.DeepMETResponseTune.pt      <  params["met_pt"])
    elif params["met_def"] == "deepmet_resolution":
        met_cut = (events.DeepMETResolutionTune.pt      <  params["met_pt"])
    else:
        met_cut = (events.PuppiMET.pt      <  params["met_pt"])
    mask = one_lep & cut_jet & loose_lep_veto & met_cut
    return ak.values_astype(mask, np.bool_)

qcd_validate_e = Cut(
    name="qcd_validate_e",
    params={
        "met_pt": 30.0,
        "nJetGood": 4,
        "met_def": "deepmet_resolution",
    },
    function=select_QCD_validate_e,
)

qcd_validate_0j_e = Cut(
    name="qcd_validate_e",
    params={
        "met_pt": 30.0,
        "nJetGood": 0,
        "met_def": "deepmet_resolution",
    },
    function=select_QCD_validate_e,
)

def select_QCD_validate_mu(events, params, **kwargs):
    one_lep = ( (events.nElectronGood == 0) & (events.nMuonGood == 1) )
    cut_jet = (events.nJetGood >= params["nJetGood"])
    loose_lep_veto = (events.nLeptonVeto < 2)
    if params["met_def"] == "deepmet_response":
        met_cut = (events.DeepMETResponseTune.pt      <  params["met_pt"])
    elif params["met_def"] == "deepmet_resolution":
        met_cut = (events.DeepMETResolutionTune.pt      <  params["met_pt"])
    else:
        met_cut = (events.PuppiMET.pt      <  params["met_pt"])
    mask = one_lep & cut_jet & loose_lep_veto & met_cut
    return ak.values_astype(mask, np.bool_)

qcd_validate_mu = Cut(
    name="qcd_validate_mu",
    params={
        "met_pt": 30.0,
        "nJetGood": 4,
        "met_def": "deepmet_resolution",
    },
    function=select_QCD_validate_mu,
)

qcd_validate_0j_mu = Cut(
    name="qcd_validate_mu",
    params={
        "met_pt": 30.0,
        "nJetGood": 0,
        "met_def": "deepmet_resolution",
    },
    function=select_QCD_validate_mu,
)
####################    Cutflow test    ####################



def second_lepton_veto(events, params, **kwargs):
    loose_lep_veto = (events.nLeptonVeto < 2)
    return loose_lep_veto

second_lepton_veto_cut = Cut(
    name="second_lepton_veto",
    params={},
    function=second_lepton_veto,
)

def gen_second_lepton_veto(events, params, **kwargs):
    loose_lep_veto = (events.nGenLeptonVeto < 2)
    return loose_lep_veto

gen_second_lepton_veto_cut = Cut(
    name="gen_second_lepton_veto",
    params={},
    function=gen_second_lepton_veto,
)



def resolved_jet(events, params, **kwargs):
    lep_ch = (events.nLeptonGood == 1)
    four_j  = (events.nJetGood >= 4)
    no_fat = (events.nFatJetCandidate == 0)
    return lep_ch & four_j & no_fat

resolved_jet_cut = Cut(
    name="resolved_jet",
    params={},
    function=resolved_jet,
)


def gen_resolved_jet(events, params, **kwargs):
    lep_ch = (events.nGenLeptonGood == 1)
    four_j  = (events.nGenJetGood >= 4)
    no_fat = (events.nGenFatJetCandidate == 0)
    return lep_ch & four_j & no_fat

gen_resolved_jet_cut = Cut(
    name="gen_resolved_jet",
    params={},
    function=gen_resolved_jet,
)

def boosted_jet(events, params, **kwargs):
    lep_ch = (events.nLeptonGood == 1)
    fat = (events.nFatJetCandidate == 1)
    return lep_ch & fat

boosted_jet_cut = Cut(
    name="boosted_jet",
    params={},
    function=boosted_jet,
)

def gen_boosted_jet(events, params, **kwargs):
    lep_ch = (events.nGenLeptonGood == 1)
    fat = (events.nGenFatJetCandidate == 1)
    return lep_ch & fat

gen_boosted_jet_cut = Cut(
    name="gen_boosted_jet",
    params={},
    function=gen_boosted_jet,
)

def resolved_forward(events, params, **kwargs):
    j1  = ak.firsts(getattr(events.vbsjets, "jet1", None))
    j2  = ak.firsts(getattr(events.vbsjets, "jet2", None))

    j1_pt_min = (j1.pt > 50)
    j2_pt_min = (j2.pt > 30)

    
    mjj_vbs   = ak.fill_none(ak.firsts(getattr(events.vbsjets, "mass", None)), np.nan)
    deta_vbs  = ak.fill_none(ak.firsts(getattr(events.vbsjets, "delta_eta", None)), np.nan)

    cut_mjj   = np.where(np.isnan(mjj_vbs),  False, mjj_vbs  > params["mjj_vbs"])
    cut_deta  = np.where(np.isnan(deta_vbs), False, deta_vbs > params["delta_eta_vbs"])

    return cut_mjj & cut_deta & j1_pt_min & j2_pt_min


resolved_forward_cut = Cut(
    name="resolved_forward",
    params={"mjj_vbs": 500.0,
            "delta_eta_vbs": 2.5},  
    function=resolved_forward,
)

def gen_resolved_forward(events, params, **kwargs):
    j1  = ak.firsts(getattr(events.gen_vbsjets, "jet1", None))
    j2  = ak.firsts(getattr(events.gen_vbsjets, "jet2", None))

    j1_pt_min = (j1.pt > 50)
    j2_pt_min = (j2.pt > 30)

    
    mjj_vbs   = ak.fill_none(ak.firsts(getattr(events.gen_vbsjets, "mass", None)), np.nan)
    deta_vbs  = ak.fill_none(ak.firsts(getattr(events.gen_vbsjets, "delta_eta", None)), np.nan)

    cut_mjj   = np.where(np.isnan(mjj_vbs),  False, mjj_vbs  > params["mjj_vbs"])
    cut_deta  = np.where(np.isnan(deta_vbs), False, deta_vbs > params["delta_eta_vbs"])

    return cut_mjj & cut_deta & j1_pt_min & j2_pt_min


gen_resolved_forward_cut = Cut(
    name="gen_resolved_forward",
    params={"mjj_vbs": 500.0,
            "delta_eta_vbs": 2.5},  
    function=gen_resolved_forward,
)


def boosted_forward(events, params, **kwargs):
    j1  = ak.firsts(getattr(events.vbsjets, "jet1", None))
    j2  = ak.firsts(getattr(events.vbsjets, "jet2", None))

    j1_pt_min = (j1.pt > 50)
    j2_pt_min = (j2.pt > 30)

    
    mjj_vbs   = ak.fill_none(ak.firsts(getattr(events.vbsjets, "mass", None)), np.nan)
    deta_vbs  = ak.fill_none(ak.firsts(getattr(events.vbsjets, "delta_eta", None)), np.nan)

    cut_mjj   = np.where(np.isnan(mjj_vbs),  False, mjj_vbs  > params["mjj_vbs"])
    cut_deta  = np.where(np.isnan(deta_vbs), False, deta_vbs > params["delta_eta_vbs"])

    return cut_mjj & cut_deta & j1_pt_min & j2_pt_min


boosted_forward_cut = Cut(
    name="boosted_forward",
    params={"mjj_vbs": 500.0,
            "delta_eta_vbs": 2.5},  
    function=boosted_forward,
)

def gen_boosted_forward(events, params, **kwargs):
    j1  = ak.firsts(getattr(events.gen_vbsjets, "jet1", None))
    j2  = ak.firsts(getattr(events.gen_vbsjets, "jet2", None))

    j1_pt_min = (j1.pt > 50)
    j2_pt_min = (j2.pt > 30)

    
    mjj_vbs   = ak.fill_none(ak.firsts(getattr(events.gen_vbsjets, "mass", None)), np.nan)
    deta_vbs  = ak.fill_none(ak.firsts(getattr(events.gen_vbsjets, "delta_eta", None)), np.nan)

    cut_mjj   = np.where(np.isnan(mjj_vbs),  False, mjj_vbs  > params["mjj_vbs"])
    cut_deta  = np.where(np.isnan(deta_vbs), False, deta_vbs > params["delta_eta_vbs"])

    return cut_mjj & cut_deta & j1_pt_min & j2_pt_min


gen_boosted_forward_cut = Cut(
    name="gen_boosted_forward",
    params={"mjj_vbs": 500.0,
            "delta_eta_vbs": 2.5},  
    function=gen_boosted_forward,
)

def b_jet_veto(events, params, **kwargs):
    b_veto = (events.nBJetLoose == 0)
    return b_veto

b_jet_veto_cut = Cut(
    name="b_jet_veto",
    params={},
    function=b_jet_veto,
)


def gen_b_jet_veto(events, params, **kwargs):
    gen_b_veto = (events.nGenBJet == 0)
    return gen_b_veto 

gen_b_jet_veto_cut = Cut(
    name="gen_b_jet_veto",
    params={},
    function=gen_b_jet_veto,
)



def resolved_SR(events, params, **kwargs):
    b_veto = (events.nBJetLoose == 0) 
    wjj_pt = ak.fill_none(ak.firsts(getattr(events.w_had_jets, "pt", None)), np.nan)
    wjj_pt_cut = np.where(np.isnan(wjj_pt),  False, wjj_pt  < 200.)
    wmass = ak.fill_none(ak.firsts(getattr(events.w_had_jets, "mass", None)), np.nan)
    within = np.where(np.isnan(wmass), False, np.abs(wmass - 85) < params["mjj_w_window"])

    return b_veto & wjj_pt_cut & within

resolved_SR_cut = Cut(
    name="resolved_SR",
    params={"mjj_w_window": 20.0},  
    function=resolved_SR,
)

def boosted_SR(events, params, **kwargs):
    b_veto = (events.nBJetLoose == 0) 
    fj1_pt = ak.fill_none(ak.firsts(getattr(events.candidate_boost, "pt", None)), np.nan)
    fj1_mass = ak.fill_none(ak.firsts(getattr(events.candidate_boost, "msoftdrop", None)), np.nan)
    pt_cut = np.where(np.isnan(fj1_pt), False, fj1_pt > 200.)
    within = np.where(np.isnan(fj1_mass), False, np.abs(fj1_mass - 92.5) < params["mass_w_window"])
    return b_veto & pt_cut & within

boosted_SR_cut = Cut(
    name="boosted_SR",
    params={"mass_w_window": 22.5},  
    function=boosted_SR,
)


def gen_resolved_SR(events, params, **kwargs):
    b_veto = (events.nGenBJet == 0) 
    wjj_pt = ak.fill_none(ak.firsts(getattr(events.gen_w_had_jets, "pt", None)), np.nan)
    wjj_pt_cut = np.where(np.isnan(wjj_pt),  False, wjj_pt  < 200.)
    wmass = ak.fill_none(ak.firsts(getattr(events.gen_w_had_jets, "mass", None)), np.nan)
    within = np.where(np.isnan(wmass), False, np.abs(wmass - 85) < params["mjj_w_window"])

    return b_veto & wjj_pt_cut & within

gen_resolved_SR_cut = Cut(
    name="gen_resolved_SR",
    params={"mjj_w_window": 20.0},  
    function=gen_resolved_SR,
)

def gen_boosted_SR(events, params, **kwargs):
    b_veto = (events.nGenBJet == 0) 
    fj1_pt = ak.fill_none(ak.firsts(getattr(events.gen_candidate_boost, "pt", None)), np.nan)
    fj1_mass = ak.fill_none(ak.firsts(getattr(events.gen_candidate_boost, "mass", None)), np.nan)
    pt_cut = np.where(np.isnan(fj1_pt), False, fj1_pt > 200.)
    within = np.where(np.isnan(fj1_mass), False, np.abs(fj1_mass - 92.5) < params["mass_w_window"])
    return b_veto & pt_cut & within

gen_boosted_SR_cut = Cut(
    name="gen_boosted_SR",
    params={"mass_w_window": 22.5},  
    function=gen_boosted_SR,
)



####################    SR, CR    ####################




def in_whad_window_mu(events, params, **kwargs):
    muon_ch = (events.nElectronGood == 0) & (events.nMuonGood == 1)
    four_j  = (events.nJetGood >= 4)
    no_fat = (events.nFatJetCandidate == 0)
    loose_lep_veto = (events.nLeptonVeto < 2)
    wjj_pt = ak.fill_none(ak.firsts(getattr(events.w_had_jets, "pt", None)), np.nan)
    wjj_pt_cut = np.where(np.isnan(wjj_pt),  False, wjj_pt  < 200.)
    wmass = ak.fill_none(ak.firsts(getattr(events.w_had_jets, "mass", None)), np.nan)
    within = np.where(np.isnan(wmass), False, np.abs(wmass - 85) < params["mjj_w_window"])
    # lead_lep_dR_cut1 = (events.lead_wlep_wjet1_dR > 0.8)
    # lead_lep_dR_cut2 = (events.lead_wlep_wjet2_dR > 0.8)

    j1  = ak.firsts(getattr(events.vbsjets, "jet1", None))
    j2  = ak.firsts(getattr(events.vbsjets, "jet2", None))

    j1_eta = ak.fill_none(getattr(j1, "eta", None), np.nan)
    j2_eta = ak.fill_none(getattr(j2, "eta", None), np.nan)

    j1_pt_min = (j1.pt > 50)
    j2_pt_min = (j2.pt > 30)

    
    mjj_vbs   = ak.fill_none(ak.firsts(getattr(events.vbsjets, "mass", None)), np.nan)
    deta_vbs  = ak.fill_none(ak.firsts(getattr(events.vbsjets, "delta_eta", None)), np.nan)

    cut_mjj   = np.where(np.isnan(mjj_vbs),  False, mjj_vbs  > params["mjj_vbs"])
    cut_deta  = np.where(np.isnan(deta_vbs), False, deta_vbs > params["delta_eta_vbs"])

    mask = four_j & within & wjj_pt_cut & cut_mjj & cut_deta & j1_pt_min & loose_lep_veto & no_fat & muon_ch & j1_pt_min & j2_pt_min 
    return ak.values_astype(mask, np.bool_)

whad_window_cut_mu = Cut(
    name="whad_window_mu",
    params={"mjj_w_window": 20.0,
            "mjj_vbs": 500.0,
            "delta_eta_vbs": 2.5},  
    function=in_whad_window_mu,
)





def in_whad_window_bveto_mu(events, params, **kwargs):
    muon_ch = (events.nElectronGood == 0) & (events.nMuonGood == 1)
    four_j  = (events.nJetGood >= 4)
    b_veto = (events.nBJetLoose == 0) 
    no_fat = (events.nFatJetCandidate == 0)
    loose_lep_veto = (events.nLeptonVeto < 2)
    wjj_pt = ak.fill_none(ak.firsts(getattr(events.w_had_jets, "pt", None)), np.nan)
    wjj_pt_cut = np.where(np.isnan(wjj_pt),  False, wjj_pt  < 200.)
    wmass = ak.fill_none(ak.firsts(getattr(events.w_had_jets, "mass", None)), np.nan)
    within = np.where(np.isnan(wmass), False, np.abs(wmass - 85) < params["mjj_w_window"])
    # lead_lep_dR_cut1 = (events.lead_wlep_wjet1_dR > 0.8)
    # lead_lep_dR_cut2 = (events.lead_wlep_wjet2_dR > 0.8)
    j1  = ak.firsts(getattr(events.vbsjets, "jet1", None))
    j2  = ak.firsts(getattr(events.vbsjets, "jet2", None))

    j1_pt_min = (j1.pt > 50)
    j2_pt_min = (j2.pt > 30)

    
    mjj_vbs   = ak.fill_none(ak.firsts(getattr(events.vbsjets, "mass", None)), np.nan)
    deta_vbs  = ak.fill_none(ak.firsts(getattr(events.vbsjets, "delta_eta", None)), np.nan)

    cut_mjj   = np.where(np.isnan(mjj_vbs),  False, mjj_vbs  > params["mjj_vbs"])
    cut_deta  = np.where(np.isnan(deta_vbs), False, deta_vbs > params["delta_eta_vbs"])

    mask = four_j & b_veto & within & wjj_pt_cut & cut_mjj & cut_deta & loose_lep_veto & no_fat & b_veto & muon_ch & j1_pt_min & j2_pt_min 
    return ak.values_astype(mask, np.bool_)

whad_window_cut_bveto_mu = Cut(
    name="whad_window_bveto_mu",
    params={"mjj_w_window": 20.0,
            "mjj_vbs": 500.0,
            "delta_eta_vbs": 2.5},  
    function=in_whad_window_bveto_mu,
)

def out_whad_window_bveto_mu(events, params, **kwargs):
    muon_ch = (events.nElectronGood == 0) & (events.nMuonGood == 1)
    four_j  = (events.nJetGood >= 4)
    b_veto = (events.nBJetLoose == 0) 
    no_fat = (events.nFatJetCandidate == 0)
    loose_lep_veto = (events.nLeptonVeto < 2)
    wjj_pt = ak.fill_none(ak.firsts(getattr(events.w_had_jets, "pt", None)), np.nan)
    wjj_pt_cut = np.where(np.isnan(wjj_pt),  False, wjj_pt  < 200.)
    wmass = ak.fill_none(ak.firsts(getattr(events.w_had_jets, "mass", None)), np.nan)
    within = np.where(np.isnan(wmass), False, np.abs(wmass - 85) < params["mjj_w_window"])
    # lead_lep_dR_cut1 = (events.lead_wlep_wjet1_dR > 0.8)
    # lead_lep_dR_cut2 = (events.lead_wlep_wjet2_dR > 0.8)
    j1  = ak.firsts(getattr(events.vbsjets, "jet1", None))
    j2  = ak.firsts(getattr(events.vbsjets, "jet2", None))

    j1_pt_min = (j1.pt > 50)
    j2_pt_min = (j2.pt > 30)

    
    mjj_vbs   = ak.fill_none(ak.firsts(getattr(events.vbsjets, "mass", None)), np.nan)
    deta_vbs  = ak.fill_none(ak.firsts(getattr(events.vbsjets, "delta_eta", None)), np.nan)

    cut_mjj   = np.where(np.isnan(mjj_vbs),  False, mjj_vbs  > params["mjj_vbs"])
    cut_deta  = np.where(np.isnan(deta_vbs), False, deta_vbs > params["delta_eta_vbs"])

    mask = four_j & b_veto & ~within & wjj_pt_cut & cut_mjj & cut_deta & loose_lep_veto & no_fat & b_veto & muon_ch & j1_pt_min & j2_pt_min 
    return ak.values_astype(mask, np.bool_)

whad_windowinvert_cut_bveto_mu = Cut(
    name="whad_window_bveto_mu",
    params={"mjj_w_window": 20.0,
            "mjj_vbs": 500.0,
            "delta_eta_vbs": 2.5},  
    function=out_whad_window_bveto_mu,
)

def in_whad_window_baccept_mu(events, params, **kwargs):
    muon_ch = (events.nElectronGood == 0) & (events.nMuonGood == 1)
    four_j  = (events.nJetGood >= 4)
    b_accept = (events.nBJetTight > 0) 
    no_fat = (events.nFatJetCandidate == 0)
    loose_lep_veto = (events.nLeptonVeto < 2)
    wjj_pt = ak.fill_none(ak.firsts(getattr(events.w_had_jets, "pt", None)), np.nan)
    wjj_pt_cut = np.where(np.isnan(wjj_pt),  False, wjj_pt  < 200.)
    wmass = ak.fill_none(ak.firsts(getattr(events.w_had_jets, "mass", None)), np.nan)
    within = np.where(np.isnan(wmass), False, np.abs(wmass - 85) < params["mjj_w_window"])
    # lead_lep_dR_cut1 = (events.lead_wlep_wjet1_dR > 0.8)
    # lead_lep_dR_cut2 = (events.lead_wlep_wjet2_dR > 0.8)
    j1  = ak.firsts(getattr(events.vbsjets, "jet1", None))
    j2  = ak.firsts(getattr(events.vbsjets, "jet2", None))

    j1_pt_min = (j1.pt > 50)
    j2_pt_min = (j2.pt > 30)


    
    mjj_vbs   = ak.fill_none(ak.firsts(getattr(events.vbsjets, "mass", None)), np.nan)
    deta_vbs  = ak.fill_none(ak.firsts(getattr(events.vbsjets, "delta_eta", None)), np.nan)

    cut_mjj   = np.where(np.isnan(mjj_vbs),  False, mjj_vbs  > params["mjj_vbs"])
    cut_deta  = np.where(np.isnan(deta_vbs), False, deta_vbs > params["delta_eta_vbs"])

    mask = four_j & b_accept & within & wjj_pt_cut & cut_mjj & cut_deta & loose_lep_veto & no_fat & b_accept & muon_ch & j1_pt_min & j2_pt_min 
    return ak.values_astype(mask, np.bool_)

whad_window_cut_baccept_mu = Cut(
    name="whad_window_bveto_mu",
    params={"mjj_w_window": 20.0,
            "mjj_vbs": 500.0,
            "delta_eta_vbs": 2.5},  
    function=in_whad_window_baccept_mu,
)



def in_msd_window_fatjet_bveto_mu(events, params, **kwargs):
    muon_ch = (events.nElectronGood == 0) & (events.nMuonGood == 1)
    #yes_fat = (events.nFatJetCentral >= 1)
    b_veto = (events.nBJetLoose == 0) 
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

    j1  = ak.firsts(getattr(events.vbsjets, "jet1", None))
    j2  = ak.firsts(getattr(events.vbsjets, "jet2", None))


    j1_pt_min = (j1.pt > 50)
    j2_pt_min = (j2.pt > 30)

    mjj_vbs   = ak.fill_none(ak.firsts(getattr(events.vbsjets, "mass", None)), np.nan)
    deta_vbs  = ak.fill_none(ak.firsts(getattr(events.vbsjets, "delta_eta", None)), np.nan)

    cut_mjj   = np.where(np.isnan(mjj_vbs),  False, mjj_vbs  > params["mjj_vbs"])
    cut_deta  = np.where(np.isnan(deta_vbs), False, deta_vbs > params["delta_eta_vbs"])

    mask = yes_fat & within & pt_cut & cut_mjj & cut_deta & loose_lep_veto & muon_ch & j1_pt_min & j2_pt_min & b_veto #& jet_dR_cut 
    return ak.values_astype(mask, np.bool_)


msd_window_cut_bveto_mu = Cut(
    name="msd_window_mu",
    params={"msd_w_window": 22.5,
            "mjj_vbs": 500.0,
            "delta_eta_vbs": 2.5}, 
    function=in_msd_window_fatjet_bveto_mu,
    )

def out_msd_window_fatjet_bveto_mu(events, params, **kwargs):
    muon_ch = (events.nElectronGood == 0) & (events.nMuonGood == 1)
    #yes_fat = (events.nFatJetCentral >= 1)
    b_veto = (events.nBJetLoose == 0) 
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

    j1  = ak.firsts(getattr(events.vbsjets, "jet1", None))
    j2  = ak.firsts(getattr(events.vbsjets, "jet2", None))

    j1_pt_min = (j1.pt > 50)
    j2_pt_min = (j2.pt > 30)

    mjj_vbs   = ak.fill_none(ak.firsts(getattr(events.vbsjets, "mass", None)), np.nan)
    deta_vbs  = ak.fill_none(ak.firsts(getattr(events.vbsjets, "delta_eta", None)), np.nan)

    cut_mjj   = np.where(np.isnan(mjj_vbs),  False, mjj_vbs  > params["mjj_vbs"])
    cut_deta  = np.where(np.isnan(deta_vbs), False, deta_vbs > params["delta_eta_vbs"])

    mask = yes_fat & ~within & pt_cut & cut_mjj & cut_deta & loose_lep_veto & muon_ch & j1_pt_min & j2_pt_min & b_veto #& jet_dR_cut 
    return ak.values_astype(mask, np.bool_)


msd_windowinvert_cut_bveto_mu = Cut(
    name="msd_window_mu",
    params={"msd_w_window": 22.5,
            "mjj_vbs": 500.0,
            "delta_eta_vbs": 2.5}, 
    function=out_msd_window_fatjet_bveto_mu,
    )


def in_msd_window_fatjet_baccept_mu(events, params, **kwargs):
    muon_ch = (events.nElectronGood == 0) & (events.nMuonGood == 1)
    #yes_fat = (events.nFatJetCentral >= 1)
    b_accept = (events.nBJetTight > 0) 
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

    j1  = ak.firsts(getattr(events.vbsjets, "jet1", None))
    j2  = ak.firsts(getattr(events.vbsjets, "jet2", None))

    j1_pt_min = (j1.pt > 50)
    j2_pt_min = (j2.pt > 30)

    

    mjj_vbs   = ak.fill_none(ak.firsts(getattr(events.vbsjets, "mass", None)), np.nan)
    deta_vbs  = ak.fill_none(ak.firsts(getattr(events.vbsjets, "delta_eta", None)), np.nan)

    cut_mjj   = np.where(np.isnan(mjj_vbs),  False, mjj_vbs  > params["mjj_vbs"])
    cut_deta  = np.where(np.isnan(deta_vbs), False, deta_vbs > params["delta_eta_vbs"])

    mask = yes_fat & within & pt_cut & cut_mjj & cut_deta & loose_lep_veto & muon_ch & j1_pt_min & j2_pt_min & b_accept #& jet_dR_cut 
    return ak.values_astype(mask, np.bool_)


msd_window_cut_baccept_mu = Cut(
    name="msd_window_mu",
    params={"msd_w_window": 22.5,
            "mjj_vbs": 500.0,
            "delta_eta_vbs": 2.5}, 
    function=in_msd_window_fatjet_baccept_mu,
    )


############################################
##### MUON CHANNEL
###########################################

def in_whad_window_e(events, params, **kwargs):
    electron_ch = (events.nElectronGood == 1) & (events.nMuonGood == 0)
    four_j  = (events.nJetGood >= 4)
    no_fat = (events.nFatJetCandidate == 0)
    loose_lep_veto = (events.nLeptonVeto < 2)
    wjj_pt = ak.fill_none(ak.firsts(getattr(events.w_had_jets, "pt", None)), np.nan)
    wjj_pt_cut = np.where(np.isnan(wjj_pt),  False, wjj_pt  < 200.)
    wmass = ak.fill_none(ak.firsts(getattr(events.w_had_jets, "mass", None)), np.nan)
    within = np.where(np.isnan(wmass), False, np.abs(wmass - 85) < params["mjj_w_window"])
    # lead_lep_dR_cut1 = (events.lead_wlep_wjet1_dR > 0.8)
    # lead_lep_dR_cut2 = (events.lead_wlep_wjet2_dR > 0.8)
    j1  = ak.firsts(getattr(events.vbsjets, "jet1", None))
    j2  = ak.firsts(getattr(events.vbsjets, "jet2", None))

    j1_pt_min = (j1.pt > 50)
    j2_pt_min = (j2.pt > 30)

    mjj_vbs   = ak.fill_none(ak.firsts(getattr(events.vbsjets, "mass", None)), np.nan)
    deta_vbs  = ak.fill_none(ak.firsts(getattr(events.vbsjets, "delta_eta", None)), np.nan)

    cut_mjj   = np.where(np.isnan(mjj_vbs),  False, mjj_vbs  > params["mjj_vbs"])
    cut_deta  = np.where(np.isnan(deta_vbs), False, deta_vbs > params["delta_eta_vbs"])

    mask = four_j & within & wjj_pt_cut & cut_mjj & cut_deta & loose_lep_veto & no_fat & electron_ch & j1_pt_min &  j2_pt_min 
    return ak.values_astype(mask, np.bool_)

whad_window_cut_e = Cut(
    name="whad_window_e",
    params={"mjj_w_window": 20.0,
            "mjj_vbs": 500.0,
            "delta_eta_vbs": 2.5},  
    function=in_whad_window_e,
)

def in_whad_window_bveto_e(events, params, **kwargs):
    electron_ch = (events.nElectronGood == 1) & (events.nMuonGood == 0)
    four_j  = (events.nJetGood >= 4)
    b_veto = (events.nBJetLoose == 0) 
    no_fat = (events.nFatJetCandidate == 0)
    loose_lep_veto = (events.nLeptonVeto < 2)
    wjj_pt = ak.fill_none(ak.firsts(getattr(events.w_had_jets, "pt", None)), np.nan)
    wjj_pt_cut = np.where(np.isnan(wjj_pt),  False, wjj_pt  < 200.)
    wmass = ak.fill_none(ak.firsts(getattr(events.w_had_jets, "mass", None)), np.nan)
    within = np.where(np.isnan(wmass), False, np.abs(wmass - 85) < params["mjj_w_window"])
    # lead_lep_dR_cut1 = (events.lead_wlep_wjet1_dR > 0.8)
    # lead_lep_dR_cut2 = (events.lead_wlep_wjet2_dR > 0.8)
    j1  = ak.firsts(getattr(events.vbsjets, "jet1", None))
    j2  = ak.firsts(getattr(events.vbsjets, "jet2", None))

    j1_pt_min = (j1.pt > 50)
    j2_pt_min = (j2.pt > 30)

    mjj_vbs   = ak.fill_none(ak.firsts(getattr(events.vbsjets, "mass", None)), np.nan)
    deta_vbs  = ak.fill_none(ak.firsts(getattr(events.vbsjets, "delta_eta", None)), np.nan)

    cut_mjj   = np.where(np.isnan(mjj_vbs),  False, mjj_vbs  > params["mjj_vbs"])
    cut_deta  = np.where(np.isnan(deta_vbs), False, deta_vbs > params["delta_eta_vbs"])

    mask = four_j & b_veto & within & wjj_pt_cut & cut_mjj & cut_deta & loose_lep_veto & no_fat & b_veto & electron_ch & j1_pt_min & j2_pt_min 
    return ak.values_astype(mask, np.bool_)

whad_window_cut_bveto_e = Cut(
    name="whad_window_bveto_e",
    params={"mjj_w_window": 20.0,
            "mjj_vbs": 500.0,
            "delta_eta_vbs": 2.5},  
    function=in_whad_window_bveto_e,
)

def out_whad_window_bveto_e(events, params, **kwargs):
    electron_ch = (events.nElectronGood == 1) & (events.nMuonGood == 0)
    four_j  = (events.nJetGood >= 4)
    b_veto = (events.nBJetLoose == 0) 
    no_fat = (events.nFatJetCandidate == 0)
    loose_lep_veto = (events.nLeptonVeto < 2)
    wjj_pt = ak.fill_none(ak.firsts(getattr(events.w_had_jets, "pt", None)), np.nan)
    wjj_pt_cut = np.where(np.isnan(wjj_pt),  False, wjj_pt  < 200.)
    wmass = ak.fill_none(ak.firsts(getattr(events.w_had_jets, "mass", None)), np.nan)
    within = np.where(np.isnan(wmass), False, np.abs(wmass - 85) < params["mjj_w_window"])
    # lead_lep_dR_cut1 = (events.lead_wlep_wjet1_dR > 0.8)
    # lead_lep_dR_cut2 = (events.lead_wlep_wjet2_dR > 0.8)
    j1  = ak.firsts(getattr(events.vbsjets, "jet1", None))
    j2  = ak.firsts(getattr(events.vbsjets, "jet2", None))

    j1_pt_min = (j1.pt > 50)
    j2_pt_min = (j2.pt > 30)

    mjj_vbs   = ak.fill_none(ak.firsts(getattr(events.vbsjets, "mass", None)), np.nan)
    deta_vbs  = ak.fill_none(ak.firsts(getattr(events.vbsjets, "delta_eta", None)), np.nan)

    cut_mjj   = np.where(np.isnan(mjj_vbs),  False, mjj_vbs  > params["mjj_vbs"])
    cut_deta  = np.where(np.isnan(deta_vbs), False, deta_vbs > params["delta_eta_vbs"])

    mask = four_j & b_veto & ~within & wjj_pt_cut & cut_mjj & cut_deta & loose_lep_veto & no_fat & b_veto & electron_ch & j1_pt_min & j2_pt_min 
    return ak.values_astype(mask, np.bool_)

whad_windowinvert_cut_bveto_e = Cut(
    name="whad_window_bveto_e",
    params={"mjj_w_window": 20.0,
            "mjj_vbs": 500.0,
            "delta_eta_vbs": 2.5},  
    function=out_whad_window_bveto_e,
)


def in_whad_window_baccept_e(events, params, **kwargs):
    electron_ch = (events.nElectronGood == 1) & (events.nMuonGood == 0)
    four_j  = (events.nJetGood >= 4)
    b_accept = (events.nBJetTight > 0) 
    no_fat = (events.nFatJetCandidate == 0)
    loose_lep_veto = (events.nLeptonVeto < 2)
    wjj_pt = ak.fill_none(ak.firsts(getattr(events.w_had_jets, "pt", None)), np.nan)
    wjj_pt_cut = np.where(np.isnan(wjj_pt),  False, wjj_pt  < 200.)
    wmass = ak.fill_none(ak.firsts(getattr(events.w_had_jets, "mass", None)), np.nan)
    within = np.where(np.isnan(wmass), False, np.abs(wmass - 85) < params["mjj_w_window"])
    # lead_lep_dR_cut1 = (events.lead_wlep_wjet1_dR > 0.8)
    # lead_lep_dR_cut2 = (events.lead_wlep_wjet2_dR > 0.8)
    j1  = ak.firsts(getattr(events.vbsjets, "jet1", None))
    j2  = ak.firsts(getattr(events.vbsjets, "jet2", None))

    j1_pt_min = (j1.pt > 50)
    j2_pt_min = (j2.pt > 30)

    mjj_vbs   = ak.fill_none(ak.firsts(getattr(events.vbsjets, "mass", None)), np.nan)
    deta_vbs  = ak.fill_none(ak.firsts(getattr(events.vbsjets, "delta_eta", None)), np.nan)

    cut_mjj   = np.where(np.isnan(mjj_vbs),  False, mjj_vbs  > params["mjj_vbs"])
    cut_deta  = np.where(np.isnan(deta_vbs), False, deta_vbs > params["delta_eta_vbs"])

    mask = four_j & b_accept & within & wjj_pt_cut & cut_mjj & cut_deta & loose_lep_veto & no_fat & electron_ch & j1_pt_min & j2_pt_min 
    return ak.values_astype(mask, np.bool_)

whad_window_cut_baccept_e = Cut(
    name="whad_window_bveto_e",
    params={"mjj_w_window": 20.0,
            "mjj_vbs": 500.0,
            "delta_eta_vbs": 2.5},  
    function=in_whad_window_baccept_e,
)


def in_msd_window_fatjet_bveto_e(events, params, **kwargs):
    electron_ch = (events.nElectronGood == 1) & (events.nMuonGood == 0)
    #yes_fat = (events.nFatJetCentral >= 1)
    b_veto = (events.nBJetLoose == 0) 
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

    j1  = ak.firsts(getattr(events.vbsjets, "jet1", None))
    j2  = ak.firsts(getattr(events.vbsjets, "jet2", None))

    j1_pt_min = (j1.pt > 50)
    j2_pt_min = (j2.pt > 30)

    mjj_vbs   = ak.fill_none(ak.firsts(getattr(events.vbsjets, "mass", None)), np.nan)
    deta_vbs  = ak.fill_none(ak.firsts(getattr(events.vbsjets, "delta_eta", None)), np.nan)

    cut_mjj   = np.where(np.isnan(mjj_vbs),  False, mjj_vbs  > params["mjj_vbs"])
    cut_deta  = np.where(np.isnan(deta_vbs), False, deta_vbs > params["delta_eta_vbs"])

    mask = yes_fat & within & pt_cut & cut_mjj & cut_deta & loose_lep_veto & electron_ch & j1_pt_min & j2_pt_min & b_veto
    return ak.values_astype(mask, np.bool_)


msd_window_cut_bveto_e = Cut(
    name="msd_window_e",
    params={"msd_w_window": 22.5,
            "mjj_vbs": 500.0,
            "delta_eta_vbs": 2.5}, 
    function=in_msd_window_fatjet_bveto_e,
    )


def out_msd_window_fatjet_bveto_e(events, params, **kwargs):
    electron_ch = (events.nElectronGood == 1) & (events.nMuonGood == 0)
    #yes_fat = (events.nFatJetCentral >= 1)
    b_veto = (events.nBJetLoose == 0) 
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

    j1  = ak.firsts(getattr(events.vbsjets, "jet1", None))
    j2  = ak.firsts(getattr(events.vbsjets, "jet2", None))

    j1_pt_min = (j1.pt > 50)
    j2_pt_min = (j2.pt > 30)

    mjj_vbs   = ak.fill_none(ak.firsts(getattr(events.vbsjets, "mass", None)), np.nan)
    deta_vbs  = ak.fill_none(ak.firsts(getattr(events.vbsjets, "delta_eta", None)), np.nan)

    cut_mjj   = np.where(np.isnan(mjj_vbs),  False, mjj_vbs  > params["mjj_vbs"])
    cut_deta  = np.where(np.isnan(deta_vbs), False, deta_vbs > params["delta_eta_vbs"])

    mask = yes_fat & ~within & pt_cut & cut_mjj & cut_deta & loose_lep_veto & electron_ch & j1_pt_min & j2_pt_min & b_veto
    return ak.values_astype(mask, np.bool_)


msd_windowinvert_cut_bveto_e = Cut(
    name="msd_window_e",
    params={"msd_w_window": 22.5,
            "mjj_vbs": 500.0,
            "delta_eta_vbs": 2.5}, 
    function=out_msd_window_fatjet_bveto_e,
    )


def in_msd_window_fatjet_baccept_e(events, params, **kwargs):
    electron_ch = (events.nElectronGood == 1) & (events.nMuonGood == 0)
    #yes_fat = (events.nFatJetCentral >= 1)
    b_accept = (events.nBJetTight > 0) 
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

    j1  = ak.firsts(getattr(events.vbsjets, "jet1", None))
    j2  = ak.firsts(getattr(events.vbsjets, "jet2", None))

    j1_pt_min = (j1.pt > 50)
    j2_pt_min = (j2.pt > 30)


    mjj_vbs   = ak.fill_none(ak.firsts(getattr(events.vbsjets, "mass", None)), np.nan)
    deta_vbs  = ak.fill_none(ak.firsts(getattr(events.vbsjets, "delta_eta", None)), np.nan)

    cut_mjj   = np.where(np.isnan(mjj_vbs),  False, mjj_vbs  > params["mjj_vbs"])
    cut_deta  = np.where(np.isnan(deta_vbs), False, deta_vbs > params["delta_eta_vbs"])

    mask = yes_fat & within & pt_cut & cut_mjj & cut_deta & loose_lep_veto & electron_ch & j1_pt_min & j2_pt_min & b_accept
    return ak.values_astype(mask, np.bool_)


msd_window_cut_baccept_e = Cut(
    name="msd_window_e",
    params={"msd_w_window": 22.5,
            "mjj_vbs": 500.0,
            "delta_eta_vbs": 2.5}, 
    function=in_msd_window_fatjet_baccept_e,
    )


def get_JetVetoMap(name="JetVetoMaps"):
    return Cut(
        name=name, params={}, function=get_JetVetoMap_Mask
    )

def get_JetVetoMap_Mask(events, params, year, processor_params, sample, isMC, **kwargs):
    jets = events.Jet
    jets["jetId_corrected"] = compute_jetId(events, "Jet", processor_params, year)
    mask_for_VetoMap = (
        ((ak.values_astype(jets.jetId_corrected, "int64") & 2)==2) # Must fulfill tight jetId
        & (abs(jets.eta) < 5.19) # Must be within HCal acceptance
        & (jets.pt*(1-jets.muonSubtrFactor) > 15.) # May no be Muons misreconstructed as jets
        & ((jets["neEmEF"]+jets["chEmEF"])<0.9) # Energy fraction not dominated by ECal
    )
    jets = jets[mask_for_VetoMap]
    cset = correctionlib.CorrectionSet.from_file(
        processor_params.jet_scale_factors.vetomaps[year]["file"]
    )
    corr = cset[processor_params.jet_scale_factors.vetomaps[year]["name"]]
    etaFlat, phiFlat, etaCounts = ak.flatten(jets.eta), ak.flatten(jets.phi), ak.num(jets.eta)
    phiFlat = np.clip(phiFlat, -3.14159, 3.14159) # Needed since no overflow included in phi binning
    weight = ak.unflatten(
        corr.evaluate("jetvetomap", etaFlat, phiFlat),
        counts=etaCounts,
    )
    eventMask = ak.sum(weight, axis=-1)==0 # if at least one jet is vetoed, reject it event
    return ak.where(ak.is_none(eventMask), False, eventMask)






def get_GenJetVetoMap(name="GenJetVetoMaps"):
    return Cut(
        name=name, params={}, function=get_GenJetVetoMap_Mask
    )

def get_GenJetVetoMap_Mask(events, params, year, processor_params, sample, isMC, **kwargs):
    if not isMC: return np.ones(len(events), dtype=bool)
    jets = events.GenJet
    mask_for_VetoMap = (abs(jets.eta) < 5.19) # Must be within HCal acceptance
    jets = jets[mask_for_VetoMap]
    cset = correctionlib.CorrectionSet.from_file(
        processor_params.jet_scale_factors.vetomaps[year]["file"]
    )
    corr = cset[processor_params.jet_scale_factors.vetomaps[year]["name"]]
    etaFlat, phiFlat, etaCounts = ak.flatten(jets.eta), ak.flatten(jets.phi), ak.num(jets.eta)
    phiFlat = np.clip(phiFlat, -3.14159, 3.14159) # Needed since no overflow included in phi binning
    weight = ak.unflatten(
        corr.evaluate("jetvetomap", etaFlat, phiFlat),
        counts=etaCounts,
    )
    eventMask = ak.sum(weight, axis=-1)==0 # if at least one jet is vetoed, reject it event
    return ak.where(ak.is_none(eventMask), False, eventMask)
