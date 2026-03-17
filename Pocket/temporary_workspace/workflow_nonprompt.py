# workflow.py
import awkward as ak
import numpy as np
from pocket_coffea.workflows.base import BaseProcessorABC
from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.objects import lepton_selection, jet_selection, btagging, soft_lepton_selection
from types import SimpleNamespace
import vector
import math
import xgboost as xgb

vector.register_awkward()

def veto_jer_forward_unmatched(jets,variations):
    """
    Run 3 AK4 Puppi jets:
    Disable JER smearing for unmatched jets in 2.5 < |eta| < 3.0
    """

    abs_eta = np.abs(jets.eta)

    veto_pt = (
        (abs_eta > 2.5)
        & (abs_eta < 3.0)
        & (jets.genJetIdx < 0)
        & (jets.pt_sf_jer != 0)
    )

    veto_mass = (
        (abs_eta > 2.5)
        & (abs_eta < 3.0)
        & (jets.genJetIdx < 0)
        & (jets.mass_sf_jer != 0)
    )
    # Nominal
    jets["pt"] = ak.where(veto_pt, jets.pt/jets.pt_sf_jer, jets.pt)
    jets["mass"] = ak.where(veto_mass, jets.mass/jets.mass_sf_jer, jets.mass)
    # Systematics (important!)
    if "pt_JER_up" in jets.fields:
        jets["pt_JER_up"] = ak.where(veto_pt, jets.pt, jets.pt_JER_up)
    if "pt_jer_down" in jets.fields:
        jets["pt_JER_down"] = ak.where(veto_pt, jets.pt, jets.pt_JER_down)
    if "mass_JER_up" in jets.fields:
        jets["mass_JER_up"] = ak.where(veto_mass, jets.mass, jets.mass_JER_up)
    if "mass_JER_down" in jets.fields:
        jets["mass_JER_down"] = ak.where(veto_mass, jets.mass, jets.mass_JER_down)
    for var in jets.fields:
        if "pt_JES" in var and ("up" in var or "down" in var):
            jets[var] = ak.where(veto_pt, jets[var] / jets.pt_sf_jer, jets[var])
        if "mass_JES" in var and ("up" in var or "down" in var):
            jets[var] = ak.where(veto_mass, jets[var] / jets.mass_sf_jer, jets[var])

    return jets



def compute_MT(lep, met):
    return np.sqrt( #CHANGED mT DEFINITION TO USE PUPPIMET
            2.0 * lep.pt * met.pt * (1.0 - np.cos(lep.delta_phi(met)))
        )
class VBSSemileptonicProcessor(BaseProcessorABC):
    """
        - Build LeptonGood and JetGood (lepton-clean)
        - Identifies VBS tagging jets as the pair with the highest mjj
        - Reconstructs the hadronic W with two non-VBS jets that minimize |m-80.4|
        - Calculates auxiliary variables for histograms (mt, pt/eta, dR, etc.)
    """
    def process(self, events):
        # --- Fix buggy genWeight here ---
        dataset = events.metadata["dataset"]
        # if "WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8" in dataset:
        if hasattr(events, "genWeight"):
            events["genWeight"] = np.sign(events.genWeight)
       
        # IMPORTANT PATCH DUE TO BUGGY GEN WEIGHT IN UL GENWEIGHT nanoaodv9!
        self.events = events
        return super().process(events)

    def __init__(self, cfg: Configurator):
        super().__init__(cfg)

    # 1) object-level preselection
    def apply_object_preselection(self, variation):
        ev = self.events
        def _tau21(fj):
            t1 = ak.fill_none(getattr(fj, "tau1", None), np.nan)
            t2 = ak.fill_none(getattr(fj, "tau2", None), np.nan)
            return ak.where((t1 > 0) & np.isfinite(t1), t2 / t1, np.nan)
        ev["Electron", "etaSC"] = ev.Electron.eta + ev.Electron.deltaEtaSC

        # Good Leptons

        tight_criteria = SimpleNamespace(
            object_preselection = {
                "Muon": {
                    "pt": 30.0,
                    "eta": 2.4,
                    "id": "tightId",
                    "iso": 0.15,
                }
            }
        )

        ev["MuonGood_0"]     = lepton_selection(ev, "Muon", tight_criteria)
        mu = ev.MuonGood_0
        mask_muon_ip = (
            (np.abs(mu.dxy) < 0.02) & (np.abs(mu.eta) < 1.479) & (np.abs(mu.dz) < 0.1)
        ) | (
            (np.abs(mu.dxy) < 0.02) & (np.abs(mu.eta) >= 1.479) & (np.abs(mu.eta) < 2.4) & (np.abs(mu.dz) < 0.1)
        )


        ev["MuonGood"] = mu[mask_muon_ip]
        #ev["MuonGood"]      = ev.MuonGood_0[(np.abs(ev.MuonGood_0.dxy) < 0.2) & np.abs(ev.MuonGood_0.dz) < 0.5]
        ev["ElectronGood_0"] = lepton_selection(ev, "Electron", self.params)
        ele = ev.Electron#Good_0
        mask2 = (
            (np.abs(ele.dxy) < 0.05) & (np.abs(ele.eta) < 1.479) & (np.abs(ele.dz) < 0.1) #& (ele.cutBased >= 4) & (ele.lostHits <= 1)
        ) | (
            (np.abs(ele.dxy) < 0.1) & (np.abs(ele.eta) >= 1.479) & (np.abs(ele.eta) < 2.4) & (np.abs(ele.dz) < 0.2) #& (ele.cutBased >= 4) & (ele.lostHits <= 1)
        )

        ev["ElectronGood"] = ele[mask2 & (ele.pt > 38) & (ele.cutBased >= 3)]
        # ELECTRONS IN SIMPLE NAMEPSACE ARE NOT USED
        veto_criteria = SimpleNamespace(
            object_preselection = {
                "Muon": {
                    "pt": 10.0,
                    "eta": 2.4,
                    "id": "looseId",
                    "iso": 500.0,
                },
                "Electron": {
                    "pt": 10.0,
                    "eta": 2.5,
                    "id": "mvaFall17V2noIso_WPL",
                    "iso": 500.0,
                }
            }
        )

        medium_criteria = SimpleNamespace(
            object_preselection = {
                "Muon": {
                    "pt": 30.0,
                    "eta": 2.4,
                    "id": "mediumId",
                    "iso": 0.20,
                },
                "Electron": {
                    "pt": 38.0,
                    "eta": 2.5,
                    "id": "mvaFall17V2noIso_WP90",
                    "iso": 500.0,
                }
            }
        )
     
 

        loose_criteria = SimpleNamespace(
            object_preselection = {
                "Electron": {
                    "pt": 38.0,
                    "eta": 2.5,
                    "iso": 500,
                    "id": "mvaFall17V2noIso_WPL",
                },
                "Muon": {
                    "pt": 30.0,
                    "eta": 2.4,
                    "id": "looseId",
                    "iso": 500,
                }
            }
        )

        #inclufake_criteria = SimpleNamespace(
        #    object_preselection = {
        #        "Muon": {
        #            "pt": 26.0,
        #            "eta": 2.4,
        #            "id": "tightId",
        #            "iso": 100.0,
        #        }
        #    }
        #)

        #loose_inclufake_criteria = SimpleNamespace(
        #    object_preselection = {
        #        "Muon": {
        #            "pt": 26.0,
        #            "eta": 2.4,
        #            "id": "looseId",
        #            "iso": 100.0,
        #        }
        #    }
        #)


        #mva90_criteria = SimpleNamespace(
        #    object_preselection = {
        #        "Electron": {
        #            "pt": 35.0,
        #            "eta": 2.4,
        #            "id": "mvaIso_WP90",
        #            "iso": 0.15
        #        }
        #    }
        #)

        #mva80_criteria = SimpleNamespace(
        #    object_preselection = {
        #        "Electron": {
        #            "pt": 35.0,
        #            "eta": 2.4,
        #            "id": "mvaIso_WP80",
        #            "iso": 0.15
        #        }
        #    }
        #)


        # Good Leptons
        ev["MuonVeto"]     = lepton_selection(ev, "Muon", veto_criteria)
        ev["MuonMedium"]     = lepton_selection(ev, "Muon", medium_criteria)
        ev["MuonLoose"]     = lepton_selection(ev, "Muon", loose_criteria)
        mask4 = (
            (np.abs(ev.MuonLoose.dxy) < 0.02) & (np.abs(ev.MuonLoose.eta) < 1.479) & (np.abs(ev.MuonLoose.dz) < 0.1) & (np.abs(ev.MuonLoose.pt) > 20)
        ) | (
            (np.abs(ev.MuonLoose.dxy) < 0.02) & (np.abs(ev.MuonLoose.eta) >= 1.479) & (np.abs(ev.MuonLoose.eta) < 2.4) & (np.abs(ev.MuonLoose.dz) < 0.1) & (np.abs(ev.MuonLoose.pt) > 20)
        ) | (
            (np.abs(ev.MuonLoose.pt) <= 20) & (np.abs(ev.MuonLoose.dxy) < 0.01) & (np.abs(ev.MuonLoose.dz) < 0.1)
        )

        mask_4 = (
            (np.abs(ev.MuonMedium.dxy) < 0.02) & (np.abs(ev.MuonMedium.eta) < 1.479) & (np.abs(ev.MuonMedium.dz) < 0.1) & (np.abs(ev.MuonMedium.pt) > 20)
        ) | (
            (np.abs(ev.MuonMedium.dxy) < 0.02) & (np.abs(ev.MuonMedium.eta) >= 1.479) & (np.abs(ev.MuonMedium.eta) < 2.4) & (np.abs(ev.MuonMedium.dz) < 0.1) & (np.abs(ev.MuonMedium.pt) > 20)
        ) | (
            (np.abs(ev.MuonMedium.pt) <= 20) & (np.abs(ev.MuonMedium.dxy) < 0.01) & (np.abs(ev.MuonMedium.dz) < 0.1)
        )

        ev["MuonLoose"] = ev.MuonLoose[mask4]
        ev["MuonMedium"] = ev.MuonMedium[mask_4]

        ev["ElectronVeto"] = ev.Electron #lepton_selection(ev, "Electron", loose_criteria)

        mask3 = (
            (np.abs(ev.ElectronVeto.dxy) < 0.05) & (np.abs(ev.ElectronVeto.eta) < 1.479) & (np.abs(ev.ElectronVeto.dz) < 0.1) 
        ) | (
            (np.abs(ev.ElectronVeto.dxy) < 0.1) & (np.abs(ev.ElectronVeto.eta) >= 1.479) & (np.abs(ev.ElectronVeto.eta) < 2.4) & (np.abs(ev.ElectronVeto.dz) < 0.2) #& (ev.ElectronVeto.sieie < 0.03) & (ev.ElectronVeto.eInvMinusPInv < 0.014)
        )
        ev["ElectronVeto"] = ev.ElectronVeto[mask3 & (ev.ElectronVeto.cutBased >= 2) & (ev.ElectronVeto.pt >= 10)]
        
        ev["ElectronLoose"] = ev.Electron #lepton_selection(ev, "Electron", loose_criteria)

        mask3_0 = (
            (np.abs(ev.ElectronLoose.dxy) < 0.05) & (np.abs(ev.ElectronLoose.eta) < 1.479) & (np.abs(ev.ElectronLoose.dz) < 0.1) 
        ) | (
            (np.abs(ev.ElectronLoose.dxy) < 0.1) & (np.abs(ev.ElectronLoose.eta) >= 1.479) & (np.abs(ev.ElectronLoose.eta) < 2.4) & (np.abs(ev.ElectronLoose.dz) < 0.2) #& (ev.ElectronLoose.sieie < 0.03) & (ev.ElectronLoose.eInvMinusPInv < 0.014)
        )
        ev["ElectronLoose"] = ev.ElectronLoose[mask3_0 & (ev.ElectronLoose.cutBased >= 2) & (ev.ElectronLoose.pt >= 38)]
       
        ev["ElectronMedium"] = ev.Electron
        mask_3 = (
            (np.abs(ev.ElectronMedium.dxy) < 0.05) & (np.abs(ev.ElectronMedium.eta) < 1.479) & (np.abs(ev.ElectronMedium.dz) < 0.1) 
        ) | (
            (np.abs(ev.ElectronMedium.dxy) < 0.1) & (np.abs(ev.ElectronMedium.eta) >= 1.479) & (np.abs(ev.ElectronMedium.eta) < 2.4) & (np.abs(ev.ElectronMedium.dz) < 0.2) #& (ev.ElectronMedium.sieie < 0.03) & (ev.ElectronMedium.eInvMinusPInv < 0.014)
        )

        ev["ElectronMedium"] = ev.ElectronMedium[mask_3 & (ev.ElectronMedium.cutBased >= 2) & (ev.ElectronMedium.pt >= 38)]

        #ev["MuonIncluFake"] = lepton_selection(ev, "Muon", inclufake_criteria)
        #mask_inclufake_impact = (
        #    (np.abs(ev.MuonIncluFake_0.dxy) < 0.2) & (np.abs(ev.MuonIncluFake_0.dz) < 0.5)
        #)
        #ev["MuonIncluFake"] = ev.MuonIncluFake_0[mask_inclufake_impact]
        #ev["MuonLooseIncluFake"] = lepton_selection(ev, "Muon",  loose_inclufake_criteria)
        #mask_looseinclufake_impact = (
        #    (np.abs(ev.MuonLooseIncluFake_0.dxy) < 0.2) & (np.abs(ev.MuonLooseIncluFake_0.dz) < 0.5)
        #)
        #ev["MuonLooseIncluFake"] = ev.MuonLooseIncluFake_0[mask_looseinclufake_impact]
        #ev["ElectronLoose"] = lepton_selection(ev, "Electron", loose_criteria) 
        
        #mask_ele_loose = (ev.Electron.pt > 35) & (np.abs(ev.Electron.eta) < 2.5) & (ev.Electron.cutBased >=1 )
        #ev["ElectronLoose"] = ev.Electron[mask_ele_loose]
        #mask_ele_medium = (ev.Electron.pt > 38) & (np.abs(ev.Electron.eta) < 2.5) & (ev.Electron.cutBased >=2 )
        #ev["ElectronMedium"] = ev.Electron[mask_ele_medium]
        #mask_ele_veto = (ev.Electron.pt > 10) & (np.abs(ev.Electron.eta) < 2.5) & (ev.Electron.cutBased >=1 )
        #ev["ElectronVeto"]     = ev.Electron[mask_ele_veto]


        # veto_criteria_e = SimpleNamespace(
        #     object_preselection = {
        #         "Electron": {
        #             "pt": 10.0,
        #             "eta": 2.5,
        #             "id": "mvaFall17V2noIso_WPL",
        #             "iso": 500.0,
        #         }
        #     }
        # )

        

        # loose_criteria_e = SimpleNamespace(
        #     object_preselection = {
        #         "Electron": {
        #             "pt": 38.0,
        #             "eta": 2.5,
        #             "id": "mvaFall17V2noIso_WPL",
        #             "iso": 500.0,
        #         }
        #     }
        # )
        
        
        #ev["ElectronWP90"] = lepton_selection(ev, "Electron", mva90_criteria)
        #ev["ElectronWP80"] = lepton_selection(ev, "Electron", mva80_criteria)
        # mu_l=ev.MuonLoose_0
        # mask3 = (
        #     (np.abs(mu_l.dxy) < 0.2) & (np.abs(mu_l.eta) < 1.4) & (np.abs(mu_l.dz) < 0.5)
        # ) | (
        #     (np.abs(mu_l.dxy) < 0.2) & (np.abs(mu_l.eta) >= 1.4) & (np.abs(mu_l.eta) < 2.4) & (np.abs(mu_l.dz) < 0.5)
        # )
        # mask4 = (
        #     (np.abs(ele_l.dxy) < 0.05) & (np.abs(ele_l.eta) < 1.5) & (np.abs(ele_l.dz) < 0.1)
        # ) | (
        #     (np.abs(ele_l.dxy) < 0.1) & (np.abs(ele_l.eta) >= 1.5) & (np.abs(ele_l.eta) < 2.4) & (np.abs(ele_l.dz) < 0.2)
        # )
         
        # ev["MuonLoose"] = mu_l[mask3]
        # ev["ElectronLoose"] = ele_l[mask4]
        # Leptóons (mu+e) and ordered in pt
        leptons = ak.with_name(
            ak.concatenate([ev.MuonGood, ev.ElectronGood], axis=1),
            "PtEtaPhiMCandidate",
        )
        medium_lep = ak.with_name(
            ak.concatenate([ev.MuonMedium, ev.ElectronMedium], axis=1),
            "PtEtaPhiMCandidate",
        )
        loose_lep = ak.with_name(
            ak.concatenate([ev.MuonLoose, ev.ElectronLoose], axis=1),
            "PtEtaPhiMCandidate",
        )
        #lep_with_fakes = ak.with_name(
        #    ak.concatenate([ev.MuonLooseIncluFake, ev.ElectronLoose], axis=1),
        #    "PtEtaPhiMCandidate",
        #)
        ev["LeptonLoose"] = loose_lep[ak.argsort(loose_lep.pt, ascending=False)]
       # ev["LeptonWithFakes"] = lep_with_fakes[ak.argsort(lep_with_fakes.pt, ascending=False)]
        ev["LeptonGood"] = leptons[ak.argsort(leptons.pt, ascending=False)]

        lead_lep = ak.firsts(ev.LeptonGood)
        lead_lep_loose = ak.firsts(ev.LeptonLoose)
        #lead_lep_with_fakes = ak.firsts(ev.LeptonWithFakes)
        #lep_i = ak.fill_none(getattr(lead_lep, "jetIdx", None), -1)
        
        #print(ev.LeptonGood.fields)
        
        ev["JetGood_0"], _ = jet_selection(ev, "Jet", self.params, self._year,"LeptonLoose") #MAYBE THIS SHOULD BE LOOSE LEPTON
        # if self._isMC:
        #     for jet_type, jet_coll_name in self.params.jets_calibration.collection[self._year].items():
        #         if jet_coll_name == "Jet":
        #             JEC_type = jet_type
        #     ev.JetGood_1 = veto_jer_forward_unmatched(ev.JetGood_0,self.params.jets_calibration.variations[JEC_type][self._year])
        # else:
        #     ev.JetGood_1 = ev.JetGood_0
        ev.JetGood_1 = ev.JetGood_0
        mask_jet_cleaning = (ev.JetGood_1.pt>30) | (abs(ev.JetGood_1.eta)<2.5) #### LOTS OF RUN2 CUSTOMIZATIONS FOR NOW
        ev["JetGood"] = ev.JetGood_1[mask_jet_cleaning]
        ev["JetGoodCentral"] = ev.JetGood[abs(ev.JetGood.eta)<2.4]
        #ev["JetGood"] = ev.JetClean[ev.JetClean.pt > 30]
        #ev["JetGood"] = ev.Jet[(ev.Jet.jetId >= 6)&(ev.Jet.pt > 30)]
        #ev["JetGood", "idx"] = ak.local_index(ev.JetGood, axis=1)
    
        #TODO: jet_selection_nanoaodv12 only used for 2022, check other versions for other years.
        #ev["FatJetGood"], _ = jet_selection(ev,"FatJet", self.params, self._year, "LeptonGood")
        ev["FatJetGood"] = ev.FatJet
        ev["FatJetGood", "idx"] = ak.local_index(ev.FatJetGood, axis=1)
        dR_fatjets_lep = ev.FatJetGood.metric_table(ev.LeptonGood)
        mask_lepjet_cleaning = ak.prod(dR_fatjets_lep > 0.8, axis=2) == 1
        #separation = ak.fill_none(ev.etGood.metric_table(ev.candidate_boost), np.nan)
        #ev["separation"] = dR_jets_jet
        #ev["separation_after_cleaning"] = ak.fill_none(ev.JetGood[mask_jet_cleaning].metric_table(ev.candidate_boost), np.nan)

        # far_enough_from_ak8 = (separation > 0.8)
        ev["FatJetGood"] = ev.FatJetGood[mask_lepjet_cleaning]
        ev["FatJetGood", "idx"] = ak.local_index(ev.FatJetGood, axis=1)


        lead_cand_fj = ak.firsts(ev.FatJetGood)
        #fj_filter_tau21 = ( _tau21(lead_cand_fj) < 0.45 )
        ev["candidate_boost"] = ev.FatJetGood[(_tau21(ev.FatJetGood) < 0.45) & (ev.FatJetGood.msoftdrop < 250) & (ev.FatJetGood.pt > 180)]
        dR_jets_jet = ev.JetGood.metric_table(ev.candidate_boost)
        mask_jet_cleaning = ak.prod(dR_jets_jet > 0.8, axis=2) == 1
        separation = ak.fill_none(ev.JetGood.metric_table(ev.candidate_boost), np.nan)
        #ev["separation"] = dR_jets_jet
        #ev["separation_after_cleaning"] = ak.fill_none(ev.JetGood[mask_jet_cleaning].metric_table(ev.candidate_boost), np.nan)

        # far_enough_from_ak8 = (separation > 0.8)
        ev["JetGood"] = ev.JetGood[mask_jet_cleaning]
        ev["JetGood", "idx"] = ak.local_index(ev.JetGood, axis=1)
        # far_enough_from_ak8 = (ev.JetGood.delta_r(ev.candidate_boost) > 0.8)
        # far_enough_from_ak8 = ak.fill_none(far_enough_from_ak8, True)
        # ev["JetGood"] = ev.JetGood[far_enough_from_ak8]

        # The jets recoiled against leptons are for computing the fake rate, so the collection of loose leptons are used.
        dR_jets_lep = ev.JetGood.metric_table(ev.LeptonLoose)
        mask_lepjet4_cleaning = ak.prod(dR_jets_lep > 1, axis=2) == 1
        
        JetGood_recoil = ev.JetGood[mask_lepjet4_cleaning]
        ev["JetGood_recoil"] = JetGood_recoil[ak.argsort(JetGood_recoil.pt, ascending=False)]
        ev["JetGood_recoil", "idx"] = ak.local_index(ev.JetGood_recoil, axis=1)

        ev["LeadJetGood_recoil"] = ak.firsts(ev.JetGood_recoil)
        ev["nJetGood_recoil"] = ak.num(ev.JetGood_recoil)
        # b-tagging 
        #b_mask = (np.abs(ev.JetGood.eta) < 2.5) & (ev.JetGood.btagDeepB > 0.15)
        #b_mask = (np.abs(ev.JetGood.eta) < 2.5) & (ev.JetGood.btagDeepB > 0.1355)
        #ev["BJet_csv"] = ev.JetGood[b_maskT]
        #ev["BJet_csv"] = ev.JetGood[b_mask]
        ev["BJetTight"] = btagging(
            ev.JetGood[np.abs(ev.JetGood.eta) < 2.5],
            self.params.btagging.working_point[self._year],
            wp="H",
        )
        ev["BJetLoose"] = btagging(
            ev.JetGood[np.abs(ev.JetGood.eta) < 2.5],
            self.params.btagging.working_point[self._year],
            wp=self.params.object_preselection.Jet.btag.wp,
        )
        #ev["BJet_genmatch"] =ev.JetGood[(np.abs(ev.JetGood.eta) < 2.5) & (np.abs(ev.JetGood.partonFlavour) == 5)]
        ev["JetGood_tagger_check"]= ev.JetGood[(np.abs(ev.JetGood.eta) < 2.5)]

        #blah = ev.JetGood_tagger_check[ak.argsort(ev.JetGood_tagger_check.btagDeepB, ascending=False)]
        ev['leading_bscore'] = ak.max(ev.JetGood_tagger_check.btagDeepFlavB, axis=1)
        #ev['nCleanJet_30'] = ak.num(ev.JetGood.pt >= 30)
        # ------------- VBS tagging jets -------------
        has4j = ak.num(ev.JetGood) >= 4
        has2j = (ak.num(ev.JetGood) >= 2) #& (ev.JetGood.idx != lep_i) #keep it at 3 st we can separate fj vs ak4 jets!
        #hasfatjet = (ak.num(ev.FatJetGood) >=1) & has2j
        has2l = ak.num(ev.LeptonGood) == 2
        jj = ak.combinations(ev.JetGood, 2, fields=["jet1", "jet2"])
        jj["mass"] = (jj.jet1 + jj.jet2).mass

        idx_vbs = ak.argmax(jj.mass, axis=1, keepdims=True)
        
        ev["vbsjets"] = ak.mask(jj[idx_vbs], has2j)
        #print("vbsjet fields ",ev.vbsjets_initial.fields)
        #ev["vbsjets"] = ev.vbsjets_initial[ev.vbsjets_initial.JetIdx >= 6]
       
        v1 = ak.firsts(ev.vbsjets.jet1)
        v2 = ak.firsts(ev.vbsjets.jet2)

        # deta and dR btw tagging jets
        ev["vbsjets", "delta_eta"] = np.abs(v1.eta - v2.eta)
        ev["vbs_dR"] = ak.fill_none(v1.delta_r(v2), np.nan)


        ##### NOW REPEAT VBS ID BUT NEED SOME BOOST CATEGORIZATION #####

        lead_cand_fj = ak.firsts(ev.candidate_boost)
        #fj_filter_tau21 = ( _tau21(lead_cand_fj) < 0.45 )
        #ev["candidate_boost"] = ak.mask(lead_cand_fj, fj_filter_tau21)
        far_enough_from_ak8 = ev.JetGood.delta_r(lead_cand_fj)

        #jet_fatjet_pairs = ak.combinations([ev.JetGood, ev.FatJetGood], 2, fields=["jet", "fatjet"])
        #ev["far_enough_from_ak8"] = jet_fatjet_pairs["jet"].deltaR(jet_fatjet_pairs["fatjet"])
        #ev["far_enough_from_ak8"] = ev.JetGood.delta_r(ev.FatJetGood)
        #ev["far_enough_from_ak8"] = ev.FatJetGood.metric_table(ev.JetGood)
        allowed_ak4_boost = ev.JetGood[far_enough_from_ak8 > 0.8]
        ev["nFarAK4Jets"] = ak.num(allowed_ak4_boost) # MAYBE NEED TO GET RID OF THIS
        jj_boost = ak.combinations(ev.JetGood, 2, fields=["jet1", "jet2"])
        jj_boost["mass"] = (jj_boost.jet1 + jj_boost.jet2).mass

        idx_vbs_boost = ak.argmax(jj_boost.mass, axis=1, keepdims=True)
        
        ev["vbsjets_boost"] = ak.mask(jj_boost[idx_vbs_boost], has2j)

        v1b = ak.firsts(ev.vbsjets_boost.jet1)
        v2b = ak.firsts(ev.vbsjets_boost.jet2)

        # deta and dR btw tagging jets
        ev["vbsjets_boost", "delta_eta"] = np.abs(v1b.eta - v2b.eta)
        ev["vbs_boost_dR"] = ak.fill_none(v1b.delta_r(v2b), np.nan)
        ## lead lep yadda yadda

        #print("lep_i: ", lep_i)
        # ------ Boosted jet -------------
        

        # Apply mask to get FatJetGood for central events only
        j1b_eta = ak.fill_none(getattr(v1b, "eta", None), np.nan)
        j2b_eta = ak.fill_none(getattr(v2b, "eta", None), np.nan)
        eta_minb = np.minimum(j1b_eta, j2b_eta)
        eta_maxb = np.maximum(j1b_eta, j2b_eta)

        fj_eta = ak.fill_none(getattr(ev.candidate_boost, "eta", None), np.nan)

        
        fj_idx = ak.fill_none(getattr(ev.candidate_boost, "idx", None), -999)

        # Broadcast lep_idx to the shape of FatJetGood
        #lep_idx_broadcasted = ak.broadcast_arrays(fj_idx, lep_i)[1]


        central_mask = ( (~np.isnan(fj_eta)) & (~np.isnan(eta_minb)) & (~np.isnan(eta_maxb)) & (fj_eta > eta_minb) & (fj_eta < eta_maxb))
        ev["candidate_boost"] = ev.candidate_boost[central_mask]
        fjc = ev.candidate_boost[ak.argsort(ev.candidate_boost.pt, ascending=False)]
        fjc_0 = ev.candidate_boost[ak.argsort(ev.candidate_boost.pt, ascending=False)]
        
        ev["nFatJetCentral"] = ak.num(fjc)

        
        fj_candidates = ( _tau21(fjc) < 0.45 ) # PICK CANDIDATES FOR V
        # ev["w_fatjet"] = ev.candidate_boost[fj_candidates]
        #ev['nFatJet_resolved'] = ak.num(ev.FatJetGood[_tau21(fjc_0) < 0.45])
        ev["nFatJetCandidate"] = ak.num(ev.candidate_boost)
        ev["candidate_boost" ,"tau21"] = _tau21(ev.candidate_boost)
        #print("KEYS: ", ev.w_fatjet.fields)
        # print(ev.FatJetCentral.phi, "FAT JET PHI")
        # print(ev.w_fatjet.phi, "FAT JET CANDIDATE PHI")
        fj1 = ak.firsts(ev.candidate_boost)
        ev["vbs1_fj_dR"] = ak.fill_none(v1b.delta_r(fj1), np.nan)
        ev["vbs2_fj_dR"] = ak.fill_none(v2b.delta_r(fj1), np.nan)
       
        # ------------- W hadronic (resolved) -------------
        vbs_i = ak.fill_none(getattr(v1, "idx", None), -1)
        vbs_j = ak.fill_none(getattr(v2, "idx", None), -1)
        
        def delta_phi(phi1, phi2):
            dphi = phi1 - phi2
            return (dphi + np.pi) % (2 * np.pi) - np.pi


        def custom_dR(j1,j2):
            dphi=delta_phi(j1.phi,j2.phi)
            deta= (j1.eta - j2.eta)
            dR = np.sqrt(dphi**2 + deta **2)
            return dR

        nonvbs_mask = (ev.JetGood.idx != vbs_i) & (ev.JetGood.idx != vbs_j) #& (ev.JetGood.idx != lep_i) #see if can better clean out dR tail at 0
        ev["CentralJets"] = ev.JetGood[nonvbs_mask]
        
        ev['CentralJetsGood']= ev.CentralJets[np.abs(ev.CentralJets.eta) < 2.4]
      
        fj_eta = ak.fill_none(ak.firsts(ev.candidate_boost.eta), np.nan)
        fj_phi = ak.fill_none(ak.firsts(ev.candidate_boost.phi), np.nan)

        cj_eta = ev.CentralJetsGood.eta
        cj_phi = ev.CentralJetsGood.phi

        ## EVALUTE B JET DISTANCE
        cj = ak.zip({
            "pt": ev.BJetLoose.pt,
            "eta": ev.BJetLoose.eta,
            "phi": ev.BJetLoose.phi,
            "mass": ev.BJetLoose.mass,
        }, with_name="Momentum4D")

        fj = ak.zip({
            "pt": lead_lep.pt,
            "eta": lead_lep.eta,
            "phi": lead_lep.phi,
            "mass": lead_lep.mass,
        }, with_name="Momentum4D")

        # Now they have matching keys: {"eta", "phi"}
        # Broadcasting will work
        
        fj_b = ak.broadcast_arrays(fj, cj)[0]
        #print("something dR", fj_b)
        #ev["bjet_lepton_separation"] = ak.fill_none(lead_lep.delta_r(ak.firsts(ev.BJetGood)), np.nan)
        dr = custom_dR(cj,fj_b)
        #has_no_fatjet = (ev.nFatJetCandidate == 0)  
        #ev["CentralJetsGood"] = ev.CentralJetsOverlay[has_no_fatjet]
        #ev["CentralJetGoodBoostedFS"] = ev.CentralJetsGood[dr > 0.8]
    
        pairs_w = ak.combinations(ev.CentralJetsGood, 2, fields=["jet1", "jet2"])
        pairs_w["mass"] = (pairs_w.jet1 + pairs_w.jet2).mass
        pairs_w["deta"] = (pairs_w.jet1 - pairs_w.jet2).eta


        target_mw = 85
        best_w_idx = ak.argmin(np.abs(pairs_w.mass - target_mw), axis=1, keepdims=True) #ADD EXTRA CUT FOR ETA MIN TOO /target_mw + np.abs(pairs_w.deta) ???

        ev["w_had_jets"] = ak.mask(pairs_w[best_w_idx], has4j)
        ev["w_had_jets", "mass"] = (ev.w_had_jets.jet1 + ev.w_had_jets.jet2).mass
        ev["w_had_jets", "pt"] = (ev.w_had_jets.jet1 + ev.w_had_jets.jet2).pt
        ev["w_had_jets", "eta"] = (ev.w_had_jets.jet1 + ev.w_had_jets.jet2).eta
        ev["w_had_jets", "phi"] = (ev.w_had_jets.jet1 + ev.w_had_jets.jet2).phi

        # dR btw the two jets for W_had
        wj1 = ak.firsts(ev.w_had_jets.jet1)
        wj2 = ak.firsts(ev.w_had_jets.jet2)

        ev["w_had_jet1_pt"]  = ak.fill_none(wj1.pt, np.nan)
        ev["w_had_jet2_pt"]  = ak.fill_none(wj2.pt, np.nan)
        ev["w_had_jet1_eta"] = ak.fill_none(wj1.eta, np.nan)
        ev["w_had_jet2_eta"] = ak.fill_none(wj2.eta, np.nan)
        ev["w_had_jet1_phi"] = ak.fill_none(wj1.phi, np.nan)
        ev["w_had_jet2_phi"] = ak.fill_none(wj2.phi, np.nan)
        ev["w_had_dR"] = ak.fill_none(wj1.delta_r(wj2), np.nan)

        # ------------- W Leptonic -------------
        #lead_lep = ak.firsts(ev.LeptonGood)
        ev["mt_w_leptonic"] = compute_MT(lead_lep, ev.PuppiMET)
        ev["mt_w_leptonic_deepMET_resolutiontune"] = compute_MT(lead_lep, ev.DeepMETResolutionTune)
        ev["mt_w_leptonic_deepMET_responsetune"] = compute_MT(lead_lep, ev.DeepMETResponseTune)
        ev["mt_w_leptonic_loose"] = compute_MT(lead_lep_loose, ev.PuppiMET)
        ev["mt_w_leptonic_deepMET_resolutiontune_loose"] = compute_MT(lead_lep_loose, ev.DeepMETResolutionTune)
        ev["mt_w_leptonic_deepMET_responsetune_loose"] = compute_MT(lead_lep_loose, ev.DeepMETResponseTune)
        w_lep = ev.PuppiMET + lead_lep
        whad = ev.w_had_jets.jet1 + ev.w_had_jets.jet2
        # print("w leptonic pT: ", w_lep.pt)

        # print(ev.LeptonGood.jetIdx, " LEP JET IDX")
        # print(ev.w_had_jets.jet1.idx, " W JET IDX")
        
        wfj = ak.firsts(ev.candidate_boost)
        #bj = ak.firsts(ev.BJet_csv)

        #lead_lep1 = lead_lep[np.abs(lead_lep.eta) < 2.5]
        #bjets = ev["BJet_csv"]

        # ΔR between lead_lep and each b-jet
        #deltaR_b = lead_lep.delta_r(bjets)
        # print("testing testing")
        # print("B VS LEP ETC: ")
        
        #deltaR_b_clean = ak.where(np.isnan(deltaR_b), -1, deltaR_b)

        # Compute min, with np.inf as fallback for empty sublists
        # min_deltaR = ak.min(deltaR_b_clean, axis=1, initial=-1)
        # print(min_deltaR)
        #badjet = ak.firsts(ev.Jet)
        # dEta, dR between lead lepton and [fat jet, resolved w jets, hadronic W]
        ev["lead_wlep_wfatjet1_dR"] = ak.fill_none(lead_lep.delta_r(wfj), np.nan)
        ev["lead_wlep_wfatjet1_deta"] = np.abs(lead_lep.eta - wfj.eta)
        ev["lead_wlep_wjet1_dR"] = ak.fill_none(lead_lep.delta_r(wj1), np.nan)
        ev["lead_wlep_wjet2_dR"] = ak.fill_none(lead_lep.delta_r(wj2), np.nan)
        ev["lead_wlep_wjet1_deta"] = np.abs(lead_lep.eta - wj1.eta)
        ev["lead_wlep_wjet2_deta"] = np.abs(lead_lep.eta - wj2.eta)
        ev["lead_wlep_w_resolved_dR"] = ak.fill_none(lead_lep.delta_r(whad), np.nan)
        ev["lead_wlep_w_resolved_deta"] = np.abs(lead_lep.eta - whad.eta)


        deltaR = lead_lep.metric_table(ev["BJetLoose"])

        # Flatten last two axes to get all lep-bjet pairs per event (usually just n_bjets per event)
        deltaR_per_event = ak.flatten(deltaR, axis=2)
        deltaR_per_event = ak.fill_none(deltaR_per_event, np.nan)
        # Flatten over events to get a 1D array of all deltaR values
        #deltaR_all = ak.flatten(deltaR_per_event)

        # Drop any NaNs (if any)
        #deltaR_clean = deltaR_all[~ak.is_none(deltaR_all)]
        ev["lep_bjet_dR"] = deltaR_per_event
        #ev["lead_wlep_badjet_dR"] = ak.fill_none(lead_lep.delta_r(badjet), np.nan)
       
        #ev["lead_wlep_MET_dR"] = ak.fill_none(lead_lep.delta_r(ev.PuppiMET), np.nan)
        #ev["lead_wlep_MET_deta"] = np.abs(lead_lep.eta - ev.PuppiMET.eta)

        #dPhi between lead lepton and MET
        ev["lead_wlep_MET_dphi"] = delta_phi(lead_lep.phi, ev.PuppiMET.phi)
        ev["lead_wlep_wfatjet1_dphi"] = delta_phi(lead_lep.phi, wfj.phi)
        ev["lead_wlep_wjet1_dphi"] = delta_phi(lead_lep.phi, wj1.phi)
        ev["lead_wlep_wjet2_dphi"] = delta_phi(lead_lep.phi, wj2.phi)
        #dPhi between lep w and had w (boost,resolved)
        ev["w_lep_w_resolved_dphi"] = delta_phi(w_lep.phi, whad.phi)
        ev["w_lep_w_boost_dphi"] = delta_phi(w_lep.phi, wfj.phi)

        #dEta, dR between lead lepton and vbs jets
        ev["lead_wlep_vbsjet1_dR"] = ak.fill_none(lead_lep.delta_r(v1), np.nan)
        ev["lead_wlep_vbsjet2_dR"] = ak.fill_none(lead_lep.delta_r(v2), np.nan)
        ev["lead_wlep_vbsjet1_deta"] = np.abs(lead_lep.eta - v1.eta)
        ev["lead_wlep_vbsjet2_deta"] = np.abs(lead_lep.eta - v2.eta)
        
        ev["lead_wlep_vbsjet1_dR_boost"] = ak.fill_none(lead_lep.delta_r(v1b), np.nan)
        ev["lead_wlep_vbsjet2_dR_boost"] = ak.fill_none(lead_lep.delta_r(v2b), np.nan)
        ev["lead_wlep_vbsjet1_deta_boost"] = np.abs(lead_lep.eta - v1b.eta)
        ev["lead_wlep_vbsjet2_deta_boost"] = np.abs(lead_lep.eta - v2b.eta)
        

        # dEta, dR between leptonic W and hadronic W (boosted and resolved)
        #ev["w_lep_w_resolved_dR"] = ak.fill_none(w_lep.delta_r(whad), np.nan)
        #ev["w_lep_w_boost_dR"] = ak.fill_none(w_lep.delta_r(fj1), np.nan)
        #ev["w_lep_w_resolved_deta"] = np.abs(w_lep.eta - whad.eta)
        #ev["w_lep_w_boost_deta"] = np.abs(w_lep.eta - fj1.eta)



        ############ mll check
        ll = ak.combinations(ev.LeptonGood, 2, fields=["lep1", "lep2"])
        ll["m_ll"] = (ll.lep1 + ll.lep2).mass

        idx_ll = ak.argmax(ll.m_ll, axis=1, keepdims=True)
        ev["ll"] = ak.mask(ll[idx_ll], has2l)
        ##############

        ev["w_lep_pt"]  = ak.fill_none(lead_lep.pt, np.nan)
        ev["w_lep_eta"] = ak.fill_none(lead_lep.eta, np.nan)
        ev["w_lep_phi"] = ak.fill_none(lead_lep.phi, np.nan)

        jets_sorted = ev.JetGood[ak.argsort(ev.JetGood.pt, ascending=False)]
        ev["jet1_pt"]  = ak.firsts(getattr(jets_sorted[:, 0:1], "pt", None))
        ev["jet2_pt"]  = ak.firsts(getattr(jets_sorted[:, 1:2], "pt", None))
        ev["jet1_eta"] = ak.firsts(getattr(jets_sorted[:, 0:1], "eta", None))
        ev["jet2_eta"] = ak.firsts(getattr(jets_sorted[:, 1:2], "eta", None))
        ev["jet1_phi"] = ak.firsts(getattr(jets_sorted[:, 0:1], "phi", None))
        ev["jet2_phi"] = ak.firsts(getattr(jets_sorted[:, 1:2], "phi", None))
        ev["jet1_idx"]  = ak.firsts(getattr(jets_sorted[:, 0:1], "idx", None))
        ev["jet2_idx"]  = ak.firsts(getattr(jets_sorted[:, 0:1], "idx", None))

        ev["vbsjet1_pt"]  = ak.fill_none(v1.pt,np.nan)
        ev["vbsjet2_pt"]  =ak.fill_none(v2.pt,np.nan)
        ev["vbsjet1_eta"] = ak.fill_none(v1.eta,np.nan)
        ev["vbsjet2_eta"] = ak.fill_none(v2.eta,np.nan)
        ev["vbsjet1_phi"] = ak.fill_none(v1.phi,np.nan)
        ev["vbsjet2_phi"] = ak.fill_none(v2.phi,np.nan)
        ev["vbsjet1_qgl"] = ak.fill_none(v1.qgl,np.nan)
        ev["vbsjet2_qgl"] = ak.fill_none(v2.qgl,np.nan)
        #ev["vbsjet1_RobustParTAK4QG"] =  ak.fill_none(v1.btagRobustParTAK4QG,np.nan) #TODO: update name at NANOAODv15
        #ev["vbsjet2_RobustParTAK4QG"] =  ak.fill_none(v2.btagRobustParTAK4QG,np.nan) #TODO: update name at NANOAODv15
        ev["vbsjet1_DeepFlavQG"] = ak.fill_none(v1.btagDeepFlavQG,np.nan)
        ev["vbsjet2_DeepFlavQG"] = ak.fill_none(v2.btagDeepFlavQG,np.nan)


        ev["vbsjet1_pt_boosted"]  = ak.fill_none(v1b.pt,np.nan)
        ev["vbsjet2_pt_boosted"]  =ak.fill_none(v2b.pt,np.nan)
        ev["vbsjet1_eta_boosted"] = ak.fill_none(v1b.eta,np.nan)
        ev["vbsjet2_eta_boosted"] = ak.fill_none(v2b.eta,np.nan)
        ev["vbsjet1_phi_boosted"] = ak.fill_none(v1b.phi,np.nan)
        ev["vbsjet2_phi_boosted"] = ak.fill_none(v2b.phi,np.nan)
        ev["vbsjet1_qgl_boosted"] = ak.fill_none(v1b.qgl,np.nan)
        ev["vbsjet2_qgl_boosted"] = ak.fill_none(v2b.qgl,np.nan)
        #ev["vbsjet1_RobustParTAK4QG_boosted"] =  ak.fill_none(v1b.btagRobustParTAK4QG,np.nan) #TODO: update name at NANOAODv15
        #ev["vbsjet2_RobustParTAK4QG_boosted"] =  ak.fill_none(v2b.btagRobustParTAK4QG,np.nan)#TODO: update name at NANOAODv15
        ev["vbsjet1_DeepFlavQG_boosted"] = ak.fill_none(v1b.btagDeepFlavQG,np.nan)
        ev["vbsjet2_DeepFlavQG_boosted"] = ak.fill_none(v2b.btagDeepFlavQG,np.nan)

        # Zeppenfeld variables (basically eta significance of V decay to VBS jet)
        def zeppenfeld(target, vbs_jet1, vbs_jet2):
            mask_valid = (
                ~ak.is_none(target) &
                ~ak.is_none(vbs_jet1) &
                ~ak.is_none(vbs_jet2)
            )
            target_eta = ak.where(mask_valid, target.eta, np.nan)
            vbs1_eta = ak.where(mask_valid, vbs_jet1.eta, np.nan)
            vbs2_eta = ak.where(mask_valid, vbs_jet2.eta, np.nan)
            numerator = target_eta - (vbs1_eta + vbs2_eta)/2
            denominator = np.abs(vbs1_eta - vbs2_eta)
            zep = numerator / denominator
            return zep
        # def zeppenfeld(target, vbs_jet1, vbs_jet2, epsilon=1e-5):
        #     target_eta = ak.fill_none(getattr(target, "eta", None), np.nan)
        #     vbs1_eta = ak.fill_none(getattr(vbs_jet1, "eta", None), np.nan)
        #     vbs2_eta = ak.fill_none(getattr(vbs_jet2, "eta", None), np.nan)

        #     mid = 0.5 * (vbs1_eta + vbs2_eta)
        #     gap = np.abs(vbs1_eta - vbs2_eta)

        #     # Only compute if gap > epsilon and all three are finite numbers (no NaN or Inf)
        #     valid = (gap > epsilon) & np.isfinite(target_eta) & np.isfinite(vbs1_eta) & np.isfinite(vbs2_eta)

        #     return ak.where(valid, (target_eta - mid) / gap, np.nan)
        
        ev['z_lep'] = ak.fill_none(zeppenfeld(lead_lep, v1,v2),np.nan)
        ev['z_fat'] = ak.fill_none(zeppenfeld(wfj, v1b,v2b),np.nan)

        #print(ev.z_lep, "zeppenfeld")
        #print(ev.z_fat, "zeppen boost")

        # zep_vals = ak.to_numpy(ak.flatten(ev.z_lep))
        # print("Min:", np.nanmin(ev.z_lep))
        # print("Max:", np.nanmax(ev.z_lep))
        # print("Total entries:", len(ev.z_lep))
        # print("Outside ±0.5:", np.sum(np.abs(ev.z_lep) > 0.5))


        def solve_neutrino_pz(lep, nu):
            m_w = 80.36
            A = m_w**2 - lep.mass**2
            delta_phi = lep.phi - nu.phi
            C = 0.5 * A + lep.pt * nu.pt * np.cos(delta_phi)
            D = lep.pz

            a = (lep.mass**2 + lep.pt**2 + lep.pz**2) - D**2
            b = -2 * C * D
            c = (lep.mass**2 + lep.pt**2 + lep.pz**2) * nu.pt**2 - C**2

            discriminant = b**2 - 4 * a * c

            a_zero_mask = abs(a) < 1e-12
            b_nonzero_mask = abs(b) > 1e-12
            disc_neg_mask = discriminant < 0

            pz_a0 = ak.where(b_nonzero_mask, -c / b, 0.0)
            pz_no_real = -b / (2 * a)
            sqrt_disc = ak.where(disc_neg_mask, 0.0, np.sqrt(discriminant))

            pz1 = (-b + sqrt_disc) / (2 * a)
            pz2 = (-b - sqrt_disc) / (2 * a)

            best_pz = ak.where(abs(pz1) < abs(pz2), pz1, pz2)

            result = ak.where(
                a_zero_mask,
                pz_a0,
                ak.where(disc_neg_mask, pz_no_real, best_pz)
            )

            return result

                
        
        def centrality(w_lep_eta,v_had,vbs1,vbs2):
            eta_plus= np.maximum(vbs1.eta,vbs2.eta) - np.maximum(w_lep_eta,v_had.eta)
            eta_minus=np.minimum(v_had.eta,w_lep_eta) - np.minimum(vbs1.eta,vbs2.eta)
            C = np.minimum(eta_plus, eta_minus)
            return C
        
        ev['neutrino_pz'] = ak.fill_none(solve_neutrino_pz(lead_lep, ev.PuppiMET),np.nan)
        ev['neutrino_eta'] = ak.fill_none(np.arcsinh(ev.neutrino_pz / ev.PuppiMET.pt),np.nan)
        ev['wleptonic_eta'] = ak.fill_none(np.arcsinh((ev.neutrino_pz+lead_lep.pz)/(w_lep.pt)),np.nan)
        ev['wleptonic_pt'] = ak.fill_none(w_lep.pt,np.nan)
        ev['wleptonic_phi'] = ak.fill_none(w_lep.phi,np.nan)
        ev['w_had_jets','centrality_resolved'] = ak.fill_none(centrality(ev.wleptonic_eta, whad,v1,v2),np.nan)
        ev['centrality_boosted'] = ak.fill_none(centrality(ev.wleptonic_eta,wfj,v1b,v2b),np.nan)

        #ev['qgl_vbs1_resolved'] = ak.fill_none(v1.qgl,np.nan)
        #ev['qgl_vbs2_resolved'] = ak.fill_none(v2.qgl,np.nan)

        #ev['qgl_vbs1_boost'] = ak.fill_none(v1b.qgl,np.nan)
        #ev['qgl_vbs2_boost'] = ak.fill_none(v2b.qgl,np.nan)

        ev["w_had_jet1_resolved_qgl"] = ak.fill_none(ev.w_had_jets.jet1.qgl,np.nan)
        ev["w_had_jet2_resolved_qgl"] = ak.fill_none(ev.w_had_jets.jet2.qgl,np.nan)
        #ev["w_had_jet1_resolved_RobustParTAK4QG"] = ak.fill_none(ev.w_had_jets.jet1.btagRobustParTAK4QG,np.nan) #TODO: update name at NANOAODv15
        #ev["w_had_jet2_resolved_RobustParTAK4QG"] = ak.fill_none(ev.w_had_jets.jet2.btagRobustParTAK4QG,np.nan)#TODO: update name at NANOAODv15
        ev["w_had_jet1_resolved_DeepFlavQG"] = ak.fill_none(ev.w_had_jets.jet1.btagDeepFlavQG,np.nan)
        ev["w_had_jet2_resolved_DeepFlavQG"] = ak.fill_none(ev.w_had_jets.jet2.btagDeepFlavQG,np.nan)
        #ev['qgl_wjet1_resolved'] = ak.fill_none(ev.w_had_jets.jet1.qgl,np.nan)
        #ev['qgl_wjet2_resolved'] = ak.fill_none(ev.w_had_jets.jet2.qgl,np.nan)


        ev["ht_sum"] = ak.sum(ev.Jet.pt, axis=1)
        if self._isMC:
            dress_lep = ak.firsts(ev.GenDressedLepton)
            gen_met = ev.GenMET
            ev["gen_HT"] = ak.fill_none(ev.LHE.HT,np.nan)
            ev["gen_w_pt_dressed"] = (dress_lep + gen_met).pt
            w_pt_dressed = ak.firsts(ev.gen_w_pt_dressed, axis=-1)
            w_pt_direct = ak.firsts(ev.GenPart[abs(ev.GenPart.pdgId) == 24].pt, axis=-1)
            ev["gen_w_pt_by_pdg"] = ak.fill_none(w_pt_direct, w_pt_dressed)
        

        #genJetIdx_nested = ev.Jet.genJetIdx

        # # Replace empty lists with [-1] (meaning: no match)
        # genJetIdx_fixed = ak.fill_none(
        #     ak.firsts(genJetIdx_nested, axis=1), 
        #     -1
        # )

        # # Optional: debug print
        # print("Fixed genJetIdx:", genJetIdx_fixed)

        # # Create mask for valid matches
        # valid_mask = (genJetIdx_fixed >= 0) & (genJetIdx_fixed < ak.num(ev.GenJet))

        # # Safe indexing
        # safe_genJetIdx = ak.where(valid_mask, genJetIdx_fixed, -1)
        # matched_genjets = ak.where(valid_mask, ev.GenJet[safe_genJetIdx], None)

        # Save result to events
        #ev['matched_gen_to_b'] = ev.GenJet[genJetIdx_nested]

        #ev['qgl_fatjet'] = ak.fill_none(wfj.qgl,np.nan)
        

    def count_objects(self, variation):
        ev = self.events
        ev["nMuonGood"]     = ak.num(ev.MuonGood)
        ev["nElectronGood"] = ak.num(ev.ElectronGood)
        ev["nLeptonGood"]   = ev.nMuonGood + ev.nElectronGood
        ev["nJetGood"]      = ak.num(ev.JetGood)
        ev["nJetAll"]      = ak.num(ev.Jet)
        ev["nJetGoodCentral"]      = ak.num(ev.JetGoodCentral)
        ev["nBJetTight"]     = ak.num(ev.BJetTight)
        ev["nBJetLoose"]     = ak.num(ev.BJetLoose)
        ev["nCentralJetsGood"] = ak.num(ev.CentralJetsGood)
        ev["nFatJetGood"] = ak.num(ev.FatJetGood)
        ev["nFatJetCentral"] = ak.num(ev.FatJetCentral) if hasattr(ev, "FatJetCentral") else 0
        ev["nMuonLoose"]     = ak.num(ev.MuonLoose)
        ev["nElectronLoose"] = ak.num(ev.ElectronLoose)
        ev["nMuonMedium"]     = ak.num(ev.MuonMedium)
        ev["nElectronMedium"] = ak.num(ev.ElectronMedium)
        ev["nMuonVeto"]     = ak.num(ev.MuonVeto)
        ev["nElectronVeto"] = ak.num(ev.ElectronVeto)
        ev["nLeptonMedium"]   = ev.nMuonMedium + ev.nElectronMedium
        ev["nLeptonLoose"]   = ev.nMuonLoose + ev.nElectronLoose
        #ev["nLeptonWithFakes"]   = ak.num(ev.LeptonWithFakes)
        ev["nLeptonVeto"]   = ev.nMuonVeto + ev.nElectronVeto
        #ev["nOtherJetsBoost"]    =ak.num(ev.CentralJetGoodBoostedFS)
