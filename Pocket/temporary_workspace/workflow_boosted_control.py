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

import correctionlib
import os as _os



METDef = "DeepMETResolutionTune"




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
                    "pt": 26.0,
                    "eta": 2.4,
                    "id": "tightId",
                    "iso": 0.15,
                }
            }
        )

        ev["MuonGood_0"]     = lepton_selection(ev, "Muon", tight_criteria)
        mu = ev.MuonGood_0
        mask_muon_ip = (
            (np.abs(mu.dxy) < 0.2) & (np.abs(mu.dz) < 0.5)
        )

        ev["MuonGood"] = mu[mask_muon_ip]
        mask_good_ele_kin = ( ev.Electron.pt > 35 ) & ( np.abs(ev.Electron.eta) < 2.4) & ( ev.Electron.cutBased >= 3 )
        ev["ElectronGood_0"] = ev.Electron[mask_good_ele_kin]
        ele = ev.ElectronGood_0
        mask_ele_ip = (
            (np.abs(ele.dxy) < 0.05) & (np.abs(ele.eta) < 1.5) & (np.abs(ele.dz) < 0.1)
        ) | (
            (np.abs(ele.dxy) < 0.1) & (np.abs(ele.eta) >= 1.5) & (np.abs(ele.eta) < 2.4) & (np.abs(ele.dz) < 0.2)
        )

        ev["ElectronGood"] = ele[mask_ele_ip]
        veto_criteria = SimpleNamespace(
            object_preselection = {
                "Muon": {
                    "pt": 10.0,
                    "eta": 2.4,
                    "id": "looseId",
                    "iso": 500.0,
                }
            }
        )

        medium_criteria = SimpleNamespace(
            object_preselection = {
                "Muon": {
                    "pt": 26.0,
                    "eta": 2.4,
                    "id": "mediumId",
                    "iso": 500.0,
                }
            }
        )

        loose_criteria = SimpleNamespace(
            object_preselection = {
                "Muon": {
                    "pt": 26.0,
                    "eta": 2.4,
                    "id": "looseId",
                    "iso": 500.0,
                }
            }
        )


        # Good Leptons
        ev["MuonVeto"]     = lepton_selection(ev, "Muon", veto_criteria)
        ev["MuonMedium"]     = lepton_selection(ev, "Muon", medium_criteria)
        ev["MuonLoose"]     = lepton_selection(ev, "Muon", loose_criteria)


        mask_ele_loose = (ev.Electron.pt > 35) & (np.abs(ev.Electron.eta) < 2.4) & (ev.Electron.cutBased >=1 )
        ev["ElectronLoose"] = ev.Electron[mask_ele_loose]
        mask_ele_medium = (ev.Electron.pt > 35) & (np.abs(ev.Electron.eta) < 2.4) & (ev.Electron.cutBased >=2 )
        ev["ElectronMedium"] = ev.Electron[mask_ele_medium]
        mask_ele_veto = (ev.Electron.pt > 10) & (np.abs(ev.Electron.eta) < 2.4) & (ev.Electron.cutBased >=1 )
        ev["ElectronVeto"]     = ev.Electron[mask_ele_veto]

                ##HEM HANDLING

        '''HEM_mask_e = (
            (ev.ElectronLoose.phi > -1.65) & (ev.ElectronLoose.phi < -0.62)
        ) & (
            (ev.ElectronLoose.eta > -3.05) & (ev.ElectronLoose.eta < -1.35)
        )

        HEM_mask_jet = (
            (ev.JetGood.phi > -1.65) & (ev.JetGood.phi < -0.62)
        ) & (
            (ev.JetGood.eta > -3.05) & (ev.JetGood.eta < -1.35)
        )
        # HEM_mask_e = (ev.ElectronLoose.phi > -1.35 & ev.ElectronLoose.phi < -0.82) & (ev.ElectronLoose.eta < -1.65 & ev.ElectronLoose.eta > -3.05)
        ev['ElectronHEM'] = ev.ElectronLoose[HEM_mask_e]
        # HEM_mask_jet = (ev.JetGood.phi > -1.35 & ev.JetGood.phi < -0.82) & (ev.JetGood.eta < -1.65 & ev.JetGood.eta > -3.05)
        ev['JetHEM'] = ev.JetGood[HEM_mask_jet]
        
        #### CLEAR RESCALING OF MC WHICH ARE EFFECTED BY HEM ISSUE IN 2018
        if hasattr(ev, "genWeight") and "2018" in ev.metadata["dataset"]:
            has_hem_object = (ak.num(ev.ElectronHEM) + ak.num(ev.JetHEM)) >= 1
            ev["genWeight"] = ak.where(has_hem_object, ev.genWeight * 0.35, ev.genWeight)
        '''

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
        ev["LeptonLoose"] = loose_lep[ak.argsort(loose_lep.pt, ascending=False)]

        ev["LeptonGood"] = leptons[ak.argsort(leptons.pt, ascending=False)]

        lead_lep = ak.firsts(ev.LeptonGood)
        lead_lep_loose = ak.firsts(ev.LeptonLoose)


        ev["JetGood_0"], _ = jet_selection(ev, "Jet", self.params, self._year,"LeptonLoose") #MAYBE THIS SHOULD BE LOOSE LEPTON
        if self._isMC:
            for jet_type, jet_coll_name in self.params.jets_calibration.collection[self._year].items():
                if jet_coll_name == "Jet":
                    JEC_type = jet_type
            ev.JetGood_1 = ev.JetGood_0#veto_jer_forward_unmatched(ev.JetGood_0,self.params.jets_calibration.variations[JEC_type][self._year])
        else:
            ev.JetGood_1 = ev.JetGood_0
        mask_jet_cleaning = (ev.JetGood_1.pt>50) | (abs(ev.JetGood_1.eta)<2.5)
        ev["JetGood"] = ev.JetGood_1[mask_jet_cleaning]



        HEM_mask_e = (
            (ev.ElectronLoose.phi > -1.65) & (ev.ElectronLoose.phi < -0.62)
        ) & (
            (ev.ElectronLoose.eta > -3.05) & (ev.ElectronLoose.eta < -1.35)
        )

        HEM_mask_jet = (
            (ev.JetGood.phi > -1.65) & (ev.JetGood.phi < -0.62)
        ) & (
            (ev.JetGood.eta > -3.05) & (ev.JetGood.eta < -1.35)
        )
        # HEM_mask_e = (ev.ElectronLoose.phi > -1.35 & ev.ElectronLoose.phi < -0.82) & (ev.ElectronLoose.eta < -1.65 & ev.ElectronLoose.eta > -3.05)                                                                                                                 
        ev['ElectronHEM'] = ev.ElectronLoose[HEM_mask_e]
        # HEM_mask_jet = (ev.JetGood.phi > -1.35 & ev.JetGood.phi < -0.82) & (ev.JetGood.eta < -1.65 & ev.JetGood.eta > -3.05)     
        ev['JetHEM'] = ev.JetGood[HEM_mask_jet]

        #### CLEAR RESCALING OF MC WHICH ARE EFFECTED BY HEM ISSUE IN 2018                                                         
        if hasattr(ev, "genWeight") and "2018" in ev.metadata["dataset"]:
            has_hem_object = (ak.num(ev.ElectronHEM) + ak.num(ev.JetHEM)) >= 1
            ev["genWeight"] = ak.where(has_hem_object, ev.genWeight * 0.35, ev.genWeight)

        ev["JetGoodCentral"] = ev.JetGood[abs(ev.JetGood.eta)<2.4]

        #TODO: jet_selection_nanoaodv12 only used for 2022, check other versions for other years.
        #ev["FatJetGood"], _ = jet_selection(ev,"FatJet", self.params, self._year, "LeptonGood")
        ev["FatJetGood"] = ev.FatJet[abs(ev.FatJet.eta)<2.4]
        ev["FatJetGood", "idx"] = ak.local_index(ev.FatJetGood, axis=1)
        dR_fatjets_lep = ev.FatJetGood.metric_table(ev.LeptonGood)
        mask_lepjet_cleaning = ak.prod(dR_fatjets_lep > 0.8, axis=2) == 1
        #separation = ak.fill_none(ev.etGood.metric_table(ev.candidate_boost), np.nan)
        #ev["separation"] = dR_jets_jet
        #ev["separation_after_cleaning"] = ak.fill_none(ev.JetGood[mask_jet_cleaning].metric_table(ev.candidate_boost), np.nan)

        # far_enough_from_ak8 = (separation > 0.8)
        ev["FatJetGood"] = ev.FatJetGood[mask_lepjet_cleaning]
        ev["FatJetGood", "idx"] = ak.local_index(ev.FatJetGood, axis=1)


        #ev["candidate_boost"] = ev.FatJetGood[(_tau21(ev.FatJetGood) < 0.45) & (ev.FatJetGood.msoftdrop < 250) & (ev.FatJetGood.msoftdrop > 40) & (ev.FatJetGood.pt > 200)]
        ev["candidate_boost"] = ev.FatJetGood[(ev.FatJetGood.msoftdrop < 250) & (ev.FatJetGood.msoftdrop > 40) & (ev.FatJetGood.pt > 200)]
        ev["candidate_boost" ,"tau21"] = _tau21(ev.candidate_boost)
        ev["candidate_boost" ,"abseta"] = abs(ev.candidate_boost.eta)
        if self._isMC:
            ev["gen_W"] = ev.GenPart[abs(ev.GenPart.pdgId)==24]
            dR_candidate_genW = ev.candidate_boost.metric_table(ev.gen_W)
            mask_matchW = ak.sum(dR_candidate_genW<0.8,axis=2) > 0
            ev["candidate_boost_matched"] = ev.candidate_boost[mask_matchW]
            ev["candidate_boost_unmatched"] = ev.candidate_boost[~mask_matchW]

        dR_jets_jet = ev.JetGood.metric_table(ev.candidate_boost)
        mask_jet_cleaning = ak.prod(dR_jets_jet > 0.8, axis=2) == 1
        separation = ak.fill_none(ev.JetGood.metric_table(ev.candidate_boost), np.nan)

        ev["JetGood"] = ev.JetGood[mask_jet_cleaning]
        ev["JetGood", "idx"] = ak.local_index(ev.JetGood, axis=1)

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


        # ------------- W Leptonic -------------
        #lead_lep = ak.firsts(ev.LeptonGood)
        ev["mt_w_leptonic"] = compute_MT(lead_lep, ev.PuppiMET)
        ev["mt_w_leptonic_deepMET_resolutiontune"] = compute_MT(lead_lep, ev.DeepMETResolutionTune)
        ev["mt_w_leptonic_deepMET_responsetune"] = compute_MT(lead_lep, ev.DeepMETResponseTune)
        ev["mt_w_leptonic_loose"] = compute_MT(lead_lep_loose, ev.PuppiMET)
        ev["mt_w_leptonic_deepMET_resolutiontune_loose"] = compute_MT(lead_lep_loose, ev.DeepMETResolutionTune)
        ev["mt_w_leptonic_deepMET_responsetune_loose"] = compute_MT(lead_lep_loose, ev.DeepMETResponseTune)


        ############ mll check
        has2l = ak.num(ev.LeptonGood) == 2
        ll = ak.combinations(ev.LeptonGood, 2, fields=["lep1", "lep2"])
        ll["m_ll"] = (ll.lep1 + ll.lep2).mass

        idx_ll = ak.argmax(ll.m_ll, axis=1, keepdims=True)
        ev["ll"] = ak.mask(ll[idx_ll], has2l)
        ##############
        ##############

        ev["w_lep_pt"]  = ak.fill_none(lead_lep.pt, np.nan)
        ev["w_lep_eta"] = ak.fill_none(lead_lep.eta, np.nan)
        ev["w_lep_phi"] = ak.fill_none(lead_lep.phi, np.nan)

        if self._isMC:
            dress_lep = ak.firsts(ev.GenDressedLepton)
            gen_met = ev.GenMET
            ev["gen_w_pt_dressed"] = (dress_lep + gen_met).pt
            w_pt_dressed = ak.firsts(ev.gen_w_pt_dressed, axis=-1)
            w_pt_direct = ak.firsts(ev.GenPart[abs(ev.GenPart.pdgId) == 24].pt, axis=-1)
            ev["gen_w_pt_by_pdg"] = ak.fill_none(w_pt_direct, w_pt_dressed)



    def count_objects(self, variation):
        ev = self.events
        ev["nMuonGood"]     = ak.num(ev.MuonGood)
        ev["nElectronGood"] = ak.num(ev.ElectronGood)
        ev["nLeptonGood"]   = ev.nMuonGood + ev.nElectronGood
        ev["nJetGood"]      = ak.num(ev.JetGood)
        ev["nJetGoodCentral"]      = ak.num(ev.JetGoodCentral)
        ev["nBoostCandidate"] = ak.num(ev.candidate_boost)

        ev["nBJetTight"]     = ak.num(ev.BJetTight)
        ev["nBJetLoose"]     = ak.num(ev.BJetLoose)

        
        ev["nMuonVeto"]     = ak.num(ev.MuonVeto)
        ev["nElectronVeto"] = ak.num(ev.ElectronVeto)
        ev["nLeptonVeto"]   = ev.nMuonVeto + ev.nElectronVeto
