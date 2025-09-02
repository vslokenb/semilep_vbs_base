# workflow.py
import awkward as ak
import numpy as np
from pocket_coffea.workflows.base import BaseProcessorABC
from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.objects import lepton_selection, jet_selection, btagging

class VBSSemileptonicProcessor(BaseProcessorABC):
    """
        - Build LeptonGood and JetGood (lepton-clean)
        - Identifies VBS tagging jets as the pair with the highest mjj
        - Reconstructs the hadronic W with two non-VBS jets that minimize |m-80.4|
        - Calculates auxiliary variables for histograms (mt, pt/eta, dR, etc.)
    """

    def __init__(self, cfg: Configurator):
        super().__init__(cfg)

    # 1) object-level preselection
    def apply_object_preselection(self, variation):
        ev = self.events
        #print(ev.fields)
        
        # Electrons: etaSC for selection ID 
        ev["Electron", "etaSC"] = ev.Electron.eta + ev.Electron.deltaEtaSC

        # Good Leptons
        ev["MuonGood"]     = lepton_selection(ev, "Muon", self.params)
        ev["ElectronGood"] = lepton_selection(ev, "Electron", self.params)

        # Leptóons (mu+e) and ordered in pt
        leptons = ak.with_name(
            ak.concatenate([ev.MuonGood, ev.ElectronGood], axis=1),
            "PtEtaPhiMCandidate",
        )
        ev["LeptonGood"] = leptons[ak.argsort(leptons.pt, ascending=False)]
        #print(ev.LeptonGood.fields)
        ev["JetGood"], _ = jet_selection(ev, "Jet", self.params, "LeptonGood")
        ev["JetGood", "idx"] = ak.local_index(ev.JetGood, axis=1)
        fj = ev.FatJet
        fj_mask = (fj.pt > 150) & (abs(fj.eta) < 2.4) 
        ev["FatJetGood"] = fj[fj_mask]

        # b-tagging 
        ev["BJetGood"] = btagging(
            ev.JetGood,
            self.params.btagging.working_point[self._year],
            wp=self.params.object_preselection.Jet.btag.wp,
        )

        # ------------- VBS tagging jets -------------
        has4j = ak.num(ev.JetGood) >= 4
        has3j = ak.num(ev.JetGood) >= 2 #keep it at 3 st we can separate fj vs ak4 jets!
        hasfatjet = ak.num(ev.FatJetGood) ==1
        has2l = ak.num(ev.LeptonGood) == 2
        jj = ak.combinations(ev.JetGood, 2, fields=["jet1", "jet2"])
        jj["mass"] = (jj.jet1 + jj.jet2).mass

        idx_vbs = ak.argmax(jj.mass, axis=1, keepdims=True)
        #if hasfatjet:
        #    ev["vbsjets"] = ak.mask(jj[idx_vbs], has3j)
        #else:
        ev["vbsjets"] = ak.mask(jj[idx_vbs], has3j)

        v1 = ak.firsts(ev.vbsjets.jet1)
        v2 = ak.firsts(ev.vbsjets.jet2)

        # deta and dR btw tagging jets
        ev["vbsjets", "delta_eta"] = np.abs(v1.eta - v2.eta)
        ev["vbs_dR"] = ak.fill_none(v1.delta_r(v2), np.nan)

        # ------ Boosted jet -------------
        

        # Apply mask to get FatJetGood for central events only
        j1_eta = ak.fill_none(getattr(v1, "eta", None), np.nan)
        j2_eta = ak.fill_none(getattr(v2, "eta", None), np.nan)
        eta_min = np.minimum(j1_eta, j2_eta)
        eta_max = np.maximum(j1_eta, j2_eta)

        fj_eta = ak.fill_none(getattr(ev.FatJetGood, "eta", None), np.nan)
        central_mask = (
            (~np.isnan(fj_eta)) & (~np.isnan(eta_min)) & (~np.isnan(eta_max))
            & (fj_eta > eta_min) & (fj_eta < eta_max)
        )
        ev["FatJetCentral"] = ev.FatJetGood[central_mask]
        fjc = ev.FatJetCentral[ak.argsort(ev.FatJetCentral.pt, ascending=False)]
        ev["nFatJetCentral"] = ak.num(fjc)

        fj1 = ak.firsts(fjc[:, 0:1])

        def _tau21(fj):
             t1 = ak.fill_none(getattr(fj, "tau1", None), np.nan)
             t2 = ak.fill_none(getattr(fj, "tau2", None), np.nan)
             return ak.where((t1 > 0) & np.isfinite(t1), t2 / t1, np.nan)
        ev["w_fatjet"] = ak.zip(
            {
                "pt":    ak.fill_none(getattr(fj1, "pt",        None), np.nan),
                "eta":   ak.fill_none(getattr(fj1, "eta",       None), np.nan),
                "msd":   ak.fill_none(getattr(fj1, "msoftdrop", None), np.nan),
                "tau21": _tau21(fj1),
            },
            depth_limit=1, ##### THIS DEPTH LIMIT THING IS KEY ELSE BREAKS
        )

            
        # ------------- W hadronic (resolved) -------------
        vbs_i = ak.fill_none(getattr(v1, "idx", None), -1)
        vbs_j = ak.fill_none(getattr(v2, "idx", None), -1)

        nonvbs_mask = (ev.JetGood.idx != vbs_i) & (ev.JetGood.idx != vbs_j)
        ev["CentralJets"] = ev.JetGood[nonvbs_mask]
        ev["CentralJetsGood"] = ev.CentralJets[np.abs(ev.CentralJets.eta) < 2.4]
        

        pairs_w = ak.combinations(ev.CentralJetsGood, 2, fields=["jet1", "jet2"])
        pairs_w["mass"] = (pairs_w.jet1 + pairs_w.jet2).mass
        pairs_w["deta"] = (pairs_w.jet1 - pairs_w.jet2).eta


        target_mw = 80.4
        best_w_idx = ak.argmin(np.abs(pairs_w.mass - target_mw)/target_mw + np.abs(pairs_w.deta), axis=1, keepdims=True) #ADD EXTRA CUT FOR ETA MIN TOO

        ev["w_had_jets"] = ak.mask(pairs_w[best_w_idx], has4j)
        ev["w_had_jets", "mass"] = (ev.w_had_jets.jet1 + ev.w_had_jets.jet2).mass
        ev["w_had_jets", "pt"] = (ev.w_had_jets.jet1 + ev.w_had_jets.jet2).pt
        

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
        lead_lep = ak.firsts(ev.LeptonGood)
        ev["mt_w_leptonic"] = np.sqrt(
            2.0 * lead_lep.pt * ev.MET.pt * (1.0 - np.cos(lead_lep.delta_phi(ev.MET)))
        )
       
        ev["lead_wlep_wfatjet1_dR"] = ak.fill_none(lead_lep.delta_r(fj1), np.nan)
        ev["lead_wlep_wjet1_dR"] = ak.fill_none(lead_lep.delta_r(wj1), np.nan)
        ev["lead_wlep_wjet2_dR"] = ak.fill_none(lead_lep.delta_r(wj2), np.nan)
        
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


    def count_objects(self, variation):
        ev = self.events
        ev["nMuonGood"]     = ak.num(ev.MuonGood)
        ev["nElectronGood"] = ak.num(ev.ElectronGood)
        ev["nLeptonGood"]   = ev.nMuonGood + ev.nElectronGood
        ev["nJetGood"]      = ak.num(ev.JetGood)
        ev["nBJetGood"]     = ak.num(ev.BJetGood)
        ev["nCentralJetsGood"] = ak.num(ev.CentralJetsGood)
        ev["nFatJetGood"] = ak.num(ev.FatJetGood)
        ev["nFatJetCentral"] = ak.num(ev.FatJetCentral) if hasattr(ev, "FatJetCentral") else 0
