# workflow.py
import awkward as ak
import numpy as np
from pocket_coffea.workflows.base import BaseProcessorABC
from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.objects import lepton_selection, jet_selection, btagging

class VBSSemileptonicProcessor(BaseProcessorABC):
    """
    - Construye LeptonGood y JetGood (limpios de leptones)
    - Identifica los VBS tagging jets como el par con mayor mjj
    - Reconstruye el W hadrónico con dos jets no-VBS que minimizan |m-80.4|
    - Calcula variables auxiliares para histogramas (mt, pt/eta, dR, etc.)
    """

    def __init__(self, cfg: Configurator):
        super().__init__(cfg)

    # 1) object-level preselection
    def apply_object_preselection(self, variation):
        ev = self.events

        # Electrones: etaSC para selección de ID (si hiciera falta en tu YAML)
        ev["Electron", "etaSC"] = ev.Electron.eta + ev.Electron.deltaEtaSC

        # Leptones buenos (según YAML)
        ev["MuonGood"]     = lepton_selection(ev, "Muon", self.params)
        ev["ElectronGood"] = lepton_selection(ev, "Electron", self.params)

        # Leptón combinado (mu+e) y ordenado en pt
        leptons = ak.with_name(
            ak.concatenate([ev.MuonGood, ev.ElectronGood], axis=1),
            "PtEtaPhiMCandidate",
        )
        ev["LeptonGood"] = leptons[ak.argsort(leptons.pt, ascending=False)]

        # Jets limpios de leptones
        ev["JetGood"], _ = jet_selection(ev, "Jet", self.params, "LeptonGood")
        ev["JetGood", "idx"] = ak.local_index(ev.JetGood, axis=1)

        # b-tagging (para veto)
        ev["BJetGood"] = btagging(
            ev.JetGood,
            self.params.btagging.working_point[self._year],
            wp=self.params.object_preselection.Jet.btag.wp,
        )

        # ------------- VBS tagging jets -------------
        has4j = ak.num(ev.JetGood) >= 4

        jj = ak.combinations(ev.JetGood, 2, fields=["jet1", "jet2"])
        jj["mass"] = (jj.jet1 + jj.jet2).mass

        idx_vbs = ak.argmax(jj.mass, axis=1, keepdims=True)
        ev["vbsjets"] = ak.mask(jj[idx_vbs], has4j)

        v1 = ak.firsts(ev.vbsjets.jet1)
        v2 = ak.firsts(ev.vbsjets.jet2)

        # Δη y ΔR entre tagging jets
        ev["vbsjets", "delta_eta"] = np.abs(v1.eta - v2.eta)
        ev["vbs_dR"] = ak.fill_none(v1.delta_r(v2), np.nan)

        # ------------- W hadrónico (resolved) -------------
        # índices de los VBS jets en la colección JetGood
        vbs_i = ak.fill_none(getattr(v1, "idx", None), -1)
        vbs_j = ak.fill_none(getattr(v2, "idx", None), -1)

        nonvbs_mask = (ev.JetGood.idx != vbs_i) & (ev.JetGood.idx != vbs_j)
        jets_nonvbs = ev.JetGood[nonvbs_mask]

        pairs_w = ak.combinations(jets_nonvbs, 2, fields=["jet1", "jet2"])
        pairs_w["mass"] = (pairs_w.jet1 + pairs_w.jet2).mass

        target_mw = 80.4
        best_w_idx = ak.argmin(np.abs(pairs_w.mass - target_mw), axis=1, keepdims=True)

        ev["w_had_jets"] = ak.mask(pairs_w[best_w_idx], has4j)
        ev["w_had_jets", "mass"] = (ev.w_had_jets.jet1 + ev.w_had_jets.jet2).mass

        # ΔR entre los dos jets del W_had
        wj1 = ak.firsts(ev.w_had_jets.jet1)
        wj2 = ak.firsts(ev.w_had_jets.jet2)
        ev["w_had_dR"] = ak.fill_none(wj1.delta_r(wj2), np.nan)

        # ------------- W leptónico (mT) -------------
        lead_lep = ak.firsts(ev.LeptonGood)
        ev["mt_w_leptonic"] = np.sqrt(
            2.0 * lead_lep.pt * ev.MET.pt * (1.0 - np.cos(lead_lep.delta_phi(ev.MET)))
        )

        # ------------- helpers para plots -------------
        jets_sorted = ev.JetGood[ak.argsort(ev.JetGood.pt, ascending=False)]
        ev["jet1_pt"]  = ak.firsts(getattr(jets_sorted[:, 0:1], "pt", None))
        ev["jet2_pt"]  = ak.firsts(getattr(jets_sorted[:, 1:2], "pt", None))
        ev["jet1_eta"] = ak.firsts(getattr(jets_sorted[:, 0:1], "eta", None))
        ev["jet2_eta"] = ak.firsts(getattr(jets_sorted[:, 1:2], "eta", None))

    # 2) contadores (se usan en cortes y plots)
    def count_objects(self, variation):
        ev = self.events
        ev["nMuonGood"]     = ak.num(ev.MuonGood)
        ev["nElectronGood"] = ak.num(ev.ElectronGood)
        ev["nLeptonGood"]   = ev.nMuonGood + ev.nElectronGood
        ev["nJetGood"]      = ak.num(ev.JetGood)
        ev["nBJetGood"]     = ak.num(ev.BJetGood)
