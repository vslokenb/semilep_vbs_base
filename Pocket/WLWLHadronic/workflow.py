import awkward as ak

from pocket_coffea.workflows.base import BaseProcessorABC
from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.hist_manager import Axis
from pocket_coffea.lib.objects import (
    jet_correction,
    lepton_selection,
    jet_selection,
    btagging,
    get_dilepton,
)


class VBSWWBaseProcessor(BaseProcessorABC):
    def __init__(self, cfg: Configurator):
        super().__init__(cfg)


    def apply_object_preselection(self, variation):

        # Include the supercluster pseudorapidity variable
        electron_etaSC = self.events.Electron.eta + self.events.Electron.deltaEtaSC
        self.events["Electron"] = ak.with_field(
            self.events.Electron, electron_etaSC, "etaSC"
        )
        # Build masks for selection of muons, electrons, jets, fatjets
        self.events["MuonGood"] = lepton_selection(
            self.events, "Muon", self.params
        )
        self.events["ElectronGood"] = lepton_selection(
            self.events, "Electron", self.params
        )
        leptons = ak.with_name(
            ak.concatenate((self.events.MuonGood, self.events.ElectronGood), axis=1),
            name='PtEtaPhiMCandidate',
        )
        self.events["LeptonGood"] = leptons[ak.argsort(leptons.pt, ascending=False)]

        self.events["ll"] = get_dilepton(
            self.events.ElectronGood, self.events.MuonGood
        )

        self.events["JetGood"], self.jetGoodMask = jet_selection(
            self.events, "Jet", self.params, "LeptonGood"
        )
        self.events["BJetGood"] = btagging(
            self.events["JetGood"], self.params.btagging.working_point[self._year], wp=self.params.object_preselection.Jet.btag.wp)

        jj_pairs = ak.combinations(self.events.JetGood, 2, fields=["jet1", "jet2"])
        jj_pairs["mass"] = (jj_pairs.jet1 + jj_pairs.jet2).mass
        idx_vbs_jets = ak.argmax(jj_pairs.mass, axis=1, keepdims=True)
        vbs_jets_pair = jj_pairs[idx_vbs_jets]

        self.events["vbsjets"] = vbs_jets_pair
        self.events["vbsjets"] = ak.with_field(self.events.vbsjets, abs(self.events.vbsjets.jet1.eta - self.events.vbsjets.jet2.eta), where="delta_eta")

        self.events["vbsjet_1"] = self.events.vbsjets.jet1
        self.events["vbsjet_2"] = self.events.vbsjets.jet2
        

    def count_objects(self, variation):
        self.events["nMuonGood"] = ak.num(self.events.MuonGood)
        self.events["nElectronGood"] = ak.num(self.events.ElectronGood)
        self.events["nLeptonGood"] = (
            self.events["nMuonGood"] + self.events["nElectronGood"]
        )
        self.events["nJetGood"] = ak.num(self.events.JetGood)
        self.events["nBJetGood"] = ak.num(self.events.BJetGood)
        # self.events["nfatjet"]   = ak.num(self.events.FatJetGood)

    def define_common_variables_before_presel(self, variation):
        pass
       # self.events["JetGood_Ht"] = ak.sum(abs(self.events.JetGood.pt), axis=1)
