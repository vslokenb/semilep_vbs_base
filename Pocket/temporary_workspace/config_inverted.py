import os, cloudpickle
from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_functions import get_HLTsel, get_nPVgood, goldenJson, eventFlags, get_JetVetoMap
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.parameters.histograms import HistConf, Axis
from pocket_coffea.lib.weights.common import common_weights
from pocket_coffea.lib.weights.common.common import SF_L1prefiring
from pocket_coffea.lib.weights.common.weights_run2_UL import SF_ele_trigger
from pocket_coffea.parameters import defaults
from pocket_coffea.lib.columns_manager import ColOut

import numpy as np
import awkward as ak
from pocket_coffea.lib.weights import WeightWrapper, WeightData, WeightDataMultiVariation, WeightLambda
from pocket_coffea.lib.scale_factors import sf_pileup_reweight


import workflow_invertlepton_noiso, custom_cut_functions, reweighting_st
from reweighting_st import ratio_function
from workflow_invertlepton_noiso import VBSSemileptonicProcessor
from custom_cut_functions import (
    nLepton_skim_cut,
    nJet_skim_cut,
    vbs_semileptonic_presel,
    # whad_window_cut_e,
    met_skim_cut,
    whad_window_cut_bveto_e,
    msd_window_cut_e,
    whad_window_cut_mu,
    whad_window_cut_bveto_mu,
    msd_window_cut_mu,
    w_cr_mu,
    w_cr_e,
    w_cr_boosted_mu,
    w_cr_boosted_e,
    vr_mu,
    vr_e,
    vr_boosted_mu,
    vr_boosted_e,
    vr_boosted_no_fwd_mu,
    vr_boosted_no_fwd_e,
    vr_qcd_enriched_mu,
    vr_qcd_enriched_e,
    vr_no_fwd_mu,
    vr_no_fwd_e,
    vr_loose_njet_mu,
    vr_loose_njet_e,
    vr_no_fwd_loose_njet_mu,
    vr_no_fwd_loose_njet_e,
    recoil_inclusive_mu,
    recoil_inclusive_e,
    recoil_fullinclusive_mu,
    recoil_fullinclusive_e,
    recoil_closure_mu,
    recoil_closure_e,
    w_cr_no_fwd_mu,
    w_cr_no_fwd_e,
    w_cr_loose_njet_mu,
    w_cr_loose_njet_e,
    w_cr_no_fwd_loose_njet_mu,
    w_cr_no_fwd_loose_njet_e,
    w_cr_sb_lo1_mu,
    w_cr_sb_lo2_mu,
    w_cr_sb_hi1_mu,
    w_cr_sb_hi2_mu,
    w_cr_sb_lo1_e,
    w_cr_sb_lo2_e,
    w_cr_sb_hi1_e,
    w_cr_sb_hi2_e,
    w_cr_incl_mu,
    w_cr_incl_e,
    # qcd_validate_mu,
    # qcd_validate_e,
    ttbar_cr_boosted_mu,
    ttbar_cr_boosted_e,
    ttbar_cr_resolved_mu,
    ttbar_cr_resolved_e,
)


cloudpickle.register_pickle_by_value(workflow_invertlepton_noiso)
cloudpickle.register_pickle_by_value(reweighting_st)
cloudpickle.register_pickle_by_value(custom_cut_functions)
localdir = os.path.dirname(os.path.abspath(__file__))


default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir + "/params")
parameters = defaults.merge_parameters_from_files(
    default_parameters,
    f"{localdir}/params/object_preselection_run2_v9.yaml",
    f"{localdir}/params/triggers.yaml",
    f"{localdir}/params/plotting.yaml",
    f"{localdir}/params/pileup.yaml",
    f"{localdir}/params/jets_calibration.yaml",
    f"{localdir}/params/jet_scale_factors.yaml",
    f"{localdir}/params/lepton_scale_factors.yaml",
    f"{localdir}/params/classifiers.yaml",
    f"{localdir}/params/variations.yaml",
    f"{localdir}/params/fakelepton_weights_noiso_3j.yaml",
    f"{localdir}/params/dphi_weights.yaml",
    f"{localdir}/params/qgtagging.yaml",
    f"{localdir}/params/fj_taggers.yaml",
    update=True,
)

PileupWeight = WeightLambda.wrap_func(
   name="PileupWeight",
   function=lambda params, metadata, events, size, shape_variations:
       sf_pileup_reweight(params, events, metadata["year"]),
   has_variations=True
   )

wjet_reweight = WeightLambda.wrap_func(
    name="wjet_reweight",
    function=lambda params, metadata, events, size, shape_variations:
       ratio_function(ak.sum(events.GenJet[events.GenJet.pt > 20].pt, axis=1)),
    has_variations=False
    )

from coffea.lookup_tools import extractor

fake_muon_weights         = {}
fake_electron_weights     = {}
fake_muon_weights_boosted     = {}
fake_electron_weights_boosted = {}

for y in parameters.fakeleptonweights.keys():
    ext = extractor()
    ext.add_weight_sets([
        f"muonFakeWeight {parameters.fakeleptonweights[y]['Muon']['nominal'][0]} {parameters.fakeleptonweights[y]['Muon']['file'][0]}",
        f"muonFakeWeight_up {parameters.fakeleptonweights[y]['Muon']['up'][0]} {parameters.fakeleptonweights[y]['Muon']['file'][0]}",
        f"muonFakeWeight_down {parameters.fakeleptonweights[y]['Muon']['down'][0]} {parameters.fakeleptonweights[y]['Muon']['file'][0]}",
        f"electronFakeWeight {parameters.fakeleptonweights[y]['Electron']['nominal'][0]} {parameters.fakeleptonweights[y]['Electron']['file'][0]}",
        f"electronFakeWeight_up {parameters.fakeleptonweights[y]['Electron']['up'][0]} {parameters.fakeleptonweights[y]['Electron']['file'][0]}",
        f"electronFakeWeight_down {parameters.fakeleptonweights[y]['Electron']['down'][0]} {parameters.fakeleptonweights[y]['Electron']['file'][0]}",
        f"muonFakeWeight_boosted {parameters.fakeleptonweights[y]['Boosted_Muon']['nominal'][0]} {parameters.fakeleptonweights[y]['Boosted_Muon']['file'][0]}",
        f"muonFakeWeight_boosted_up {parameters.fakeleptonweights[y]['Boosted_Muon']['up'][0]} {parameters.fakeleptonweights[y]['Boosted_Muon']['file'][0]}",
        f"muonFakeWeight_boosted_down {parameters.fakeleptonweights[y]['Boosted_Muon']['down'][0]} {parameters.fakeleptonweights[y]['Boosted_Muon']['file'][0]}",
        f"electronFakeWeight_boosted {parameters.fakeleptonweights[y]['Boosted_Electron']['nominal'][0]} {parameters.fakeleptonweights[y]['Boosted_Electron']['file'][0]}",
        f"electronFakeWeight_boosted_up {parameters.fakeleptonweights[y]['Boosted_Electron']['up'][0]} {parameters.fakeleptonweights[y]['Boosted_Electron']['file'][0]}",
        f"electronFakeWeight_boosted_down {parameters.fakeleptonweights[y]['Boosted_Electron']['down'][0]} {parameters.fakeleptonweights[y]['Boosted_Electron']['file'][0]}",
    ])
    ext.finalize()
    ev = ext.make_evaluator()

    fake_muon_weights[y] = {
        "nominal": ev["muonFakeWeight"],
        "up":      ev["muonFakeWeight_up"],
        "down":    ev["muonFakeWeight_down"],
    }
    fake_electron_weights[y] = {
        "nominal": ev["electronFakeWeight"],
        "up":      ev["electronFakeWeight_up"],
        "down":    ev["electronFakeWeight_down"],
    }
    fake_muon_weights_boosted[y] = {
        "nominal": ev["muonFakeWeight_boosted"],
        "up":      ev["muonFakeWeight_boosted_up"],
        "down":    ev["muonFakeWeight_boosted_down"],
    }
    fake_electron_weights_boosted[y] = {
        "nominal": ev["electronFakeWeight_boosted"],
        "up":      ev["electronFakeWeight_boosted_up"],
        "down":    ev["electronFakeWeight_boosted_down"],
    }

import correctionlib

nonprompt_dphi_weights_mu = {}
nonprompt_dphi_weights_e  = {}

for y in parameters.dphi_weights.keys():
    mu_path  = parameters.dphi_weights[y]['Muon']['file'][0]
    mu_name  = parameters.dphi_weights[y]['Muon']['correction_name'][0]
    e_path   = parameters.dphi_weights[y]['Electron']['file'][0]
    e_name   = parameters.dphi_weights[y]['Electron']['correction_name'][0]

    nonprompt_dphi_weights_mu[y] = correctionlib.CorrectionSet.from_file(mu_path)[mu_name]
    nonprompt_dphi_weights_e[y]  = correctionlib.CorrectionSet.from_file(e_path)[e_name]

import awkward as ak
from pocket_coffea.lib.weights import WeightWrapper

class MuonGoodLeadWeight(WeightWrapper):
    name = "muon_inverttight_to_fake"
    has_variations = True
    isMC_only = False
    def compute(self, events, *args, **kwargs):
        year = events.metadata["year"]
        mu = events.MuonGoodLead
        has_mu = ~ak.is_none(mu)
        pt  = ak.where(has_mu, mu.pt,      0.0)
        eta = ak.where(has_mu, abs(mu.eta), 0.0)

        in_eta_range = (eta >= 0.0) & (eta <= 2.4)

        pt = ak.where(has_mu, np.clip(pt, 26.0, 100.0), pt)
        eta_for_lookup = np.clip(eta, 0.0, 2.4)

        nominal = fake_muon_weights[year]["nominal"](pt, eta_for_lookup)
        up      = fake_muon_weights[year]["up"](pt, eta_for_lookup)
        down    = fake_muon_weights[year]["down"](pt, eta_for_lookup)

        valid = has_mu & in_eta_range
        nominal = ak.where(valid, nominal, 0.0)
        up      = ak.fill_none(ak.where(valid, up,   0.0), 0.0)
        down    = ak.fill_none(ak.where(valid, down, 0.0), 0.0)

        print("nominal ", nominal)
        print("up ",      up)
        print("down ",    down)
        return WeightData(self.name, nominal, up, down)

class ElectronGoodLeadWeight(WeightWrapper):
    name = "electron_inverttight_to_fake"
    has_variations = True
    isMC_only = False
    def compute(self, events, *args, **kwargs):
        year = events.metadata["year"]
        ele = events.ElectronGoodLead
        has_ele = ~ak.is_none(ele)
        pt  = ak.where(has_ele, ele.pt,       0.0)
        eta = ak.where(has_ele, abs(ele.eta), 0.0)

        in_eta_range = (eta >= 0.0) & (eta <= 2.4)

        pt = ak.where(has_ele, np.clip(pt, 35.0, 100.0), pt)
        # clip eta only for safe table lookup (avoid out-of-domain evaluation);
        # the actual cut is applied to the output weight below via in_eta_range
        eta_for_lookup = np.clip(eta, 0.0, 2.4)

        nominal = fake_electron_weights[year]["nominal"](pt, eta_for_lookup)
        up      = fake_electron_weights[year]["up"](pt, eta_for_lookup)
        down    = fake_electron_weights[year]["down"](pt, eta_for_lookup)

        nominal = ak.where(has_ele, ak.where(in_eta_range, nominal, 0.0), 0.0)
        up      = ak.fill_none(ak.where(has_ele, ak.where(in_eta_range, up,   0.0), 1.0), 0.0)
        down    = ak.fill_none(ak.where(has_ele, ak.where(in_eta_range, down, 0.0), 1.0), 0.0)

        print("nominal ", nominal)
        print("up ",      up)
        print("down ",    down)
        return WeightData(self.name, nominal, up, down)


class DPHI_SF(WeightWrapper):
    name = "dphi_sf"
    has_variations = True
    isMC_only = False

    def compute(self, events, *args, **kwargs):
        year = events.metadata["year"]

        mu     = events.MuonGoodLead
        el     = events.ElectronGoodLead
        has_mu = ~ak.is_none(mu)
        has_el = ~ak.is_none(el)

        # shared observable
        abs_dphi_np = np.abs(
            ak.to_numpy(
                ak.fill_none(events.dphi.lepton1_DeepMETResolutionTune, 0.0)
            ).astype(np.float64)
        )
        has_mu_np = ak.to_numpy(ak.fill_none(has_mu, False))
        has_el_np = ak.to_numpy(ak.fill_none(has_el, False))

        corr_mu = nonprompt_dphi_weights_mu[year]
        corr_e  = nonprompt_dphi_weights_e[year]

        def _eval(systematic: str) -> ak.Array:
            vals = np.ones(len(abs_dphi_np), dtype=np.float64)
            # muon-flavored events
            if has_mu_np.any():
                vals[has_mu_np] = corr_mu.evaluate(
                    year, systematic, abs_dphi_np[has_mu_np],
                )
            # electron-flavored events
            if has_el_np.any():
                vals[has_el_np] = corr_e.evaluate(
                    year, systematic, abs_dphi_np[has_el_np],
                )
            # fall back to 1.0 where neither flavor present
            has_lep = has_mu | has_el
            return ak.where(has_lep, ak.Array(vals), 1.0)

        nominal = _eval("nominal")
        up      = ak.fill_none(_eval("up"),   1.0)
        down    = ak.fill_none(_eval("down"), 1.0)

        print("nominal ", nominal)
        print("up ",      up)
        print("down ",    down)

        return WeightData(self.name, nominal, up, down)

class MuonGoodLeadWeightBoosted(WeightWrapper):
    name = "muon_inverttight_to_fake_boosted"
    has_variations = True
    isMC_only = False

    def compute(self, events, *args, **kwargs):
        year = events.metadata["year"]
        mu = events.MuonGoodLead
        has_mu = ~ak.is_none(mu)
        pt  = ak.where(has_mu, mu.pt,      0.0)
        eta = ak.where(has_mu, abs(mu.eta), 0.0)

        in_eta_range = (eta >= 0.0) & (eta <= 2.4)

        pt = ak.where(has_mu, np.clip(pt, 26.0, 100.0), pt)
        eta_for_lookup = np.clip(eta, 0.0, 2.4)

        nominal = fake_muon_weights[year]["nominal"](pt, eta_for_lookup)
        up      = fake_muon_weights[year]["up"](pt, eta_for_lookup)
        down    = fake_muon_weights[year]["down"](pt, eta_for_lookup)

        valid = has_mu & in_eta_range
        nominal = ak.where(valid, nominal, 0.0)
        up      = ak.fill_none(ak.where(valid, up,   0.0), 0.0)
        down    = ak.fill_none(ak.where(valid, down, 0.0), 0.0)

        print("nominal ", nominal)
        print("up ",      up)
        print("down ",    down)
        return WeightData(self.name, nominal, up, down)


class ElectronGoodLeadWeightBoosted(WeightWrapper):
    name = "electron_inverttight_to_fake_boosted"
    has_variations = True
    isMC_only = False

    def compute(self, events, *args, **kwargs):
        year = events.metadata["year"]
        ele = events.ElectronGoodLead
        has_ele = ~ak.is_none(ele)
        pt  = ak.where(has_ele, ele.pt,       0.0)
        eta = ak.where(has_ele, abs(ele.eta), 0.0)

        in_eta_range = (eta >= 0.0) & (eta <= 2.4)

        pt = ak.where(has_ele, np.clip(pt, 35.0, 100.0), pt)
        # clip eta only for safe table lookup (avoid out-of-domain evaluation);
        # the actual cut is applied to the output weight below via in_eta_range
        eta_for_lookup = np.clip(eta, 0.0, 2.4)

        nominal = fake_electron_weights[year]["nominal"](pt, eta_for_lookup)
        up      = fake_electron_weights[year]["up"](pt, eta_for_lookup)
        down    = fake_electron_weights[year]["down"](pt, eta_for_lookup)

        nominal = ak.where(has_ele, ak.where(in_eta_range, nominal, 0.0), 0.0)
        up      = ak.fill_none(ak.where(has_ele, ak.where(in_eta_range, up,   0.0), 1.0), 0.0)
        down    = ak.fill_none(ak.where(has_ele, ak.where(in_eta_range, down, 0.0), 1.0), 0.0)

        print("nominal ", nominal)
        print("up ",      up)
        print("down ",    down)
        return WeightData(self.name, nominal, up, down)



class LHEScaleWeightWrapper(WeightWrapper):
    """LHE renormalization and factorization scale uncertainties.
    LHEScaleWeight is already normalized as w_var/w_nominal in NanoAOD.
    For the standard 9-member set:
      renorm_scale: down=LHEScaleWeight[:,1], up=LHEScaleWeight[:,7]
      fact_scale:   down=LHEScaleWeight[:,3], up=LHEScaleWeight[:,5]
    Samples with fewer than 9 members get unit weights.
    """
    name = "LHEScaleWeight"
    has_variations = True
    isMC_only = True
    _variations = ["renorm_scale", "fact_scale"]

    def __init__(self, parameters, metadata):
        super().__init__(parameters, metadata)

    def compute(self, events, size, shape_variation):
        if shape_variation != "nominal":
            return WeightData(name=self.name, nominal=np.ones(size))

        ones = np.ones(size)
        if not hasattr(events, "LHEScaleWeight"):
            return WeightDataMultiVariation(
                name=self.name, nominal=ones, variations=self._variations,
                up=[ones, ones], down=[ones, ones],
            )

        w = events.LHEScaleWeight
        # pad so index 8 always exists; missing members -> 1.0 (no variation)
        w = ak.fill_none(ak.pad_none(w, 9, axis=1, clip=False), 1.0)
        n = ak.to_numpy(ak.num(events.LHEScaleWeight))
        ok = n >= 9  # per-event guard (uniform within a sample in practice)

        def member(i):
            return np.where(ok, ak.to_numpy(w[:, i]), 1.0)

        return WeightDataMultiVariation(
            name=self.name,
            nominal=np.ones(size),
            variations=self._variations,
            up=[member(7), member(5)],
            down=[member(1), member(3)],
        )


class LHEPdfWeightWrapper(WeightWrapper):
    """LHE PDF and alpha_S uncertainties from LHEPdfWeight.
    For the standard 103-member set:
      [1]-[100]: PDF eigenvariations (up=w[:,i], down=2-w[:,i])
      [101]/[102]: alpha_S up/down.
    Samples with shorter sets (101 members: no alpha_S; 33 members: fewer
    eigenvectors) get unit weights for the missing members.
    """
    name = "LHEPdfWeight"
    has_variations = True
    isMC_only = True
    _variations = [f"pdf_{i}" for i in range(1, 101)] + ["alpha_S"]

    def __init__(self, parameters, metadata):
        super().__init__(parameters, metadata)

    def compute(self, events, size, shape_variation):
        if shape_variation != "nominal":
            return WeightData(name=self.name, nominal=np.ones(size))

        ones = np.ones(size)
        if not hasattr(events, "LHEPdfWeight"):
            return WeightDataMultiVariation(
                name=self.name, nominal=ones, variations=self._variations,
                up=[ones] * len(self._variations),
                down=[ones] * len(self._variations),
            )

        w_raw = events.LHEPdfWeight
        n = ak.to_numpy(ak.num(w_raw))
        # pad so indices up to 102 always exist; padded members -> 1.0
        w = ak.fill_none(ak.pad_none(w_raw, 103, axis=1, clip=False), 1.0)

        def member(i, transform=None):
            vals = ak.to_numpy(w[:, i])
            if transform is not None:
                vals = transform(vals)
            # only trust members the sample actually has
            return np.where(n > i, vals, 1.0)

        pdf_up   = [member(i)                    for i in range(1, 101)]
        pdf_down = [member(i, lambda v: 2 - v)   for i in range(1, 101)]
        alphas_up   = member(101)
        alphas_down = member(102)

        return WeightDataMultiVariation(
            name=self.name,
            nominal=np.ones(size),
            variations=self._variations,
            up=pdf_up + [alphas_up],
            down=pdf_down + [alphas_down],
        )

############################################
##### QG TAGGING SCALE FACTOR (correctionlib)
##### Shape SFs for qgl / btagDeepFlavQG / particleNetAK4_QvsG,
##### one correction file per year (see params/qgtagging.yaml).
##### Per-jet SF(QvG, flavor, |eta|, pt) multiplied over analysis jets.
############################################
import correctionlib

# year -> CorrectionSet, loaded from params/qgtagging.yaml (all files same format)
_qgtagging_csets = {}
for _y in parameters.qgtagging.keys():
    _qg_file = parameters.qgtagging[_y]["file"]
    if not os.path.isabs(_qg_file):
        _qg_file = f"{localdir}/{_qg_file}"
    if os.path.exists(_qg_file):
        _qgtagging_csets[_y] = correctionlib.CorrectionSet.from_file(_qg_file)

# Full-shape SF: inputs (systematic, QvG, absflavor, abseta, pt).
# Per discriminant bin; only |partonFlavour| 1-3 and 21 are corrected
# (c/b/undefined get SF=1). Domain: QvG [0,1), |eta| [0,2.5), pt [30,8000).
# Systematics: central, up/down (total), and {source}_{up,down} for
# stat, fsr, isr, pu, jes, jer, L1prefiring, herwig, scale, PDF.
_QG_CORR_NAME = "deepJet_fullshape"   # also available: qgl_fullshape, particleNet_fullshape
_QG_JET_FIELD = "btagDeepFlavQG"      # jet field matching the correction

# systematic sources exposed as weight variations ("total" = combined up/down)
_QG_SOURCES = ["total", "stat", "fsr", "isr", "pu", "jes", "jer",
               "L1prefiring", "herwig", "scale", "PDF"]


class QGTaggingWeight(WeightWrapper):
    name = "sf_qgtagging"
    has_variations = True
    isMC_only = True
    _variations = [f"sf_qgtagging_{s}" for s in _QG_SOURCES]

    def compute(self, events, size, shape_variation):
        if shape_variation != "nominal":
            return WeightData(self.name, np.ones(size))

        year = events.metadata["year"]
        if year not in _qgtagging_csets:
            raise KeyError(
                f"No qgtagging correction file for year '{year}' "
                f"(available: {list(_qgtagging_csets)}); check params/qgtagging.yaml"
            )
        corr = _qgtagging_csets[year][_QG_CORR_NAME]
        jets = events.JetGood30

        qvg    = ak.fill_none(getattr(jets, _QG_JET_FIELD, None), -1.0)
        flav   = np.abs(ak.fill_none(jets.partonFlavour, 0))
        abseta = np.abs(jets.eta)
        pt     = jets.pt

        # only light quarks (1-3) and gluons (21) are corrected
        corrected_flav = ((flav >= 1) & (flav <= 3)) | (flav == 21)
        in_domain = (qvg >= 0.0) & corrected_flav & (abseta < 2.5) & (pt >= 30.0)

        counts   = ak.num(jets)
        qvg_f    = np.clip(ak.to_numpy(ak.flatten(qvg)), 0.0, 0.999999)
        flav_f   = ak.to_numpy(ak.flatten(flav)).astype(int)
        abseta_f = np.clip(ak.to_numpy(ak.flatten(abseta)), 0.0, 2.499)
        pt_f     = np.clip(ak.to_numpy(ak.flatten(pt)), 30.0, 7999.0)
        dom_f    = ak.to_numpy(ak.flatten(in_domain))

        def _per_event(systematic):
            sf_f = np.ones(len(qvg_f))
            if dom_f.any():
                sf_f[dom_f] = corr.evaluate(
                    systematic,
                    qvg_f[dom_f], flav_f[dom_f], abseta_f[dom_f], pt_f[dom_f],
                )
            # product over jets in each event
            return ak.to_numpy(ak.prod(ak.unflatten(sf_f, counts), axis=1))

        nominal = _per_event("central")
        ups, downs = [], []
        for src in _QG_SOURCES:
            if src == "total":
                ups.append(_per_event("up"))
                downs.append(_per_event("down"))
            else:
                ups.append(_per_event(f"{src}_up"))
                downs.append(_per_event(f"{src}_down"))

        return WeightDataMultiVariation(
            name=self.name,
            nominal=nominal,
            variations=self._variations,
            up=ups,
            down=downs,
        )


############################################
##### AK8 FAT-JET TAGGER SCALE FACTORS (correctionlib)
##### Per-event SFs from the leading candidate fat jet, one file per year
##### (see params/fj_taggers.yaml). fjtype = matched/unmatched to a gen W
##### (dR < 0.8). Events without a candidate fat jet get SF = 1.
#####   fj_tau21_SF            : SF(tau21), coarse bins [0, 0.45, 1.0]
#####   fj_WvsQCD_SF_tau21cut  : SF(msoftdrop, WvsQCD), tau21 < 0.45 applied
############################################

# year -> {tagger_name: CorrectionSet}
_fj_tagger_csets = {}
for _y in parameters.fj_taggers.keys():
    _fj_tagger_csets[_y] = {}
    for _tag in parameters.fj_taggers[_y].keys():
        _fj_file = parameters.fj_taggers[_y][_tag]["file"]
        if not os.path.isabs(_fj_file):
            _fj_file = f"{localdir}/{_fj_file}"
        if os.path.exists(_fj_file):
            _fj_tagger_csets[_y][_tag] = correctionlib.CorrectionSet.from_file(_fj_file)

_FJ_SOURCES = ["stat", "pileup", "sf_btag",
               "sf_partonshower_isr", "sf_partonshower_fsr", "JES", "JER"]


def _fj_lead_and_match(events):
    """Leading candidate fat jet, presence mask, and gen-W match (dR < 0.8)."""
    fj = ak.firsts(events.candidate_boost)
    has_fj = ~ak.is_none(fj)
    genw = events.GenPart[np.abs(events.GenPart.pdgId) == 24]
    dr = genw.delta_r(fj)
    matched = ak.fill_none(ak.any(dr < 0.8, axis=1), False) & has_fj
    return fj, ak.to_numpy(has_fj), ak.to_numpy(matched)


def _fj_sf_multivariation(name, corr, obs_arrays, has_fj, matched):
    """Evaluate SF per event, split matched/unmatched, all systematic sources."""
    n = len(has_fj)

    def _eval(systematic):
        sf = np.ones(n)
        for fjtype, sel in (("matched", has_fj & matched), ("unmatched", has_fj & ~matched)):
            if sel.any():
                sf[sel] = corr.evaluate(systematic, fjtype, *[a[sel] for a in obs_arrays])
        return sf

    nominal = _eval("nominal")
    ups   = [_eval(f"{s}_up")   for s in _FJ_SOURCES]
    downs = [_eval(f"{s}_down") for s in _FJ_SOURCES]
    return WeightDataMultiVariation(
        name=name,
        nominal=nominal,
        variations=[f"{name}_{s}" for s in _FJ_SOURCES],
        up=ups,
        down=downs,
    )


class FatJetTau21Weight(WeightWrapper):
    name = "sf_fj_tau21"
    has_variations = True
    isMC_only = True
    _variations = [f"sf_fj_tau21_{s}" for s in _FJ_SOURCES]

    def compute(self, events, size, shape_variation):
        if shape_variation != "nominal":
            return WeightData(self.name, np.ones(size))
        year = events.metadata["year"]
        cset = _fj_tagger_csets.get(year, {})
        if "fj_tau21_SF" not in cset:
            raise KeyError(f"No fj_tau21_SF file for year '{year}'; check params/fj_taggers.yaml")
        corr = cset["fj_tau21_SF"]["fj_tau21_SF"]

        fj, has_fj, matched = _fj_lead_and_match(events)
        tau21 = np.clip(ak.to_numpy(ak.fill_none(fj.tau21, 0.0)), 0.0, 0.999)
        return _fj_sf_multivariation(self.name, corr, [tau21], has_fj, matched)


class FatJetWvsQCDWeight(WeightWrapper):
    name = "sf_fj_WvsQCD"
    has_variations = True
    isMC_only = True
    _variations = [f"sf_fj_WvsQCD_{s}" for s in _FJ_SOURCES]

    def compute(self, events, size, shape_variation):
        if shape_variation != "nominal":
            return WeightData(self.name, np.ones(size))
        year = events.metadata["year"]
        cset = _fj_tagger_csets.get(year, {})
        if "fj_WvsQCD_SF_tau21cut" not in cset:
            raise KeyError(f"No fj_WvsQCD_SF_tau21cut file for year '{year}'; check params/fj_taggers.yaml")
        corr = cset["fj_WvsQCD_SF_tau21cut"]["fj_WvsQCD_SF_tau21cut"]

        fj, has_fj, matched = _fj_lead_and_match(events)
        # candidate_boost already has tau21 < 0.45 applied (SF derivation selection)
        msd    = np.clip(ak.to_numpy(ak.fill_none(fj.msoftdrop, 40.0)), 40.0, 199.9)
        wvsqcd = np.clip(ak.to_numpy(ak.fill_none(fj.particleNet_WvsQCD, 0.0)), 0.0, 0.999)
        return _fj_sf_multivariation(self.name, corr, [msd, wvsqcd], has_fj, matched)


cfg = Configurator(
    parameters=parameters,
    datasets={
        "jsons": [
            # f"{localdir}/datasets/WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            # f"{localdir}/datasets/WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            # f"{localdir}/datasets/WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            # f"{localdir}/datasets/WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            # f"{localdir}/datasets/WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            # f"{localdir}/datasets/WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            # f"{localdir}/datasets/WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            # f"{localdir}/datasets/WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            # f"{localdir}/datasets/WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            # f"{localdir}/datasets/WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8.json",
            # f"{localdir}/datasets/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8.json",
            # f"{localdir}/datasets/SingleMuon.json",
            # f"{localdir}/datasets/EGamma.json",
            # f"{localdir}/datasets/DYJetsToLL_M-10to50_TuneCP5_13TeV-amcatnloFXFX-pythia8_fast.json",
            # f"{localdir}/datasets/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_fast.json",
            # f"{localdir}/datasets/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_fast.json",
            # f"{localdir}/datasets/WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8_fast.json",
            # f"{localdir}/datasets/skimmed.json",
            f"{localdir}/datasets/skimmed_rescale.json",
            f"{localdir}/datasets/rare_skim.json",
        ],

        "filter": {
            "samples": [
                # "WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8",
                "WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8",
                "WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8",
                "WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8",
                "WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8",
                "WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8",
                "WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8",
                "WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8",
                "WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8",
                "DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8",
                "DYJetsToLL_M-10to50_TuneCP5_13TeV-amcatnloFXFX-pythia8",
                "TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8",
                "TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8",

                "SingleMuon",
                "EGamma",

                "ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8",
                "ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8",
                "ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8",
                "ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8",
                "ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8",

                "ttWJets_TuneCP5_13TeV_madgraphMLM_pythia8",
                "ttZJets_TuneCP5_13TeV_madgraphMLM_pythia8",

                "GluGluWWToLNuQQ_TuneCP5_13TeV_madgraph-pythia8",
                "WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8",
                "WWZ_4F_TuneCP5_13TeV-amcatnlo-pythia8",
                "WZTo3LNu_mllmin01_NNPDF31_TuneCP5_13TeV_powheg_pythia8",
                "WZZ_TuneCP5_13TeV-amcatnlo-pythia8",
                "ZGToLLG_01J_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8",
                "ZZZ_TuneCP5_13TeV-amcatnlo-pythia8",

                "WminusTo2JZTo2LJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
                "WminusToLNuWminusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
                "WminusToLNuZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
                "WplusTo2JZTo2LJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
                "WplusTo2JWminusToLNuJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
                "WplusToLNuWminusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
                "WplusToLNuWplusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
                "WplusToLNuZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
                "ZTo2LZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
           
                # ###### SIGNAL #########
                "WminusTo2JZTo2LJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
                "WminusToLNuWminusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
                "WminusToLNuZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
                "WplusTo2JWminusToLNuJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8", #WplusTo2JWminusToLNuJJ missing in QCD
                "WplusTo2JZTo2LJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
                "WplusToLNuWminusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
                "WplusToLNuWplusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
                "WplusToLNuZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
                "ZTo2LZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",

            ],
            "year": ["2018"],
        },
    },
    workflow=VBSSemileptonicProcessor,

    skim=[
        get_nPVgood(1),
        eventFlags,
        goldenJson,
        nLepton_skim_cut,
        nJet_skim_cut,
        met_skim_cut,
        get_HLTsel(primaryDatasets=["SingleMuon", "EGamma"]),
    ],

    preselections=[vbs_semileptonic_presel],

    categories={
        "baseline": [passthrough],

        # ------------------------------------------------------------------
        # W control region
        # SR structure + inverted hadronic W mass window + mT > 30
        # ------------------------------------------------------------------
        "w_cr_mu": [w_cr_mu],
        "w_cr_e":  [w_cr_e],
        "w_cr_boosted_mu": [w_cr_boosted_mu],
        "w_cr_boosted_e": [w_cr_boosted_e],

        "ttbar_cr_boosted_mu": [ttbar_cr_boosted_mu],
        "ttbar_cr_boosted_e": [ttbar_cr_boosted_e],
        "ttbar_cr_resolved_mu": [ttbar_cr_resolved_mu],
        "ttbar_cr_resolved_e": [ttbar_cr_resolved_e],

        # "w_cr_no_fwd_mu": [w_cr_no_fwd_mu],
        # "w_cr_no_fwd_e": [w_cr_no_fwd_e],
        # "w_cr_loose_njet_mu": [w_cr_loose_njet_mu],
        # "w_cr_loose_njet_e": [w_cr_loose_njet_e],
        # "w_cr_no_fwd_loose_njet_mu": [w_cr_no_fwd_loose_njet_mu],
        # "w_cr_no_fwd_loose_njet_e": [w_cr_no_fwd_loose_njet_e],
        # "w_cr_sb_lo1_mu": [w_cr_sb_lo1_mu],
        # "w_cr_sb_lo2_mu": [w_cr_sb_lo2_mu],
        # "w_cr_sb_hi1_mu": [w_cr_sb_hi1_mu],
        # "w_cr_sb_hi2_mu": [w_cr_sb_hi2_mu],
        # "w_cr_sb_lo1_e":  [w_cr_sb_lo1_e],
        # "w_cr_sb_lo2_e":  [w_cr_sb_lo2_e],
        # "w_cr_sb_hi1_e":  [w_cr_sb_hi1_e],
        # "w_cr_sb_hi2_e":  [w_cr_sb_hi2_e],

        # ------------------------------------------------------------------
        # Validation region
        # SR structure exactly (no W mass window), mT in [20, 30]
        # ------------------------------------------------------------------
        "vr_mu": [vr_mu],
        "vr_e":  [vr_e],
        "vr_boosted_mu": [vr_boosted_mu],
        "vr_boosted_e": [vr_boosted_e],
        # "vr_boosted_no_fwd_mu": [vr_boosted_no_fwd_mu],
        # "vr_boosted_no_fwd_e": [vr_boosted_no_fwd_e],
        # "vr_qcd_enriched_mu": [vr_qcd_enriched_mu],
        # "vr_qcd_enriched_e": [vr_qcd_enriched_e],
        # "vr_no_fwd_mu": [vr_no_fwd_mu],
        # "vr_no_fwd_e": [vr_no_fwd_e],
        # "vr_loose_njet_mu": [vr_loose_njet_mu],
        # "vr_loose_njet_e": [vr_loose_njet_e],
        # "vr_no_fwd_loose_njet_mu": [vr_no_fwd_loose_njet_mu],
        # "vr_no_fwd_loose_njet_clip e": [vr_no_fwd_loose_njet_e],
        # "w_cr_incl_mu": [w_cr_incl_mu],
        # "w_cr_incl_e": [w_cr_incl_e],
        # "recoil_inclusive_mu": [recoil_inclusive_mu],
        # "recoil_inclusive_e": [recoil_inclusive_e],
        # "recoil_fullinclusive_mu": [recoil_fullinclusive_mu],
        # "recoil_fullinclusive_e": [recoil_fullinclusive_e],
        # "recoil_closure_mu": [recoil_closure_mu],
        # "recoil_closure_e": [recoil_closure_e],

        "boosted_e": [msd_window_cut_e],
        "boosted_mu": [msd_window_cut_mu],
        "resolved_mu":  [whad_window_cut_bveto_mu],
        "resolved_e": [whad_window_cut_bveto_e],
        
    },

    weights_classes=common_weights + [MuonGoodLeadWeight, ElectronGoodLeadWeight] + [PileupWeight] + [SF_L1prefiring] + [wjet_reweight]+[SF_ele_trigger]+[DPHI_SF]+[MuonGoodLeadWeightBoosted]+[ElectronGoodLeadWeightBoosted]+ [LHEScaleWeightWrapper, LHEPdfWeightWrapper]+[FatJetTau21Weight,FatJetWvsQCDWeight,QGTaggingWeight],
    weights={
        "common": {
            "inclusive": ["genWeight", "lumi", "XS", "PileupWeight", "sf_mu_id","sf_mu_iso","sf_ele_id","sf_ele_reco","sf_mu_trigger","sf_ele_trigger","sf_L1prefiring","sf_jet_puId","sf_partonshower_isr", "sf_partonshower_fsr", "sf_btag",  "LHEScaleWeight", "LHEPdfWeight", "sf_fj_WvsQCD","sf_fj_tau21","sf_qgtagging"],
            "bycategory": {
                "resolved_mu":           ["muon_inverttight_to_fake"],
                "resolved_e":            ["electron_inverttight_to_fake"],
                "ttbar_cr_resolved_mu":  ["muon_inverttight_to_fake"],
                "ttbar_cr_resolved_e":   ["electron_inverttight_to_fake"],
                "w_cr_mu":               ["muon_inverttight_to_fake"],
                "w_cr_e":                ["electron_inverttight_to_fake"],
                "vr_mu":                 ["muon_inverttight_to_fake"],
                "vr_e":                  ["electron_inverttight_to_fake"],
                "boosted_mu":            ["muon_inverttight_to_fake_boosted"],
                "boosted_e":             ["electron_inverttight_to_fake_boosted"],
                "ttbar_cr_boosted_mu":   ["muon_inverttight_to_fake_boosted"],
                "ttbar_cr_boosted_e":    ["electron_inverttight_to_fake_boosted"],
                "w_cr_boosted_mu":       ["muon_inverttight_to_fake_boosted"],
                "w_cr_boosted_e":        ["electron_inverttight_to_fake_boosted"],
                "vr_boosted_mu":         ["muon_inverttight_to_fake_boosted"],
                "vr_boosted_e":          ["electron_inverttight_to_fake_boosted"],
            }
        },
    },
    variations={
        "weights": {
            "common": {
                "inclusive": [],
                "bycategory": {
                    "resolved_mu":           ["muon_inverttight_to_fake"],
                    "resolved_e":            ["electron_inverttight_to_fake"],
                    "ttbar_cr_resolved_mu":  ["muon_inverttight_to_fake"],
                    "ttbar_cr_resolved_e":   ["electron_inverttight_to_fake"],
                    "w_cr_mu":               ["muon_inverttight_to_fake"],
                    "w_cr_e":                ["electron_inverttight_to_fake"],
                    "vr_mu":                 ["muon_inverttight_to_fake"],
                    "vr_e":                  ["electron_inverttight_to_fake"],
                    "boosted_mu":            ["muon_inverttight_to_fake_boosted"],
                    "boosted_e":             ["electron_inverttight_to_fake_boosted"],
                    "ttbar_cr_boosted_mu":   ["muon_inverttight_to_fake_boosted"],
                    "ttbar_cr_boosted_e":    ["electron_inverttight_to_fake_boosted"],
                    "w_cr_boosted_mu":       ["muon_inverttight_to_fake_boosted"],
                    "w_cr_boosted_e":        ["electron_inverttight_to_fake_boosted"],
                    "vr_boosted_mu":         ["muon_inverttight_to_fake_boosted"],
                    "vr_boosted_e":          ["electron_inverttight_to_fake_boosted"],
                }
            },
        },
        # "shape": {"common": {"inclusive": ['jet_calibration', 'electron_scale_and_smearing', 'muons_scale_and_resolution']}}
    },
    variables={
        "mT_lep_pt_corr": HistConf([
            Axis(coll="events", field="mt_w_leptonic_deepMET_resolutiontune", bins=30, start=0, stop=120, label=r"$m_T(W_{lep})$ [GeV]"),
            Axis(coll="events", field="w_lep_pt", bins=40, start=30.0, stop=200.0, label=r"$p_T^{lep_1}$ [GeV]"),
        ], storage="weight"),
        "mT_lep_eta_corr": HistConf([
            Axis(coll="events", field="mt_w_leptonic_deepMET_resolutiontune", bins=30, start=0, stop=120, label=r"$m_T(W_{lep})$ [GeV]"),
            Axis(coll="events", field="w_lep_eta", bins=32, start=-4.0, stop=4.0, label=r"$\eta^{lep_1}$ "),
        ], storage="weight"),
        "mT_lep_phi_corr": HistConf([
            Axis(coll="events", field="mt_w_leptonic_deepMET_resolutiontune", bins=30, start=0, stop=120, label=r"$m_T(W_{lep})$ [GeV]"),
            Axis(coll="events", field="w_lep_phi", bins=32, start=-4.0, stop=4.0, label=r"$\phi^{lep_1}$ "),
        ], storage="weight"),
        "mT_lep_dphi_corr": HistConf([
            Axis(coll="events", field="mt_w_leptonic_deepMET_resolutiontune", bins=30, start=0, stop=120, label=r"$m_T(W_{lep})$ [GeV]"),
            Axis(coll="events", field="lead_wlep_MET_dphi", bins=32, start=-4, stop=4.0, label=r"$|\Delta\phi_{l,MET}|$"),
        ], storage="weight"),
        "mT_MET_corr": HistConf([
            Axis(coll="events", field="mt_w_leptonic_deepMET_resolutiontune", bins=30, start=0, stop=120, label=r"$m_T(W_{lep})$ [GeV]"),
            Axis(coll="DeepMETResolutionTune", field="phi", bins=50, start=-4, stop=4, label=r"$\phi^{miss}$ [GeV]"),
        ], storage="weight"),
        "mT_METphi_corr": HistConf([
            Axis(coll="events", field="mt_w_leptonic_deepMET_resolutiontune", bins=30, start=0, stop=120, label=r"$m_T(W_{lep})$ [GeV]"),
            Axis(coll="DeepMETResolutionTune", field="phi", bins=50, start=-4, stop=4, label=r"$\phi^{miss}$ [GeV]"),
        ], storage="weight"),
        "lep_pt_MET_corr": HistConf([
            Axis(coll="events", field="w_lep_pt", bins=40, start=30.0, stop=200.0, label=r"$p_T^{lep_1}$ [GeV]"),
            Axis(coll="DeepMETResolutionTune", field="pt", bins=40, start=30.0, stop=200.0, label=r"$p_T^{miss}$ [GeV]"),
        ], storage="weight"),
        "lep_phi_METphi_corr": HistConf([
            Axis(coll="events", field="w_lep_phi", bins=32, start=-4.0, stop=4.0, label=r"$\phi^{lep_1}$ "),
            Axis(coll="DeepMETResolutionTune", field="phi", bins=50, start=-4, stop=4, label=r"$\phi^{miss}$ [GeV]"),
        ], storage="weight"),
        "nJets":          HistConf([Axis(coll="events", field="nJetGood",           bins=12, start=0,    stop=12,   label="N(jets)")]),
        "nBJets":         HistConf([Axis(coll="events", field="nBJetGood",          bins=8,  start=0,    stop=8,    label="N(bjets)")]),
        "nCentralJets":   HistConf([Axis(coll="events", field="nCentralJetsGood",   bins=12, start=0,    stop=12,   label="N(Central Jets)")]),
        "nFatJets":       HistConf([Axis(coll="events", field="nFatJetGood",        bins=4,  start=0,    stop=4,    label="N(Fat Jets)")]),
        "nFatJetCentral": HistConf([Axis(coll="events", field="nFatJetCentral",     bins=4,  start=0,    stop=4,    label="N(Central Fat Jets)")]),
        "nLeptonLoose":   HistConf([Axis(coll="events", field="nLeptonLoose",       bins=4,  start=0,    stop=4,    label="N(Lepton Loose)")]),
        "nFatJetCandidate": HistConf([Axis(coll="events", field="nFatJetCandidate", bins=4,  start=0,    stop=4,    label="N(Candidate Fat Jets)")]),
        "nFarAK4Jets":    HistConf([Axis(coll="events", field="nFarAK4Jets",        bins=8,  start=0,    stop=8,    label="N(nFarAK4Jets)")]),
        "nMuonGood":      HistConf([Axis(coll="events", field="nMuonGood",          bins=6,  start=0,    stop=6,    label="N(muon good)")]),
        "nElectronGood":  HistConf([Axis(coll="events", field="nElectronGood",      bins=6,  start=0,    stop=6,    label="N(electron good)")]),
        "nLeptonGood":    HistConf([Axis(coll="events", field="nLeptonGood",        bins=6,  start=0,    stop=6,    label="N(lepton good)")]),
        "btagDeepFlavB":  HistConf([Axis(coll="JetGood", field="btagDeepFlavB",     bins=20, start=0,    stop=1,    label="deepFlavB discrim score")]),
        "btagDeepB":      HistConf([Axis(coll="JetGood", field="btagDeepB",         bins=20, start=0,    stop=1,    label="deepCSV discrim score")]),
        "leading_bscore": HistConf([Axis(coll="events", field="leading_bscore",     bins=33, start=0,    stop=1,    label="max(deepFlavB discrim score)")]),
        "met":            HistConf([Axis(coll="MET",    field="pt",                  bins=50, start=0,    stop=250,  label=r"$p_T^{miss}$ [GeV]")]),
        "met_phi":        HistConf([Axis(coll="MET",    field="phi",                 bins=50, start=-4,   stop=4,    label=r"$\phi^{miss}$ [GeV]")]),
        "puppimet":       HistConf([Axis(coll="PuppiMET", field="pt",               bins=50, start=0,    stop=250,  label=r"$p_T^{miss}$ [GeV]")]),
        "puppimet_phi":   HistConf([Axis(coll="PuppiMET", field="phi",              bins=50, start=-4,   stop=4,    label=r"$\phi^{miss}$ [GeV]")]),
        "DeepMETResponseTune_pt":  HistConf([Axis(coll="DeepMETResponseTune",  field="pt",  bins=50, start=0,  stop=250, label=r"$p_T^{miss}$ [GeV]")]),
        "DeepMETResponseTune_phi": HistConf([Axis(coll="DeepMETResponseTune",  field="phi", bins=50, start=-4, stop=4,   label=r"$\phi^{miss}$ [GeV]")]),
        "DeepMETResolutionTune_pt":  HistConf([Axis(coll="DeepMETResolutionTune", field="pt",  bins=50, start=0,  stop=250, label=r"$p_T^{miss}$ [GeV]")]),
        "DeepMETResolutionTune_phi": HistConf([Axis(coll="DeepMETResolutionTune", field="phi", bins=50, start=-4, stop=4,   label=r"$\phi^{miss}$ [GeV]")]),
        "mt_w_lep":       HistConf([Axis(coll="events", field="mt_w_leptonic", bins=30, start=0, stop=200, label=r"$m_T(W_{lep})$ [GeV]")]),
        "neutrino_pz":    HistConf([Axis(coll="events", field="neutrino_pz",               bins=50, start=0,    stop=250, label=r"$p_z^{\nu}$ [GeV]")]),
        "neutrino_eta":   HistConf([Axis(coll="events", field="neutrino_eta",              bins=32, start=-4.0, stop=4.0, label=r"$\eta^{\nu}$")]),
        "neutrino_deta":  HistConf([Axis(coll="events", field="lead_wlep_neutrino_deta",   bins=32, start=0,    stop=4.0, label=r"$\delta\eta^{l,\nu}$")]),
        "neutrino_dR":    HistConf([Axis(coll="events", field="lead_wlep_neutrino_dR",     bins=32, start=0.0,  stop=4,   label=r"$\delta R^{l,\nu}$")]),
        "mjj_vbs":        HistConf([Axis(coll="vbsjets", field="mass",        bins=50, start=300, stop=4000, label=r"$M_{jj}^{forward}$ [GeV]")]),
        "deta_vbs":       HistConf([Axis(coll="vbsjets", field="delta_eta",   bins=36, start=0,   stop=9.0,  label=r"$|\Delta\eta_{jj}^{forward}|$")]),
        "dR_vbs":         HistConf([Axis(coll="events", field="vbs_dR",       bins=40, start=0.0, stop=7.0,  label=r"$\Delta R(jj)^{forward}$")]),
        "dR_fj_vbs1":     HistConf([Axis(coll="events", field="vbs1_fj_dR",   bins=40, start=0.0, stop=7.0,  label=r"$\Delta R(AK8, j_{forward_1})$")]),
        "dR_fj_vbs2":     HistConf([Axis(coll="events", field="vbs2_fj_dR",   bins=40, start=0.0, stop=7.0,  label=r"$\Delta R(AK8, j_{forward_2})$")]),
        "jet_eta":        HistConf([Axis(coll="JetGood", field="eta",          bins=48, start=-4.8, stop=4.8, label="JetGood eta")]),
        "jet_id":         HistConf([Axis(coll="JetGood", field="jetId",        bins=10, start=0,   stop=10,   label="Jet id")]),
        "jet_rel_iso":    HistConf([Axis(coll="LeptonGood", field="jetRelIso", bins=50, start=0,   stop=2,    label="Jet iso in lep")]),
        "dxy_mu":         HistConf([Axis(coll="LeptonGood", field="dxy",       bins=50, start=0,   stop=0.5,  label="dxy mu")]),
        "dxy_ele":        HistConf([Axis(coll="LeptonGood", field="dxy",       bins=50, start=0,   stop=0.2,  label="dxy ele")]),
        "dz_mu":          HistConf([Axis(coll="LeptonGood", field="dz",        bins=50, start=0,   stop=1,    label="dz mu")]),
        "dz_ele":         HistConf([Axis(coll="LeptonGood", field="dz",        bins=50, start=0,   stop=0.5,  label="dz ele")]),
        "m_jj_w":         HistConf([Axis(coll="w_had_jets", field="mass",      bins=40, start=65,  stop=105,  label=r"$M_{jj}^{had. \, V}$ [GeV]")]),
        "pt_jj_w":        HistConf([Axis(coll="w_had_jets", field="pt",        bins=40, start=40,  stop=210,  label=r"$p_T(jj)^{had. \, V}$ [GeV]")]),
        "dR_w_had":       HistConf([Axis(coll="events", field="w_had_dR",      bins=40, start=0.0, stop=4.0,  label=r"$\Delta R(jj)^{had. \, V}$")]),
        "eta_w_had1":     HistConf([Axis(coll="events", field="w_had_jet1_eta", bins=48, start=-4.0, stop=4.0, label=r"$\eta(j1^{had. \, V})$")]),
        "eta_w_had2":     HistConf([Axis(coll="events", field="w_had_jet2_eta", bins=48, start=-4.0, stop=4.0, label=r"$\eta(j2^{had. \, V})$")]),
        "pt_w_had1":      HistConf([Axis(coll="events", field="w_had_jet1_pt",  bins=60, start=0.0, stop=300.0, label=r"$p_T(j1^{had. \, V})$ [GeV]")]),
        "pt_w_had2":      HistConf([Axis(coll="events", field="w_had_jet2_pt",  bins=60, start=0.0, stop=300.0, label=r"$p_T(j2^{had. \, V})$ [GeV]")]),
        "pt_tag1":        HistConf([Axis(coll="events", field="jet1_pt",  bins=60, start=0, stop=300, label=r"$p_T(j_1)$ [GeV]")]),
        "pt_tag2":        HistConf([Axis(coll="events", field="jet2_pt",  bins=60, start=0, stop=300, label=r"$p_T(j_2)$ [GeV]")]),
        "eta_tag1":       HistConf([Axis(coll="events", field="jet1_eta", bins=48, start=-4.8, stop=4.8, label=r"$\eta(j_1)$")]),
        "eta_tag2":       HistConf([Axis(coll="events", field="jet2_eta", bins=48, start=-4.8, stop=4.8, label=r"$\eta(j_2)$")]),
        "eta_w_lep":      HistConf([Axis(coll="events", field="w_lep_eta", bins=32, start=-4.0, stop=4.0, label=r"$\eta^{lep_1}$")]),
        "pt_w_lep":       HistConf([Axis(coll="events", field="w_lep_pt",  bins=40, start=0.0,  stop=300.0, label=r"$p_T^{lep_1}$ [GeV]")]),
        "phi_w_lep":      HistConf([Axis(coll="events", field="w_lep_phi", bins=32, start=-4.0, stop=4.0,   label=r"$\phi^{lep_1}$")]),
        "m_ll":           HistConf([Axis(coll="ll", field="m_ll", bins=50, start=50, stop=125.0, label=r"$m_{ll}$ [GeV]")]),
        "lead_wlep_wjet1_dR":       HistConf([Axis(coll="events", field="lead_wlep_wjet1_dR",       bins=40, start=0.0, stop=4.0, label=r"$\Delta R(l,j_1)^{had. \, V}$")]),
        "lead_wlep_wjet2_dR":       HistConf([Axis(coll="events", field="lead_wlep_wjet2_dR",       bins=40, start=0.0, stop=4.0, label=r"$\Delta R(l,j_2)^{had. \, V}$")]),
        "lead_wlep_wfatjet1_dR":    HistConf([Axis(coll="events", field="lead_wlep_wfatjet1_dR",    bins=40, start=0.0, stop=4.0, label=r"$\Delta R(l,AK8)$")]),
        "lead_wlep_w_resolved_dR":  HistConf([Axis(coll="events", field="lead_wlep_w_resolved_dR",  bins=40, start=0.0, stop=4.0, label=r"$\Delta R(l,W_{resolved})$")]),
        "lead_wlep_vbsjet1_dR":     HistConf([Axis(coll="events", field="lead_wlep_vbsjet1_dR",     bins=40, start=0.0, stop=4.0, label=r"$\Delta R(l,j_1)^{forward}$")]),
        "lead_wlep_vbsjet2_dR":     HistConf([Axis(coll="events", field="lead_wlep_vbsjet2_dR",     bins=40, start=0.0, stop=4.0, label=r"$\Delta R(l,j_2)^{forward}$")]),
        "lead_wlep_wfatjet1_deta":  HistConf([Axis(coll="events", field="lead_wlep_wfatjet1_deta",  bins=24, start=0.0, stop=9.0, label=r"$|\Delta\eta_{l,J}^{W}|$")]),
        "lead_wlep_wjet1_deta":     HistConf([Axis(coll="events", field="lead_wlep_wjet1_deta",     bins=24, start=0.0, stop=9.0, label=r"$|\Delta\eta_{l,j_1^{had. \, V}}|$")]),
        "lead_wlep_wjet2_deta":     HistConf([Axis(coll="events", field="lead_wlep_wjet2_deta",     bins=24, start=0.0, stop=9.0, label=r"$|\Delta\eta_{l,j_2^{had. \, V}}|$")]),
        "lead_wlep_w_resolved_deta": HistConf([Axis(coll="events", field="lead_wlep_w_resolved_deta", bins=24, start=2.0, stop=9.0, label=r"$|\Delta\eta_{l,W_{resolved}}|$")]),
        "lead_wlep_MET_dphi":       HistConf([Axis(coll="events", field="lead_wlep_MET_dphi",       bins=32, start=-4, stop=4.0, label=r"$|\Delta\phi_{l,MET}|$")]),
        "lead_wlep_wfatjet1_dphi":  HistConf([Axis(coll="events", field="lead_wlep_wfatjet1_dphi",  bins=32, start=-4, stop=4.0, label=r"$|\Delta\phi_{l,AK8}|$")]),
        "lead_wlep_wjet1_dphi":     HistConf([Axis(coll="events", field="lead_wlep_wjet1_dphi",     bins=32, start=-4, stop=4.0, label=r"$|\Delta\phi_{l,j_1^{had. \, V}}|$")]),
        "lead_wlep_wjet2_dphi":     HistConf([Axis(coll="events", field="lead_wlep_wjet2_dphi",     bins=32, start=-4, stop=4.0, label=r"$|\Delta\phi_{l,j_2^{had. \, V}}|$")]),
        "w_lep_w_resolved_dphi":    HistConf([Axis(coll="events", field="w_lep_w_resolved_dphi",    bins=32, start=-4, stop=4.0, label=r"$|\Delta\phi_{W_{lep},W_{res}}|$")]),
        "w_lep_w_boost_dphi":       HistConf([Axis(coll="events", field="w_lep_w_boost_dphi",       bins=32, start=-4, stop=4.0, label=r"$|\Delta\phi_{W_{lep},W_{boost}}|$")]),
        "wleptonic_eta":  HistConf([Axis(coll="events", field="wleptonic_eta", bins=48, start=-2.4, stop=2.4, label=r"$\eta(W_{leptonic})$")]),
        "wleptonic_pt":   HistConf([Axis(coll="events", field="wleptonic_pt",  bins=48, start=0,    stop=500, label=r"$p_T(W_{leptonic})$")]),
        "fj_pt":          HistConf([Axis(coll="candidate_boost", field="pt",           bins=60, start=150, stop=1000, label=r"$p_T(AK8)$ [GeV]")]),
        "fj_eta":         HistConf([Axis(coll="candidate_boost", field="eta",          bins=48, start=-2.4, stop=2.4,  label=r"$\eta(AK8)$")]),
        "fj_msd":         HistConf([Axis(coll="candidate_boost", field="msoftdrop",    bins=40, start=0,    stop=200,  label=r"$m_{SD}(AK8)$ [GeV]")]),
        "fj_t21":         HistConf([Axis(coll="candidate_boost", field="tau21",        bins=32, start=0,    stop=1.1,  label=r"$\tau_{21}$")]),
        "fj_W_vs_QCD":    HistConf([Axis(coll="candidate_boost", field="particleNet_WvsQCD", bins=32, start=0, stop=1.1, label=r"particleNet_WvsQCD")]),
        "fj_Z_vs_QCD":    HistConf([Axis(coll="candidate_boost", field="particleNet_ZvsQCD", bins=32, start=0, stop=1.1, label=r"particleNet_ZvsQCD")]),
        "fj_pn_mass":     HistConf([Axis(coll="candidate_boost", field="particleNet_mass",   bins=15, start=40, stop=115, label=r"particleNet_mass")]),
        "fj_W_vs_QCD_deeptag":   HistConf([Axis(coll="candidate_boost", field="deepTag_WvsQCD",   bins=32, start=0, stop=1.1, label=r"deepTag_WvsQCD")]),
        "fj_Z_vs_QCD_deeptag":   HistConf([Axis(coll="candidate_boost", field="deepTag_ZvsQCD",   bins=32, start=0, stop=1.1, label=r"deepTag_ZvsQCD")]),
        "fj_W_vs_QCD_deeptagMD": HistConf([Axis(coll="candidate_boost", field="deepTagMD_WvsQCD", bins=32, start=0, stop=1.1, label=r"deepTagMD_WvsQCD")]),
        "fj_Z_vs_QCD_deeptagMD": HistConf([Axis(coll="candidate_boost", field="deepTagMD_ZvsQCD", bins=32, start=0, stop=1.1, label=r"deepTagMD_ZvsQCD")]),
        "z_lep":          HistConf([Axis(coll="events", field="z_lep", bins=40, start=-1.0, stop=1.0, label=r"$Zepp. lepton$")]),
        "z_fat":          HistConf([Axis(coll="events", field="z_fat", bins=40, start=-1.0, stop=1.0, label=r"$Zepp. boosted jet$")]),
        "centrality_resolved": HistConf([Axis(coll="w_had_jets", field="centrality_resolved", bins=40, start=-5.0, stop=5.0, label=r"$Centrality_{resolved}$")]),
        "centrality_boosted":  HistConf([Axis(coll="events",    field="centrality_boosted",  bins=40, start=-5.0, stop=5.0, label=r"$Centrality_{boosted}$")]),
        "qgl_vbs1_resolved": HistConf([Axis(coll="events", field="qgl_vbs1_resolved", bins=40, start=0, stop=1.0, label=r"QGL VBS jet1 (resolved)")]),
        "qgl_vbs2_resolved": HistConf([Axis(coll="events", field="qgl_vbs2_resolved", bins=40, start=0, stop=1.0, label=r"QGL VBS jet2 (resolved)")]),
        "qgl_wjet1_resolved": HistConf([Axis(coll="w_had_jets", field="qgl_wjet1_resolved", bins=40, start=0, stop=1.0, label=r"QGL had. W jet 1")]),
        "qgl_wjet2_resolved": HistConf([Axis(coll="w_had_jets", field="qgl_wjet2_resolved", bins=40, start=0, stop=1.0, label=r"QGL had. W jet 2")]),
        "pt_vbsjet1":     HistConf([Axis(coll="events", field="vbsjet1_pt",  bins=60, start=0,    stop=300, label=r"$p_T(j_1)^{forward}$ [GeV]")]),
        "pt_vbsjet2":     HistConf([Axis(coll="events", field="vbsjet2_pt",  bins=60, start=0,    stop=300, label=r"$p_T(j_2)^{forward}$ [GeV]")]),
        "eta_vbsjet1":    HistConf([Axis(coll="events", field="vbsjet1_eta", bins=48, start=-4.8, stop=4.8, label=r"$\eta(j_1)^{forward}$")]),
        "eta_vbsjet2":    HistConf([Axis(coll="events", field="vbsjet2_eta", bins=48, start=-4.8, stop=4.8, label=r"$\eta(j_2)^{forward}$")]),
        "bjet_lepton_separation": HistConf([Axis(coll="events", field="lep_bjet_dR", bins=32, start=0, stop=9.0, label=r"$\Delta R_{lep,b}$")]),
        "LeadJetIdx":     HistConf([Axis(coll="events", field="jet1_idx", bins=10, start=-1, stop=10, label="tag 1 idx")]),
        "SecondJetIdx":   HistConf([Axis(coll="events", field="jet2_idx", bins=10, start=-1, stop=10, label="tag 2 idx")]),
        "jet_new_neHEF":  HistConf([Axis(coll="JetGood", field="neHEF", bins=10,  start=0, stop=1.2, label="neHEF (good jets)")]),
        "jet_new_chHEF":  HistConf([Axis(coll="JetGood", field="chHEF", bins=100, start=0, stop=1.2, label="chHEF (good jets)")]),
        "jet_new_muEF":   HistConf([Axis(coll="JetGood", field="muEF",  bins=10,  start=0, stop=1.2, label="muEF (good jets)")]),
        "jet_new_neEmEF": HistConf([Axis(coll="JetGood", field="neEmEF",bins=10,  start=0, stop=1.2, label="neEmEF (good jets)")]),
        "jet_new_eta":    HistConf([Axis(coll="JetGood", field="eta",   bins=10,  start=-4.8, stop=4.8, label=r"$\eta(jet_{good})$")]),
        "HT_sum":         HistConf([Axis(coll="events", field="ht_sum", bins=35, start=0, stop=3500, label="reco HT [GeV]")]),
        "bdt_boosted_mu":   HistConf([Axis(coll="events", field="bdt_boosted_mu",   bins=40, start=0, stop=1, label="BDT mu boosted")]),
        "bdt_resolved_mu":  HistConf([Axis(coll="events", field="bdt_resolved_mu",  bins=40, start=0, stop=1, label="BDT mu resolved")]),
        "bdt_boosted_e":    HistConf([Axis(coll="events", field="bdt_boosted_e",    bins=40, start=0, stop=1, label="BDT e boosted")]),
        "bdt_resolved_e":   HistConf([Axis(coll="events", field="bdt_resolved_e",   bins=40, start=0, stop=1, label="BDT e resolved")]),
        "mass_jet1_jet2":   HistConf([Axis(coll="mass", field="jet1_jet2",  bins=100, start=0, stop=2000, label="mass jet1 jet2")]),
        "mass_jet1_jet3":   HistConf([Axis(coll="mass", field="jet1_jet3",  bins=100, start=0, stop=2000, label="mass jet1 jet3")]),
        "mass_jet1_jet4":   HistConf([Axis(coll="mass", field="jet1_jet4",  bins=100, start=0, stop=2000, label="mass jet1 jet4")]),
        "mass_jet1_jet5":   HistConf([Axis(coll="mass", field="jet1_jet5",  bins=100, start=0, stop=2000, label="mass jet1 jet5")]),
        "mass_jet1_jet6":   HistConf([Axis(coll="mass", field="jet1_jet6",  bins=100, start=0, stop=2000, label="mass jet1 jet6")]),
        "mass_jet1_lepton1": HistConf([Axis(coll="mass", field="jet1_lepton1", bins=100, start=0, stop=2000, label="mass jet1 lepton1")]),
        "mass_jet2_jet3":   HistConf([Axis(coll="mass", field="jet2_jet3",  bins=100, start=0, stop=2000, label="mass jet2 jet3")]),
        "mass_jet2_jet4":   HistConf([Axis(coll="mass", field="jet2_jet4",  bins=100, start=0, stop=2000, label="mass jet2 jet4")]),
        "mass_jet2_jet5":   HistConf([Axis(coll="mass", field="jet2_jet5",  bins=100, start=0, stop=2000, label="mass jet2 jet5")]),
        "mass_jet2_jet6":   HistConf([Axis(coll="mass", field="jet2_jet6",  bins=100, start=0, stop=2000, label="mass jet2 jet6")]),
        "mass_jet2_lepton1": HistConf([Axis(coll="mass", field="jet2_lepton1", bins=100, start=0, stop=2000, label="mass jet2 lepton1")]),
        "mass_jet3_jet4":   HistConf([Axis(coll="mass", field="jet3_jet4",  bins=100, start=0, stop=2000, label="mass jet3 jet4")]),
        "mass_jet3_jet5":   HistConf([Axis(coll="mass", field="jet3_jet5",  bins=100, start=0, stop=2000, label="mass jet3 jet5")]),
        "mass_jet3_jet6":   HistConf([Axis(coll="mass", field="jet3_jet6",  bins=100, start=0, stop=2000, label="mass jet3 jet6")]),
        "mass_jet3_lepton1": HistConf([Axis(coll="mass", field="jet3_lepton1", bins=100, start=0, stop=2000, label="mass jet3 lepton1")]),
        "mass_jet4_jet5":   HistConf([Axis(coll="mass", field="jet4_jet5",  bins=100, start=0, stop=2000, label="mass jet4 jet5")]),
        "mass_jet4_jet6":   HistConf([Axis(coll="mass", field="jet4_jet6",  bins=100, start=0, stop=2000, label="mass jet4 jet6")]),
        "mass_jet4_lepton1": HistConf([Axis(coll="mass", field="jet4_lepton1", bins=100, start=0, stop=2000, label="mass jet4 lepton1")]),
        "mass_jet5_jet6":   HistConf([Axis(coll="mass", field="jet5_jet6",  bins=100, start=0, stop=2000, label="mass jet5 jet6")]),
        "mass_jet5_lepton1": HistConf([Axis(coll="mass", field="jet5_lepton1", bins=100, start=0, stop=2000, label="mass jet5 lepton1")]),
        "mass_jet6_lepton1": HistConf([Axis(coll="mass", field="jet6_lepton1", bins=100, start=0, stop=2000, label="mass jet6 lepton1")]),
        "mass_jet1_DeepMETResolutionTune": HistConf([Axis(coll="mass", field="jet1_DeepMETResolutionTune", bins=100, start=0, stop=2000, label="mass jet1 met")]),
        "mass_jet2_DeepMETResolutionTune": HistConf([Axis(coll="mass", field="jet2_DeepMETResolutionTune", bins=100, start=0, stop=2000, label="mass jet2 met")]),
        "mass_jet3_DeepMETResolutionTune": HistConf([Axis(coll="mass", field="jet3_DeepMETResolutionTune", bins=100, start=0, stop=2000, label="mass jet3 met")]),
        "mass_jet4_DeepMETResolutionTune": HistConf([Axis(coll="mass", field="jet4_DeepMETResolutionTune", bins=100, start=0, stop=2000, label="mass jet4 met")]),
        "mass_jet5_DeepMETResolutionTune": HistConf([Axis(coll="mass", field="jet5_DeepMETResolutionTune", bins=100, start=0, stop=2000, label="mass jet5 met")]),
        "mass_jet6_DeepMETResolutionTune": HistConf([Axis(coll="mass", field="jet6_DeepMETResolutionTune", bins=100, start=0, stop=2000, label="mass jet6 met")]),
        "mass_lepton1_DeepMETResolutionTune": HistConf([Axis(coll="mass", field="lepton1_DeepMETResolutionTune", bins=100, start=0, stop=2000, label="mass lepton1 met")]),
        "dR_jet1_jet2":    HistConf([Axis(coll="dR", field="jet1_jet2",    bins=50, start=0, stop=6, label="dR jet1 jet2")]),
        "dR_jet1_jet3":    HistConf([Axis(coll="dR", field="jet1_jet3",    bins=50, start=0, stop=6, label="dR jet1 jet3")]),
        "dR_jet1_jet4":    HistConf([Axis(coll="dR", field="jet1_jet4",    bins=50, start=0, stop=6, label="dR jet1 jet4")]),
        "dR_jet1_jet5":    HistConf([Axis(coll="dR", field="jet1_jet5",    bins=50, start=0, stop=6, label="dR jet1 jet5")]),
        "dR_jet1_jet6":    HistConf([Axis(coll="dR", field="jet1_jet6",    bins=50, start=0, stop=6, label="dR jet1 jet6")]),
        "dR_jet1_lepton1": HistConf([Axis(coll="dR", field="jet1_lepton1", bins=50, start=0, stop=6, label="dR jet1 lepton1")]),
        "dR_jet2_jet3":    HistConf([Axis(coll="dR", field="jet2_jet3",    bins=50, start=0, stop=6, label="dR jet2 jet3")]),
        "dR_jet2_jet4":    HistConf([Axis(coll="dR", field="jet2_jet4",    bins=50, start=0, stop=6, label="dR jet2 jet4")]),
        "dR_jet2_jet5":    HistConf([Axis(coll="dR", field="jet2_jet5",    bins=50, start=0, stop=6, label="dR jet2 jet5")]),
        "dR_jet2_jet6":    HistConf([Axis(coll="dR", field="jet2_jet6",    bins=50, start=0, stop=6, label="dR jet2 jet6")]),
        "dR_jet2_lepton1": HistConf([Axis(coll="dR", field="jet2_lepton1", bins=50, start=0, stop=6, label="dR jet2 lepton1")]),
        "dR_jet3_jet4":    HistConf([Axis(coll="dR", field="jet3_jet4",    bins=50, start=0, stop=6, label="dR jet3 jet4")]),
        "dR_jet3_jet5":    HistConf([Axis(coll="dR", field="jet3_jet5",    bins=50, start=0, stop=6, label="dR jet3 jet5")]),
        "dR_jet3_jet6":    HistConf([Axis(coll="dR", field="jet3_jet6",    bins=50, start=0, stop=6, label="dR jet3 jet6")]),
        "dR_jet3_lepton1": HistConf([Axis(coll="dR", field="jet3_lepton1", bins=50, start=0, stop=6, label="dR jet3 lepton1")]),
        "dR_jet4_jet5":    HistConf([Axis(coll="dR", field="jet4_jet5",    bins=50, start=0, stop=6, label="dR jet4 jet5")]),
        "dR_jet4_jet6":    HistConf([Axis(coll="dR", field="jet4_jet6",    bins=50, start=0, stop=6, label="dR jet4 jet6")]),
        "dR_jet4_lepton1": HistConf([Axis(coll="dR", field="jet4_lepton1", bins=50, start=0, stop=6, label="dR jet4 lepton1")]),
        "dR_jet5_jet6":    HistConf([Axis(coll="dR", field="jet5_jet6",    bins=50, start=0, stop=6, label="dR jet5 jet6")]),
        "dR_jet5_lepton1": HistConf([Axis(coll="dR", field="jet5_lepton1", bins=50, start=0, stop=6, label="dR jet5 lepton1")]),
        "dR_jet6_lepton1": HistConf([Axis(coll="dR", field="jet6_lepton1", bins=50, start=0, stop=6, label="dR jet6 lepton1")]),
        "dR_jet1_DeepMETResolutionTune": HistConf([Axis(coll="dR", field="jet1_DeepMETResolutionTune", bins=50, start=0, stop=6, label="dR jet1 met")]),
        "dR_jet2_DeepMETResolutionTune": HistConf([Axis(coll="dR", field="jet2_DeepMETResolutionTune", bins=50, start=0, stop=6, label="dR jet2 met")]),
        "dR_jet3_DeepMETResolutionTune": HistConf([Axis(coll="dR", field="jet3_DeepMETResolutionTune", bins=50, start=0, stop=6, label="dR jet3 met")]),
        "dR_jet4_DeepMETResolutionTune": HistConf([Axis(coll="dR", field="jet4_DeepMETResolutionTune", bins=50, start=0, stop=6, label="dR jet4 met")]),
        "dR_jet5_DeepMETResolutionTune": HistConf([Axis(coll="dR", field="jet5_DeepMETResolutionTune", bins=50, start=0, stop=6, label="dR jet5 met")]),
        "dR_jet6_DeepMETResolutionTune": HistConf([Axis(coll="dR", field="jet6_DeepMETResolutionTune", bins=50, start=0, stop=6, label="dR jet6 met")]),
        "dR_lepton1_DeepMETResolutionTune": HistConf([Axis(coll="dR", field="lepton1_DeepMETResolutionTune", bins=50, start=0, stop=6, label="dR lepton1 met")]),
        "dphi_jet1_jet2":    HistConf([Axis(coll="dphi", field="jet1_jet2",    bins=64, start=-3.2, stop=3.2, label="dphi jet1 jet2")]),
        "dphi_jet1_jet3":    HistConf([Axis(coll="dphi", field="jet1_jet3",    bins=64, start=-3.2, stop=3.2, label="dphi jet1 jet3")]),
        "dphi_jet1_jet4":    HistConf([Axis(coll="dphi", field="jet1_jet4",    bins=64, start=-3.2, stop=3.2, label="dphi jet1 jet4")]),
        "dphi_jet1_lepton1": HistConf([Axis(coll="dphi", field="jet1_lepton1", bins=64, start=-3.2, stop=3.2, label="dphi jet1 lepton1")]),
        "dphi_jet2_jet3":    HistConf([Axis(coll="dphi", field="jet2_jet3",    bins=64, start=-3.2, stop=3.2, label="dphi jet2 jet3")]),
        "dphi_jet2_jet4":    HistConf([Axis(coll="dphi", field="jet2_jet4",    bins=64, start=-3.2, stop=3.2, label="dphi jet2 jet4")]),
        "dphi_jet2_lepton1": HistConf([Axis(coll="dphi", field="jet2_lepton1", bins=64, start=-3.2, stop=3.2, label="dphi jet2 lepton1")]),
        "dphi_jet3_jet4":    HistConf([Axis(coll="dphi", field="jet3_jet4",    bins=64, start=-3.2, stop=3.2, label="dphi jet3 jet4")]),
        "dphi_jet3_lepton1": HistConf([Axis(coll="dphi", field="jet3_lepton1", bins=64, start=-3.2, stop=3.2, label="dphi jet3 lepton1")]),
        "dphi_jet4_lepton1": HistConf([Axis(coll="dphi", field="jet4_lepton1", bins=64, start=-3.2, stop=3.2, label="dphi jet4 lepton1")]),
        "dphi_jet1_DeepMETResolutionTune": HistConf([Axis(coll="dphi", field="jet1_DeepMETResolutionTune", bins=64, start=-3.2, stop=3.2, label="dphi jet1 met")]),
        "dphi_jet2_DeepMETResolutionTune": HistConf([Axis(coll="dphi", field="jet2_DeepMETResolutionTune", bins=64, start=-3.2, stop=3.2, label="dphi jet2 met")]),
        "dphi_jet3_DeepMETResolutionTune": HistConf([Axis(coll="dphi", field="jet3_DeepMETResolutionTune", bins=64, start=-3.2, stop=3.2, label="dphi jet3 met")]),
        "dphi_jet4_DeepMETResolutionTune": HistConf([Axis(coll="dphi", field="jet4_DeepMETResolutionTune", bins=64, start=-3.2, stop=3.2, label="dphi jet4 met")]),
        "dphi_lepton1_DeepMETResolutionTune": HistConf([Axis(coll="dphi", field="lepton1_DeepMETResolutionTune", bins=64, start=-3.2, stop=3.2, label="dphi lepton1 met")]),
        "deta_jet1_jet2":    HistConf([Axis(coll="deta", field="jet1_jet2",    bins=50, start=-5, stop=5, label="deta jet1 jet2")]),
        "deta_jet1_jet3":    HistConf([Axis(coll="deta", field="jet1_jet3",    bins=50, start=-5, stop=5, label="deta jet1 jet3")]),
        "deta_jet1_jet4":    HistConf([Axis(coll="deta", field="jet1_jet4",    bins=50, start=-5, stop=5, label="deta jet1 jet4")]),
        "deta_jet1_lepton1": HistConf([Axis(coll="deta", field="jet1_lepton1", bins=50, start=-5, stop=5, label="deta jet1 lepton1")]),
        "deta_jet2_jet3":    HistConf([Axis(coll="deta", field="jet2_jet3",    bins=50, start=-5, stop=5, label="deta jet2 jet3")]),
        "deta_jet2_jet4":    HistConf([Axis(coll="deta", field="jet2_jet4",    bins=50, start=-5, stop=5, label="deta jet2 jet4")]),
        "deta_jet2_lepton1": HistConf([Axis(coll="deta", field="jet2_lepton1", bins=50, start=-5, stop=5, label="deta jet2 lepton1")]),
        "deta_jet3_jet4":    HistConf([Axis(coll="deta", field="jet3_jet4",    bins=50, start=-5, stop=5, label="deta jet3 jet4")]),
        "deta_jet3_lepton1": HistConf([Axis(coll="deta", field="jet3_lepton1", bins=50, start=-5, stop=5, label="deta jet3 lepton1")]),
        "deta_jet4_lepton1": HistConf([Axis(coll="deta", field="jet4_lepton1", bins=50, start=-5, stop=5, label="deta jet4 lepton1")]),
        "deta_jet1_DeepMETResolutionTune": HistConf([Axis(coll="deta", field="jet1_DeepMETResolutionTune", bins=50, start=-5, stop=5, label="deta jet1 met")]),
        "deta_jet2_DeepMETResolutionTune": HistConf([Axis(coll="deta", field="jet2_DeepMETResolutionTune", bins=50, start=-5, stop=5, label="deta jet2 met")]),
        "deta_jet3_DeepMETResolutionTune": HistConf([Axis(coll="deta", field="jet3_DeepMETResolutionTune", bins=50, start=-5, stop=5, label="deta jet3 met")]),
        "deta_jet4_DeepMETResolutionTune": HistConf([Axis(coll="deta", field="jet4_DeepMETResolutionTune", bins=50, start=-5, stop=5, label="deta jet4 met")]),
        "deta_lepton1_DeepMETResolutionTune": HistConf([Axis(coll="deta", field="lepton1_DeepMETResolutionTune", bins=50, start=-5, stop=5, label="deta lepton1 met")]),
        "jet1_eta":  HistConf([Axis(coll="jet1", field="eta",           bins=50,  start=-5,   stop=5,   label="jet1 eta")]),
        "jet1_phi":  HistConf([Axis(coll="jet1", field="phi",           bins=64,  start=-3.2, stop=3.2, label="jet1 phi")]),
        "jet1_pt":   HistConf([Axis(coll="jet1", field="pt",            bins=100, start=0,    stop=1000, label="jet1 pt")]),
        "jet1_qgl":  HistConf([Axis(coll="jet1", field="qgl",           bins=50,  start=0,    stop=1,   label="jet1 qgl")]),

        "jet1_btagDeepFlavQG": HistConf([Axis(coll="jet1", field="btagDeepFlavQG", bins=25, start=0, stop=1, label="jet1 btagDeepFlavQG")]),
        "jet2_btagDeepFlavQG": HistConf([Axis(coll="jet2", field="btagDeepFlavQG", bins=25, start=0, stop=1, label="jet2 btagDeepFlavQG")]),
        "jet3_btagDeepFlavQG": HistConf([Axis(coll="jet3", field="btagDeepFlavQG", bins=25, start=0, stop=1, label="jet3 btagDeepFlavQG")]),
        "jet4_btagDeepFlavQG": HistConf([Axis(coll="jet4", field="btagDeepFlavQG", bins=25, start=0, stop=1, label="jet4 btagDeepFlavQG")]),
        "jet5_btagDeepFlavQG": HistConf([Axis(coll="jet5", field="btagDeepFlavQG", bins=25, start=0, stop=1, label="jet5 btagDeepFlavQG")]),
        "jet6_btagDeepFlavQG": HistConf([Axis(coll="jet6", field="btagDeepFlavQG", bins=25, start=0, stop=1, label="jet6 btagDeepFlavQG")]),

        "jet1_btagDeepFlavB": HistConf([Axis(coll="jet1", field="btagDeepFlavB", bins=50, start=0, stop=1, label="jet1 btagDeepFlavB")]),
        "jet2_eta":  HistConf([Axis(coll="jet2", field="eta",           bins=50,  start=-5,   stop=5,   label="jet2 eta")]),
        "jet2_phi":  HistConf([Axis(coll="jet2", field="phi",           bins=64,  start=-3.2, stop=3.2, label="jet2 phi")]),
        "jet2_pt":   HistConf([Axis(coll="jet2", field="pt",            bins=100, start=0,    stop=1000, label="jet2 pt")]),
        "jet2_qgl":  HistConf([Axis(coll="jet2", field="qgl",           bins=50,  start=0,    stop=1,   label="jet2 qgl")]),
        "jet2_btagDeepFlavB": HistConf([Axis(coll="jet2", field="btagDeepFlavB", bins=50, start=0, stop=1, label="jet2 btagDeepFlavB")]),
        "jet3_eta":  HistConf([Axis(coll="jet3", field="eta",           bins=50,  start=-5,   stop=5,   label="jet3 eta")]),
        "jet3_phi":  HistConf([Axis(coll="jet3", field="phi",           bins=64,  start=-3.2, stop=3.2, label="jet3 phi")]),
        "jet3_pt":   HistConf([Axis(coll="jet3", field="pt",            bins=100, start=0,    stop=1000, label="jet3 pt")]),
        "jet3_qgl":  HistConf([Axis(coll="jet3", field="qgl",           bins=50,  start=0,    stop=1,   label="jet3 qgl")]),
        "jet3_btagDeepFlavB": HistConf([Axis(coll="jet3", field="btagDeepFlavB", bins=50, start=0, stop=1, label="jet3 btagDeepFlavB")]),
        "jet4_eta":  HistConf([Axis(coll="jet4", field="eta",           bins=50,  start=-5,   stop=5,   label="jet4 eta")]),
        "jet4_phi":  HistConf([Axis(coll="jet4", field="phi",           bins=64,  start=-3.2, stop=3.2, label="jet4 phi")]),
        "jet4_pt":   HistConf([Axis(coll="jet4", field="pt",            bins=100, start=0,    stop=1000, label="jet4 pt")]),
        "jet4_qgl":  HistConf([Axis(coll="jet4", field="qgl",           bins=50,  start=0,    stop=1,   label="jet4 qgl")]),
        "jet4_btagDeepFlavB": HistConf([Axis(coll="jet4", field="btagDeepFlavB", bins=50, start=0, stop=1, label="jet4 btagDeepFlavB")]),
        "jet5_eta":  HistConf([Axis(coll="jet5", field="eta",           bins=50,  start=-5,   stop=5,   label="jet5 eta")]),
        "jet5_phi":  HistConf([Axis(coll="jet5", field="phi",           bins=64,  start=-3.2, stop=3.2, label="jet5 phi")]),
        "jet5_pt":   HistConf([Axis(coll="jet5", field="pt",            bins=100, start=0,    stop=1000, label="jet5 pt")]),
        "jet5_qgl":  HistConf([Axis(coll="jet5", field="qgl",           bins=50,  start=0,    stop=1,   label="jet5 qgl")]),
        "jet5_btagDeepFlavB": HistConf([Axis(coll="jet5", field="btagDeepFlavB", bins=50, start=0, stop=1, label="jet5 btagDeepFlavB")]),
        "jet6_eta":  HistConf([Axis(coll="jet6", field="eta",           bins=50,  start=-5,   stop=5,   label="jet6 eta")]),
        "jet6_phi":  HistConf([Axis(coll="jet6", field="phi",           bins=64,  start=-3.2, stop=3.2, label="jet6 phi")]),
        "jet6_pt":   HistConf([Axis(coll="jet6", field="pt",            bins=100, start=0,    stop=1000, label="jet6 pt")]),
        "jet6_qgl":  HistConf([Axis(coll="jet6", field="qgl",           bins=50,  start=0,    stop=1,   label="jet6 qgl")]),
        "jet6_btagDeepFlavB": HistConf([Axis(coll="jet6", field="btagDeepFlavB", bins=50, start=0, stop=1, label="jet6 btagDeepFlavB")]),
        "lepton1_eta": HistConf([Axis(coll="lepton1", field="eta", bins=50,  start=-3,   stop=3,   label="lepton1 eta")]),
        "lepton1_phi": HistConf([Axis(coll="lepton1", field="phi", bins=64,  start=-3.2, stop=3.2, label="lepton1 phi")]),
        "lepton1_pt":  HistConf([Axis(coll="lepton1", field="pt",  bins=100, start=0,    stop=500, label="lepton1 pt")]),
    },
)
