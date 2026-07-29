# example_config_semileptonic.py
import os, cloudpickle
from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_functions import get_HLTsel, get_nPVgood, goldenJson, eventFlags, get_JetVetoMap
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.parameters.histograms import HistConf, Axis
from pocket_coffea.lib.weights.common import common_weights
# from pocket_coffea.lib.weights.common.weights_run3 import SF_ele_trigger
from pocket_coffea.lib.weights.common.common import SF_L1prefiring
from pocket_coffea.lib.weights.common.weights_run2_UL import SF_ele_trigger
from pocket_coffea.parameters import defaults
import numpy as np
import awkward as ak
from pocket_coffea.lib.weights import WeightWrapper, WeightData, WeightDataMultiVariation, WeightLambda
from pocket_coffea.lib.scale_factors import sf_pileup_reweight


import workflow_boosted_control, custom_cut_sf
from workflow_boosted_control import VBSSemileptonicProcessor
from custom_cut_sf import (
    TT_boosted_sel,
    zjets_boosted_sel

)


cloudpickle.register_pickle_by_value(workflow_boosted_control)
cloudpickle.register_pickle_by_value(custom_cut_sf)

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
    f"{localdir}/params/lepton_scale_factors.yaml",
    f"{localdir}/params/jet_scale_factors.yaml",
    f"{localdir}/params/classifiers.yaml",
    f"{localdir}/params/variations.yaml",
    f"{localdir}/params/fakelepton_weights_noiso_3j.yaml",
    f"{localdir}/params/fj_taggers.yaml",
    update=True,
)
PileupWeight = WeightLambda.wrap_func(
   name="PileupWeight",
   function=lambda params, metadata, events, size, shape_variations:
       sf_pileup_reweight(params, events, metadata["year"]),
   has_variations=True
   )
_PT_BINS  = [200, 240, 300, 380, 8000]
_ETA_BINS = [0,1.3,2.4]
_MASS_BINS = [40,55,70, 80, 90, 200]
_SCORE_BINS = [0.0,0.025,0.05,0.1,0.2,0.5,0.8,0.9,0.95,0.975,1.0]
_T21_BINS = [0.0, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 1.0]
_T21_COARSE_BINS = [0.0, 0.45, 1.0]

import correctionlib
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


def _gen_norm_weight(events, size):
    """Generator weight, used as the normalization proxy for shape corrections."""
    if hasattr(events, "genWeight"):
        return ak.to_numpy(ak.fill_none(events.genWeight, 0.0))
    return np.ones(size)


def _fj_sf_multivariation(name, corr, obs_arrays, has_fj, matched, w):
    """
    Evaluate the per-event SF (split matched/unmatched, all systematic sources)
    and apply it as a SHAPE correction: each variation is renormalized so the
    total yield is preserved -> w_new = w * sf * sum(w) / sum(w * sf), with w
    the generator weight and the sum taken over the current chunk.
    """
    n = len(has_fj)
    sum_w = np.sum(w)

    def _eval(systematic):
        sf = np.ones(n)
        for fjtype, sel in (("matched", has_fj & matched), ("unmatched", has_fj & ~matched)):
            if sel.any():
                sf[sel] = corr.evaluate(systematic, fjtype, *[a[sel] for a in obs_arrays])
        denom = np.sum(w * sf)
        norm = (sum_w / denom) if denom != 0.0 else 1.0
        return sf * norm

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
        w = _gen_norm_weight(events, size)
        return _fj_sf_multivariation(self.name, corr, [tau21], has_fj, matched, w)


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
        w = _gen_norm_weight(events, size)
        return _fj_sf_multivariation(self.name, corr, [msd, wvsqcd], has_fj, matched, w)


cfg = Configurator(
    parameters=parameters,
    datasets={
        "jsons": [
            #######
            ## RUN 2 BKG
            # #########
            # f"{localdir}/datasets/WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            # f"{localdir}/datasets/WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8_17.json",
            #
            #f"{localdir}/datasets/WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            #f"{localdir}/datasets/WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            #f"{localdir}/datasets/WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            # XSEC STUDIES
            #f"{localdir}/datasets/WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8_17.json",
            #f"{localdir}/datasets/WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_17.json",
            #f"{localdir}/datasets/WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8_17.json",
            
            #f"{localdir}/datasets/WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8_17_2.json",
            #f"{localdir}/datasets/WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_17_2.json",
            #f"{localdir}/datasets/WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8_17_2.json",
            

            #f"{localdir}/datasets/WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8_fix.json",
            #f"{localdir}/datasets/WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_fix.json",
            #f"{localdir}/datasets/WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8_fix.json",
            
            #f"{localdir}/datasets/WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8_fix2.json",
            #f"{localdir}/datasets/WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_fix2.json",
            #f"{localdir}/datasets/WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8_fix2.json",
            
            #END XSEC STUDIES

            #f"{localdir}/datasets/WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            #f"{localdir}/datasets/WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            #f"{localdir}/datasets/WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            #f"{localdir}/datasets/WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            #f"{localdir}/datasets/WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8.json",

            # f"{localdir}/datasets/WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8.json",
            # f"{localdir}/datasets/WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8_17.json",
            

            f"{localdir}/datasets/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8.json",
            #f"{localdir}/datasets/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_17.json",

            #f"{localdir}/datasets/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            #f"{localdir}/datasets/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8_17.json",

            f"{localdir}/datasets/DYJetsToLL_M-10to50_TuneCP5_13TeV-amcatnloFXFX-pythia8.json",

            #f"{localdir}/datasets/DYJetsToLL_M-50_HT-70to100_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8.json",
            #f"{localdir}/datasets/DYJetsToLL_M-50_HT-100to200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8.json",
            #f"{localdir}/datasets/DYJetsToLL_M-50_HT-200to400_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8.json",
            #f"{localdir}/datasets/DYJetsToLL_M-50_HT-400to600_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8.json",
            #f"{localdir}/datasets/DYJetsToLL_M-50_HT-600to800_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8.json",
            #f"{localdir}/datasets/DYJetsToLL_M-50_HT-800to1200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8.json",
            #f"{localdir}/datasets/DYJetsToLL_M-50_HT-1200to2500_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8.json",
            #f"{localdir}/datasets/DYJetsToLL_M-50_HT-2500toInf_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8.json",

            f"{localdir}/datasets/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8.json",
            f"{localdir}/datasets/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8.json",
            # f"{localdir}/datasets/ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8.json",
            # f"{localdir}/datasets/ST_t-channel_top_4f_inclusiveDecays_TuneCP5_13TeV-powhegV2-madspin-pythia8.json",
            # f"{localdir}/datasets/ST_t-channel_antitop_4f_inclusiveDecays_TuneCP5_13TeV-powhegV2-madspin-pythia8.json",
            # f"{localdir}/datasets/ttZJets_TuneCP5_13TeV_madgraphMLM_pythia8.json",
            # f"{localdir}/datasets/ttWJets_TuneCP5_13TeV_madgraphMLM_pythia8.json",
            # f"{localdir}/datasets/WplusTo2JZTo2LJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            # f"{localdir}/datasets/WplusToLNuWminusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV.json",
            # f"{localdir}/datasets/WplusToLNuWplusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            # f"{localdir}/datasets/WminusTo2JZTo2LJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            # f"{localdir}/datasets/WminusToLNuWminusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            # f"{localdir}/datasets/WminusToLNuZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            # f"{localdir}/datasets/WplusTo2JWminusToLNuJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV.json",
            # f"{localdir}/datasets/WplusToLNuZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            # f"{localdir}/datasets/ZTo2LZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            # f"{localdir}/datasets/WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8.json",
            # f"{localdir}/datasets/WZZ_TuneCP5_13TeV-amcatnlo-pythia8.json",
            # f"{localdir}/datasets/ZZZ_TuneCP5_13TeV-amcatnlo-pythia8.json",
            # f"{localdir}/datasets/WGToLNuG_TuneCP5_13TeV-madgraphMLM-pythia8.json",
            # f"{localdir}/datasets/ZGToLLG_01J_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8.json",
            # f"{localdir}/datasets/WZTo3LNu_mllmin01_NNPDF31_TuneCP5_13TeV_powheg_pythia8.json",
            
            # #########
            # ## RUN 2 SIGNAL
            # ########
            # f"{localdir}/datasets/WplusTo2JWminusToLNuJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            # f"{localdir}/datasets/WplusToLNuWminusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            # f"{localdir}/datasets/WminusToLNuWminusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            # f"{localdir}/datasets/WplusToLNuWplusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            # f"{localdir}/datasets/WminusToLNuZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
            # f"{localdir}/datasets/WplusToLNuZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
 
            ########
            ## RUN 3 BKG
            ########
            #f"{localdir}/datasets/WWtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8.json",
            #f"{localdir}/datasets/WtoLNu-2Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            #f"{localdir}/datasets/ZZto2L2Q_TuneCP5_13p6TeV_powheg-pythia8.json",
            #f"{localdir}/datasets/TTto2L2Nu_TuneCP5_ERDOn_13p6TeV_powheg-pythia8.json",
            #f"{localdir}/datasets/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8.json",
            #f"{localdir}/datasets/DYto2L-2Jets_MLL-10to50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
            #f"{localdir}/datasets/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",'''

          #   f"{localdir}/datasets/WtoLNu-2Jets_0J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
          #   f"{localdir}/datasets/WtoLNu-2Jets_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
          #   f"{localdir}/datasets/WtoLNu-2Jets_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
          #   f"{localdir}/datasets/WtoLNu-2Jets_PTLNu-40to100_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
          #   f"{localdir}/datasets/WtoLNu-2Jets_PTLNu-100to200_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
          #   f"{localdir}/datasets/WtoLNu-2Jets_PTLNu-200to400_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
          #   f"{localdir}/datasets/WtoLNu-2Jets_PTLNu-400to600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
          #   f"{localdir}/datasets/WtoLNu-2Jets_PTLNu-600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
          #   f"{localdir}/datasets/WtoLNu-2Jets_PTLNu-40to100_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
          #   f"{localdir}/datasets/WtoLNu-2Jets_PTLNu-100to200_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
          #   f"{localdir}/datasets/WtoLNu-2Jets_PTLNu-200to400_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
          #   f"{localdir}/datasets/WtoLNu-2Jets_PTLNu-400to600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
          #   f"{localdir}/datasets/WtoLNu-2Jets_PTLNu-600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
          #   f"{localdir}/datasets/WtoLNu-2Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
          #   f"{localdir}/datasets/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
          #   f"{localdir}/datasets/DYto2L-2Jets_MLL-10to50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
          #   f"{localdir}/datasets/DYto2L-2Jets_MLL-50_0J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
          #   f"{localdir}/datasets/DYto2L-2Jets_MLL-50_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
          #   f"{localdir}/datasets/DYto2L-2Jets_MLL-50_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
          #   f"{localdir}/datasets/DYto2L-2Jets_MLL-50_PTLL-40to100_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
          #   f"{localdir}/datasets/DYto2L-2Jets_MLL-50_PTLL-100to200_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
          #   f"{localdir}/datasets/DYto2L-2Jets_MLL-50_PTLL-200to400_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
          #   f"{localdir}/datasets/DYto2L-2Jets_MLL-50_PTLL-400to600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
          #   f"{localdir}/datasets/DYto2L-2Jets_MLL-50_PTLL-600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
          #   f"{localdir}/datasets/DYto2L-2Jets_MLL-50_PTLL-40to100_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
          #   f"{localdir}/datasets/DYto2L-2Jets_MLL-50_PTLL-100to200_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
          #   f"{localdir}/datasets/DYto2L-2Jets_MLL-50_PTLL-200to400_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
          #   f"{localdir}/datasets/DYto2L-2Jets_MLL-50_PTLL-400to600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
          #   f"{localdir}/datasets/DYto2L-2Jets_MLL-50_PTLL-600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
          #   f"{localdir}/datasets/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8.json",
          #   f"{localdir}/datasets/TT_TuneCP5_13p6TeV_powheg-pythia8.json",
          #   f"{localdir}/datasets/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8.json",
          #   f"{localdir}/datasets/TbarBQ_t-channel_4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8.json",
          #   f"{localdir}/datasets/TBbarQ_t-channel_4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8.json",
          #   f"{localdir}/datasets/TBbarto2Q-s-channel_TuneCP5_13p6TeV_powheg-pythia8.json",
          #   f"{localdir}/datasets/TBbartoLplusNuBbar-s-channel-4FS_TuneCP5_13p6TeV_amcatnlo-pythia8.json",
          #   f"{localdir}/datasets/TbarBtoLminusNuB-s-channel-4FS_TuneCP5_13p6TeV_amcatnlo-pythia8.json",
          #   f"{localdir}/datasets/TbarWplus_DR_AtLeastOneLepton_TuneCP5_13p6TeV_powheg-pythia8.json",
          #   f"{localdir}/datasets/TWminus_DR_AtLeastOneLepton_TuneCP5_13p6TeV_powheg-pythia8.json",
          #   f"{localdir}/datasets/WW_TuneCP5_13p6TeV_pythia8.json",
          #   f"{localdir}/datasets/WZ_TuneCP5_13p6TeV_pythia8.json",
          #   f"{localdir}/datasets/ZZ_TuneCP5_13p6TeV_pythia8.json",
          #   f"{localdir}/datasets/WWZ_4F_TuneCP5_13p6TeV_amcatnlo-pythia8.json",
          #   f"{localdir}/datasets/WWW_4F_TuneCP5_13p6TeV_amcatnlo-madspin-pythia8.json",
          #   f"{localdir}/datasets/WZZ_TuneCP5_13p6TeV_amcatnlo-pythia8.json",
          #   f"{localdir}/datasets/ZZZ_TuneCP5_13p6TeV_amcatnlo-pythia8.json",
          #   f"{localdir}/datasets/WGtoLNuG-1Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
          #   f"{localdir}/datasets/ZGto2LG-1Jets_ntgc_5f_TuneCP5_13p6TeV_madgraphMLM-pythia8.json",
          #   f"{localdir}/datasets/VBFtoLNu_TuneCP5_13p6TeV_madgraph-pythia8.json",


            # #########
            # ## RUN 3 SIGNAL
            # ########
          #   f"{localdir}/datasets/ssWWunpolarized_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
          #   f"{localdir}/datasets/osWWunpolarized_Wptojj_Wmtolv_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
          #   f"{localdir}/datasets/osWWunpolarized_Wptolv_Wmtojj_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
          #   f"{localdir}/datasets/WZunpolarized_Wmtolv_Ztojj_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",
          #   f"{localdir}/datasets/WZunpolarized_Wptolv_Ztojj_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.json",


            #f"{localdir}/datasets/WpWpJJ-EWK_TuneCP5_13p6TeV-powheg-pythia8.json",
            #f"{localdir}/datasets/WmWmJJ-EWK_TuneCP5_13p6TeV-powheg-pythia8.json",
            #########
            ## SOME DATA
            #########
            f"{localdir}/datasets/SingleMuon.json", ## 2017B Single Muon dataset
            f"{localdir}/datasets/EGamma.json",
          #   f"{localdir}/datasets/Muon.json"
            
        ],
        "filter": {
            "samples": [
                
            #########
            ## RUN 2 BKG
            #########
            # "WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8",

            # "WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8_17",
            #"WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8",
            #"WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8",
            #"WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8",
            #"WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8",
            #"WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8",
            #"WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8",
            #"WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8",
            #"WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8",

            # "WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8_17",
            # "WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_17",
            # "WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8_17",
            
            # "WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8_17_2",
            # "WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_17_2",
            # "WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8_17_2",
            

            # "WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8_fix",
            # "WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_fix",
            # "WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8_fix",
            
            # "WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8_fix2",
            # "WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_fix2",
            # "WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8_fix2",
            
            #"WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8", 
            #"WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8_17", 
            #"DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8_17",
            #"DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8",
            "DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8", 
            #"DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_17", 
            "DYJetsToLL_M-10to50_TuneCP5_13TeV-amcatnloFXFX-pythia8", 

            # "DYJetsToLL_M-50_HT-70to100_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
            # "DYJetsToLL_M-50_HT-100to200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
            # "DYJetsToLL_M-50_HT-200to400_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
            # "DYJetsToLL_M-50_HT-400to600_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
            # "DYJetsToLL_M-50_HT-600to800_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
            # "DYJetsToLL_M-50_HT-800to1200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
            # "DYJetsToLL_M-50_HT-1200to2500_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
            # "DYJetsToLL_M-50_HT-2500toInf_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",

            # "TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8", 
            "TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8", 
            # "ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8", 
            # "ST_t-channel_top_4f_inclusiveDecays_TuneCP5_13TeV-powhegV2-madspin-pythia8",
            # "ST_t-channel_antitop_4f_inclusiveDecays_TuneCP5_13TeV-powhegV2-madspin-pythia8", 
            # "ttZJets_TuneCP5_13TeV_madgraphMLM_pythia8", 
            # "ttWJets_TuneCP5_13TeV_madgraphMLM_pythia8", 
            # # "WplusTo2JZTo2LJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8", 
            # # "WplusToLNuWminusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV", 
            # # "WplusToLNuWplusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8", 
            # # "WminusTo2JZTo2LJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8", 
            # # "WminusToLNuWminusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8", 
            # # "WminusToLNuZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8", 
            # # "WplusTo2JWminusToLNuJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV", 
            # # "WplusToLNuZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8", 
            # # "ZTo2LZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8", 
            # # "WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8", 
            # # "WZZ_TuneCP5_13TeV-amcatnlo-pythia8", 
            # # "ZZZ_TuneCP5_13TeV-amcatnlo-pythia8", 
            # # "WGToLNuG_TuneCP5_13TeV-madgraphMLM-pythia8", 
            # # "ZGToLLG_01J_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8", 
            # # "WZTo3LNu_mllmin01_NNPDF31_TuneCP5_13TeV_powheg_pythia8", 
        
                        
            # #########
            # ## RUN 2 SIGNAL
            # ########
            #"WminusToLNuWminusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            #"WplusTo2JWminusToLNuJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            #"WplusToLNuWminusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            #"WplusToLNuWplusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            #"WminusToLNuZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
            #"WplusToLNuZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",

            #########
            ## SOME DATA
            #########
            "SingleMuon", ## 2017B Single Muon dataset
            "EGamma",
            #"Muon"
            ],
            "year": ["2018"]
        },
    },
    workflow=VBSSemileptonicProcessor,

    
    skim=[
        get_nPVgood(1),    # nPV>0
        eventFlags,        # PileupID
        goldenJson,        
        get_HLTsel(primaryDatasets=["SingleMuon", "EGamma"]),
     #    get_JetVetoMap()
    ],

    # 2) preselections 
    preselections=[passthrough],

   
    categories={
        "TT_boosted": [TT_boosted_sel],
        "zjets_boosted": [zjets_boosted_sel],
    },

   
    weights_classes=common_weights+[SF_ele_trigger]+[PileupWeight] + [SF_L1prefiring]+[FatJetTau21Weight,FatJetWvsQCDWeight],
    #weights={"common": {"inclusive": ["genWeight", "lumi", "XS", "PileupWeight", "sf_mu_id", "sf_mu_iso", "sf_ele_id", "sf_ele
    #weights={"common": {"inclusive": ["genWeight", "lumi", "XS", "pileup", "sf_mu_id", "sf_mu_iso", "sf_ele_id", "sf_ele_reco","sf_mu_trigger","sf_ele_trigger","sf_btag","sf_btag_calib"]}},
    weights={"common": {"inclusive": ["genWeight", "lumi", "XS", "pileup", "sf_mu_id", "sf_mu_iso", "sf_ele_id", "sf_ele_reco","sf_mu_trigger","sf_ele_trigger","sf_btag", "sf_partonshower_isr","sf_partonshower_fsr","sf_L1prefiring","sf_fj_tau21", "sf_fj_WvsQCD"]}},
    #variations={"weights": {"common": {"inclusive": ["pileup", "sf_mu_id","sf_mu_iso","sf_ele_id","sf_ele_reco","sf_mu_trigger","sf_ele_trigger","sf_btag"]}}}, #"pileup"
    variations={
     "weights": {
          "common": {
               "inclusive": ["pileup","sf_btag","sf_partonshower_isr","sf_partonshower_fsr","sf_mu_id","sf_mu_iso","sf_ele_id","sf_ele_reco","sf_mu_trigger","sf_ele_trigger","sf_L1prefiring","sf_fj_tau21", "sf_fj_WvsQCD"]
               }
          },
          "shape": {
            "common": {
                "inclusive": ['jet_calibration','electron_scale_and_smearing','muons_scale_and_resolution']
            },
        }
    },
    variables={

        "nJets":      HistConf([Axis(coll="events", field="nJetGood", bins=12, start=0, stop=12, label="N(jets)")]),
        "nMuonGood":     HistConf([Axis(coll="events", field="nMuonGood", bins=6, start=0, stop=6, label="N(muon good)")]),
        "nElectronGood":      HistConf([Axis(coll="events", field="nElectronGood", bins=6, start=0, stop=6, label="N(electron good)")]),
        "nLeptonGood":    HistConf([Axis(coll="events", field="nLeptonGood", bins=6, start=0, stop=6, label="N(lepton good)")]),    
        # MET and mT
        "met":        HistConf([Axis(coll="MET", field="pt", bins=50, start=0, stop=250, label=r"$p_T^{miss}$ [GeV]")]),
        "met_phi":    HistConf([Axis(coll="MET", field="phi", bins=50, start=-4, stop=4, label=r"$\phi^{miss}$ [GeV]")]),
        "puppimet_phi":    HistConf([Axis(coll="PuppiMET", field="phi", bins=50, start=-4, stop=4, label=r"$\phi^{miss}$ [GeV]")]),
        "puppimet":        HistConf([Axis(coll="PuppiMET", field="pt", bins=50, start=0, stop=250, label=r"$p_T^{miss}$ [GeV]")]),
        "deepmet_resolution_tune_phi":    HistConf([Axis(coll="DeepMETResolutionTune", field="phi", bins=50, start=-4, stop=4, label=r"deep $ \phi^{miss}$ resolution tune [GeV]")]),
        "deepmet_resolution_tune":        HistConf([Axis(coll="DeepMETResolutionTune", field="pt", bins=50, start=0, stop=250, label=r"deep $p_T^{miss}$ resolution tune [GeV]")]),
        "deepmet_response_tune_phi":    HistConf([Axis(coll="DeepMETResponseTune", field="phi", bins=50, start=-4, stop=4, label=r"deep $ \phi^{miss}$ response tune [GeV]")]),
        "deepmet_response_tune":        HistConf([Axis(coll="DeepMETResponseTune", field="pt", bins=50, start=0, stop=250, label=r"deep $p_T^{miss}$ response tune [GeV]")]),
        "mt_w_lep":   HistConf([Axis(coll="events", field="mt_w_leptonic", bins=30, start=0, stop=200, label=r"$m_T(W_{lep})$ [GeV]")]),
        "mt_w_lep_deepresolution": HistConf([Axis(coll="events", field="mt_w_leptonic_deepMET_resolutiontune", bins=30, start=0, stop=200, label=r"$m_T(W_{lep})$ (deep MET resolution tune) [GeV]")]),
        "mt_w_lep_deepresponse": HistConf([Axis(coll="events", field="mt_w_leptonic_deepMET_responsetune", bins=30, start=0, stop=200, label=r"$m_T(W_{lep})$ (deep MET response tune) [GeV]")]),
        # lead lepton
        "eta_w_lep":   HistConf([Axis(coll="events", field="w_lep_eta", bins=32, start=-4.0, stop=4.0, label=r"$\eta^{lead\ lep}$ ")]),
        "pt_w_lep":   HistConf([Axis(coll="events", field="w_lep_pt", bins=40, start=0.0, stop=300.0, label=r"$p_T^{lead\ lep}$ [GeV]")]),
        "phi_w_lep":   HistConf([Axis(coll="events", field="w_lep_phi", bins=32, start=-4.0, stop=4.0, label=r"$\phi^{lead\ lep}$ ")]),
        "m_ll":   HistConf([Axis(coll="ll", field="m_ll", bins=150, start=0, stop=200.0, label=r"$m_{ll}$ [GeV]")]),
        # # W fat jet
        # "fj_pt":    HistConf([Axis(coll="candidate_boost", field="pt",  bins=50, start=200, stop=1000, label=r"$p_T(J^{W})$ [GeV]")]),
        # "fj_eta":   HistConf([Axis(coll="candidate_boost", field="eta", bins=48, start=-2.4, stop=2.4,   label=r"$\eta(J^{W})$")]),
        # "fj_msd":   HistConf([Axis(coll="candidate_boost", field="msoftdrop", bins=40, start=0,   stop=200,   label=r"$m_{SD}(J^{W})$ [GeV]")]),
        "fj_t21":   HistConf([Axis(coll="candidate_boost", field="tau21", bins=32, start=0, stop=1.1,   label=r"$\tau_{21}$")]),

        # "fj_XqqVsQCD":   HistConf([Axis(coll="candidate_boost", field="particleNet_XqqVsQCD", bins=40, start=0, stop=1,   label=r"fatjet XqqVsQCD")]),

        "fj_WvsQCD":   HistConf([Axis(coll="candidate_boost", field="particleNet_WvsQCD", bins=40, start=0, stop=1,   label=r"fatjet WvsQCD")]),
        "fj_ZvsQCD":   HistConf([Axis(coll="candidate_boost", field="particleNet_ZvsQCD", bins=40, start=0, stop=1,   label=r"fatjet ZvsQCD")]),
        # "fj_WvsQCD_pt_eta": HistConf([
        #     Axis(coll="candidate_boost", field="particleNet_WvsQCD",
        #          bins=_SCORE_BINS,
        #          label=r"fatjet WvsQCD"),
        #     Axis(coll="candidate_boost", field="pt",
        #          bins=_PT_BINS,
        #          label=r"$p_T$"),
        #     Axis(coll="candidate_boost", field="abseta",
        #          bins=_ETA_BINS,
        #          label=r"$|\eta|$"),
        #     Axis(coll="candidate_boost", field="msoftdrop",
        #          bins=_MASS_BINS,
        #          label=r"$m_{SD}(J^{W})$ [GeV]"),
        #     Axis(coll="candidate_boost", field="tau21",
        #          bins=_T21_COARSE_BINS,
        #          label=r"$\tau_{21}$"),
        # ]),
        # # "fj_ZvsQCD_pt_eta": HistConf([
        # #     Axis(coll="candidate_boost", field="particleNetWithMass_ZvsQCD",
        # #          bins=_SCORE_BINS,
        # #          label=r"fatjet ZvsQCD"),
        # #     Axis(coll="candidate_boost", field="pt",
        # #          bins=_PT_BINS,
        # #          label=r"$p_T$"),
        # #     Axis(coll="candidate_boost", field="abseta",
        # #          bins=_ETA_BINS,
        # #          label=r"$|\eta|$"),
        # #     Axis(coll="candidate_boost", field="msoftdrop",
        # #          bins=_MASS_BINS,
        # #          label=r"$m_{SD}(J^{W})$ [GeV]"),
        # #     Axis(coll="candidate_boost", field="tau21",
        # #          bins=_T21_COARSE_BINS,
        # #          label=r"$\tau_{21}$"),
        # # ]),
        # # "fj_XqqVsQCD_pt_eta": HistConf([
        # #     Axis(coll="candidate_boost", field="particleNet_XqqVsQCD",
        # #          bins=_SCORE_BINS,
        # #          label=r"fatjet XqqVsQCD"),
        # #     Axis(coll="candidate_boost", field="pt",
        # #          bins=_PT_BINS,
        # #          label=r"$p_T$"),
        # #     Axis(coll="candidate_boost", field="abseta",
        # #          bins=_ETA_BINS,
        # #          label=r"$|\eta|$"),
        # #     Axis(coll="candidate_boost", field="msoftdrop",
        # #          bins=_MASS_BINS,
        # #          label=r"$m_{SD}(J^{W})$ [GeV]"),
        # #     Axis(coll="candidate_boost", field="tau21",
        # #          bins=_T21_COARSE_BINS,
        # #          label=r"$\tau_{21}$"),
        # # ]),
        # "fj_WvsQCD_tau21": HistConf([
        #      Axis(coll="candidate_boost", field="particleNet_WvsQCD",
        #           bins=_SCORE_BINS,
        #           label=r"fatjet WvsQCD"),
        #      Axis(coll="candidate_boost", field="tau21",
        #           bins=_T21_BINS,
        #           label=r"$\tau_{21}$"),
        #  ]),
        # "fj_ZvsQCD_tau21": HistConf([
        #     Axis(coll="candidate_boost", field="particleNetWithMass_ZvsQCD",
        #          bins=_SCORE_BINS,
        #          label=r"fatjet ZvsQCD"),
        #     Axis(coll="candidate_boost", field="tau21",
        #          bins=_T21_BINS,
        #          label=r"$\tau_{21}$"),
        # ]),
        # "fj_XqqVsQCD_tau21": HistConf([
        #     Axis(coll="candidate_boost", field="particleNet_XqqVsQCD",
        #          bins=_SCORE_BINS,
        #          label=r"fatjet XqqVsQCD"),
        #     Axis(coll="candidate_boost", field="tau21",
        #          bins=_T21_BINS,
        #          label=r"$\tau_{21}$"),
        # ]),
        # # W fat jet
     #    "fj_matched_pt":    HistConf([Axis(coll="candidate_boost_matched", field="pt",  bins=50, start=200, stop=1000, label=r"$p_T(J^{W})$ [GeV]")]),
     #    "fj_matched_eta":   HistConf([Axis(coll="candidate_boost_matched", field="eta", bins=48, start=-2.4, stop=2.4,   label=r"$\eta(J^{W})$")]),
     #    "fj_matched_msd":   HistConf([Axis(coll="candidate_boost_matched", field="msoftdrop", bins=40, start=0,   stop=200,   label=r"$m_{SD}(J^{W})$ [GeV]")]),
     #    "fj_matched_t21":   HistConf([Axis(coll="candidate_boost_matched", field="tau21", bins=32, start=0, stop=1.1,   label=r"$\tau_{21}$")]),

     #    #"fj_matched_XqqVsQCD":   HistConf([Axis(coll="candidate_boost_matched", field="particleNet_XqqVsQCD", bins=40, start=0, stop=1,   label=r"fatjet XqqVsQCD")]),

     #    "fj_matched_WvsQCD":   HistConf([Axis(coll="candidate_boost_matched", field="particleNet_WvsQCD", bins=40, start=0, stop=1,   label=r"fatjet WvsQCD")]),
     #    "fj_matched_ZvsQCD":   HistConf([Axis(coll="candidate_boost_matched", field="particleNet_ZvsQCD", bins=40, start=0, stop=1,   label=r"fatjet ZvsQCD")]),
     #    "fj_matched_WvsQCD_pt_eta": HistConf([
     #        Axis(coll="candidate_boost_matched", field="particleNet_WvsQCD",
     #             bins=_SCORE_BINS,
     #             label=r"fatjet WvsQCD"),
     #        Axis(coll="candidate_boost_matched", field="pt",
     #             bins=_PT_BINS,
     #             label=r"$p_T$"),
     #        Axis(coll="candidate_boost_matched", field="abseta",
     #             bins=_ETA_BINS,
     #             label=r"$|\eta|$"),
     #        Axis(coll="candidate_boost_matched", field="msoftdrop",
     #             bins=_MASS_BINS,
     #             label=r"$m_{SD}(J^{W})$ [GeV]"),
     #        Axis(coll="candidate_boost_matched", field="tau21",
     #             bins=_T21_COARSE_BINS,
     #             label=r"$\tau_{21}$"),
     #    ]),
     #    "fj_matched_ZvsQCD_pt_eta": HistConf([
     #        Axis(coll="candidate_boost_matched", field="particleNet_ZvsQCD",
     #             bins=_SCORE_BINS,
     #             label=r"fatjet ZvsQCD"),
     #        Axis(coll="candidate_boost_matched", field="pt",
     #             bins=_PT_BINS,
     #             label=r"$p_T$"),
     #        Axis(coll="candidate_boost_matched", field="abseta",
     #             bins=_ETA_BINS,
     #             label=r"$|\eta|$"),
     #        Axis(coll="candidate_boost_matched", field="msoftdrop",
     #             bins=_MASS_BINS,
     #             label=r"$m_{SD}(J^{W})$ [GeV]"),
     #        Axis(coll="candidate_boost_matched", field="tau21",
     #             bins=_T21_COARSE_BINS,
     #             label=r"$\tau_{21}$"),
     #    ]),
     #    "fj_matched_ZvsQCD_pt_eta": HistConf([
     #        Axis(coll="candidate_boost_matched", field="particleNet_ZvsQCD",
     #             bins=_SCORE_BINS,
     #             label=r"fatjet ZvsQCD"),
     #        Axis(coll="candidate_boost_matched", field="pt",
     #             bins=_PT_BINS,
     #             label=r"$p_T$"),
     #        Axis(coll="candidate_boost_matched", field="abseta",
     #             bins=_ETA_BINS,
     #             label=r"$|\eta|$"),
     #        Axis(coll="candidate_boost_matched", field="msoftdrop",
     #             bins=_MASS_BINS,
     #             label=r"$m_{SD}(J^{W})$ [GeV]"),
     #        Axis(coll="candidate_boost_matched", field="tau21",
     #             bins=_T21_COARSE_BINS,
     #             label=r"$\tau_{21}$"),
     #    ]),

     #    "fj_matched_WvsQCD_tau21": HistConf([
     #        Axis(coll="candidate_boost_matched", field="particleNet_WvsQCD",
     #             bins=_SCORE_BINS,
     #             label=r"fatjet WvsQCD"),
     #        Axis(coll="candidate_boost_matched", field="tau21",
     #             bins=_T21_BINS,
     #             label=r"$\tau_{21}$"),
     #    ]),
     #    "fj_matched_ZvsQCD_tau21": HistConf([
     #        Axis(coll="candidate_boost_matched", field="particleNet_ZvsQCD",
     #             bins=_SCORE_BINS,
     #             label=r"fatjet ZvsQCD"),
     #        Axis(coll="candidate_boost_matched", field="tau21",
     #             bins=_T21_BINS,
     #             label=r"$\tau_{21}$"),
     #    ]),
        #"fj_matched_XqqVsQCD_tau21": HistConf([
        #    Axis(coll="candidate_boost_matched", field="particleNet_XqqVsQCD",
        #         bins=_SCORE_BINS,
        #         label=r"fatjet XqqVsQCD"),
        #    Axis(coll="candidate_boost_matched", field="tau21",
        #         bins=_T21_BINS,
        #         label=r"$\tau_{21}$"),
        #]),
                
         # # W fat jet
     #    "fj_unmatched_pt":    HistConf([Axis(coll="candidate_boost_unmatched", field="pt",  bins=50, start=200, stop=1000, label=r"$p_T(J^{W})$ [GeV]")]),
     #    "fj_unmatched_eta":   HistConf([Axis(coll="candidate_boost_unmatched", field="eta", bins=48, start=-2.4, stop=2.4,   label=r"$\eta(J^{W})$")]),
     #    "fj_unmatched_msd":   HistConf([Axis(coll="candidate_boost_unmatched", field="msoftdrop", bins=40, start=0,   stop=200,   label=r"$m_{SD}(J^{W})$ [GeV]")]),
     #    "fj_unmatched_t21":   HistConf([Axis(coll="candidate_boost_unmatched", field="tau21", bins=32, start=0, stop=1.1,   label=r"$\tau_{21}$")]),

     #    #"fj_unmatched_XqqVsQCD":   HistConf([Axis(coll="candidate_boost_unmatched", field="particleNet_XqqVsQCD", bins=40, start=0, stop=1,   label=r"fatjet XqqVsQCD")]),
     #    "fj_unmatched_WvsQCD":   HistConf([Axis(coll="candidate_boost_unmatched", field="particleNet_WvsQCD", bins=40, start=0, stop=1,   label=r"fatjet WvsQCD")]),
     #    "fj_unmatched_ZvsQCD":   HistConf([Axis(coll="candidate_boost_unmatched", field="particleNet_ZvsQCD", bins=40, start=0, stop=1,   label=r"fatjet ZvsQCD")]),
     #    "fj_unmatched_WvsQCD_pt_eta": HistConf([
     #        Axis(coll="candidate_boost_unmatched", field="particleNet_WvsQCD",
     #             bins=_SCORE_BINS,
     #             label=r"fatjet WvsQCD"),
     #        Axis(coll="candidate_boost_unmatched", field="pt",
     #             bins=_PT_BINS,
     #             label=r"$p_T$"),
     #        Axis(coll="candidate_boost_unmatched", field="abseta",
     #             bins=_ETA_BINS,
     #             label=r"$|\eta|$"),
     #        Axis(coll="candidate_boost_unmatched", field="msoftdrop",
     #             bins=_MASS_BINS,
     #             label=r"$m_{SD}(J^{W})$ [GeV]"),
     #        Axis(coll="candidate_boost_unmatched", field="tau21",
     #             bins=_T21_COARSE_BINS,
     #             label=r"$\tau_{21}$"),
     #    ]),
     #    "fj_unmatched_ZvsQCD_pt_eta": HistConf([
     #        Axis(coll="candidate_boost_unmatched", field="particleNet_ZvsQCD",
     #             bins=_SCORE_BINS,
     #             label=r"fatjet ZvsQCD"),
     #        Axis(coll="candidate_boost_unmatched", field="pt",
     #             bins=_PT_BINS,
     #             label=r"$p_T$"),
     #        Axis(coll="candidate_boost_unmatched", field="abseta",
     #             bins=_ETA_BINS,
     #             label=r"$|\eta|$"),
     #        Axis(coll="candidate_boost_unmatched", field="msoftdrop",
     #             bins=_MASS_BINS,
     #             label=r"$m_{SD}(J^{W})$ [GeV]"),
     #        Axis(coll="candidate_boost_unmatched", field="tau21",
     #             bins=_T21_COARSE_BINS,
     #             label=r"$\tau_{21}$"),
     #    ]),
        #"fj_unmatched_XqqVsQCD_pt_eta": HistConf([
        #    Axis(coll="candidate_boost_unmatched", field="particleNet_XqqVsQCD",
        #         bins=_SCORE_BINS,
        #         label=r"fatjet XqqVsQCD"),
        #    Axis(coll="candidate_boost_unmatched", field="pt",
        #         bins=_PT_BINS,
        #         label=r"$p_T$"),
        #    Axis(coll="candidate_boost_unmatched", field="abseta",
        #         bins=_ETA_BINS,
        #         label=r"$|\eta|$"),
        #    Axis(coll="candidate_boost_unmatched", field="msoftdrop",
        #         bins=_MASS_BINS,
        #         label=r"$m_{SD}(J^{W})$ [GeV]"),
        #    Axis(coll="candidate_boost_unmatched", field="tau21",
        #         bins=_T21_COARSE_BINS,
        #         label=r"$\tau_{21}$"),
        #]),
     #    "fj_unmatched_WvsQCD_tau21": HistConf([
     #        Axis(coll="candidate_boost_unmatched", field="particleNet_WvsQCD",
     #             bins=_SCORE_BINS,
     #             label=r"fatjet WvsQCD"),
     #        Axis(coll="candidate_boost_unmatched", field="tau21",
     #             bins=_T21_BINS,
     #             label=r"$\tau_{21}$"),
     #    ]),
     #    "fj_unmatched_ZvsQCD_tau21": HistConf([
     #        Axis(coll="candidate_boost_unmatched", field="particleNet_ZvsQCD",
     #             bins=_SCORE_BINS,
     #             label=r"fatjet ZvsQCD"),
     #        Axis(coll="candidate_boost_unmatched", field="tau21",
     #             bins=_T21_BINS,
     #             label=r"$\tau_{21}$"),
     #    ]),
        #"fj_unmatched_XqqVsQCD_tau21": HistConf([
        #    Axis(coll="candidate_boost_unmatched", field="particleNet_XqqVsQCD",
        ##         bins=_SCORE_BINS,
        #         label=r"fatjet XqqVsQCD"),
        #    Axis(coll="candidate_boost_unmatched", field="tau21",
        #         bins=_T21_BINS,
        #         label=r"$\tau_{21}$"),
        #]),
    },
)
