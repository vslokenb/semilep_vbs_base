#!/usr/bin/env python3
"""
Read coffea files and write a ROOT file for CMS Combine containing histograms
for all 12 channels: 4 SRs x {SR, TTCR, WCR}.

Unlike the SRonly version, NO scale factors are pre-applied.  Instead,
free-floating rateParams (norm_top, norm_wjet) are used in the datacards so
that the fit to data in the CRs constrains the TTbar and W+jets normalization
simultaneously with the signal extraction in the SRs.

v28 INPUTS (multi-sample output_all.coffea files)
  The v28 production (V28_COMBINE_HANDOFF.md) packs many samples into one
  <stage>_<YEAR>_<GROUP>/output_all.coffea, with several config processes
  pointing at the SAME file. load_process_hists() already selects by sample
  key and sums the inner dataset level (eras), so the structure is handled;
  the v28-specific additions are:
    - load_coffea_cached(): one multi-GB file is shared by a whole group of
      processes, and without a cache it would be deserialized once per
      (process, region) -- hundreds of times per run. V28_CACHE_SIZE env var
      bounds the number of outputs held in memory (default 4; lower it on
      RAM-tight nodes, raise it for speed).
    - a year guard on the inner dataset keys (year= threads down from the
      config), so datasets of another year can never be silently summed in
      (the handoff section 4.2 pitfall). v28 files are single-year, so this
      is belt-and-suspenders; legacy callers that pass no year are unchanged.
    - data sample names: v28 uses DATA_SingleMuon / DATA_EGamma /
      DATA_SingleElectron (and 2016/2017 have SingleElectron where 2018 has
      EGamma) -- the ELECTRON/MUON_DATA_PROCESSES sets accept both namings.
  Generate the inputs YAMLs from the manifest with
  make_inputs_from_manifest.py -- one file per year, processes ordered by
  group so consecutive processes share the cached file.

Process merging:
  - TTBAR_PROCESSES  -> 'TTbar'
  - WJET_PROCESSES   -> 'WJets'
  - DY_PROCESSES     -> 'DY'
  Single-top processes keep their individual names.

Region names vs coffea category names:
  The ROOT channel / datacard bin names follow ALL_REGIONS below.  The coffea
  'cat' axis does not always use the same strings -- see CATEGORY_OVERRIDES.

Output ROOT file structure:
  {sr}/{proc}                          <- nominal (SR, BDT shape)
  {sr}/{proc}_{syst}Up/Down            <- shape variations
  {sr}/data_obs                        <- observed data (SR)
  ttbar_cr_{sr}/{proc}                 <- nominal (TTCR, 1 bin)
  ttbar_cr_{sr}/{proc}_{syst}Up/Down   <- shape variations (1 bin)
  ttbar_cr_{sr}/data_obs               <- observed data (TTCR, 1 bin)
  w_cr_{sr}/{proc}                     <- nominal (WCR, 1 bin)
  w_cr_{sr}/{proc}_{syst}Up/Down       <- shape variations (1 bin)
  w_cr_{sr}/data_obs                   <- observed data (WCR, 1 bin)

Signal processes are NOT included in CRs (their contribution is negligible).

--scale-systs controls which LHEScaleWeight_* variations become nuisances.
The default keeps the renorm/fact scale variations named in _SCALE_KEEP and
drops the envelope members. The run prints which were kept and which dropped,
and warns if none survived -- that state is what silently removed the scale
uncertainty from the datacards.

--variable selects what is histogrammed. The default 'bdt' reads the per-SR
discriminant bdt_{sr}. Passing an explicit coffea variable name (e.g. mjj_vbs)
reads that same axis for every region, which drives the mjj sanity check: the
card-building and Asimov-significance path then runs on a variable with no BDT
training in it. In that mode --binning is ignored (its edges describe the BDT
score) and the native binning is kept unless --sr-rebin is given.
"""

import argparse
import os
from collections import defaultdict, OrderedDict

import numpy as np
import yaml
import coffea.util
import boost_histogram as bh
import uproot
import re


SIGNAL_REGIONS = ['resolved_mu', 'resolved_e', 'boosted_mu', 'boosted_e']
CR_SUFFIXES     = ['ttbar_cr', 'w_cr']
ALL_REGIONS     = SIGNAL_REGIONS + [
    f'{c}_{sr}' for sr in SIGNAL_REGIONS for c in CR_SUFFIXES
]

# ---------------------------------------------------------------------------
# Region name -> coffea 'cat' axis name.
#
# The resolved W control regions are not topology-labelled in the coffea
# output: they are called w_cr_mu / w_cr_e rather than w_cr_resolved_mu /
# w_cr_resolved_e.  Anything not listed here maps to itself.
#
# This matters because load_process_hists() skips silently when the requested
# category is absent, so an unmapped region yields no data_obs, and
# make_datacards.py then drops that channel from the card without an error.
# The visible symptom is a rateParam with nothing constraining it (norm_wjet
# floating to an arbitrary value with an uncertainty larger than its own
# allowed range).
# ---------------------------------------------------------------------------
CATEGORY_OVERRIDES = {
    'w_cr_resolved_mu': 'w_cr_mu',
    'w_cr_resolved_e':  'w_cr_e',
}


def coffea_cat(region):
    """Coffea 'cat' axis value to read for a given canonical region name."""
    return CATEGORY_OVERRIDES.get(region, region)


# Data sample names, both namings:
#   - legacy per-sample files: EGamma / SingleMuon
#   - v28 manifest samples:    DATA_EGamma / DATA_SingleMuon /
#                              DATA_SingleElectron
# EGamma is the 2018 electron primary dataset; 2016/2017 use SingleElectron
# (handoff section 2), so both must be accepted for the electron channels.
ELECTRON_DATA_PROCESSES = {'EGamma', 'DATA_EGamma',
                           'SingleElectron', 'DATA_SingleElectron'}
MUON_DATA_PROCESSES     = {'SingleMuon', 'DATA_SingleMuon'}

# TTbar (dileptonic + semileptonic) - to be merged into 'TTbar'
TTBAR_PROCESSES = {
    "TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8",
    "TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8",
}
TTBAR_MERGED_NAME = 'TTbar'

# Single top - kept with individual names (NOT merged into TTbar)
SINGLETOP_PROCESSES = {
    "ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8",
    "ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8",
    "ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8",
    "ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8",
    "ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8",
}

WJET_PROCESSES = {
    "WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8",
    "WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8",
    "WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8",
    "WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8",
    "WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8",
    "WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8",
    "WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8",
    "WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8",
}
WJET_MERGED_NAME = 'WJets'

DY_PROCESSES = {
    "DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8",
    "DYJetsToLL_M-10to50_TuneCP5_13TeV-amcatnloFXFX-pythia8",
}
DY_MERGED_NAME = 'DY'

QCD_PROCESSES = {
    "WminusTo2JZTo2LJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
    "WminusToLNuWminusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
    "WminusToLNuZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
    "WplusTo2JZTo2LJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
    "WplusTo2JWminusToLNuJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
    "WplusToLNuWminusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
    "WplusToLNuWplusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
    "WplusToLNuZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
    "ZTo2LZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
}

QCD_MERGED_NAME = 'QCD-VV'

MULTI_PROCESSES = {
    "GluGluWWToLNuQQ_TuneCP5_13TeV_madgraph-pythia8",
    "WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8",
    "WWZ_4F_TuneCP5_13TeV-amcatnlo-pythia8",
    "WZTo3LNu_mllmin01_NNPDF31_TuneCP5_13TeV_powheg_pythia8",
    "WZZ_TuneCP5_13TeV-amcatnlo-pythia8",
    "ZGToLLG_01J_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8",
    "ZZZ_TuneCP5_13TeV-amcatnlo-pythia8",
}

MULTI_MERGED_NAME = 'Multiboson'

MIN_YIELD = 1e-3

# Shape systematics to symmetrize: replace (up, down) with
# (nominal + delta, nominal - delta) where delta = max(|up-nom|, |down-nom|)
# applied bin-by-bin.  Variances are taken from the nominal histogram.
#
# MATCHING, incl. the per-year variations: an entry selects the variation with
# EXACTLY its name, plus any variation that is the entry with a year suffix
# appended -- 'AK4PFchs_JER' also selects AK4PFchs_JER_2017 /
# AK4PFchs_JER_2016_PreVFP, because the v28 JES/JER families carry their year
# IN the raw coffea variation name. An entry that itself ends in a year
# (AK4PFchs_JES_Regrouped_EC2_2017) selects only that one; a base entry
# (AK4PFchs_JES_Regrouped_EC2) selects the base AND its year-specific
# siblings, since symmetrization is a per-family choice. What each entry
# matched is printed once, and an entry that matches NOTHING in the whole run
# aborts (check_symmetrize_entries_matched) -- the silent-skip behavior is
# exactly how a renamed family would otherwise quietly lose its
# symmetrization. The QvG names below carry no year in the coffea files (the
# _<year> on their nuisances is added later by make_datacards.py).
SYMMETRIZE_SYSTS = {
    # 'AK4PFchs_JES_Total',
    # 'AK8PFPuppi_JES_Total',
    # 'AK4PFchs_JER',
    # 'AK8PFPuppi_JER',
    'sf_qgtagging_sf_qgtagging_herwig',
    'sf_qgtagging_sf_qgtagging_total',
    'sf_qgtagging_sf_qgtagging_fsr',
    # 'sf_partonshower_fsr',
}

# Raw coffea variation names that end in a year, e.g. _2017 / _2016_PreVFP.
# (No 'yr' form here: _yr tags exist only in datacard nuisance names,
# downstream of this script.)
_YEAR_SUFFIX_RE = re.compile(r'_20\d{2}(_(Pre|Post)VFP)?$')

# entry -> set of actual variation base names it symmetrized, filled across
# the whole run; read once by check_symmetrize_entries_matched().
_SYMM_MATCHED = {}


def symmetrize_targets(entry, systs_present):
    """The systematics in `systs_present` that SYMMETRIZE_SYSTS entry selects.

    Exact name, or the entry plus a year suffix (see the SYMMETRIZE_SYSTS
    comment). Records every hit for the end-of-run dead-entry check.
    """
    hits = {s for s in systs_present
            if s == entry
            or (_YEAR_SUFFIX_RE.search(s)
                and _YEAR_SUFFIX_RE.sub('', s) == entry)}
    _SYMM_MATCHED.setdefault(entry, set()).update(hits)
    return hits


def check_symmetrize_entries_matched():
    """Report what each SYMMETRIZE_SYSTS entry matched; abort on dead entries.

    Call ONCE after the symmetrization loop. Same philosophy as
    check_renorm_patterns_matched: an opt-in entry that selected nothing in
    the entire run means the family was renamed (e.g. gained a year suffix
    this script does not know) or the name is mistyped, and silently
    symmetrizing nothing is the failure mode this check removes.
    """
    if not SYMMETRIZE_SYSTS:
        return
    print('\nSymmetrized systematics:')
    dead = []
    for entry in sorted(SYMMETRIZE_SYSTS):
        hits = sorted(_SYMM_MATCHED.get(entry, ()))
        if not hits:
            dead.append(entry)
            print(f'    {entry} -> MATCHED NOTHING')
        elif hits == [entry]:
            print(f'    {entry}')
        else:
            print(f'    {entry} -> ' + ', '.join(hits))
    if dead:
        raise SystemExit(
            'ERROR: these SYMMETRIZE_SYSTS entries matched NO variation in '
            'any SR:\n         ' + '\n         '.join(dead) + '\n'
            '       An entry that selects nothing means the family was '
            'renamed (per-year\n       suffix?) or mistyped -- fix it or '
            'remove it.')

RENORM_INCLUDE_SYSTS = [
    # r"sf_btag.*",
    r"LHEScaleWeight_.*",
    r"sf_partonshower_.*",
]
# Target number of bins per SR (original coffea histograms have 40 bins).
# Used as fallback when no --binning config is provided.
REBIN_TARGETS = {
    'resolved_mu': 40,
    'resolved_e':  40,
    'boosted_mu':  40,
    'boosted_e':   40,
}


# ---------------------------------------------------------------------------
# Rebinning utilities
# ---------------------------------------------------------------------------

def rebin(vals, variances, edges, n_target):
    """Merge adjacent bins so the histogram has n_target bins.

    n_orig must be divisible by n_target.
    Returns (new_vals, new_variances, new_edges).
    """
    n_orig = len(vals)
    if n_orig == n_target:
        return vals, variances, edges
    if n_orig % n_target != 0:
        raise ValueError(f'Cannot rebin {n_orig} bins into {n_target} bins '
                         f'({n_orig} is not divisible by {n_target})')
    factor = n_orig // n_target
    new_vals = vals.reshape(n_target, factor).sum(axis=1)
    new_vars = variances.reshape(n_target, factor).sum(axis=1)
    new_edges = edges[::factor]
    # Ensure last edge is preserved exactly
    new_edges = np.append(new_edges[:n_target], edges[-1])
    return new_vals, new_vars, new_edges


def rebin_hists_dict(hists_dict, n_target):
    """Apply rebin() to every variation in a hists_dict."""
    out = {}
    for var_key, (vals, variances, edges) in hists_dict.items():
        out[var_key] = rebin(vals, variances, edges, n_target)
    return out


def rebin_to_edges(vals, variances, orig_edges, new_edges):
    """Rebin onto arbitrary new_edges by summing original bins.

    new_edges must be a subset of orig_edges (matched within 1e-6 relative
    tolerance to handle floating-point rounding).  All original bins that fall
    between consecutive new edges are summed into one output bin.

    Returns (new_vals, new_variances, new_edges).
    """
    new_edges = np.asarray(new_edges, dtype=float)
    orig_edges = np.asarray(orig_edges, dtype=float)

    indices = np.searchsorted(orig_edges, new_edges)
    indices = np.clip(indices, 0, len(orig_edges) - 1)

    # Snap to nearest orig edge to handle float rounding
    for i in range(len(indices)):
        idx = indices[i]
        if not np.isclose(orig_edges[idx], new_edges[i], rtol=1e-6, atol=1e-9):
            if idx > 0 and np.isclose(orig_edges[idx - 1], new_edges[i], rtol=1e-6, atol=1e-9):
                indices[i] = idx - 1
            else:
                raise ValueError(
                    f'Bin edge {new_edges[i]} not found in original edges '
                    f'(nearest: {orig_edges[idx]:.6g}). '
                    f'new_edges must be a subset of the original coffea bin edges.'
                )

    n_new = len(new_edges) - 1
    new_vals = np.zeros(n_new)
    new_vars = np.zeros(n_new)
    for i in range(n_new):
        lo, hi = indices[i], indices[i + 1]
        new_vals[i] = vals[lo:hi].sum()
        new_vars[i] = variances[lo:hi].sum()

    return new_vals, new_vars, new_edges


def rebin_hists_to_edges(hists_dict, new_edges):
    """Apply rebin_to_edges() to every variation in a hists_dict."""
    out = {}
    for var_key, (vals, variances, edges) in hists_dict.items():
        out[var_key] = rebin_to_edges(vals, variances, edges, new_edges)
    return out


def rebin_to_one(hists_dict):
    """Rebin all variations to a single bin (total yield).

    Used for CR channels where only the total count matters for the
    rateParam constraint.
    """
    out = {}
    for var_key, (vals, variances, edges) in hists_dict.items():
        out[var_key] = (
            np.array([vals.sum()]),
            np.array([variances.sum()]),
            np.array([0.0, 1.0]),
        )
    return out


# ---------------------------------------------------------------------------
# Region / variable helpers
# ---------------------------------------------------------------------------

CR_PREFIXES = tuple(f'{c}_' for c in CR_SUFFIXES)   # ('ttbar_cr_', 'w_cr_')


def sr_for_cr(region):
    """Return the base SR name for a CR region, or None if it IS an SR."""
    for pre in CR_PREFIXES:
        if region.startswith(pre):
            return region[len(pre):]
    return None


# The default discriminant is one BDT per SR, so the coffea variable name
# depends on the region. Any other variable (mjj_vbs for the sanity check) is a
# single axis shared by every category, so its name is used verbatim.
DEFAULT_VARIABLE = 'bdt'


def bdt_var_for_region(region, variable=DEFAULT_VARIABLE):
    """Coffea variable to read for a region.

    variable='bdt'  -> per-SR discriminant, bdt_{sr}; CRs read the same
                       variable as their parent SR.
    anything else   -> that name for every region (e.g. 'mjj_vbs').
    """
    if variable != DEFAULT_VARIABLE:
        return variable
    base = sr_for_cr(region) or region
    return f'bdt_{base}'


def is_electron_region(region):
    # resolved_e, boosted_e, and their CRs
    return region.endswith('_e')


# ---------------------------------------------------------------------------
# Histogram loading
# ---------------------------------------------------------------------------

# LRU cache for coffea outputs. v28 group files hold every sample of their
# group-year and are referenced by many processes in the config: without a
# cache, the SAME multi-GB file is deserialized once per (process, region) --
# hundreds of times per run. Each cached output can be large, so the cache is
# small and bounded: V28_CACHE_SIZE (default 4; set to 1 on RAM-tight nodes,
# raise it on fat ones). The manifest-generated configs order processes by
# group, so consecutive processes hit the cached file.
_COFFEA_CACHE = OrderedDict()
_COFFEA_CACHE_SIZE = int(os.environ.get('V28_CACHE_SIZE', '4'))


def load_coffea_cached(fpath):
    key = os.path.abspath(fpath)
    if key in _COFFEA_CACHE:
        _COFFEA_CACHE.move_to_end(key)
        return _COFFEA_CACHE[key]
    out = coffea.util.load(fpath)
    _COFFEA_CACHE[key] = out
    while len(_COFFEA_CACHE) > _COFFEA_CACHE_SIZE:
        _COFFEA_CACHE.popitem(last=False)
    return out


def _dataset_matches_year(ds_name, sample, year):
    """True when dataset key sample_year[_Era] belongs to `year`.

    Handles 2016_PreVFP/2016_PostVFP (the year itself contains '_') and era
    suffixes. Falls back to a substring test when the dataset key does not
    start with the sample name.
    """
    y = str(year)
    if ds_name.startswith(sample):
        tail = ds_name[len(sample):]
        return tail == f'_{y}' or tail.startswith(f'_{y}_')
    return f'_{y}' in ds_name


def load_process_hists(file_list, process_name, var_name, cat, year=None):
    """Sum histograms across multiple coffea files.

    Returns variation -> [vals, vars, edges].

    v28 notes: files may hold MANY samples (no --process-separately) --
    selection is by the process_name sample key, so groupmates in the same
    file are never absorbed. The inner loop runs over the sample's dataset
    keys (sample_year[_Era]): data eras are summed; when `year` is given,
    datasets of any other year are excluded (v28 files are single-year, so
    this guard should never fire -- if it does, the config points at the
    wrong file).
    """
    result = {}
    for fpath in file_list:
        if not fpath or not os.path.exists(fpath):
            print(f'  [warn] not found: {fpath}')
            continue
        try:
            f = load_coffea_cached(fpath)
        except Exception as e:
            print(f'  [warn] load failed {fpath}: {e}')
            continue
        variables = f.get('variables', {})
        if var_name not in variables or process_name not in variables[var_name]:
            continue
        for ds_name, h in variables[var_name][process_name].items():
            if year is not None and not _dataset_matches_year(
                    ds_name, process_name, year):
                continue
            if cat not in list(h.axes['cat']):
                continue
            for var in list(h.axes['variation']):
                try:
                    sl = h[{'cat': cat, 'variation': var}]
                    v = sl.values().copy()
                    e = (sl.variances() if sl.variances() is not None
                         else np.zeros_like(v)).copy()
                    edges = sl.axes[0].edges.copy()
                except Exception:
                    continue
                if var not in result:
                    result[var] = [v, e, edges]
                else:
                    result[var][0] += v
                    result[var][1] += e
    return result


def available_categories(file_list, var_name):
    """Every 'cat' axis value present for var_name, for diagnostics."""
    cats = set()
    for fpath in file_list:
        if not fpath or not os.path.exists(fpath):
            continue
        try:
            f = load_coffea_cached(fpath)
        except Exception:
            continue
        for proc_hists in f.get('variables', {}).get(var_name, {}).values():
            for h in proc_hists.values():
                try:
                    cats.update(list(h.axes['cat']))
                except Exception:
                    continue
    return cats


# Which LHEScaleWeight_* variations become nuisances.
#
# The renormalization and factorization scale variations are the physics ones.
# The rest of the LHEScaleWeight block is the 6- or 9-point envelope members,
# which must NOT each become an independent nuisance -- that would count the
# same effect many times over.
#
# If this set matches nothing while LHEScaleWeight variations do exist in the
# coffea files, report_scale_systs() prints the names that ARE there. That is
# the case to watch: the names are generator/producer dependent, and a set that
# matches nothing silently drops the scale uncertainty from every datacard.
_SCALE_KEEP = {'LHEScaleWeight_fact_scale', 'LHEScaleWeight_renorm_scale'}

# 'selected' -> only _SCALE_KEEP; 'all' -> every LHEScaleWeight_* Up/Down pair
# (use only to see what exists -- envelope members are not independent);
# 'none' -> drop them all.
SCALE_SYST_MODE = 'selected'

# Filled by get_standard_systs() so main() can report once instead of per file.
_SCALE_SEEN = set()
_SCALE_KEPT = set()


def _scale_base(v):
    """Strip the Up/Down suffix from a variation name."""
    if v.endswith('Up'):
        return v[:-2]
    if v.endswith('Down'):
        return v[:-4]
    return None


def get_standard_systs(variations):
    systs = set()
    for v in variations:
        if v == 'nominal':
            continue
        if v.startswith('LHEPdfWeight_pdf_'):
            continue

        if v.startswith('LHEScaleWeight_'):
            base = _scale_base(v)
            if base is not None:
                _SCALE_SEEN.add(base)
            if SCALE_SYST_MODE == 'none':
                continue
            if SCALE_SYST_MODE == 'selected' and base not in _SCALE_KEEP:
                continue
            if base is not None:
                _SCALE_KEPT.add(base)
            # falls through, so the Up/Down pair is registered below

        if v.endswith('Up'):
            systs.add(v[:-2])
        elif v.endswith('Down'):
            systs.add(v[:-4])
    return systs


def report_scale_systs():
    """Say what happened to the LHEScaleWeight variations. Called once."""
    if not _SCALE_SEEN:
        print('\nLHEScaleWeight: no such variations found in the coffea files.')
        return
    print(f'\nLHEScaleWeight variations (mode={SCALE_SYST_MODE!r}):')
    for base in sorted(_SCALE_SEEN):
        mark = 'KEPT   ' if base in _SCALE_KEPT else 'dropped'
        print(f'    {mark}  {base}')
    if not _SCALE_KEPT:
        print('  WARNING: every LHEScaleWeight variation was dropped, so the')
        print('           scale uncertainty will be ABSENT from the datacards.')
        print('           _SCALE_KEEP currently holds:')
        for n in sorted(_SCALE_KEEP):
            print(f'             {n}')
        print('           None of those match the names listed above. Edit')
        print('           _SCALE_KEEP to the right names, or pass')
        print('           --scale-systs all to keep every variation.')



# Patterns from include_systs that matched at least one systematic anywhere,
# across every call. check_renorm_patterns_matched() reads this at the end:
# a pattern with zero hits in the whole run is a renamed/mistyped systematic
# family, and silently renormalizing nothing is exactly the failure mode the
# opt-in list must not have.
_RENORM_PATTERN_HITS = {}


def renormalize_rateparam_procs_per_category(all_hists, category_ch_keys, procs,
                                             include_systs=()):
    """For each category (a group of channels sharing one free-floating
    rateParam per process -- e.g. SR + TTCR + WCR of 'boosted_mu', all
    covered by the same norm_top_boosted_mu/norm_wjet_boosted_mu rateParam)
    and each of `procs` (the processes that HAVE such a rateParam: TTbar,
    WtoLNu-2Jets), rescale the SELECTED systematics' Up/Down variations by a
    single per-category factor so the SR+TTCR+WCR combined yield for that
    variation equals the SR+TTCR+WCR combined nominal yield.

    Why: norm_top/norm_wjet are fit freely from the SR+CR combination, so
    they will already absorb any fully-correlated overall rate shift on
    TTbar/WtoLNu-2Jets across the category, regardless of what caused it.
    If another systematic's Up/Down variation also carries such an overall
    rate shift (e.g. btagging SF pushing the category-wide TTbar yield up
    or down), that shift is redundant with what norm_top/norm_wjet already
    covers -- attributing it to the other systematic double-counts it.

    What's preserved: this uses ONE scale factor per (category, proc, syst,
    direction), applied identically to all channels in the category, so the
    RELATIVE split between SR/TTCR/WCR from the systematic is untouched --
    only the overall category-wide level is pulled back to nominal. A
    systematic that shifts more of proc's yield into the SR at the expense
    of the CRs (a genuine, non-redundant migration/anti-correlation effect)
    keeps that redistribution; only the part every channel moves together
    is removed.

    include_systs: OPT-IN list of regex patterns selecting which systematic
    base names get this treatment; everything unmatched is left untouched.
    Patterns are anchored (re.fullmatch, the same convention as combine's
    rgx{} and uncertainty_breakdown.py), so 'sf_btag' matches only that
    literal name and the family needs 'sf_btag.*'. An empty list means this
    function does nothing. Do not list systematics already made shape-only
    PER CHANNEL upstream (make_shape_only: SYMMETRIZE_SYSTS/QvG and
    WVSQCD_SHAPE_ONLY_SYSTS/WvsQCD tagger) -- each channel's yield is
    already pinned to its own nominal there, so the category-wide total is
    already at nominal and rescaling would be a redundant no-op at best.

    After the last call, run check_renorm_patterns_matched(include_systs)
    once: a pattern that matched nothing in the entire run aborts, because
    an opt-in entry that silently selects nothing is a renamed or mistyped
    systematic family, not a harmless extra.

    category_ch_keys: list of ch_key strings for one category's channels
    (e.g. ['boosted_mu', 'boosted_mu_TTCR', 'boosted_mu_WCR']).
    """
    if not include_systs:
        return
    try:
        patterns = [re.compile(p) for p in include_systs]
    except re.error as exc:
        raise SystemExit(f"ERROR: bad regex in include_systs: {exc}")
    for p in patterns:
        _RENORM_PATTERN_HITS.setdefault(p.pattern, 0)

    present = [k for k in category_ch_keys if k in all_hists]
    if len(present) < 2:
        return

    for proc in procs:
        proc_chans = [k for k in present if proc in all_hists[k]]
        if not proc_chans:
            continue

        nom_total = sum(all_hists[k][proc]['nominal'][0].sum() for k in proc_chans)
        if nom_total <= 0:
            continue

        systs = set()
        for k in proc_chans:
            for var_key in all_hists[k][proc]:
                if var_key == 'nominal':
                    continue
                if var_key.endswith('Up'):
                    systs.add(var_key[:-2])
                elif var_key.endswith('Down'):
                    systs.add(var_key[:-4])

        selected = set()
        for pat in patterns:
            hits = {s for s in systs if pat.fullmatch(s)}
            _RENORM_PATTERN_HITS[pat.pattern] += len(hits)
            selected |= hits

        for syst in selected:
            for direction in ('Up', 'Down'):
                var_key = f'{syst}{direction}'
                have_var = [k for k in proc_chans if var_key in all_hists[k][proc]]
                if not have_var:
                    continue
                # Channels without this variation contribute their nominal
                # (unaffected) yield to the category total, same as nominal.
                no_var_nom_total = sum(
                    all_hists[k][proc]['nominal'][0].sum()
                    for k in proc_chans if k not in have_var)
                var_total = no_var_nom_total + sum(
                    all_hists[k][proc][var_key][0].sum() for k in have_var)
                if var_total <= 0:
                    continue
                # The factor applies only to the have_var channels, so it must
                # bring THEIR summed yield to (nom_total - no_var_nom_total):
                # the no-variation channels already sit at nominal and are not
                # scaled. Using nom_total/var_total here would under-correct
                # whenever no_var_nom_total > 0 (new total would be
                # nom_total + no_var_nom_total*(1 - nom_total/var_total)).
                # After merge_processes() every variation key exists in every
                # channel, so no_var_nom_total is 0 in practice and the two
                # formulas agree -- this is the correct general form.
                scaled_nom = nom_total - no_var_nom_total
                scaled_var = var_total - no_var_nom_total
                if scaled_var <= 0 or scaled_nom <= 0:
                    continue
                scale = scaled_nom / scaled_var
                for k in have_var:
                    v, var, edges = all_hists[k][proc][var_key]
                    all_hists[k][proc] = dict(all_hists[k][proc])
                    all_hists[k][proc][var_key] = (v * scale, var * scale ** 2, edges)


def check_renorm_patterns_matched(include_systs):
    """Abort if any include_systs pattern matched nothing in the entire run.

    Call ONCE, after renormalize_rateparam_procs_per_category has run over
    every category. Checked globally rather than per call because a pattern
    can legitimately match nothing in one category (sf_fj.* has no target in
    the resolved channels) while being essential in another.
    """
    dead = [p for p in include_systs
            if _RENORM_PATTERN_HITS.get(str(p), 0) == 0]
    if dead:
        raise SystemExit(
            "ERROR: these include_systs patterns matched NO systematic in any "
            "category:\n         " + "\n         ".join(dead) + "\n"
            "       An opt-in renormalization entry that selects nothing means "
            "the family\n       was renamed or the pattern is wrong -- fix it "
            "or remove it.")

def build_process_hists(file_list, process_name, var_name, cat, is_signal=False,
                        year=None):
    """Return variation_key -> (vals, vars, edges) for all standard systematics.

    The 100 LHEPdfWeight_pdf_N variations are Hessian eigenvariations, so each
    is kept as an independent shape systematic for signal processes.  Because the
    eigenvariations share the same name across processes, Combine correctly treats
    them as fully correlated nuisance parameters.

    year= is threaded to load_process_hists for the v28 dataset-year guard;
    callers that omit it (legacy per-sample files) are unchanged.
    """
    raw = load_process_hists(file_list, process_name, var_name, cat, year=year)
    if not raw or 'nominal' not in raw:
        return {}
    nom_vals, nom_vars, edges = raw['nominal']
    out = {'nominal': (nom_vals.copy(), nom_vars.copy(), edges.copy())}

    for syst in get_standard_systs(raw.keys()):
        if f'{syst}Up' in raw and f'{syst}Down' in raw:
            out[f'{syst}Up']   = (raw[f'{syst}Up'][0].copy(),   raw[f'{syst}Up'][1].copy(),   edges.copy())
            out[f'{syst}Down'] = (raw[f'{syst}Down'][0].copy(), raw[f'{syst}Down'][1].copy(), edges.copy())

    # Individual PDF eigenvariations for signals only - independent nuisances,
    # correlated across signal processes by shared name; not applied to backgrounds
    if is_signal:
        for v in raw:
            if (v.startswith('LHEPdfWeight_pdf_')
                    and 'alpha_S' not in v
                    and v.endswith('Up')):
                syst = v[:-2]
                if f'{syst}Down' in raw:
                    out[f'{syst}Up']   = (raw[f'{syst}Up'][0].copy(),   raw[f'{syst}Up'][1].copy(),   edges.copy())
                    out[f'{syst}Down'] = (raw[f'{syst}Down'][0].copy(), raw[f'{syst}Down'][1].copy(), edges.copy())

    return out


# ---------------------------------------------------------------------------
# Merging / symmetrization utilities
# ---------------------------------------------------------------------------

def merge_processes(ch_hists, source_procs, merged_name):
    """Sum histograms from source_procs into merged_name, then remove sources.

    For each variation key, processes that lack the variation contribute their
    nominal histogram (i.e. no shape effect from that missing variation).
    """
    present = [p for p in source_procs if p in ch_hists]
    if not present:
        return

    # Collect all variation keys across all source processes
    all_keys = set()
    for p in present:
        all_keys.update(ch_hists[p].keys())

    merged = {}
    for var_key in all_keys:
        vals = variances = edges = None
        for p in present:
            ph = ch_hists[p]
            if var_key in ph:
                v, e, ed = ph[var_key]
            else:
                # Process has no shape for this variation: use its nominal
                v, e, ed = ph['nominal']
            if vals is None:
                vals, variances, edges = v.copy(), e.copy(), ed.copy()
            else:
                vals      = vals      + v
                variances = variances + e
        merged[var_key] = (vals, variances, edges)

    ch_hists[merged_name] = merged
    for p in present:
        del ch_hists[p]


def symmetrize_syst(hists_dict, syst):
    """Replace up/down shape variations with a symmetric envelope around nominal.

    new_up[i]   = nom[i] + max(|up[i]-nom[i]|, |down[i]-nom[i]|)
    new_down[i] = nom[i] - max(|up[i]-nom[i]|, |down[i]-nom[i]|)

    Variances for the new shapes are taken from the nominal histogram.
    Skips silently if any of up/down/nominal is missing.
    """
    up_key   = f'{syst}Up'
    down_key = f'{syst}Down'
    if up_key not in hists_dict or down_key not in hists_dict or 'nominal' not in hists_dict:
        return hists_dict

    nom_vals, nom_vars, edges = hists_dict['nominal']
    up_vals,  _,        _     = hists_dict[up_key]
    down_vals, _,       _     = hists_dict[down_key]

    delta = np.maximum(np.abs(up_vals - nom_vals), np.abs(down_vals - nom_vals))

    out = dict(hists_dict)
    out[up_key]   = (nom_vals + delta, nom_vars.copy(), edges.copy())
    out[down_key] = (nom_vals - delta, nom_vars.copy(), edges.copy())
    return out


# ---------------------------------------------------------------------------
# ROOT histogram helper
# ---------------------------------------------------------------------------

def integral(vals):
    return float(np.sum(np.maximum(vals, 0.0)))


def make_bh(values, variances, edges):
    h = bh.Histogram(bh.axis.Variable(edges), storage=bh.storage.Weight())
    h.view().value    = np.maximum(values,    0.0)
    h.view().variance = np.maximum(variances, 0.0)
    return h


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Histogram extraction for SR+CR simultaneous fit (no pre-applied SFs)')
    parser.add_argument('--config',   required=True,
                        help='YAML inputs config (inputs_2018.yaml, or a '
                             'manifest-generated inputs_v28_<year>.yaml)')
    parser.add_argument('--output',   default='histograms/histograms.root')
    parser.add_argument('--year',     default=None)
    parser.add_argument('--scale-systs', choices=['selected', 'all', 'none'],
                        default='selected',
                        help="Which LHEScaleWeight_* variations become "
                             "nuisances. 'selected' (default) keeps those in "
                             "_SCALE_KEEP -- the renorm/fact scale variations; "
                             "'all' keeps every one, including the envelope "
                             "members, which are NOT independent; 'none' drops "
                             "them. A run that keeps none warns loudly.")
    parser.add_argument('--variable', default=DEFAULT_VARIABLE,
                        help="Coffea variable to histogram. Default 'bdt' uses "
                             "the per-SR discriminant bdt_{sr}. Pass an explicit "
                             "name (e.g. mjj_vbs) to run the card-building and "
                             "significance path on another variable as a sanity "
                             "check.")
    parser.add_argument('--sr-rebin', type=int, default=None,
                        help='Rebin SR histograms to this many bins. Default: '
                             'REBIN_TARGETS for the BDT, and no rebinning at all '
                             'for any other --variable, whose native bin count '
                             'need not be divisible by 40.')
    parser.add_argument('--binning',  default=None,
                        help='YAML file with custom bin edges per SR. '
                             'Format: resolved_mu: [0.0, 0.2, 0.4, ...]. '
                             'Edges must be a subset of the original coffea bin edges. '
                             'SRs not listed fall back to REBIN_TARGETS uniform rebinning.')
    args = parser.parse_args()

    global SCALE_SYST_MODE
    SCALE_SYST_MODE = args.scale_systs

    is_bdt = (args.variable == DEFAULT_VARIABLE)
    if not is_bdt:
        print(f'Variable override: reading {args.variable!r} for every region '
              f'instead of the per-SR BDT.')

    # Load custom bin edges if provided.
    # The --binning config lists BDT-score edges; they do not describe any other
    # axis, so it is ignored rather than silently applied to the wrong variable
    # (where rebin_to_edges would either raise or produce nonsense).
    custom_edges = {}
    if args.binning and not is_bdt:
        print(f'  [note] ignoring --binning {args.binning}: those edges describe '
              f'the BDT score, not {args.variable!r}.')
    elif args.binning:
        with open(args.binning) as fh:
            raw_binning = yaml.safe_load(fh)
        for sr, edges in (raw_binning or {}).items():
            custom_edges[sr] = np.asarray(edges, dtype=float)
        print(f'Custom binning loaded from {args.binning}:')
        for sr, edges in custom_edges.items():
            print(f'  {sr}: {len(edges)-1} bins, edges = {edges.tolist()}')

    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    years = [args.year] if args.year else list(config['years'].keys())
    multi_year = len(years) > 1

    print('\nRegion -> coffea category map:')
    for region in ALL_REGIONS:
        cat = coffea_cat(region)
        flag = '  (override)' if cat != region else ''
        var = bdt_var_for_region(region, args.variable)
        print(f'  {region:<26} -> {cat:<20} [{var}]{flag}')

    # ---------------------------------------------------------------------- #
    # Step 1: Load histograms for ALL 12 regions (4 SRs + 4 TTCRs + 4 WCRs)  #
    # ---------------------------------------------------------------------- #
    # all_hists[channel][process][var_key] = (vals, vars, edges)
    all_hists = {}
    no_data   = []          # regions where no data_obs could be built

    for year in years:
        year_cfg = config['years'][year]

        for region in ALL_REGIONS:
            channel  = f'{region}_{year}' if multi_year else region
            var_name = bdt_var_for_region(region, args.variable)
            cat      = coffea_cat(region)
            electron = is_electron_region(region)
            is_cr    = sr_for_cr(region) is not None

            if channel not in all_hists:
                all_hists[channel] = {}

            # ---- Backgrounds ----
            for proc, pcfg in year_cfg.get('background', {}).items():
                flist = pcfg.get('files', [pcfg.get('file')])
                hists = build_process_hists(flist, proc, var_name, cat, year=year)
                if not hists:
                    continue
                if proc not in all_hists[channel]:
                    all_hists[channel][proc] = hists
                else:
                    # Sum across years
                    for vk in hists:
                        if vk in all_hists[channel][proc]:
                            v1, e1, ed = all_hists[channel][proc][vk]
                            v2, e2, _  = hists[vk]
                            all_hists[channel][proc][vk] = (v1 + v2, e1 + e2, ed)
                        else:
                            all_hists[channel][proc][vk] = hists[vk]

            # ---- Signals (SRs only - negligible in CRs) ----
            if not is_cr:
                for proc, pcfg in year_cfg.get('signal', {}).items():
                    flist = pcfg.get('files', [pcfg.get('file')])
                    hists = build_process_hists(flist, proc, var_name, cat,
                                                is_signal=True, year=year)
                    if not hists:
                        continue
                    if proc not in all_hists[channel]:
                        all_hists[channel][proc] = hists
                    else:
                        for vk in hists:
                            if vk in all_hists[channel][proc]:
                                v1, e1, ed = all_hists[channel][proc][vk]
                                v2, e2, _  = hists[vk]
                                all_hists[channel][proc][vk] = (v1 + v2, e1 + e2, ed)
                            else:
                                all_hists[channel][proc][vk] = hists[vk]

            # ---- Real data (SRs and CRs alike) ----
            data_acc = None
            data_files = []
            for proc, pcfg in year_cfg.get('data', {}).items():
                if electron and proc not in ELECTRON_DATA_PROCESSES:
                    continue
                if not electron and proc not in MUON_DATA_PROCESSES:
                    continue
                flist = pcfg.get('files', [pcfg.get('file')])
                data_files.extend(flist)
                raw = load_process_hists(flist, proc, var_name, cat, year=year)
                if not raw or 'nominal' not in raw:
                    continue
                v, e, edges = raw['nominal']
                if data_acc is None:
                    data_acc = [v.copy(), e.copy(), edges]
                else:
                    data_acc[0] += v
                    data_acc[1] += e

            if data_acc is not None:
                all_hists[channel]['data_obs'] = {'nominal': tuple(data_acc)}
            else:
                # A channel with no data_obs is silently DROPPED by
                # make_datacards.py, which leaves the corresponding rateParam
                # unconstrained. Report it loudly, with the categories that do
                # exist, so a name mismatch is obvious.
                no_data.append((region, cat, var_name, tuple(data_files)))

    if no_data:
        print('\n' + '=' * 70)
        print('WARNING: no data_obs built for the following regions.')
        print('These channels will be DROPPED from the datacards.')
        print('=' * 70)
        for region, cat, var_name, flist in no_data:
            print(f'\n  region {region!r} -> looked for cat {cat!r} in {var_name!r}')
            cats = available_categories(list(flist), var_name)
            if cats:
                print(f'    categories present: {sorted(cats)}')
                print( '    -> add the right name to CATEGORY_OVERRIDES')
            else:
                print(f'    variable {var_name!r} has no categories at all '
                      f'in the data files')
        print('=' * 70 + '\n')

    report_scale_systs()

    # ---------------------------------------------------------------------- #
    # Step 2: Merge process groups (TTbar, WJets, DY) for ALL channels        #
    # ---------------------------------------------------------------------- #
    for region in ALL_REGIONS:
        channel = f'{region}_{years[0]}' if multi_year else region
        if channel not in all_hists:
            continue
        merge_processes(all_hists[channel], TTBAR_PROCESSES,  TTBAR_MERGED_NAME)
        merge_processes(all_hists[channel], WJET_PROCESSES,   WJET_MERGED_NAME)
        merge_processes(all_hists[channel], DY_PROCESSES,     DY_MERGED_NAME)

    # ---------------------------------------------------------------------- #
    # Step 3: Symmetrize shape systematics (SRs only)                        #
    # ---------------------------------------------------------------------- #
    # Entries are resolved against the variation names actually present, so
    # per-year families (AK4PFchs_JER -> AK4PFchs_JER_2017) are picked up;
    # check_symmetrize_entries_matched() then aborts on any entry that
    # selected nothing anywhere.
    for sr in SIGNAL_REGIONS:
        channel = f'{sr}_{years[0]}' if multi_year else sr
        if channel not in all_hists:
            continue
        for proc in list(all_hists[channel].keys()):
            if proc == 'data_obs':
                continue
            present = {k[:-2] for k in all_hists[channel][proc]
                       if k.endswith('Up')}
            for entry in SYMMETRIZE_SYSTS:
                for syst in symmetrize_targets(entry, present):
                    all_hists[channel][proc] = symmetrize_syst(
                        all_hists[channel][proc], syst)
    check_symmetrize_entries_matched()
    # ---------------------------------------------------------------------- #
    # Step 3b: Renormalize selected systematics on the rateParam processes    #
    # ---------------------------------------------------------------------- #
    # Groups are built PER TOPOLOGY, not per lepton category, because that is
    # what the rateParams cover: norm_top_resolved / norm_wjet_resolved float
    # freely across resolved_mu AND resolved_e (SRs and CRs alike -- the
    # datacard rateParam bin field is "*resolved*"). Only a rate shift common
    # to that whole group is redundant with the norm; renormalizing smaller
    # groups would remove shifts the shared norm cannot absorb. If the norms
    # are ever decorrelated further (per lepton flavor), change the grouping
    # here to match, at the same commit.
    if RENORM_INCLUDE_SYSTS:
        print('\nRenormalizing the per-topology rate component of selected '
              f'systematics on {TTBAR_MERGED_NAME}/{WJET_MERGED_NAME}:')
        for pat in RENORM_INCLUDE_SYSTS:
            print(f'  {pat}')
        for topo in ('resolved', 'boosted'):
            cat_regions = []
            for sr in SIGNAL_REGIONS:
                if not sr.startswith(topo):
                    continue
                cat_regions.append(sr)
                cat_regions += [r for r in ALL_REGIONS if sr_for_cr(r) == sr]
            cat_keys = [f'{r}_{years[0]}' if multi_year else r
                        for r in cat_regions]
            renormalize_rateparam_procs_per_category(
                all_hists, cat_keys,
                [TTBAR_MERGED_NAME, WJET_MERGED_NAME],
                include_systs=RENORM_INCLUDE_SYSTS)
        check_renorm_patterns_matched(RENORM_INCLUDE_SYSTS)
    else:
        print('\n  [note] RENORM_INCLUDE_SYSTS is empty: no systematic is '
              'renormalized against the norm_* rateParams.')
    # ---------------------------------------------------------------------- #
    # Step 4: Write ROOT file                                                 #
    # ---------------------------------------------------------------------- #
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    print(f'Writing ROOT file: {args.output}')

    n_sr_ch = 0
    n_cr_ch = 0

    with uproot.recreate(args.output) as root_file:
        for region in ALL_REGIONS:
            channel = f'{region}_{years[0]}' if multi_year else region
            if channel not in all_hists or not all_hists[channel]:
                continue

            is_cr = sr_for_cr(region) is not None
            sr    = sr_for_cr(region) if is_cr else region

            for proc, hists in all_hists[channel].items():
                if is_cr:
                    # CR channels: rebin to 1 bin for counting
                    rebinned = rebin_to_one(hists)
                else:
                    # SR channels: apply variable binning from --binning config
                    if sr in custom_edges:
                        rebinned = rebin_hists_to_edges(hists, custom_edges[sr])
                    elif args.sr_rebin is not None:
                        rebinned = rebin_hists_dict(hists, args.sr_rebin)
                    elif is_bdt:
                        rebinned = rebin_hists_dict(hists, REBIN_TARGETS[sr])
                    else:
                        # Keep the native binning: REBIN_TARGETS is 40 because
                        # that is the BDT's bin count, and rebin() raises unless
                        # the target divides the input exactly.
                        rebinned = hists

                for var_key, (vals, errs, edges) in rebinned.items():
                    if var_key == 'nominal':
                        path = f'{channel}/{proc}'
                    else:
                        path = f'{channel}/{proc}_{var_key}'
                    root_file[path] = make_bh(vals, errs, edges)

            if is_cr:
                n_cr_ch += 1
            else:
                n_sr_ch += 1

    # ---------------------------------------------------------------------- #
    # Summary: per-channel yields, so a missing or mis-mapped CR is visible   #
    # ---------------------------------------------------------------------- #
    print('\nChannel summary:')
    print(f'  {"channel":<26}{"TTbar":>12}{"WJets":>12}{"total MC":>12}'
          f'{"data":>12}{"data/MC":>10}')
    print('  ' + '-' * 84)
    for region in ALL_REGIONS:
        channel = f'{region}_{years[0]}' if multi_year else region
        chh = all_hists.get(channel, {})
        if not chh:
            print(f'  {channel:<26}{"-- absent --":>58}')
            continue
        tot = sum(integral(h['nominal'][0])
                  for p, h in chh.items() if p != 'data_obs')
        top = integral(chh[TTBAR_MERGED_NAME]['nominal'][0]) if TTBAR_MERGED_NAME in chh else 0.0
        wj  = integral(chh[WJET_MERGED_NAME]['nominal'][0])  if WJET_MERGED_NAME  in chh else 0.0
        dat = integral(chh['data_obs']['nominal'][0]) if 'data_obs' in chh else None
        ratio = f'{dat / tot:.3f}' if (dat is not None and tot > 0) else 'NO DATA'
        dat_s = f'{dat:.1f}' if dat is not None else '--'
        print(f'  {channel:<26}{top:>12.1f}{wj:>12.1f}{tot:>12.1f}'
              f'{dat_s:>12}{ratio:>10}')

    print(f'\nDone.')
    print(f'  SR channels written: {n_sr_ch}')
    print(f'  CR channels written: {n_cr_ch}')
    print(f'  Total ROOT file: {args.output}')


if __name__ == '__main__':
    main()
