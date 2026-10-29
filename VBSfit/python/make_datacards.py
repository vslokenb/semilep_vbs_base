#!/usr/bin/env python3
"""
Create CMS Combine datacards for a simultaneous SR + CR fit.

Each per-SR datacard combines three channels:
  {sr}        — signal region (multi-bin BDT shape)
  {sr}_TTCR   — TTbar control region (1 counting bin)
  {sr}_WCR    — W+jets control region (1 counting bin)

The combined datacard uses all 12 channels (4 SRs x 3).

CROSS-YEAR CORRELATION POLICY (--decorrelate-years, default 'year')
-------------------------------------------------------------------
One datacard per category PER YEAR. When per-year cards are later combined
with combineCards.py, nuisances correlate across years purely by sharing a
name — so by default NOTHING is correlated across years: every nuisance
name is suffixed with the year (sf_btag_hf -> sf_btag_hf_2017) unless it
already carries it (the AK4/AK8 *_<YEAR> JES families) or matches
YEAR_CORRELATED_PATTERNS below — the "unless otherwise specified" hook.
This covers the shape systematics (via renamed templates in the augmented
histogram file, same alias mechanism as the shower decorrelation), the
luminosity lnN (lumi_<year>, value from LUMI_UNC_BY_YEAR) and the norm_*
rateParams (norm_top_resolved_<year>, ...).

CAVEAT for the future cross-year combination step: the rateParam bin field
is a wildcard ('*resolved*'), which in a cross-year combined card would
match EVERY year's resolved bins. The per-year parameter NAMES are already
distinct, but each must bind only its own year's bins — when building the
cross-year card, either give combineCards.py year-carrying bin labels and
tighten the patterns, or edit the rateParam lines accordingly. Do not
combine years without handling this.

Free-floating rateParams constrained by CR yields, decorrelated per topology
by default (--decorrelate-norms topology) and per year by default:
  norm_top_resolved_<year>   rateParam *resolved* TTbar 1
  norm_top_boosted_<year>    rateParam *boosted*  TTbar 1
  norm_wjet_resolved_<year>  rateParam *resolved* WJets 1
  norm_wjet_boosted_<year>   rateParam *boosted*  WJets 1

Pass --decorrelate-norms none for a single shared pair (per year).

The parton shower uncertainties are decorrelated per MC group by default
(--decorrelate-shower group), giving

  sf_partonshower_isr_signal_<year> / _TTbar_<year> / ...

instead of one correlated nuisance each. Renamed templates live in an
augmented copy of the histogram file, {outdir}/histograms_decorrelated.root;
the cards point at that copy whenever ANY renaming is active (shower groups
and/or year tags). --decorrelate-shower none and --decorrelate-years none
together restore the original file.

Process naming in the ROOT file (after merging in make_histograms.py):
  TTbar          <- TTto2L2Nu_... + TTtoLNu2Q_...  (merged)
  WtoLNu-2Jets   <- all WJet sub-processes          (merged)
  DYto2L-2Jets   <- all DY sub-processes            (merged)
  Single-top processes keep their individual names.
  Signal processes keep their individual names.

Output structure:
  {outdir}/combined/datacard.txt
  {outdir}/resolved_mu/datacard.txt
  {outdir}/resolved_e/datacard.txt
  {outdir}/boosted_mu/datacard.txt
  {outdir}/boosted_e/datacard.txt
"""

import argparse
import fnmatch
import os
import shutil
from collections import defaultdict

import numpy as np
import uproot
import yaml


SIGNAL_REGIONS = ['resolved_mu', 'resolved_e', 'boosted_mu', 'boosted_e']
CR_SUFFIXES    = ['ttbar_cr', 'w_cr']

# Individual process config names that are merged into single names in the ROOT file
WJET_CONFIG_NAMES = {
    "WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8",
    "WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8",
    "WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8",
    "WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8",
    "WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8",
    "WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8",
    "WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8",
    "WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8",
}

DY_CONFIG_NAMES = {
    "DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8",
    "DYJetsToLL_M-10to50_TuneCP5_13TeV-amcatnloFXFX-pythia8",
}
# TTbar config names (merged into 'TTbar' in the ROOT file)
TTBAR_CONFIG_NAMES = {
    "TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8",
    "TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8",
}

# Merged names as they appear in the ROOT file
TTBAR_MERGED_NAME = 'TTbar'
WJET_MERGED_NAME  = 'WJets'
DY_MERGED_NAME    = 'DY'

# --------------------------------------------------------------------------
# Normalization rateParams
#
# 'topology' (default) gives the resolved and boosted categories INDEPENDENT
# normalizations; 'none' restores the single shared pair.
#
# Why decorrelate: the CR-only fits to data measure norm_top = 0.93 in the
# resolved topology and 0.66 in the boosted one, far apart compared with their
# uncertainties. A single shared parameter is forced to average that, which
# biases both topologies and understates the normalization uncertainty in each.
# The two topologies use different objects (AK4 pairs vs a single AK8 jet) and
# different selections, so there is no reason to expect one scale factor.
#
# The bin field uses a WILDCARD rather than explicit channel names on purpose:
# combineCards.py prefixes bins with ch1_/ch2_ when per-SR cards are merged
# into a topology card, and '*resolved*' still matches 'ch1_resolved_mu' while
# an explicit 'resolved_mu' would only survive if combineCards rewrote it.
# The pattern also tolerates the '_{year}' suffix added in multi-year mode.
#
# Channel names it relies on: SRs are '<topology>_<lepton>' and CRs are
# '<cr>_<topology>_<lepton>', so every channel carries its topology in its name.
# SEE the cross-year caveat in the module docstring before combining years.
# --------------------------------------------------------------------------
TOPOLOGY_PATTERNS = [
    ('resolved', '*resolved*'),
    ('boosted',  '*boosted*'),
]

NORM_SPECS = [
    ('norm_top',  TTBAR_MERGED_NAME),
    ('norm_wjet', WJET_MERGED_NAME),
]

NORM_RANGE = '[0.2,5]'


def channel_topology(channel):
    """Return 'resolved'/'boosted' for a channel name, or None if unknown."""
    for name, _ in TOPOLOGY_PATTERNS:
        if name in channel:
            return name
    return None

# --------------------------------------------------------------------------
# Cross-year correlation allowlist.
#
# Default policy: NO nuisance is correlated across years. Every nuisance name
# is suffixed _<year> unless it (a) already ends with the year (the
# *_<YEAR>-suffixed JES families arrive per-year from the coffea files), or
# (b) matches one of these fnmatch patterns. Add a pattern here ONLY as a
# deliberate, documented decision that a source is correlated across years.
#
# JES/JER are allowlisted (decision 2026-10-28): the JME Regrouped recipe
# already encodes the cross-year correlation structure in the names
# themselves -- the base components (Absolute, BBEC1, EC2, HF, RelativeBal,
# FlavorQCD) are correlated across years BY DESIGN and must share one name,
# while the *_<year> components decorrelate on their own because their names
# differ per year. Tagging the bases (_yr<year>) was double bookkeeping; the
# recipe's own scheme is used as-is. The patterns also match the *_<year>
# components, which is harmless -- those return unchanged earlier anyway --
# and it keeps the 2016 halves correct: a *_2016 component appears in BOTH
# the PreVFP and PostVFP cards and must stay one nuisance across them.
# --------------------------------------------------------------------------
YEAR_CORRELATED_PATTERNS = [
    'AK4PFchs_JES_*',
    'AK8PFPuppi_JES_*',
    'AK4PFchs_JER*',
    'AK8PFPuppi_JER*',
    # e.g. 'LHEPdfWeight_*',          # PDF: same eigenset across years
    # e.g. 'LHEScaleWeight_*_signal', # ME scales on the same generator setup
]


def year_tagged(name, year, existing=frozenset()):
    """Nuisance name with the year suffix applied per the correlation policy.

    `existing` is the set of ALL systematic base names in the input file.
    It matters because the CMS Regrouped JES scheme ships both a
    year-correlated component and a year-specific one as two INDEPENDENT
    nuisances, e.g.
        AK4PFchs_JES_Regrouped_EC2        (correlated across years by design)
        AK4PFchs_JES_Regrouped_EC2_2017   (2017 only)
    Naively tagging the base one as ..._EC2_2017 collides with the real
    ..._EC2_2017: the datacard then carries two rows with one name, and the
    alias templates overwrite the genuine EC2_2017 templates in the shapes
    file (duplicate ROOT keys -- combine silently reads whichever cycle was
    written last, and crashes with "Bogus norm" when that one happens to be
    a negative-weight cancellation). So a tag that would collide with a
    DIFFERENT existing nuisance uses _yr<year> instead: the base component
    becomes ..._EC2_yr2017, still decorrelated per year per the default
    policy. (Since 2026-10-28 the JES/JER families are allowlisted in
    YEAR_CORRELATED_PATTERNS -- the recipe's own naming carries the
    correlation structure -- so this fallback is dormant for them; it stays
    as a guard for any future family with the same base/base_<year> shape.)
    """
    if year is None:
        return name
    if name.endswith(f'_{year}'):
        return name
    if any(fnmatch.fnmatchcase(name, p) for p in YEAR_CORRELATED_PATTERNS):
        return name
    cand = f'{name}_{year}'
    if cand in existing:
        cand = f'{name}_yr{year}'
    return cand

# --------------------------------------------------------------------------
# Per-MC-group decorrelation of the parton shower uncertainties
#
# ISR and FSR are modelled by the generator's own shower variations. They are
# NOT a common detector effect: the shower uncertainty on a ttbar sample says
# nothing about the shower uncertainty on the EWK signal, and treating them as
# one nuisance lets a well-constrained process (ttbar, huge CR yields) pull the
# shower variation of a process with little statistical weight (the signal).
# One nuisance per MC group removes that.
#
# HOW THE RENAME IS MADE TO WORK
# The nuisance becomes sf_partonshower_isr_TTbar(_<year>), but the shapes line
# resolves $SYSTEMATIC to the NUISANCE name, so combine would look for a
# template called
#     <channel>/TTbar_sf_partonshower_isr_TTbar_2018Up
# which does not exist -- the file has TTbar_sf_partonshower_isrUp. So the cards
# point at an augmented copy of the histogram file (histograms_decorrelated.root,
# written next to the cards) that carries the renamed templates in addition to
# the originals. The same mechanism serves the year tagging: every shape
# nuisance whose final name differs from its template name gets an alias.
# --------------------------------------------------------------------------
DECORRELATE_PER_GROUP = ['sf_partonshower_isr', 'sf_partonshower_fsr', 'LHEScaleWeight_fact_scale','LHEScaleWeight_renorm_scale']

# First match wins; anything unmatched falls into MC_GROUP_DEFAULT.
# Patterns are fnmatch-style against the process name in the ROOT file.
MC_GROUP_RULES = [
    ('signal',    ['*dipoleRecoil*']),   # EWK signal; QCD look-alikes are not
    ('TTbar',     [TTBAR_MERGED_NAME]),
    ('WJets',     [WJET_MERGED_NAME]),
    ('DY',        [DY_MERGED_NAME]),
    ('SingleTop', ['ST_*']),
    ('Nonprompt', ['nonprompt']),
]
MC_GROUP_DEFAULT = 'Diboson'

DECORR_HISTFILE_NAME = 'histograms_decorrelated.root'


def mc_group(process):
    """Which MC group a process belongs to, for shower decorrelation."""
    for name, patterns in MC_GROUP_RULES:
        if any(fnmatch.fnmatchcase(process, p) for p in patterns):
            return name
    return MC_GROUP_DEFAULT


def final_syst_name(syst, process, year, do_shower, do_year,
                    existing=frozenset()):
    """Nuisance name for `syst` on `process` after all decorrelations."""
    name = syst
    if do_shower and syst in DECORRELATE_PER_GROUP:
        name = '%s_%s' % (name, mc_group(process))
    if do_year:
        name = year_tagged(name, year, existing)
    return name


def all_syst_names(hists):
    """Every shape systematic base name anywhere in the file."""
    out = set()
    for channel in hists:
        for proc in hists[channel]:
            for key in hists[channel][proc]:
                if key.endswith('Up') and key not in ('nominal', '_raw'):
                    out.add(key[:-2])
    return frozenset(out)


def collect_alias_templates(hists, year, do_shower, do_year,
                            existing=frozenset()):
    """(source key, alias key) for every template whose nuisance is renamed.

    Covers both decorrelations: shower-per-group and year tagging. E.g.
      ('resolved_mu/TTbar_sf_btag_hfUp',
       'resolved_mu/TTbar_sf_btag_hf_2017Up')
    """
    out = []
    for channel in sorted(hists):
        for proc in sorted(hists[channel]):
            if proc == 'data_obs':
                continue
            seen = set()
            for key in hists[channel][proc]:
                if key in ('nominal', '_raw'):
                    continue
                if key.endswith('Up'):
                    syst = key[:-2]
                elif key.endswith('Down'):
                    syst = key[:-4]
                else:
                    continue
                if syst in seen:
                    continue
                seen.add(syst)
                new = final_syst_name(syst, proc, year, do_shower, do_year,
                                      existing)
                if new == syst:
                    continue
                for direction in ('Up', 'Down'):
                    var = syst + direction
                    if var in hists[channel][proc]:
                        out.append((
                            '%s/%s_%s' % (channel, proc, var),
                            '%s/%s_%s%s' % (channel, proc, new, direction),
                        ))
    return out


def write_decorrelated_histfile(src_path, dst_path, aliases):
    """Rewrite the histogram file with the renamed templates added.

    Every histogram is round-tripped through to_hist(), which carries the full
    per-bin variances (sumw2) into the output -- autoMCStats sees the same bin
    errors as in the original file. A filesystem copy + uproot.update was used
    before, but uproot.update cannot re-open large files (FreeSegmentsData
    AssertionError once the histfile outgrew it), so the file is rebuilt from
    scratch instead: read everything, write everything plus the aliases.
    """
    if not aliases:
        print('  No templates need renaming; using the original file.')
        return src_path

    alias_map = {}
    for old_key, new_key in aliases:
        alias_map.setdefault(old_key, []).append(new_key)

    # A renamed template must never land on a key that already exists (or on
    # another alias): duplicate ROOT keys make combine silently read whichever
    # cycle was written last -- corrupt cards, not an error. Abort instead.
    targets = [t for ts in alias_map.values() for t in ts]
    dup_targets = {t for t in targets if targets.count(t) > 1}
    if dup_targets:
        raise SystemExit(
            '  ERROR: two renamed templates map to the same key, e.g. '
            + sorted(dup_targets)[0]
            + '\n  This is a naming-scheme bug in make_datacards.py.')

    tmp_path = dst_path + '.tmp'
    written = 0
    seen = set()
    with uproot.open(src_path) as fin, uproot.recreate(tmp_path) as fout:
        original_keys = {k for k, c in fin.classnames(cycle=False).items()
                         if not c.startswith('TDirectory')}
        clashes = original_keys & set(targets)
        if clashes:
            raise SystemExit(
                '  ERROR: renamed template(s) would overwrite existing '
                'histograms, e.g.\n    ' + sorted(clashes)[0]
                + '\n  (two distinct nuisances collapsing to one name, like '
                'the Regrouped JES\n  base vs year-specific components). This '
                'is a naming-scheme bug in\n  make_datacards.py -- fix '
                'year_tagged(), do not ship these cards.')
        for key, classname in fin.classnames(cycle=False).items():
            if classname.startswith('TDirectory'):
                continue
            h = fin[key].to_hist()
            fout[key] = h
            if key in alias_map:
                seen.add(key)
                for new_key in alias_map[key]:
                    fout[new_key] = h
                    written += 1

    missing = sorted(set(alias_map) - seen)
    if missing:
        os.remove(tmp_path)
        print(f'  ERROR: {len(missing)} alias source template(s) not found in '
              f'{src_path}, e.g.')
        for key in missing[:3]:
            print(f'    {key}')
        raise SystemExit(
            '  Refusing to continue: the datacard would name nuisances whose\n'
            '  templates are missing, and combine would fail to build the\n'
            '  workspace. Re-run with --decorrelate-shower none and\n'
            '  --decorrelate-years none.')

    os.replace(tmp_path, dst_path)
    print(f'  Wrote {dst_path}')
    print(f'    {written} renamed template(s) added alongside the originals')
    return dst_path


# Per-year luminosity lnN. The nuisance is named lumi_<year>, so the lumi
# uncertainties of different years are independent in a cross-year
# combination (consistent with the everything-decorrelated default; add a
# correlated-component scheme deliberately if ever needed).
# 1.073 for every year per the analyst's instruction (2026-10; replaces the
# earlier 2018-only 1.0084 -- single-year 2018 cards change too).
LUMI_UNC_BY_YEAR = {
    '2018':          1.073,
    '2017':          1.073,
    '2016_PreVFP':   1.073,
    '2016_PostVFP':  1.073,
}
LUMI_UNC_FALLBACK = 1.073

MIN_YIELD = 1e-3


# ---------------------------------------------------------------------------
# ROOT file reader
# ---------------------------------------------------------------------------

def read_histograms(root_file_path):
    """Read all histograms from ROOT file.

    Returns hists[channel][process][variation] = (values, edges)
    """
    hists = defaultdict(lambda: defaultdict(dict))

    with uproot.open(root_file_path) as rf:
        for key in rf.keys(cycle=False):
            parts = key.split('/')
            if len(parts) != 2:
                continue
            channel, name = parts
            h     = rf[key]
            vals  = h.values()
            edges = h.axes[0].edges

            if name == 'data_obs':
                hists[channel]['data_obs']['nominal'] = (vals, edges)
            elif name.endswith('Up') or name.endswith('Down'):
                hists[channel][name]['_raw'] = (vals, edges)
            else:
                hists[channel][name]['nominal'] = (vals, edges)

    # Attach Up/Down variations to their owning process
    for channel in list(hists.keys()):
        proc_names = {p for p in hists[channel] if 'nominal' in hists[channel][p]}
        raw_keys   = {k for k in hists[channel] if '_raw' in hists[channel][k]}
        for raw_key in raw_keys:
            vals, edges = hists[channel][raw_key]['_raw']
            match = None
            for proc in sorted(proc_names, key=len, reverse=True):
                if raw_key.startswith(proc + '_'):
                    match = proc
                    break
            if match is None:
                hists[channel][raw_key]['nominal'] = (vals, edges)
            else:
                hists[channel][match][raw_key[len(match) + 1:]] = (vals, edges)
            del hists[channel][raw_key]

    return hists


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def hist_integral(vals):
    return float(np.sum(np.maximum(vals, 0.0)))


# Minimum ratio of a variation's signed integral to the nominal's for the
# template to be usable. Any smaller and the integral is negative-weight
# cancellation noise, not a physical shape: combine would either reject the
# card ("Bogus norm" at <= 0) or build an absurd normalization kappa from it.
KAPPA_MIN = 1e-4


def hist_raw_integral(vals):
    """Signed sum over bins -- the integral combine itself uses.

    Differs from hist_integral for templates with negative bins (negative
    generator weights, e.g. amcatnlo samples like ZZZ): combine computes the
    Up/Down normalization kappas from the raw TH1::Integral(), signed and
    unclipped, and refuses the whole card when one is <= 0
    ("RuntimeError: Bogus norm").

    The sum is forced to float64: TH1F templates come out of uproot as
    float32 arrays, and np.sum would then accumulate in float32, where a
    cancellation that is exactly zero in ROOT's double-precision Integral()
    can leave an O(1e-8) rounding residue and sneak past the check.
    """
    return float(np.sum(np.asarray(vals, dtype=np.float64)))


def collect_shape_systs(hists, channels, all_procs):
    """Collect all shape systematic base names (anything with an Up variation)."""
    systs = set()
    for ch in channels:
        for proc in all_procs:
            if proc not in hists[ch]:
                continue
            for var in hists[ch][proc]:
                if var not in ('nominal', '_raw') and var.endswith('Up'):
                    systs.add(var[:-2])
    return sorted(systs)


# ---------------------------------------------------------------------------
# Datacard writer
# ---------------------------------------------------------------------------

def write_datacard(signal_procs, channels, hists, all_bkg_procs,
                   root_file_rel_path, out_path, decorrelate='topology',
                   decorrelate_shower=True, year=None, decorrelate_years=True,
                   lumi_unc=LUMI_UNC_FALLBACK, existing_systs=frozenset()):
    """Write one combined Combine datacard.

    signal_procs : ordered list of all signal process names.
                   Index 0 -> first signal, -1 -> second, -2 -> third, ...
                   All scale with the same POI (r/mu).
    channels     : list of channel names to include in this datacard.
                   For a per-SR card: [sr, sr_TTCR, sr_WCR].
                   For the combined card: all 12 channels.
    year / decorrelate_years : when set, every nuisance name is suffixed
                   with the year per YEAR_CORRELATED_PATTERNS, so per-year
                   cards share no nuisances across years once combined.
    """
    valid_channels = [ch for ch in channels if 'data_obs' in hists.get(ch, {})]
    if not valid_channels:
        print(f'  No valid channels, skipping datacard: {out_path}')
        return

    do_year = decorrelate_years and year is not None

    def tagged(name):
        return year_tagged(name, year, existing_systs) if do_year else name

    # Build per-channel process lists and rates. A process also needs a
    # positive SIGNED nominal integral: combine computes its norm from the
    # raw TH1::Integral(), and a negative-weight cancellation there aborts
    # text2workspace with "Null norm" just like a bogus variation does.
    ch_procs = {}   # channel -> [(proc_name, is_signal)]
    ch_rates = {}   # channel -> {proc_name: rate}
    dropped_procs = []

    for ch in valid_channels:
        procs_here = []
        rates_here = {}

        for sig_proc in signal_procs:
            if (sig_proc in hists[ch]
                    and 'nominal' in hists[ch][sig_proc]):
                vals = hists[ch][sig_proc]['nominal'][0]
                y = hist_integral(vals)
                if y >= MIN_YIELD:
                    if hist_raw_integral(vals) <= 0:
                        dropped_procs.append((ch, sig_proc))
                        continue
                    procs_here.append((sig_proc, True))
                    rates_here[sig_proc] = y

        for bkg in all_bkg_procs:
            if bkg not in hists[ch] or 'nominal' not in hists[ch][bkg]:
                continue
            vals = hists[ch][bkg]['nominal'][0]
            y = hist_integral(vals)
            if y >= MIN_YIELD:
                if hist_raw_integral(vals) <= 0:
                    dropped_procs.append((ch, bkg))
                    continue
                procs_here.append((bkg, False))
                rates_here[bkg] = y

        ch_procs[ch] = procs_here
        ch_rates[ch] = rates_here

    all_bkg_present = {p for ch in ch_procs
                       for p, is_sig in ch_procs[ch] if not is_sig}
    all_sig_present = {p for ch in ch_procs
                       for p, is_sig in ch_procs[ch] if is_sig}

    # Collect shape systematics across all channels and processes
    all_procs_for_systs = list(all_bkg_present) + list(all_sig_present)
    shape_systs = collect_shape_systs(hists, valid_channels, all_procs_for_systs)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Process-index map: signals get 0, -1, -2, ...; backgrounds get 1, 2, ...
    sig_idx  = {p: -i for i, p in enumerate(signal_procs)}  # first signal = 0
    proc_idx = dict(sig_idx)
    bkg_ctr  = 1
    for p in all_bkg_procs:
        if p in all_bkg_present:
            proc_idx[p] = bkg_ctr
            bkg_ctr += 1

    # ---- Work out which normalization rateParams this card needs ---------
    # Only emit a line when the topology AND the process are actually present:
    # a rateParam naming a bin or process that does not exist in the card makes
    # combine fail, and a per-SR card only ever contains one topology.
    rate_param_lines, rate_param_notes = [], []

    unknown_ch = [ch for ch in valid_channels if channel_topology(ch) is None]
    if unknown_ch and decorrelate == 'topology':
        raise SystemExit(
            'ERROR: these channels contain neither "resolved" nor "boosted", '
            'so their topology cannot be determined:\n    '
            + '\n    '.join(unknown_ch)
            + '\n  Rename them, extend TOPOLOGY_PATTERNS, or pass '
              '--decorrelate-norms none.')

    if decorrelate == 'topology':
        scopes = [(name, pattern) for name, pattern in TOPOLOGY_PATTERNS
                  if any(channel_topology(ch) == name for ch in valid_channels)]
    else:
        scopes = [('', '*')]

    for base, proc in NORM_SPECS:
        for topo, pattern in scopes:
            covered = [ch for ch in valid_channels
                       if (topo == '' or channel_topology(ch) == topo)
                       and proc in ch_rates.get(ch, {})]
            if not covered:
                rate_param_notes.append(
                    f'    [note] no {base} for "{topo or "all"}": '
                    f'{proc} is absent from those channels')
                continue
            pname = f'{base}_{topo}' if topo else base
            if do_year:
                pname = f'{pname}_{year}'
            rate_param_lines.append(
                f'{pname:<20} rateParam  {pattern:<12} {proc:<14} 1  {NORM_RANGE}')
            rate_param_notes.append(
                f'    {pname:<20} -> {proc} in ' + ', '.join(covered))

    shower_rows = []

    col = lambda s: f'{s:<42}'

    with open(out_path, 'w') as f:
        n_ch   = len(valid_channels)
        n_jmax = len(all_bkg_present) + len(all_sig_present) - 1

        f.write('# VBS simultaneous SR+CR fit datacard\n')
        f.write('# TTbar and W+jets normalization constrained by CR yields via rateParams\n')
        if do_year:
            f.write(f'# Year {year}: all nuisances are year-tagged -- nothing '
                    f'correlates across years\n'
                    f'# unless listed in YEAR_CORRELATED_PATTERNS '
                    f'(make_datacards.py)\n')
        f.write(f'# Signal processes ({len(signal_procs)}):\n')
        for i, sp in enumerate(signal_procs):
            f.write(f'#   index {-i:2d}  {sp}\n')
        f.write(f'imax {n_ch}\n')
        f.write(f'jmax {n_jmax}\n')
        f.write('kmax *\n')
        f.write('-' * 80 + '\n')

        f.write(f'shapes * * {root_file_rel_path} $CHANNEL/$PROCESS '
                f'$CHANNEL/$PROCESS_$SYSTEMATIC\n')
        f.write(f'shapes data_obs * {root_file_rel_path} $CHANNEL/data_obs\n')
        f.write('-' * 80 + '\n')

        # Observation
        f.write('bin         ' + '  '.join(col(ch) for ch in valid_channels) + '\n')
        obs_row = []
        for ch in valid_channels:
            obs = (hist_integral(hists[ch]['data_obs']['nominal'][0])
                   if 'data_obs' in hists[ch] and 'nominal' in hists[ch]['data_obs']
                   else 0.0)
            obs_row.append(col(f'{obs:.4f}'))
        f.write('observation ' + '  '.join(obs_row) + '\n')
        f.write('-' * 80 + '\n')

        # Flat list of (channel, process, is_signal, rate) for all table columns
        entries = [
            (ch, proc, is_sig, ch_rates[ch][proc])
            for ch in valid_channels
            for proc, is_sig in ch_procs[ch]
        ]

        f.write('bin       ' + '  '.join(col(ch)           for ch, _, _, _  in entries) + '\n')
        f.write('process   ' + '  '.join(col(p)            for _,  p, _, _  in entries) + '\n')
        f.write('process   ' + '  '.join(col(proc_idx[p])  for _,  p, _, _  in entries) + '\n')
        f.write('rate      ' + '  '.join(col(f'{r:.6f}')   for _,  _, _, r  in entries) + '\n')
        f.write('-' * 80 + '\n')

        # ---- Luminosity lnN (year-tagged, per-year value) ----
        lumi_name = f'lumi_{year}' if year is not None else 'lumi_2018'
        f.write(f'{lumi_name}  lnN  '
                + '  '.join(col(f'{lumi_unc:.3f}') for _ in entries) + '\n')

        # ---- Shape systematics ----
        dropped_variations = set()
        emitted_rows = set()

        def guard_unique(name):
            # Two rows with one nuisance name silently merge in combine's
            # parser -- always a naming-scheme bug here, never acceptable.
            if name in emitted_rows:
                raise SystemExit(
                    f'  ERROR: duplicate nuisance name {name!r} in {out_path} '
                    f'-- two systematics collapsed to one name (naming-scheme '
                    f'bug in make_datacards.py).')
            emitted_rows.add(name)

        for syst in shape_systs:

            def usable(ch, proc, _syst=syst):
                # Mirror combine's acceptance: the SIGNED Up/Down integrals
                # must be positive, or text2workspace aborts with
                # "Bogus norm" on the whole card. Require them to be at least
                # KAPPA_MIN of the nominal as well -- below that the integral
                # is cancellation noise, not a shape.
                ph = hists[ch].get(proc, {})
                up, dn = f'{_syst}Up', f'{_syst}Down'
                if up not in ph or dn not in ph:
                    return False
                nom_raw = (hist_raw_integral(ph['nominal'][0])
                           if 'nominal' in ph else 0.0)
                if nom_raw <= 0:
                    return False
                if (hist_raw_integral(ph[up][0]) > KAPPA_MIN * nom_raw
                        and hist_raw_integral(ph[dn][0]) > KAPPA_MIN * nom_raw):
                    return True
                dropped_variations.add((ch, proc, _syst))
                return False

            if decorrelate_shower and syst in DECORRELATE_PER_GROUP:
                # One nuisance per MC group. Each column carries '1' only in
                # its own group's row, so the groups are independent and every
                # process is still covered exactly once.
                groups = []
                for ch, proc, _, _ in entries:
                    if usable(ch, proc):
                        g = mc_group(proc)
                        if g not in groups:
                            groups.append(g)
                for g in groups:
                    row = [col('1') if (usable(ch, proc) and mc_group(proc) == g)
                           else col('-')
                           for ch, proc, _, _ in entries]
                    name = tagged(f'{syst}_{g}')
                    guard_unique(name)
                    f.write(f'{name}  shape  ' + '  '.join(row) + '\n')
                    shower_rows.append(name)
            else:
                row = []
                has_any = False
                for ch, proc, _, _ in entries:
                    if usable(ch, proc):
                        row.append(col('1'))
                        has_any = True
                    else:
                        row.append(col('-'))
                if has_any:
                    name = tagged(syst)
                    guard_unique(name)
                    f.write(f'{name}  shape  ' + '  '.join(row) + '\n')

        f.write('* autoMCStats 10\n')
        f.write('\n')

        # ---- Free-floating rateParams for TTbar and W+jets normalization ----
        # See the TOPOLOGY_PATTERNS comment at the top for why these are split
        # per topology by default, and the module docstring for the cross-year
        # wildcard caveat.
        for line in rate_param_lines:
            f.write(line + '\n')

    print(f'  Written: {out_path}')
    if dropped_procs:
        print(f'    [note] {len(dropped_procs)} process(es) dropped from '
              f'channels (signed nominal integral <= 0 despite positive '
              f'clipped yield -- would be a combine "Null norm"):')
        for ch, proc in dropped_procs[:10]:
            print(f'      {ch} / {proc}')
    if dropped_variations:
        print(f'    [note] {len(dropped_variations)} shape variation(s) '
              f'dropped (signed Up/Down integral < {KAPPA_MIN:g} x nominal: '
              f'negative-weight cancellation in low-stat bins):')
        for ch, proc, s in sorted(dropped_variations)[:10]:
            print(f'      {ch} / {proc} / {s}')
        if len(dropped_variations) > 10:
            print(f'      ... and {len(dropped_variations) - 10} more')
    for note in rate_param_notes:
        print(note)
    if shower_rows:
        print(f'    shower nuisances decorrelated into {len(shower_rows)}: '
              + ', '.join(shower_rows))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Create SR+CR datacards for simultaneous fit with rateParams')
    parser.add_argument('--histfile', required=True,
                        help='ROOT file from make_histograms.py')
    parser.add_argument('--config',   required=True,
                        help='YAML inputs config (inputs_v28_<year>.yaml)')
    parser.add_argument('--outdir',   default='cards')
    parser.add_argument('--year',     default=None)
    parser.add_argument('--decorrelate-shower', choices=['group', 'none'],
                        default='group',
                        help='"group" (default) gives each MC group its own '
                             'sf_partonshower_isr/_fsr nuisance, writing an '
                             'augmented histogram file next to the cards; '
                             '"none" keeps one correlated nuisance each.')
    parser.add_argument('--decorrelate-norms', choices=['topology', 'none'],
                        default='topology',
                        help='"topology" (default) gives resolved and boosted '
                             'independent norm_top/norm_wjet; "none" restores '
                             'a single shared pair across all categories.')
    parser.add_argument('--require-lumi', action='store_true',
                        help='fail instead of falling back to the 2018 '
                             'luminosity uncertainty when the year has no '
                             'entry in LUMI_UNC_BY_YEAR (used by the '
                             'cross-year combination pipeline).')
    parser.add_argument('--decorrelate-years', choices=['year', 'none'],
                        default='year',
                        help='"year" (default) suffixes EVERY nuisance (shape, '
                             'lumi, rateParams) with the year so nothing '
                             'correlates across years when per-year cards are '
                             'combined -- exceptions go into '
                             'YEAR_CORRELATED_PATTERNS. "none" keeps the bare '
                             'names (pre-v28 single-year behavior).')
    args = parser.parse_args()

    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    years = [args.year] if args.year else list(config['years'].keys())
    multi_year = len(years) > 1

    year0 = years[0]
    do_year = args.decorrelate_years == 'year'
    lumi_unc = LUMI_UNC_BY_YEAR.get(year0)
    if lumi_unc is None:
        if args.require_lumi:
            raise SystemExit(
                f'ERROR: no luminosity uncertainty set for {year0!r} in '
                f'LUMI_UNC_BY_YEAR (make_datacards.py) and --require-lumi '
                f'was passed. Fill in the per-era value before building '
                f'cross-year cards -- a placeholder here would silently '
                f'misconstrain lumi_{year0} in every combined fit.')
        lumi_unc = LUMI_UNC_FALLBACK
        print(f'  [WARN] no luminosity uncertainty set for {year0!r}; using '
              f'{lumi_unc} (the 2018 value). Set LUMI_UNC_BY_YEAR in '
              f'make_datacards.py before quoting {year0} results.')

    # Build ordered lists of signal and background process names.
    # Individual config names are mapped to the merged names used in the ROOT file.
    all_signal_procs  = []
    seen_sig          = set()
    all_bkg_procs_ord = []
    seen_bkg          = set()

    for year in years:
        yc = config['years'][year]
        for p in yc.get('signal', {}):
            if p not in seen_sig:
                all_signal_procs.append(p)
                seen_sig.add(p)
        for p in yc.get('background', {}):
            # Map individual config names to merged names
            if p in TTBAR_CONFIG_NAMES:
                p = TTBAR_MERGED_NAME
            elif p in WJET_CONFIG_NAMES:
                p = WJET_MERGED_NAME
            elif p in DY_CONFIG_NAMES:
                p = DY_MERGED_NAME
            if p not in seen_bkg:
                all_bkg_procs_ord.append(p)
                seen_bkg.add(p)

    print(f'Reading histograms from {args.histfile} ...')
    hists = read_histograms(args.histfile)
    print(f'  Found channels: {sorted(hists.keys())}')
    print(f'  Signal processes ({len(all_signal_procs)}):')
    for i, p in enumerate(all_signal_procs):
        print(f'    index {-i:2d}  {p}')
    print(f'  Background processes ({len(all_bkg_procs_ord)}):')
    for p in all_bkg_procs_ord:
        print(f'    {p}')

    histfile_abs = os.path.abspath(args.histfile)

    # Any renaming (shower groups and/or year tags) needs the renamed
    # templates in the shapes file, so build the augmented copy before any
    # card points at it.
    do_shower = args.decorrelate_shower == 'group'
    if do_shower:
        print('\nDecorrelating parton shower uncertainties per MC group ...')
        groups_seen = {}
        for ch in hists:
            for proc in hists[ch]:
                if proc != 'data_obs':
                    groups_seen.setdefault(mc_group(proc), set()).add(proc)
        for g in sorted(groups_seen):
            print(f'    {g:<10} {len(groups_seen[g])} process(es): '
                  + ', '.join(sorted(groups_seen[g])[:3])
                  + (' ...' if len(groups_seen[g]) > 3 else ''))
    if do_year:
        print(f'\nYear-tagging every nuisance with _{year0} '
              f'({len(YEAR_CORRELATED_PATTERNS)} cross-year-correlated '
              f'exception pattern(s) configured).')

    existing_systs = all_syst_names(hists)
    aliases = collect_alias_templates(hists, year0 if do_year else None,
                                      do_shower, do_year, existing_systs)
    if aliases:
        os.makedirs(args.outdir, exist_ok=True)
        histfile_abs = os.path.abspath(write_decorrelated_histfile(
            histfile_abs,
            os.path.join(args.outdir, DECORR_HISTFILE_NAME),
            aliases))

    # Resolve channel names in the ROOT file (may have year suffix in multi-year mode)
    def ch_name(region):
        return f'{region}_{years[0]}' if multi_year else region

    # ------------------------------------------------------------------ #
    # Combined datacard: all 12 channels                                  #
    # ------------------------------------------------------------------ #
    all_channels = []
    for sr in SIGNAL_REGIONS:
        all_channels.append(ch_name(sr))
        for cr in CR_SUFFIXES:
            all_channels.append(ch_name(f'{cr}_{sr}'))

    out_path = os.path.join(args.outdir, 'combined', 'datacard.txt')
    root_rel = os.path.relpath(histfile_abs, os.path.dirname(out_path))
    print(f'\nCreating combined datacard ({len(all_channels)} channels) ...')
    write_datacard(
        signal_procs       = all_signal_procs,
        channels           = all_channels,
        hists              = hists,
        all_bkg_procs      = all_bkg_procs_ord,
        root_file_rel_path = root_rel,
        out_path           = out_path,
        decorrelate        = args.decorrelate_norms,
        decorrelate_shower = do_shower,
        year               = year0,
        decorrelate_years  = do_year,
        lumi_unc           = lumi_unc,
        existing_systs     = existing_systs,
    )

    # ------------------------------------------------------------------ #
    # Per-SR datacards: SR + TTCR + WCR (3 channels each)                #
    # ------------------------------------------------------------------ #
    for sr in SIGNAL_REGIONS:
        sr_channels = [
            ch_name(sr),
            ch_name(f'ttbar_cr_{sr}'),
            ch_name(f'w_cr_{sr}'),
        ]
        out_path = os.path.join(args.outdir, sr, 'datacard.txt')
        root_rel = os.path.relpath(histfile_abs, os.path.dirname(out_path))
        print(f'\nCreating {sr} datacard (3 channels: SR + TTCR + WCR) ...')
        write_datacard(
            signal_procs       = all_signal_procs,
            channels           = sr_channels,
            hists              = hists,
            all_bkg_procs      = all_bkg_procs_ord,
            root_file_rel_path = root_rel,
            out_path           = out_path,
            decorrelate        = args.decorrelate_norms,
            decorrelate_shower = do_shower,
            year               = year0,
            decorrelate_years  = do_year,
            lumi_unc           = lumi_unc,
            existing_systs     = existing_systs,
        )

    print('\nDone.')


if __name__ == '__main__':
    main()
