"""
Extract kinematic variable histograms from coffea files for all 4 SRs.
Uses pre-computed scale factors from JSON (from make_histograms_SRonly.py).

v28 INPUTS (multi-sample output_all.coffea files)
  Shares make_histograms.py's v28 machinery (imported below):
    - load_coffea_cached(): the v28 group files are multi-GB and shared by
      many processes; without the cache each would be deserialized once per
      (process, region). V28_CACHE_SIZE bounds memory (default 4).
    - the dataset-year guard: the inner level under each sample holds
      sample_year[_Era] dataset keys; eras are summed, other years are
      excluded when year= is given (v28 files are single-year, so this is
      belt-and-suspenders).
    - ELECTRON/MUON_DATA_PROCESSES accept both the legacy names (EGamma,
      SingleMuon) and the v28 ones (DATA_EGamma, DATA_SingleElectron,
      DATA_SingleMuon; 2016/2017 use SingleElectron where 2018 has EGamma).
  Selection is by sample key (variables[var][sample]), so processes sharing
  one output_all.coffea never absorb their groupmates.

Only keeps the systematics needed for the 6-panel systematic ratio plots:
  sf_qgtagging_fsr/isr/stat, sf_qgtagging_jes/jer,
  AK4PFPuppi_JES_Total, AK4PFPuppi_JER,
  ele_scale, ele_smear, muon_scale, muon_smear,
  AK8PFPuppi_JES_Total, AK8PFPuppi_JER

Output ROOT file structure:
  {channel}__{var_name}/{process}              <- nominal
  {channel}__{var_name}/{process}_{syst}Up     <- up variation
  {channel}__{var_name}/{process}_{syst}Down   <- down variation
  {channel}__{var_name}/data_obs               <- real data

Run inside apptainer environment.
Usage:
  python3 python/make_histograms_kinematics.py \
      --config config/inputs_2022postEE.yaml \
      --sffile histograms/scale_factors.json \
      --output histograms/histograms_kinematics.root
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import yaml
import coffea.util
import boost_histogram as bh
import uproot

# Single implementation of the v28 input handling (file cache, dataset-year
# guard, data sample namings) lives in make_histograms.py -- same directory,
# so this import works however the pipeline invokes the script.
from make_histograms import (
    load_coffea_cached, _dataset_matches_year,
    ELECTRON_DATA_PROCESSES, MUON_DATA_PROCESSES,
)


SIGNAL_REGIONS = ['resolved_mu', 'resolved_e', 'boosted_mu', 'boosted_e']

TOP_PROCESSES = {
    'TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8',
    'TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8',
    'ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8',
    'ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8',
    'ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8',
    'ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8',
    'ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8',
}
WJET_PROCESSES = {
    'WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8',
    'WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8',
    'WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8',
    'WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8',
    'WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8',
    'WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8',
    'WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8',
    'WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8',
}
WJET_MERGED_NAME = 'WJetsToLNu'

DY_PROCESSES = {
    'DYJetsToLL_M-10to50_TuneCP5_13TeV-amcatnloFXFX-pythia8',
    'DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8',
}
DY_MERGED_NAME = 'DYJetsToLL'

# QvG shape systematics to symmetrize (same logic as make_histograms_SRonly.py) ## ?? idk
# SYMMETRIZE_SYSTS = {'sf_qvg_fsr', 'sf_qvg_isr', 'sf_qvg_jes', 'sf_qvg_jer', 'sf_qvg_pu'}
SYMMETRIZE_SYSTS = { 'AK4PFchs_JER', 'AK8PFPuppi_JER', 'sf_qgtagging_sf_qgtagging_herwig','sf_qgtagging_sf_qgtagging_total','sf_qgtagging_sf_qgtagging_fsr'}

# Only keep these systematics (+ nominal)
SYSTS_FOR_PANELS = {
    'AK4PFchs_JES_Regrouped_FlavorQCD',
    'AK4PFchs_JES_Regrouped_RelativeBal',
    'AK4PFchs_JES_Regrouped_HF',
    'AK4PFchs_JES_Regrouped_BBEC1',
    'AK4PFchs_JES_Regrouped_EC2',
    'AK4PFchs_JES_Regrouped_Absolute',
    'AK4PFchs_JES_Regrouped_Absolute_2018',
    'AK4PFchs_JES_Regrouped_HF_2018',
    'AK4PFchs_JES_Regrouped_EC2_2018',
    'AK4PFchs_JES_Regrouped_RelativeSample_2018',
    'AK4PFchs_JES_Regrouped_BBEC1_2018',
    'AK4PFchs_JER',
    'electron_scale_and_smearing_legacy',
    'muon_roccor',
    'AK8PFPuppi_JES_Regrouped_FlavorQCD',
    'AK8PFPuppi_JES_Regrouped_RelativeBal',
    'AK8PFPuppi_JES_Regrouped_HF',
    'AK8PFPuppi_JES_Regrouped_BBEC1',
    'AK8PFPuppi_JES_Regrouped_EC2',
    'AK8PFPuppi_JES_Regrouped_Absolute',
    'AK8PFPuppi_JES_Regrouped_Absolute_2018',
    'AK8PFPuppi_JES_Regrouped_HF_2018',
    'AK8PFPuppi_JES_Regrouped_EC2_2018',
    'AK8PFPuppi_JES_Regrouped_RelativeSample_2018',
    'AK8PFPuppi_JES_Regrouped_BBEC1_2018',
    'AK8PFPuppi_JER',
    "PileupWeight", "sf_mu_id","sf_mu_iso","sf_ele_id","sf_ele_reco","sf_mu_trigger","sf_ele_trigger","sf_L1prefiring","sf_jet_puId","sf_partonshower_isr", "sf_partonshower_fsr",
    "muon_inverttight_to_fake","electron_inverttight_to_fake", "muon_inverttight_to_fake_boosted", "muon_inverttight_to_fake_boosted", "sf_btag", "sf_qgtagging_sf_qgtagging_herwig","sf_qgtagging_sf_qgtagging_stat","sf_qgtagging_sf_qgtagging_isr","sf_qgtagging_sf_qgtagging_fsr", "LHEScaleWeight_renorm_scale", "LHEScaleWeight_fact_scale"
}

KINEMATIC_VARS = [
    'pt_w_lep', 'eta_w_lep',
    'mjj_vbs', 'mjj_vbs_boost',
    'DeepMETResolutionTune_pt',
    'mt_w_leptonic',
    'z_lep', 'z_fat',
    'centrality_resolved', 'centrality_boosted',
    'lead_wlep_MET_dphi', 'nJets', 'nFatJetCandidate'
]


def is_electron_region(region):
    return '_e' in region


def load_file_all_vars(fpath, var_names, process_name, regions, year=None):
    """Load one coffea file and extract all (var, region) combinations at once.

    Returns: result[var_name][region][variation] = [vals, vars, edges]

    v28 notes: the file may hold MANY samples -- selection is by the
    process_name sample key only. The inner loop runs over the sample's
    dataset keys (sample_year[_Era]): data eras are summed, and datasets of
    another year are excluded when year= is given.
    """
    result = defaultdict(lambda: defaultdict(dict))
    try:
        f = load_coffea_cached(fpath)
    except Exception as e:
        print(f'  [warn] load failed {fpath}: {e}')
        return result

    variables = f.get('variables', {})

    for var_name in var_names:
        if var_name not in variables:
            continue
        var_data = variables[var_name]
        if process_name not in var_data:
            continue
        for ds_name, h in var_data[process_name].items():
            if year is not None and not _dataset_matches_year(
                    ds_name, process_name, year):
                continue
            available_cats = list(h.axes['cat'])
            for region in regions:
                if region not in available_cats:
                    continue
                for var in list(h.axes['variation']):
                    try:
                        sl = h[{'cat': region, 'variation': var}]
                        v = sl.values().copy()
                        e = (sl.variances() if sl.variances() is not None
                             else np.zeros_like(v)).copy()
                        edges = sl.axes[0].edges.copy()
                    except Exception:
                        continue
                    if var not in result[var_name][region]:
                        result[var_name][region][var] = [v, e, edges]
                    else:
                        result[var_name][region][var][0] += v
                        result[var_name][region][var][1] += e
    return result


def filter_systs(raw_vars):
    """Keep only nominal + SYSTS_FOR_PANELS variations."""
    out = {}
    for var_key, (vals, errs, edges) in raw_vars.items():
        if var_key == 'nominal':
            out['nominal'] = (vals.copy(), errs.copy(), edges.copy())
            continue
        # Derive base syst name
        if var_key.endswith('Up'):
            base = var_key[:-2]
        elif var_key.endswith('Down'):
            base = var_key[:-4]
        else:
            continue
        if base in SYSTS_FOR_PANELS:
            out[var_key] = (vals.copy(), errs.copy(), edges.copy())
    return out


def merge_processes(ch_var_hists, source_procs, merged_name):
    """Sum histograms from source_procs into merged_name across all vars."""
    for var_name in list(ch_var_hists.keys()):
        present = [p for p in source_procs if p in ch_var_hists[var_name]]
        if not present:
            continue
        all_keys = set()
        for p in present:
            all_keys.update(ch_var_hists[var_name][p].keys())
        merged = {}
        for var_key in all_keys:
            vals = variances = edges = None
            for p in present:
                ph = ch_var_hists[var_name][p]
                v, e, ed = ph.get(var_key, ph['nominal'])
                if vals is None:
                    vals, variances, edges = v.copy(), e.copy(), ed.copy()
                else:
                    vals += v
                    variances += e
            merged[var_key] = (vals, variances, edges)
        ch_var_hists[var_name][merged_name] = merged
        for p in present:
            del ch_var_hists[var_name][p]


def apply_scale(hists_dict, sf):
    return {k: (v * sf, e * sf * sf, ed) for k, (v, e, ed) in hists_dict.items()}


def symmetrize_syst(hists_dict, syst):
    """Replace up/down variations with symmetric envelope around nominal.

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


def make_bh(values, variances, edges):
    h = bh.Histogram(bh.axis.Variable(edges), storage=bh.storage.Weight())
    h.view().value    = np.maximum(values,    0.0)
    h.view().variance = np.maximum(variances, 0.0)
    return h


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',  required=True)
    parser.add_argument('--sffile',  default=None,
                        help='JSON file with SF_top/SF_wjet. '
                             'If omitted, both scale factors default to 1.0 '
                             '(appropriate for the simultaneous SR+CR fit where '
                             'normalization is free-floating via rateParams).')
    parser.add_argument('--output',  default='histograms/histograms_kinematics.root')
    parser.add_argument('--year',    default=None)
    args = parser.parse_args()

    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    if args.sffile is not None:
        with open(args.sffile) as fh:
            sf_info = json.load(fh)
        sf_top  = sf_info['SF_top']['value']
        sf_wjet = sf_info['SF_wjet']['value']
        print(f'Scale factors: SF_top={sf_top:.4f}, SF_wjet={sf_wjet:.4f}')
    else:
        sf_top  = 1.0
        sf_wjet = 1.0
        print('No --sffile provided; using SF_top=1.0, SF_wjet=1.0')

    years = [args.year] if args.year else list(config['years'].keys())

    # all_hists[channel][var_name][process][variation] = (vals, vars, edges)
    all_hists = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for year in years:
        year_cfg = config['years'][year]

        for region in SIGNAL_REGIONS:
            channel  = f'{region}_{year}' if len(years) > 1 else region
            electron = is_electron_region(region)
            bdt_var  = f'bdt_{region}'
            all_vars = [bdt_var] + KINEMATIC_VARS

            # --- backgrounds ---
            for proc, pcfg in year_cfg.get('background', {}).items():
                flist = pcfg.get('files', [pcfg.get('file')])
                print(f'  Loading bkg {proc} / {region}')
                for fpath in flist:
                    if not fpath or not os.path.exists(fpath):
                        continue
                    raw = load_file_all_vars(fpath, all_vars, proc, [region],
                                             year=year)
                    for var_name, reg_data in raw.items():
                        if region not in reg_data:
                            continue
                        filtered = filter_systs(reg_data[region])
                        if 'nominal' not in filtered:
                            continue
                        if proc not in all_hists[channel][var_name]:
                            all_hists[channel][var_name][proc] = filtered
                        else:
                            for vk, (v, e, ed) in filtered.items():
                                if vk in all_hists[channel][var_name][proc]:
                                    pv, pe, ped = all_hists[channel][var_name][proc][vk]
                                    all_hists[channel][var_name][proc][vk] = (pv+v, pe+e, ped)
                                else:
                                    all_hists[channel][var_name][proc][vk] = (v, e, ed)

            # --- signals ---
            for proc, pcfg in year_cfg.get('signal', {}).items():
                flist = pcfg.get('files', [pcfg.get('file')])
                print(f'  Loading sig {proc} / {region}')
                for fpath in flist:
                    if not fpath or not os.path.exists(fpath):
                        continue
                    raw = load_file_all_vars(fpath, all_vars, proc, [region],
                                             year=year)
                    for var_name, reg_data in raw.items():
                        if region not in reg_data:
                            continue
                        filtered = filter_systs(reg_data[region])
                        if 'nominal' not in filtered:
                            continue
                        if proc not in all_hists[channel][var_name]:
                            all_hists[channel][var_name][proc] = filtered
                        else:
                            for vk, (v, e, ed) in filtered.items():
                                if vk in all_hists[channel][var_name][proc]:
                                    pv, pe, ped = all_hists[channel][var_name][proc][vk]
                                    all_hists[channel][var_name][proc][vk] = (pv+v, pe+e, ped)
                                else:
                                    all_hists[channel][var_name][proc][vk] = (v, e, ed)

            # --- data ---
            data_acc = defaultdict(lambda: None)  # var_name -> [vals, vars, edges]
            for proc, pcfg in year_cfg.get('data', {}).items():
                if electron and proc not in ELECTRON_DATA_PROCESSES:
                    continue
                if not electron and proc not in MUON_DATA_PROCESSES:
                    continue
                flist = pcfg.get('files', [pcfg.get('file')])
                print(f'  Loading data {proc} / {region}')
                for fpath in flist:
                    if not fpath or not os.path.exists(fpath):
                        continue
                    raw = load_file_all_vars(fpath, all_vars, proc, [region],
                                             year=year)
                    for var_name, reg_data in raw.items():
                        if region not in reg_data or 'nominal' not in reg_data[region]:
                            continue
                        v, e, edges = reg_data[region]['nominal']
                        if data_acc[var_name] is None:
                            data_acc[var_name] = [v.copy(), e.copy(), edges]
                        else:
                            data_acc[var_name][0] += v
                            data_acc[var_name][1] += e

            for var_name, acc in data_acc.items():
                if acc is not None:
                    all_hists[channel][var_name]['data_obs'] = {
                        'nominal': tuple(acc)
                    }

    # --- apply scale factors ---
    for sr in SIGNAL_REGIONS:
        ch = sr if len(years) == 1 else f'{sr}_{years[0]}'
        if ch not in all_hists:
            continue
        for var_name in all_hists[ch]:
            for proc in list(all_hists[ch][var_name].keys()):
                if proc == 'data_obs':
                    continue
                if proc in TOP_PROCESSES:
                    all_hists[ch][var_name][proc] = apply_scale(
                        all_hists[ch][var_name][proc], sf_top)
                elif proc in WJET_PROCESSES:
                    all_hists[ch][var_name][proc] = apply_scale(
                        all_hists[ch][var_name][proc], sf_wjet)

    # --- merge subprocesses ---
    for sr in SIGNAL_REGIONS:
        ch = sr if len(years) == 1 else f'{sr}_{years[0]}'
        if ch not in all_hists:
            continue
        merge_processes(all_hists[ch], WJET_PROCESSES, WJET_MERGED_NAME)
        merge_processes(all_hists[ch], DY_PROCESSES,   DY_MERGED_NAME)

    # --- symmetrize selected QvG shape systematics ---
    for sr in SIGNAL_REGIONS:
        ch = sr if len(years) == 1 else f'{sr}_{years[0]}'
        if ch not in all_hists:
            continue
        for var_name in all_hists[ch]:
            for proc in list(all_hists[ch][var_name].keys()):
                if proc == 'data_obs':
                    continue
                for syst in SYMMETRIZE_SYSTS:
                    all_hists[ch][var_name][proc] = symmetrize_syst(
                        all_hists[ch][var_name][proc], syst)

    # --- write ROOT file ---
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    print(f'\nWriting {args.output} ...')
    n_written = 0
    with uproot.recreate(args.output) as root_file:
        for ch, var_dict in all_hists.items():
            for var_name, proc_dict in var_dict.items():
                dir_key = f'{ch}__{var_name}'
                for proc, hists in proc_dict.items():
                    for var_key, (vals, errs, edges) in hists.items():
                        if var_key == 'nominal':
                            path = f'{dir_key}/{proc}'
                        else:
                            path = f'{dir_key}/{proc}_{var_key}'
                        root_file[path] = make_bh(vals, errs, edges)
                        n_written += 1

    print(f'Done. {n_written} histograms written to {args.output}')


if __name__ == '__main__':
    main()
