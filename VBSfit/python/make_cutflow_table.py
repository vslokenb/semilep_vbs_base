#!/usr/bin/env python3
"""
Cutflow and efficiency tables from coffea output files.

For each MC process reads:
  - xsec * luminosity  (from datasets_metadata, for the 'initial' row)
  - sumw per selection region  (from sumw accumulator)

Writes two text files:
  cutflow.txt   — absolute expected yields per group
  efficiency.txt — yield / initial per group (background-only, no nonprompt)
"""
import argparse
import os
import yaml
import coffea.util

# ---------------------------------------------------------------------------
# Process → group mapping
# ---------------------------------------------------------------------------
WJET_PROCS = {
    "WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8",
    "WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8",
    "WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8",
    "WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8",
    "WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8",
    "WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8",
    "WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8",
    "WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8",
}

TTBAR_PROCS = {
    "TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8",
    "TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8",
}

SINGLETOP_PROCS = {
    "ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8",
    "ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8",
    "ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8",
    "ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8",
    "ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8",
}

DY_PROCS = {
    "DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8",
    "DYJetsToLL_M-10to50_TuneCP5_13TeV-amcatnloFXFX-pythia8",
}

DIBOSON_PROCS = {
    
    "WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8",
    "WWZ_4F_TuneCP5_13TeV-amcatnlo-pythia8",
    "WZTo3LNu_mllmin01_NNPDF31_TuneCP5_13TeV_powheg_pythia8",
    "WZZ_TuneCP5_13TeV-amcatnlo-pythia8",
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
}

VGAMMA_VBF_PROCS = {
    "GluGluWWToLNuQQ_TuneCP5_13TeV_madgraph-pythia8",
    "ZGToLLG_01J_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8",

}

NONPROMPT_PROCS = {'nonprompt'}

SIGNAL_PROCS = {
    "WminusTo2JZTo2LJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
    "WminusToLNuWminusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
    "WminusToLNuZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
    "WplusTo2JWminusToLNuJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8", #WplusTo2JWminusToLNuJJ missing in QCD
    "WplusTo2JZTo2LJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
    "WplusToLNuWminusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
    "WplusToLNuWplusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
    "WplusToLNuZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
    "ZTo2LZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
}

DATA_PROCS = {'EGamma', 'SingleMuon'}

# High-pT stitched slices: included in sumw accumulation but their xsec
# must NOT be added to the initial row (the inclusive 0J/1J/2J slices already
# cover the full phase space; the PTLNu/PTLL slices are overlap-exclusive
# samples used only to improve high-pT statistics).
def skip_initial_xsec(proc_name):
    return (proc_name.startswith('WJetsToLNu') or
            proc_name.startswith('DY'))

_PROC_GROUP = {}
for _p in WJET_PROCS:        _PROC_GROUP[_p] = 'Wjet'
for _p in TTBAR_PROCS:       _PROC_GROUP[_p] = 'TTbar'
for _p in SINGLETOP_PROCS:   _PROC_GROUP[_p] = 'Single Top'
for _p in DY_PROCS:          _PROC_GROUP[_p] = 'DY'
for _p in DIBOSON_PROCS:     _PROC_GROUP[_p] = 'Diboson&Triboson'
for _p in VGAMMA_VBF_PROCS:  _PROC_GROUP[_p] = 'Vgamma&VBF'
for _p in NONPROMPT_PROCS:   _PROC_GROUP[_p] = 'Nonprompt'
for _p in SIGNAL_PROCS:      _PROC_GROUP[_p] = 'Signal'

BKG_GROUPS = ['Wjet', 'TTbar', 'Nonprompt', 'Single Top', 'DY', 'Diboson&Triboson', 'Vgamma&VBF']

# coffea sumw region keys and their table row labels
# ('baseline' in coffea = pre-selection, labelled 'presele' in the table)
REGION_KEYS = ['baseline', 'boosted_e', 'boosted_mu', 'resolved_e', 'resolved_mu']
ROW_LABELS  = ['presele',  'boosted_e', 'boosted_mu', 'resolved_e', 'resolved_mu']
_ROW_TO_REGION_KEY = dict(zip(ROW_LABELS, REGION_KEYS))

# BDT > 0.95 rows: sum of histogram bins with left edge >= 0.95
BDT_P95_THRESHOLD = 0.95
BDT_REGIONS       = ['boosted_e', 'boosted_mu', 'resolved_e', 'resolved_mu']
BDT_ROW_LABELS    = [f'{r}_bdt_p95' for r in BDT_REGIONS]

ROW_NAMES    = ['initial'] + ROW_LABELS + BDT_ROW_LABELS

CUTFLOW_COLS = ['Total Background', 'Wjet', 'TTbar', 'Nonprompt',
                'Single Top', 'DY', 'Diboson&Triboson', 'Vgamma&VBF', 'Signal']
EFF_COLS     = ['Wjet', 'TTbar', 'Single Top', 'DY', 'Diboson&Triboson', 'Vgamma&VBF', 'Signal']


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------
def get_process_list(config):
    """Return list of (proc_name, year_key, [coffea_files])."""
    entries = []
    for year_key, year_data in config['years'].items():
        for dtype in ('background', 'signal'):
            for proc_name, proc_data in year_data.get(dtype, {}).items():
                if proc_name in DATA_PROCS:
                    continue
                if 'file' in proc_data:
                    files = [proc_data['file']]
                else:
                    files = proc_data.get('files', [])
                entries.append((proc_name, year_key, files))
    return entries


# ---------------------------------------------------------------------------
# Coffea reading
# ---------------------------------------------------------------------------
def read_proc(proc_name, year_key, coffea_files, lumi_invpb):
    """Load one process; return (xsec_times_lumi_or_None, sumw_dict, bdt_p95_dict)."""
    dataset_key = f"{proc_name}_{year_key}"
    xsec_lumi   = None
    sumw        = {r: 0.0 for r in REGION_KEYS}
    bdt_p95     = {r: 0.0 for r in BDT_REGIONS}

    for fpath in coffea_files:
        if not os.path.exists(fpath):
            print(f'  [warn] file not found: {fpath}')
            continue
        try:
            f = coffea.util.load(fpath)
        except Exception as exc:
            print(f'  [warn] load error {fpath}: {exc}')
            continue

        # xsec (first file wins; nonprompt and stitched high-pT slices excluded)
        if xsec_lumi is None and proc_name not in NONPROMPT_PROCS and not skip_initial_xsec(proc_name):
            try:
                xsec = f['datasets_metadata']['by_dataset'][dataset_key]['xsec']
                xsec_lumi = float(xsec) * lumi_invpb
            except (KeyError, TypeError):
                pass

        # sumw per region (coffea key is e.g. 'baseline', not 'presele')
        for region_key in REGION_KEYS:
            try:
                val = f['sumw'][region_key][dataset_key][proc_name]['nominal']
                sumw[region_key] += float(val)
            except (KeyError, TypeError):
                pass

        # BDT > 0.95: sum histogram bins with left edge >= threshold
        variables = f.get('variables', {})
        for region in BDT_REGIONS:
            try:
                h = variables[f'bdt_{region}'][proc_name][dataset_key]
                sl = h[{'cat': region, 'variation': 'nominal'}]
                vals  = sl.values()
                edges = sl.axes[0].edges
                mask  = edges[:-1] >= BDT_P95_THRESHOLD
                bdt_p95[region] += float(vals[mask].sum())
            except (KeyError, TypeError, IndexError, Exception):
                pass

    return xsec_lumi, sumw, bdt_p95


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------
def fmt_cutflow(v):
    if v is None:
        return '-'
    return f'{v:,.1f}'

def fmt_eff(v):
    if v is None:
        return '-'
    return f'{v:.4e}'

def render_table(data, row_names, col_names, cell_fmt_fn):
    """Render an ASCII box table; data[row][col] may be float or None."""
    # Convert to strings
    str_rows = [[cell_fmt_fn(v) for v in row] for row in data]
    # Column widths
    col_w = [max(len(cn), max(len(r[j]) for r in str_rows))
             for j, cn in enumerate(col_names)]
    row_label_w = max(len(n) for n in [''] + row_names)
    sep = '+' + '+'.join('-' * (w + 2) for w in [row_label_w] + col_w) + '+'

    def line(label, vals):
        s = '| ' + label.ljust(row_label_w) + ' |'
        for v, w in zip(vals, col_w):
            s += ' ' + v.rjust(w) + ' |'
        return s

    out = [sep, line('', col_names), sep]
    for name, row in zip(row_names, str_rows):
        out.append(line(name, row))
    out.append(sep)
    return '\n'.join(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',  required=True,
                        help='inputs YAML config (e.g. config/inputs_2022postEE.yaml)')
    parser.add_argument('--lumi',    type=float, default=59830.0,
                        help='Luminosity in pb^-1 (default: 59830 = 59.83 fb^-1)')
    parser.add_argument('--outdir',  default='.',
                        help='Directory to write cutflow.txt and efficiency.txt')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    config = yaml.safe_load(open(args.config))
    proc_list = get_process_list(config)

    # Accumulators: initial (xsec*lumi), sumw per region, BDT > 0.95 per region
    group_names = ['Wjet', 'TTbar', 'Nonprompt', 'Single Top',
                   'DY', 'Diboson&Triboson', 'Vgamma&VBF', 'Signal']
    initial  = {g: 0.0 for g in group_names}
    initial['Nonprompt'] = None          # no xsec for nonprompt
    sumw_acc = {g: {r: 0.0 for r in REGION_KEYS} for g in group_names}
    bdt_acc  = {g: {r: 0.0 for r in BDT_REGIONS}  for g in group_names}

    for proc_name, year_key, files in proc_list:
        group = _PROC_GROUP.get(proc_name)
        if group is None:
            print(f'  [skip] unknown process: {proc_name}')
            continue
        print(f'  {proc_name}')
        xsec_lumi, sumw, bdt_p95 = read_proc(proc_name, year_key, files, args.lumi)

        if group != 'Nonprompt' and xsec_lumi is not None:
            initial[group] += xsec_lumi

        for rk in REGION_KEYS:
            sumw_acc[group][rk] += sumw[rk]
        for r in BDT_REGIONS:
            bdt_acc[group][r] += bdt_p95[r]

    # -------------------------------------------------------------------
    # Cutflow table
    # -------------------------------------------------------------------
    def cutflow_cell(col, row_label):
        if row_label == 'initial':
            if col in ('Total Background', 'Nonprompt'):
                return None
            return initial[col]
        if row_label in _ROW_TO_REGION_KEY:
            rk = _ROW_TO_REGION_KEY[row_label]
            if col == 'Total Background':
                return sum(sumw_acc[g][rk] for g in BKG_GROUPS)
            return sumw_acc[col][rk]
        # BDT p95 row
        r = row_label.replace('_bdt_p95', '')
        if col == 'Total Background':
            return sum(bdt_acc[g][r] for g in BKG_GROUPS)
        return bdt_acc[col][r]

    cf_data  = [[cutflow_cell(col, rl) for col in CUTFLOW_COLS] for rl in ROW_NAMES]
    cf_table = render_table(cf_data, ROW_NAMES, CUTFLOW_COLS, fmt_cutflow)

    cf_path = os.path.join(args.outdir, 'cutflow.txt')
    with open(cf_path, 'w') as fout:
        fout.write(cf_table + '\n')
    print(f'\nCutflow table -> {cf_path}')
    print(cf_table)

    # -------------------------------------------------------------------
    # Efficiency table  (no Total Background, no Nonprompt)
    # -------------------------------------------------------------------
    def eff_cell(col, row_label):
        init = initial[col]
        if init is None or init == 0.0:
            return None
        if row_label == 'initial':
            return 1.0
        if row_label in _ROW_TO_REGION_KEY:
            rk = _ROW_TO_REGION_KEY[row_label]
            return sumw_acc[col][rk] / init
        # BDT p95 row
        r = row_label.replace('_bdt_p95', '')
        return bdt_acc[col][r] / init

    eff_data  = [[eff_cell(col, rl) for col in EFF_COLS] for rl in ROW_NAMES]
    eff_table = render_table(eff_data, ROW_NAMES, EFF_COLS, fmt_eff)

    eff_path = os.path.join(args.outdir, 'efficiency.txt')
    with open(eff_path, 'w') as fout:
        fout.write(eff_table + '\n')
    print(f'\nEfficiency table -> {eff_path}')
    print(eff_table)


if __name__ == '__main__':
    main()
