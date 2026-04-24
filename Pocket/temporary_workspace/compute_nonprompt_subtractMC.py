#!/usr/bin/env python3
import yaml
import coffea.util as util
from coffea.hist import Hist
import sys
import awkward as ak
import numpy as np
from copy import deepcopy


def safe_divide(num, den):
    """Avoid division-by-zero; return 0 where den=0"""
    out = np.zeros_like(num)
    mask = den != 0
    num[num<0] = 0 
    out[mask] = num[mask] / den[mask]
    for i in range(len(out)):
        out[i] = np.clip(out[i], 0, None)
    return out

def division_variance(num,den):
    out = np.zeros_like(num)
    mask = den != 0
    out[mask] = num[mask]/den[mask]**2 - num[mask]**2/den[mask]**3
    return out


def scale_histogram(hist: Hist, factor: float) -> Hist:
    """Return a scaled copy of a coffea.hist.Hist"""
    hnew = deepcopy(hist)
    hnew.values()[...] *= factor
    hnew.variances()[...] *= abs(factor)
    return hnew


def merge_hist_list(hist_list):
    """Sum a list of histograms with identical structure."""
    merged = deepcopy(hist_list[0])
    for h in hist_list[1:]:
        merged.values()[...] += h.values()[...]
    return merged


def compute_weight_hist(merged_accumulator, numerator_name, denominator_name, output_name):
    """Compute ratio histogram and store inside accumulator."""
    num = merged_accumulator[numerator_name]
    den = merged_accumulator[denominator_name]

    # Extract numpy arrays
    num_vals = num.values()[()]
    den_vals = den.values()[()]
    ratio_vals = safe_divide(num_vals, den_vals)
    ratio_vars = division_variance(num_vals, den_vals)
    # Clone histogram structure and insert ratio values
    ratio_hist = deepcopy(num)
    ratio_hist.values()[...] = ratio_vals
    ratio_hist.variances()[...] = ratio_vars

    weight_hist = deepcopy(ratio_hist)
    weight_hist.values()[...] = safe_divide(ratio_hist.values()[()], 1.0-ratio_hist.values()[()])
    weight_hist.variances()[...] = ratio_hist.variances()[()] / (1.0-ratio_hist.values()[()]**4)
    # Save into accumulator
    merged_accumulator[output_name] = weight_hist
    print("weight_hist ",weight_hist)
    print(f"✔ Created weight histogram '{output_name}'")



def main():
    if len(sys.argv) < 4:
        print("Usage: compute_fakelepton_weights_coffea.py factors.yaml output.coffea input1.coffea [input2.coffea ...]")
        sys.exit(1)

    yaml_file = sys.argv[1]
    output_file = sys.argv[2]
    input_files = sys.argv[3:]

    print("📘 Reading YAML:", yaml_file)
    with open(yaml_file) as f:
        cfg = yaml.safe_load(f)

    factors = cfg.get("factors", {})
    weight_pairs = cfg.get("weight_pairs", [])

    # --- Load input coffea pickles ---
    print("📦 Loading input coffea files...")
    print(input_files)
    accs = [util.load(f) for f in input_files]

    # Extract dataset names from filename (customize if needed)
    def dataset_name_from_file(acc):
        period = list(acc['datasets_metadata']['by_datataking_period'].keys())[0]
        dataset = list(acc['datasets_metadata']['by_datataking_period'][period].keys())[0]
        return dataset,period

    def dataset_name_era_from_file(acc):
        dataset_era = list(acc['datasets_metadata']['by_dataset'].keys())[0]
        dataset = acc['datasets_metadata']['by_dataset'][dataset_era]['sample']
        period = acc['datasets_metadata']['by_dataset'][dataset_era]['year']
        return dataset_era


    # --- Collect histograms per dataset ---
    merged_acc = {
            'sum_genweights': {
                },
            'sum_signOf_genweights': {
                },
            'sumw': {
                },
            'sumw2': {
                },
            'cutflow': {},
            'variables': {},
            'columns': {},
            'processing_metadata': {},
            'datasets_metadata': {}
            }

    for fname, acc in zip(input_files, accs):
        dsname,year = dataset_name_from_file(acc)
        dsname_era = dataset_name_era_from_file(acc)
        if dsname not in factors:
            print(f"⚠ WARNING: dataset {dsname} missing in YAML 'factors'. Using factor=1.")
            scale = -1.0
        else:
            scale = factors[dsname]
        if len(merged_acc["sumw"].keys()) == 0:
            merged_acc["sum_genweights"] = {
                    "nonprompt_"+year: 0.0
                    }
            merged_acc["sum_signOf_genweights"] = {
                    "nonprompt_"+year: 0.0
                    }
            for key in acc['cutflow'].keys():
                if key in ['initial','skim']:
                    merged_acc['cutflow'][key] = {"nonprompt_"+year: 0.0}
                elif key == "presel":
                    merged_acc['cutflow'][key] = {"nonprompt_"+year:{"nominal": 0.0}}
                else:
                    merged_acc['cutflow'][key] = {"nonprompt_"+year: {"nonprompt": {"nominal": 0.0}}}
            for key in acc['sumw'].keys():
                merged_acc['sumw'][key] = {"nonprompt_"+year: {"nonprompt": {"nominal": 0.0}}}
                merged_acc['sumw2'][key] = {"nonprompt_"+year: {"nonprompt": {"nominal": 0.0}}}
            merged_acc['datasets_metadata']['by_datataking_period'] = {
                    year: {
                        "nonprompt": {"nonprompt_"+year} 
                        }
                    }
            merged_acc['datasets_metadata']['by_dataset'] = {
                    "nonprompt_"+year: {
                        'das_names': "none",
                        'sample': "nonprompt",
                        'year': year,
                        'isMC': 'True',
                        'xsec': '1.0',
                        'nevents': '0',
                        'size': '0'
                        }
                    }
        for hname, hist_dic in acc['variables'].items():
            hist = hist_dic[dsname][dsname_era]
            scaled_hist = scale_histogram(hist, scale)
            slicing_variations = {
                    'variation': ['electron_inverttight_to_fakeDown', 'electron_inverttight_to_fakeUp', 'muon_inverttight_to_fakeDown', 'muon_inverttight_to_fakeUp','nominal'],
                    #'cat': ["baseline","boosted_e","boosted_e_WCR","boosted_e_TTCR","resolved_e","resolved_e_WCR","resolved_e_TTCR","boosted_mu","resolved_mu","resolved_mu_WCR","resolved_mu_TTCR","boosted_mu_WCR","boosted_mu_TTCR"]
                    }
            if hname not in merged_acc['variables'].keys():
                # print("scaled_hist",scaled_hist)
                merged_acc['variables'][hname] = {
                        "nonprompt": {
                            "nonprompt_"+year: deepcopy(scaled_hist)[slicing_variations]
                            }
                        }
            else:
                merged_acc['variables'][hname]['nonprompt']['nonprompt_'+year].values()[...] += scaled_hist[slicing_variations].values()[...]
                merged_acc['variables'][hname]['nonprompt']['nonprompt_'+year].variances()[...] += scaled_hist[slicing_variations].variances()[...]
        if len(acc['sum_genweights'].keys()) > 0:
            merged_acc['sum_genweights']["nonprompt_"+year] += acc['sum_genweights'][dsname_era]*scale
            merged_acc['sum_signOf_genweights']["nonprompt_"+year] += acc['sum_signOf_genweights'][dsname_era]*scale
        else:
            sumw_base = acc['sumw']['baseline']
            sumw_base = sumw_base[list(sumw_base.keys())[0]]
            sumw_base = sumw_base[list(sumw_base.keys())[0]]
            sumw_base = sumw_base[list(sumw_base.keys())[0]]
            merged_acc['sum_genweights']["nonprompt_"+year] += sumw_base*scale
            merged_acc['sum_signOf_genweights']["nonprompt_"+year] += sumw_base*scale
        for key in merged_acc['cutflow'].keys():
            if key in ['initial','skim']:
                merged_acc['cutflow'][key]["nonprompt_"+year] += acc['cutflow'][key][dsname_era]*scale
            elif key =="presel":
                merged_acc['cutflow'][key]["nonprompt_"+year]["nominal"] += acc['cutflow'][key][dsname_era]["nominal"]*scale
            else:
                merged_acc['cutflow'][key]["nonprompt_"+year]["nonprompt"]["nominal"] += acc['cutflow'][key][dsname_era][dsname]["nominal"]*scale
        for key in merged_acc['sumw'].keys():
            merged_acc['sumw'][key]["nonprompt_"+year]["nonprompt"]["nominal"] += acc['sumw'][key][dsname_era][dsname]["nominal"]*scale
        for key in merged_acc['sumw2'].keys():
            merged_acc['sumw2'][key]["nonprompt_"+year]["nonprompt"]["nominal"] += acc['sumw2'][key][dsname_era][dsname]["nominal"]*scale
    # print("merged_acc['cutflow']",merged_acc['cutflow'])
    # print("merged_acc['sumw']",merged_acc['sumw'])
    # print("merged_acc['sumw2']",merged_acc['sumw2'])
    # print("✔ Finished linear-combination merging.")


    import collections

    def to_defaultdict(obj):
        """Recursively convert all dicts to defaultdict(None) for coffea compatibility."""
        if isinstance(obj, (dict, collections.defaultdict)):
            return collections.defaultdict(None, {k: to_defaultdict(v) for k, v in obj.items()})
        return obj

    # Just before util.save:
    merged_acc = to_defaultdict(merged_acc)
    print("💾 Writing output coffea:", output_file)
    util.save(merged_acc, output_file)

    print("🎉 Done!")


if __name__ == "__main__":
    main()
