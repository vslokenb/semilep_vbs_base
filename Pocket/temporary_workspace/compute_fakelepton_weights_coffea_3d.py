#!/usr/bin/env python3
import yaml
import coffea.util as util
from coffea.hist import Hist
import sys
import awkward as ak
import boost_histogram as bh
import numpy as np
from copy import deepcopy


def safe_divide(num, den):
    """Avoid division-by-zero; return 0 where den=0"""
    out = np.zeros_like(num)
    mask = den != 0
    out[mask] = num[mask] / den[mask]
    for i in range(len(out)):
        out[i] = np.clip(out[i], 0, None)
    return out

def division_variance(num,den):
    out = np.zeros_like(num)
    mask = ((den > 0) & (num > 0))
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
    
    new_edges_pt_mu = np.array([26,30,32,35,40,100])
    new_edges_pt_e = np.array([35,40,50,60,100])
    new_edges_eta = np.array([-2.4,-2.15,-1.479,-0.5,0.5,1.479,2.15,2.4], dtype=float)
    if "muon" in numerator_name:
        if len(num.axes)==3 and "pt" in numerator_name:
            num = num[:,:,bh.rebin(bh.axis.Variable(new_edges_pt_mu))]
            den = den[:,:,bh.rebin(bh.axis.Variable(new_edges_pt_mu))]
    # elif "muon" in numerator_name and "eta" in numerator_name:
        if len(num.axes)==3 and "eta" in numerator_name:
            num = num[:,:,bh.rebin(bh.axis.Variable(new_edges_eta))]
            den = den[:,:,bh.rebin(bh.axis.Variable(new_edges_eta))]
        elif len(num.axes)==5:
            num = num[:,:,bh.rebin(bh.axis.Variable(new_edges_pt_mu)),bh.rebin(bh.axis.Variable(new_edges_eta)),:]
            den = den[:,:,bh.rebin(bh.axis.Variable(new_edges_pt_mu)),bh.rebin(bh.axis.Variable(new_edges_eta)),:]
            # print(num[0,0,:,:,:])
            # print(den[0,0,:,:,:])
    elif "electron" in numerator_name:
        if len(num.axes)==3 and "pt" in numerator_name:
            num = num[:,:,bh.rebin(bh.axis.Variable(new_edges_pt_e))]
            den = den[:,:,bh.rebin(bh.axis.Variable(new_edges_pt_e))]
    # elif "muon" in numerator_name and "eta" in numerator_name:
        if len(num.axes)==3 and "eta" in numerator_name:
            num = num[:,:,bh.rebin(bh.axis.Variable(new_edges_eta))]
            den = den[:,:,bh.rebin(bh.axis.Variable(new_edges_eta))]
        elif len(num.axes)==5:
            num = num[:,:,bh.rebin(bh.axis.Variable(new_edges_pt_e)),bh.rebin(bh.axis.Variable(new_edges_eta)),:]
            den = den[:,:,bh.rebin(bh.axis.Variable(new_edges_pt_e)),bh.rebin(bh.axis.Variable(new_edges_eta)),:]
            # print(num[0,0,:,:,:])
            # print(den[0,0,:,:,:])
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
    accs = [util.load(f) for f in input_files]

    # Extract dataset names from filename (customize if needed)
    def dataset_name_from_file(acc):
        period = list(acc['datasets_metadata']['by_datataking_period'].keys())[0]
        dataset = list(acc['datasets_metadata']['by_datataking_period'][period].keys())[0]
        return dataset

    def dataset_name_era_from_file(acc):
        dataset_era = list(acc['datasets_metadata']['by_dataset'].keys())[0]
        dataset = acc['datasets_metadata']['by_dataset'][dataset_era]['sample']
        period = acc['datasets_metadata']['by_dataset'][dataset_era]['year']
        return dataset_era

    # --- Collect histograms per dataset ---
    merged_acc = {}

    for fname, acc in zip(input_files, accs):
        dsname = dataset_name_from_file(acc)
        dsname_era = dataset_name_era_from_file(acc)
        if dsname not in factors:
            print(f"⚠ WARNING: dataset {dsname} missing in YAML 'factors'. Using factor=1.")
            scale = 1.0
        else:
            scale = factors[dsname]
        for hname, hist_dic in acc['variables'].items():
            hist = hist_dic[dsname][dsname_era]
            scaled_hist = scale_histogram(hist, scale)
            if hname not in merged_acc:
                merged_acc[hname] = deepcopy(scaled_hist)
            else:
                merged_acc[hname].values()[...] += scaled_hist.values()[...]
                merged_acc[hname].variances()[...] += scaled_hist.variances()[...]
                #if dsname in ['EGamma','Muon','SingleMuon']: 
                #    print(scaled_hist.values()[...][:, np.newaxis, :].shape)
                #    merged_acc[hname].values()[...] += scaled_hist.values()[...][:, np.newaxis, :]
                #else:
                #    merged_acc[hname].values()[...] += scaled_hist.values()[...]

    print("✔ Finished linear-combination merging.")

    # --- Compute ratio histograms ---
    if weight_pairs:
        print("📊 Computing weight histograms...")
        for pair in weight_pairs:
            print("pair numerator ",pair["numerator"])
            print("pair denominator ",pair["denominator"])
            print("pair output ",pair["output"])
            compute_weight_hist(
                merged_accumulator=merged_acc,
                numerator_name=pair["numerator"],
                denominator_name=pair["denominator"],
                output_name=pair["output"],
            )

    # --- Save combined output ---
    print("💾 Writing output coffea:", output_file)
    util.save(merged_acc, output_file)

    print("🎉 Done!")


if __name__ == "__main__":
    main()
