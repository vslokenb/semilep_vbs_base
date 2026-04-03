#!/usr/bin/env python3
import yaml
import coffea.util as util
from coffea.hist import Hist
import boost_histogram as bh
import sys
import awkward as ak
import numpy as np
from copy import deepcopy


def safe_divide(num, den):
    """Avoid division-by-zero; return 0 where den=0"""
    out = np.zeros_like(num)
    mask = den != 0
    out[mask] = num[mask] / den[mask]
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


def compute_ratio_hist(merged_accumulator, numerator_name, denominator_name, output_name):
    """Compute ratio histogram and store inside accumulator."""
    num = merged_accumulator[numerator_name]
    den = merged_accumulator[denominator_name]
    print("num.axes",num.axes)
    if len(num.axes)==3:
        print("num ",num[1,0,:].values())
        print("den ",den[1,0,:].values())
    else:
        print("num ",num[1,0,:,:].values())
        print("den ",den[1,0,:,:].values())
    # Extract numpy arrays
    if "muon" in numerator_name and "pt" in numerator_name:
        new_edges = np.array([26,28,30,32,35,40,45,100])
        if len(num.axes)==3:
            num = num[:,:,bh.rebin(bh.axis.Variable(new_edges))]
            den = den[:,:,bh.rebin(bh.axis.Variable(new_edges))]
        else:
            num = num[:,:,bh.rebin(bh.axis.Variable(new_edges)),:]
            den = den[:,:,bh.rebin(bh.axis.Variable(new_edges)),:]
            #num = num[:,:,Hist.rebin(num.axes[2],Hist.new.Variable(new_edges, name=num.axes[2].name, label=num.axes[2].label))]
            #den = den[:,:,Hist.rebin(den.axes[2],Hist.new.Variable(new_edges, name=den.axes[2].name, label=den.axes[2].label))]
        #else:
        #    num = num[:,:,Hist.rebin(num.axes[2],Hist.new.Variable(new_edges, name=num.axes[2].name, label=num.axes[2].label)),:]
        #    den = den[:,:,Hist.rebin(den.axes[2],Hist.new.Variable(new_edges, name=den.axes[2].name, label=den.axes[2].label)),:]
        #num = num.rebin(num.axes[2],Hist.new.Variable(new_edges, name=num.axes[2].name, label=num.axes[2].label))
        #den = den.rebin(num.axes[2],Hist.new.Variable(new_edges, name=den.axes[2].name, label=den.axes[2].label))
        #num = num.rebin(num.axes[2].name,Hist.new.Variable(new_edges, name=num.axes[2].name, label=num.axes[2].label))
        #den = den.rebin(den.axes[2].name,Hist.new.Variable(new_edges, name=den.axes[2].name, label=den.axes[2].label))
    num_vals = num.values()[()]
    den_vals = den.values()[()]
    print("num.values()",num.values())
    #print("num_vals",num_vals)
    #print("den_vals",den_vals)
    ratio_vals = safe_divide(num_vals, den_vals)
    ratio_vars = division_variance(num_vals, den_vals)
    print("ratio_vals ", ratio_vals)
    # Clone histogram structure and insert ratio values
    ratio_hist = deepcopy(num)
    ratio_hist.values()[...] = ratio_vals
    ratio_hist.variances()[...] = ratio_vars

    # Save into accumulator
    merged_accumulator[output_name] = ratio_hist
    print("ratio_hist ",ratio_hist)
    print(f"✔ Created ratio histogram '{output_name}'")



def main():
    if len(sys.argv) < 4:
        print("Usage: compute_fake_rate_coffea.py factors.yaml output.coffea input1.coffea [input2.coffea ...]")
        sys.exit(1)

    yaml_file = sys.argv[1]
    output_file = sys.argv[2]
    input_files = sys.argv[3:]

    print("📘 Reading YAML:", yaml_file)
    with open(yaml_file) as f:
        cfg = yaml.safe_load(f)

    factors = cfg.get("factors", {})
    ratio_pairs = cfg.get("ratio_pairs", [])

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
    hists_per_dataset = {}
    for fname, acc in zip(input_files, accs):
        dsname = dataset_name_from_file(acc)
        dsname_era = dataset_name_era_from_file(acc)
        if dsname not in factors:
            print(f"⚠ WARNING: dataset {dsname} missing in YAML 'factors'. Using factor=1.")
            scale = 1.0
        else:
            scale = factors[dsname]
        print("fname ",fname)
        print("dsname ",dsname)
        print("dsname_era ",dsname_era)
        print("scale ",scale)

        for hname, hist_dic in acc['variables'].items():
            hist = hist_dic[dsname][dsname_era]
            hists_per_dataset[hname+"_"+dsname_era] = hist
            scaled_hist = scale_histogram(hist, scale)
            if hname=="nJets":
                print("scaled_hist ",scaled_hist)
            if hname not in merged_acc:
                merged_acc[hname] = deepcopy(scaled_hist)
            else:
                merged_acc[hname].values()[...] += scaled_hist.values()[...]
                merged_acc[hname].variances()[...] += scaled_hist.variances()[...]
                #if dsname in ['EGamma','Muon','SingleMuon']:
                #    if hname=="nJets":
                #        print("scaled_hist.values()[...].shape",scaled_hist.values()[...].shape)
                #        print("scaled_hist.values()[...][:, np.newaxis, :].shape",scaled_hist.values()[...][:, np.newaxis, :].shape)
                #    merged_acc[hname].values()[...] += scaled_hist.values()[...][:, np.newaxis, :]
                #else:
                #    merged_acc[hname].values()[...] += scaled_hist.values()[...]

    print("✔ Finished linear-combination merging.")

    # --- Compute ratio histograms ---
    if ratio_pairs:
        print("📊 Computing ratio histograms...")
        for pair in ratio_pairs:
            print("pair numerator ",pair["numerator"])
            print("pair denominator ",pair["denominator"])
            print("pair output ",pair["output"])
            compute_ratio_hist(
                merged_accumulator=merged_acc,
                numerator_name=pair["numerator"],
                denominator_name=pair["denominator"],
                output_name=pair["output"],
            )

    # --- Save combined output ---
    print("💾 Writing output coffea:", output_file)
    #util.save({**merged_acc,**hists_per_dataset}, output_file)
    util.save({**merged_acc}, output_file)

    print("🎉 Done!")


if __name__ == "__main__":
    main()
