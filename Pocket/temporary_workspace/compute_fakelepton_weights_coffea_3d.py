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
    out = np.zeros_like(num)
    mask = den != 0
    num[num<0] = 0 
    out[mask] = num[mask] / den[mask]
    for i in range(len(out)):
        out[i] = np.clip(out[i], 0, None)
    return out

def division_variance(num, den):
    out = np.zeros_like(num)
    mask = ((den > 0) & (num > 0))
    out[mask] = num[mask]/den[mask]**2 - num[mask]**2/den[mask]**3
    return out

def scale_histogram(hist: Hist, factor: float) -> Hist:
    hnew = deepcopy(hist)
    hnew.values()[...] *= factor
    hnew.variances()[...] *= abs(factor)
    return hnew

def merge_hist_list(hist_list):
    merged = deepcopy(hist_list[0])
    for h in hist_list[1:]:
        merged.values()[...] += h.values()[...]
    return merged

def drop_variation_axis(h):
    """Strip the trivial size-1 variation axis that PocketCoffea adds to MC."""
    axes = [ax.name for ax in h.axes]
    if 'variation' in axes:
        assert h.values().shape[axes.index('variation')] == 1, \
            f"Refusing to drop variation axis with {h.values().shape[axes.index('variation')]} bins"
        return h[{'variation': 'nominal'}]
    return h  # data passes through unchanged

def fold_abs_eta(hist):
    """
    Find the signed-eta axis (any axis with edges symmetric around 0),
    sum mirrored negative+positive bins, and return a new histogram
    with abs(eta) edges.
    """
    eta_idx = None
    for i, ax in enumerate(hist.axes):
        edges = ax.edges
        # symmetric around 0: first edge negative, last positive, same magnitude
        if edges[0] < 0 and np.isclose(edges[0], -edges[-1], atol=1e-4):
            eta_idx = i
            break

    if eta_idx is None:
        print("  [fold_abs_eta] No symmetric eta axis found – skipping fold.")
        return hist

    edges  = hist.axes[eta_idx].edges
    n_bins = len(edges) - 1
    half   = n_bins // 2

    # abs(eta) edges are just the positive half
    abs_edges = edges[half:]          # shape (half+1,)

    # indices: positive half [half .. n_bins-1]
    # mirrored negative half [half-1 .. 0]  (reversed so bin0_neg <-> bin0_pos)
    pos_idx = np.arange(half, n_bins)
    neg_idx = np.arange(half - 1, -1, -1)

    vals  = hist.values()
    varis = hist.variances()

    folded_vals  = (np.take(vals,  pos_idx, axis=eta_idx)
                  + np.take(vals,  neg_idx, axis=eta_idx))
    folded_vars  = (np.take(varis, pos_idx, axis=eta_idx)
                  + np.take(varis, neg_idx, axis=eta_idx))

    # Preserve original axis name (may be "" or "eta" or similar)
    old_name  = hist.axes[eta_idx].name
    new_axes  = list(hist.axes)
    new_axes[eta_idx] = bh.axis.Variable(abs_edges)

    new_hist = bh.Histogram(*new_axes, storage=bh.storage.Weight())
    new_hist.values()[...]    = folded_vals
    new_hist.variances()[...] = folded_vars

    print(f"  [fold_abs_eta] axis {eta_idx} folded: "
          f"{edges[0]:.2f}..{edges[-1]:.2f} → 0..{abs_edges[-1]:.2f} "
          f"({n_bins} bins → {half} bins)")
    return new_hist

def compute_weight_hist(merged_accumulator, numerator_name, denominator_name, output_name):
    """Compute ratio histogram and store inside accumulator."""
    num = merged_accumulator[numerator_name]
    den = merged_accumulator[denominator_name]

    new_edges_pt_mu = np.array([26, 30, 40, 100])
    new_edges_pt_e  = np.array([35, 40, 50, 60, 100])
    new_edges_eta   = np.array([-2.4, -2.15, -1.479, 0, 1.479, 2.15, 2.4], dtype=float)

    if "muon" in numerator_name:
        if len(num.axes) == 2 and "pt" in numerator_name:        # was 3
            num = num[:, bh.rebin(bh.axis.Variable(new_edges_pt_mu))]
            den = den[:, bh.rebin(bh.axis.Variable(new_edges_pt_mu))]
        if len(num.axes) == 2 and "eta" in numerator_name:       # was 3
            num = num[:, bh.rebin(bh.axis.Variable(new_edges_eta))]
            den = den[:, bh.rebin(bh.axis.Variable(new_edges_eta))]
        elif len(num.axes) == 4:                                  # was 5
            num = num[:, bh.rebin(bh.axis.Variable(new_edges_pt_mu)),
                          bh.rebin(bh.axis.Variable(new_edges_eta)), :]
            den = den[:, bh.rebin(bh.axis.Variable(new_edges_pt_mu)),
                          bh.rebin(bh.axis.Variable(new_edges_eta)), :]

    elif "electron" in numerator_name:
        if len(num.axes) == 2 and "pt" in numerator_name:        # was 3
            num = num[:, bh.rebin(bh.axis.Variable(new_edges_pt_e))]
            den = den[:, bh.rebin(bh.axis.Variable(new_edges_pt_e))]
        if len(num.axes) == 2 and "eta" in numerator_name:       # was 3
            num = num[:, bh.rebin(bh.axis.Variable(new_edges_eta))]
            den = den[:, bh.rebin(bh.axis.Variable(new_edges_eta))]
        elif len(num.axes) == 4:                                  # was 5
            num = num[:, bh.rebin(bh.axis.Variable(new_edges_pt_e)),
                          bh.rebin(bh.axis.Variable(new_edges_eta)), :]
            den = den[:, bh.rebin(bh.axis.Variable(new_edges_pt_e)),
                          bh.rebin(bh.axis.Variable(new_edges_eta)), :]

    num = fold_abs_eta(num)
    den = fold_abs_eta(den)

    num_vals   = num.values()[()]
    den_vals   = den.values()[()]
    ratio_vals = safe_divide(num_vals, den_vals)
    ratio_vars = division_variance(num_vals, den_vals)

    ratio_hist = deepcopy(num)
    ratio_hist.values()[...]    = ratio_vals
    ratio_hist.variances()[...] = ratio_vars

    weight_hist = deepcopy(ratio_hist)
    weight_hist.values()[...]    = safe_divide(ratio_hist.values()[()], 1.0 - ratio_hist.values()[()])
    weight_hist.variances()[...] = ratio_hist.variances()[()] / (1.0 - ratio_hist.values()[()]**4)

    merged_accumulator[output_name] = weight_hist
    print("weight_hist ", weight_hist)
    print(f"✔ Created weight histogram '{output_name}'")


def main():
    if len(sys.argv) < 4:
        print("Usage: compute_fakelepton_weights_coffea.py factors.yaml output.coffea input1.coffea [input2.coffea ...]")
        sys.exit(1)

    yaml_file    = sys.argv[1]
    output_file  = sys.argv[2]
    input_files  = sys.argv[3:]

    print("📘 Reading YAML:", yaml_file)
    with open(yaml_file) as f:
        cfg = yaml.safe_load(f)

    factors      = cfg.get("factors", {})
    weight_pairs = cfg.get("weight_pairs", [])

    print("📦 Loading input coffea files...")
    accs = [util.load(f) for f in input_files]

    def dataset_name_from_file(acc):
        period  = list(acc['datasets_metadata']['by_datataking_period'].keys())[0]
        dataset = list(acc['datasets_metadata']['by_datataking_period'][period].keys())[0]
        return dataset

    def dataset_name_era_from_file(acc):
        dataset_era = list(acc['datasets_metadata']['by_dataset'].keys())[0]
        return dataset_era

    merged_acc = {}

    for fname, acc in zip(input_files, accs):
        dsname     = dataset_name_from_file(acc)
        dsname_era = dataset_name_era_from_file(acc)
        if dsname not in factors:
            print(f"⚠ WARNING: dataset {dsname} missing in YAML 'factors'. Using factor=1.")
            scale = 1.0
        else:
            scale = factors[dsname]

        for hname, hist_dic in acc['variables'].items():
            hist        = hist_dic[dsname][dsname_era]
            hist        = drop_variation_axis(hist)          # ← strip MC-only axis
            scaled_hist = scale_histogram(hist, scale)
            if hname not in merged_acc:
                merged_acc[hname] = deepcopy(scaled_hist)
            else:
                merged_acc[hname].values()[...]    += scaled_hist.values()[...]
                merged_acc[hname].variances()[...] += scaled_hist.variances()[...]

    print("✔ Finished linear-combination merging.")

    if weight_pairs:
        print("📊 Computing weight histograms...")
        for pair in weight_pairs:
            print("pair numerator ",  pair["numerator"])
            print("pair denominator ", pair["denominator"])
            print("pair output ",      pair["output"])
            compute_weight_hist(
                merged_accumulator=merged_acc,
                numerator_name=pair["numerator"],
                denominator_name=pair["denominator"],
                output_name=pair["output"],
            )

    print("💾 Writing output coffea:", output_file)
    util.save(merged_acc, output_file)
    print("🎉 Done!")


if __name__ == "__main__":
    main()