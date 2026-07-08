#!/usr/bin/env python3
import yaml
import coffea.util as util
from coffea.hist import Hist
import sys
import awkward as ak
import boost_histogram as bh
import numpy as np
from copy import deepcopy
import collections


def safe_divide(num, den):
    """Avoid division-by-zero; return 0 where den=0"""
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


def scale_histogram(hist, factor: float):
    """Return a scaled copy using coffea's * operator."""
    return hist * factor


def add_histograms(h_existing, h_new):
    v_e = h_existing.view()
    v_n = h_new.view()

    if v_e.shape == v_n.shape:
        v_e.value[...] += v_n.value
        v_e.variance[...] += v_n.variance
        return h_existing

    def expand_and_broadcast(arr, target_shape):
        """Insert missing axes at position 1 (after category axis) then broadcast."""
        while arr.ndim < len(target_shape):
            arr = np.expand_dims(arr, axis=1)  # insert after category axis
        return np.broadcast_to(arr, target_shape).copy()

    if v_n.value.size <= v_e.value.size:
        try:
            v_e.value[...] += expand_and_broadcast(v_n.value, v_e.value.shape)
            v_e.variance[...] += expand_and_broadcast(v_n.variance, v_e.variance.shape)
        except ValueError as err:
            print(f"  WARNING: could not broadcast {v_n.value.shape} -> {v_e.value.shape}: {err}")
        return h_existing
    else:
        result = deepcopy(h_new)
        v_r = result.view()
        try:
            v_r.value[...] += expand_and_broadcast(v_e.value, v_n.value.shape)
            v_r.variance[...] += expand_and_broadcast(v_e.variance, v_n.variance.shape)
        except ValueError as err:
            print(f"  WARNING: could not broadcast {v_e.value.shape} -> {v_n.value.shape}: {err}")
        return result


def merge_hist_list(hist_list):
    """Sum a list of histograms with identical structure."""
    merged = deepcopy(hist_list[0])
    for h in hist_list[1:]:
        merged = add_histograms(merged, h)
    return merged


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

    new_axes  = list(hist.axes)
    new_axes[eta_idx] = bh.axis.Variable(abs_edges)

    new_hist = bh.Histogram(*new_axes, storage=bh.storage.Weight())
    new_hist.values()[...]    = folded_vals
    new_hist.variances()[...] = folded_vars

    print(f"  [fold_abs_eta] axis {eta_idx} folded: "
          f"{edges[0]:.2f}..{edges[-1]:.2f} → 0..{abs_edges[-1]:.2f} "
          f"({n_bins} bins → {half} bins)")
    return new_hist


def compute_weight_hist(merged_variables, numerator_name, denominator_name, output_name):
    """Compute weight histogram and store inside merged_variables flat dict."""
    num = merged_variables[numerator_name]
    den = merged_variables[denominator_name]

    print("num.axes", num.axes)
    if len(num.axes) == 3:
        print("num ", num[1, 0, :].values())
        print("den ", den[1, 0, :].values())
    else:
        print("num ", num[1, 0, :, :].values())
        print("den ", den[1, 0, :, :].values())

    # ── pt rebinning ─────────────────────────────────────────────────────────
    if "muon" in numerator_name and "pt" in numerator_name:
        new_edges = np.array([26, 28, 30, 32, 35, 40, 45, 100])
        if len(num.axes) == 3:
            num = num[:, :, bh.rebin(bh.axis.Variable(new_edges))]
            den = den[:, :, bh.rebin(bh.axis.Variable(new_edges))]
        else:
            num = num[:, :, bh.rebin(bh.axis.Variable(new_edges)), :]
            den = den[:, :, bh.rebin(bh.axis.Variable(new_edges)), :]

    # ── fold signed eta → abs(eta) ────────────────────────────────────────────
    num = fold_abs_eta(num)
    den = fold_abs_eta(den)

    # ── ratio ─────────────────────────────────────────────────────────────────
    num_vals   = num.values()[()]
    den_vals   = den.values()[()]
    ratio_vals = safe_divide(num_vals, den_vals)
    ratio_vars = division_variance(num_vals, den_vals)

    ratio_hist = deepcopy(num)
    ratio_hist.values()[...]    = ratio_vals
    ratio_hist.variances()[...] = ratio_vars

    weight_hist = deepcopy(ratio_hist)
    weight_hist.values()[...] = safe_divide(ratio_hist.values()[()], 1.0 - ratio_hist.values()[()])
    weight_hist.variances()[...] = ratio_hist.variances()[()] / (1.0 - ratio_hist.values()[()]**4)

    merged_variables[output_name] = weight_hist
    print("weight_hist ", weight_hist)
    print(f"✔ Created weight histogram '{output_name}'")


def to_defaultdict(obj):
    """Recursively convert all dicts to defaultdict(None) for coffea compatibility."""
    if isinstance(obj, (dict, collections.defaultdict)):
        return collections.defaultdict(None, {k: to_defaultdict(v) for k, v in obj.items()})
    return obj


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
    output_sample = cfg.get("output_sample", "merged")

    print("📦 Loading input coffea files...")
    accs = [util.load(f) for f in input_files]

    def get_all_datasets(acc):
        """Return list of (dsname_era, dsname, year) for every dataset in accumulator."""
        result = []
        for dsname_era, meta in acc['datasets_metadata']['by_dataset'].items():
            result.append((dsname_era, meta['sample'], meta['year']))
        return result

    merged_acc = {
        'sum_genweights': {},
        'sum_signOf_genweights': {},
        'sumw': {},
        'sumw2': {},
        'cutflow': {},
        'variables': {},
        'columns': {},
        'processing_metadata': {},
        'datasets_metadata': {}
    }

    initialized_years = set()
    # flat {hname: merged_bh_hist} accumulated across all years for weight computation
    merged_vars_total = {}

    for fname, acc in zip(input_files, accs):
        all_datasets = get_all_datasets(acc)
        print(f"\n📂 File: {fname}  ({len(all_datasets)} dataset(s))")
        for dsname_era, dsname, year in all_datasets:
            scale_val = factors.get(dsname, None)
            flag = "  ← MISSING in YAML" if scale_val is None else ""
            print(f"   • {dsname_era}  (sample={dsname}, year={year}, scale={scale_val}){flag}")

        for dsname_era, dsname, year in all_datasets:
            if dsname not in factors:
                print(f"⚠ WARNING: dataset {dsname} missing in YAML 'factors'. Using factor=1.")
                scale = 1.0
            else:
                scale = factors[dsname]

            out_key = f"{output_sample}_{year}"

            if year not in initialized_years:
                initialized_years.add(year)
                merged_acc["sum_genweights"][out_key] = 0.0
                merged_acc["sum_signOf_genweights"][out_key] = 0.0
                for key in acc['cutflow'].keys():
                    if key not in merged_acc['cutflow']:
                        merged_acc['cutflow'][key] = {}
                    if key in ['initial', 'skim']:
                        merged_acc['cutflow'][key][out_key] = 0.0
                    elif key == "presel":
                        merged_acc['cutflow'][key][out_key] = {"nominal": 0.0}
                    else:
                        merged_acc['cutflow'][key][out_key] = {output_sample: {"nominal": 0.0}}
                for key in acc['sumw'].keys():
                    if key not in merged_acc['sumw']:
                        merged_acc['sumw'][key] = {}
                        merged_acc['sumw2'][key] = {}
                    merged_acc['sumw'][key][out_key] = {output_sample: {"nominal": 0.0}}
                    merged_acc['sumw2'][key][out_key] = {output_sample: {"nominal": 0.0}}
                merged_acc['datasets_metadata'].setdefault('by_datataking_period', {})[year] = {
                    output_sample: {out_key}
                }
                merged_acc['datasets_metadata'].setdefault('by_dataset', {})[out_key] = {
                    'das_names': "none",
                    'sample': output_sample,
                    'year': year,
                    'isMC': 'True',
                    'xsec': '1.0',
                    'nevents': '0',
                    'size': '0'
                }

            for hname, hist_dic in acc['variables'].items():
                hist = hist_dic[dsname][dsname_era]
                scaled_hist = scale_histogram(hist, scale)

                if hname not in merged_acc['variables']:
                    merged_acc['variables'][hname] = {output_sample: {out_key: deepcopy(scaled_hist)}}
                elif out_key not in merged_acc['variables'][hname].get(output_sample, {}):
                    merged_acc['variables'][hname].setdefault(output_sample, {})[out_key] = deepcopy(scaled_hist)
                else:
                    merged_acc['variables'][hname][output_sample][out_key] = add_histograms(
                        merged_acc['variables'][hname][output_sample][out_key], scaled_hist
                    )

                # accumulate flat total for weight computation (all years combined)
                if hname not in merged_vars_total:
                    merged_vars_total[hname] = deepcopy(scaled_hist)
                else:
                    merged_vars_total[hname] = add_histograms(merged_vars_total[hname], scaled_hist)

            if dsname_era in acc.get('sum_genweights', {}):
                merged_acc['sum_genweights'][out_key] += acc['sum_genweights'][dsname_era] * scale
                merged_acc['sum_signOf_genweights'][out_key] += acc['sum_signOf_genweights'][dsname_era] * scale
            else:
                sumw_val = acc['sumw']['baseline'][dsname_era][dsname]['nominal']
                merged_acc['sum_genweights'][out_key] += sumw_val * scale
                merged_acc['sum_signOf_genweights'][out_key] += sumw_val * scale

            for key in merged_acc['cutflow'].keys():
                if key in ['initial', 'skim']:
                    merged_acc['cutflow'][key][out_key] += acc['cutflow'][key][dsname_era] * scale
                elif key == "presel":
                    merged_acc['cutflow'][key][out_key]["nominal"] += acc['cutflow'][key][dsname_era]["nominal"] * scale
                else:
                    merged_acc['cutflow'][key][out_key][output_sample]["nominal"] += acc['cutflow'][key][dsname_era][dsname]["nominal"] * scale
            for key in merged_acc['sumw'].keys():
                merged_acc['sumw'][key][out_key][output_sample]["nominal"] += acc['sumw'][key][dsname_era][dsname]["nominal"] * scale
            for key in merged_acc['sumw2'].keys():
                merged_acc['sumw2'][key][out_key][output_sample]["nominal"] += acc['sumw2'][key][dsname_era][dsname]["nominal"] * scale

    print("✔ Finished linear-combination merging.")

    if weight_pairs:
        print("📊 Computing weight histograms...")
        for pair in weight_pairs:
            print("pair numerator ", pair["numerator"])
            print("pair denominator ", pair["denominator"])
            print("pair output ", pair["output"])
            compute_weight_hist(
                merged_variables=merged_vars_total,
                numerator_name=pair["numerator"],
                denominator_name=pair["denominator"],
                output_name=pair["output"],
            )
            # slot the computed weight histogram into the coffea variables structure
            # weight is derived from all-years total, replicated per year
            wname = pair["output"]
            weight = merged_vars_total[wname]
            merged_acc['variables'][wname] = {output_sample: {}}
            for year in initialized_years:
                out_key = f"{output_sample}_{year}"
                merged_acc['variables'][wname][output_sample][out_key] = deepcopy(weight)

    merged_acc = to_defaultdict(merged_acc)

    print("💾 Writing output coffea:", output_file)
    util.save(merged_acc, output_file)

    print("🎉 Done!")

    def category_audit(merged_acc, hname):
        if hname not in merged_acc['variables']:
            print(f"  [category_audit] '{hname}' not found, skipping.")
            return
        sample_dict = merged_acc['variables'][hname].get(output_sample, {})
        if not sample_dict:
            return
        out_key = next(iter(sample_dict))
        h = sample_dict[out_key]
        cat_axis = h.axes[0]
        print(f"\n=== {hname} ===")
        for ci, cname in enumerate(cat_axis):
            total = h[ci, ...].values().sum()
            print(f"  [{ci}] {cname:28s}  sum={total:10.1f}")

    for hn in ("electron_tight_pt_eta", "muon_loose_pt_eta", "muon_tight_pt", "muon_loose_pt"):
        category_audit(merged_acc, hn)


if __name__ == "__main__":
    main()