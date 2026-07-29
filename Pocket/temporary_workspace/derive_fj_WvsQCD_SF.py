#!/usr/bin/env python3
"""
Derive WvsQCD and tau21 fatjet tagger scale factors.

Part 1 (original): WvsQCD SFs in two msoftdrop bins, inclusive tau21.
Part 2 (new):      WvsQCD SFs in two msoftdrop bins after tau21 < 0.45 cut.
Part 3 (new):      tau21 SFs in two coarse tau21 bins [0, 0.45, 1.0].

For WvsQCD SFs (Parts 1 & 2), a 2x2 linear system per WvsQCD score bin:
    SF_unm * N_unm_TT + SF_mat * N_mat_TT = N_data_TT
    SF_unm * N_unm_Z  + SF_mat * N_mat_Z  = N_data_Z

For tau21 SFs (Part 3): simple ratio after normalising MC to data,
    SF[tau21_bin] = N_data[tau21_bin] / N_MC_norm[tau21_bin]
summed over both categories and all WvsQCD bins.

Output: three correctionlib-compatible JSON files.

Usage:
    python derive_fj_WvsQCD_SF.py \\
        --mc  output_boosted_control/output_MC.coffea \\
        --data output_boosted_control/output_EGamma*.coffea \\
               output_boosted_control/output_Muon*.coffea \\
        --output           output_boosted_control/fj_WvsQCD_SF.json \\
        --output-tau21cut  output_boosted_control/fj_WvsQCD_SF_tau21cut.json \\
        --output-tau21sf   output_boosted_control/fj_tau21_SF.json
"""
import argparse
import glob
import json

import cloudpickle as pickle
import hist
import lz4.frame
import numpy as np

# ── msd bins ──────────────────────────────────────────────────────────────────
MSD_BINS  = [(40., 80.), (80., 200.)]
MSD_EDGES = [40., 80., 200.]

# ── tau21 coarse bins for SF ──────────────────────────────────────────────────
T21_COARSE_EDGES = [0.0, 0.45, 1.0]
T21_CUT = 0.45   # upper tau21 cut for Part 2

# ── Systematic groups ─────────────────────────────────────────────────────────
_BTAG_SUBS = ["cferr1", "cferr2", "hf", "hfstats1", "hfstats2", "lf", "lfstats1", "lfstats2"]
SYST_GROUPS = {
    "pileup":              ([("pileupUp",                     "pileupDown")],                   "nominal"),
    "sf_btag":             ([(f"sf_btag_{s}Up", f"sf_btag_{s}Down") for s in _BTAG_SUBS],      "nominal"),
    "sf_partonshower_isr": ([("sf_partonshower_isrUp",        "sf_partonshower_isrDown")],      "nominal"),
    "sf_partonshower_fsr": ([("sf_partonshower_fsrUp",        "sf_partonshower_fsrDown")],      "nominal"),
    "JES":                 ([("AK8PFPuppi_JES_TotalUp",       "AK8PFPuppi_JES_TotalDown")],   "nominal"),
    "JER":                 ([("AK8PFPuppi_JERUp",             "AK8PFPuppi_JERDown")],          "nominal"),
}
CATS = ["TT_boosted", "zjets_boosted"]


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_coffea(path):
    with lz4.frame.open(path, "rb") as f:
        return pickle.load(f)


def sum_hist_over_datasets(coffea, key):
    total = None
    for ds_dict in coffea["variables"][key].values():
        for h in ds_dict.values():
            total = h.copy() if total is None else total + h
    return total


def integrate_kinematics(h, obj, msd_lo, msd_hi, tau21_bin=None):
    """
    Integrate over pt and abseta; select msoftdrop in [msd_lo, msd_hi].
    If the histogram has a tau21 axis and tau21_bin is None, integrate over it.
    If tau21_bin is an integer, select that single coarse tau21 bin index.
    """
    msd_ax   = f"{obj}.msoftdrop"
    tau21_ax = f"{obj}.tau21"
    ax_names = {ax.name for ax in h.axes}

    actions = {
        f"{obj}.pt":     sum,
        f"{obj}.abseta": sum,
        msd_ax:          slice(hist.loc(msd_lo), hist.loc(msd_hi), sum),
    }
    if tau21_ax in ax_names:
        if tau21_bin is not None:
            actions[tau21_ax] = slice(tau21_bin, tau21_bin + 1, sum)
        else:
            actions[tau21_ax] = sum
    return h[actions]


def cat_slice(h, cat_name, var_name="nominal"):
    sliced = h[{"cat": hist.loc(cat_name), "variation": hist.loc(var_name)}]
    return sliced.values(), sliced.variances()


# ── 2×2 WvsQCD solver ────────────────────────────────────────────────────────

def solve_sf(A, d, sig_A, sig_d):
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    if abs(det) < 1e-10:
        return np.full(2, np.nan), np.full(2, np.nan)
    Ainv = np.linalg.inv(A)
    SF = Ainv @ d
    V = np.zeros(2)
    for k in range(2):
        for m in range(2):
            V[k] += Ainv[k, m] ** 2 * sig_d[m] ** 2
        for i in range(2):
            for j in range(2):
                V[k] += (Ainv[k, i] * SF[j]) ** 2 * sig_A[i, j] ** 2
    return SF, np.sqrt(V)


def extract_and_solve(h_unm, h_mat, h_data, n, mc_var, data_var):
    A_val = np.zeros((2, n, 2))
    A_var = np.zeros((2, n, 2))
    d_val = np.zeros((2, n))
    d_var = np.zeros((2, n))

    for ci, cat in enumerate(CATS):
        A_val[ci, :, 0], A_var[ci, :, 0] = cat_slice(h_unm,  cat, mc_var)
        A_val[ci, :, 1], A_var[ci, :, 1] = cat_slice(h_mat,  cat, mc_var)
        d_val[ci, :],    d_var[ci, :]     = cat_slice(h_data, cat, data_var)

    for ci in range(2):
        total_mc   = np.sum(A_val[ci, :, 0] + A_val[ci, :, 1])
        total_data = np.sum(d_val[ci, :])
        if total_mc > 0:
            s = total_data / total_mc
            A_val[ci] *= s
            A_var[ci] *= s ** 2

    SF_unm = np.zeros(n); unc_unm = np.zeros(n)
    SF_mat = np.zeros(n); unc_mat = np.zeros(n)
    for i in range(n):
        SF, sigma = solve_sf(
            A_val[:, i, :], d_val[:, i],
            np.sqrt(np.clip(A_var[:, i, :], 0, None)),
            np.sqrt(np.clip(d_var[:, i],    0, None)),
        )
        SF_unm[i], SF_mat[i]   = SF
        unc_unm[i], unc_mat[i] = sigma
    return SF_unm, unc_unm, SF_mat, unc_mat


# ── WvsQCD correctionlib helpers ──────────────────────────────────────────────

def binning_node(edges, values):
    return {
        "nodetype": "binning",
        "input": "WvsQCD",
        "edges":   [float(e) for e in edges],
        "content": [float(v) for v in values],
        "flow": "clamp",
    }


def msd_node(msd_edges, score_edges, sf_per_msd):
    return {
        "nodetype": "binning",
        "input": "msoftdrop",
        "edges":   [float(e) for e in msd_edges],
        "content": [binning_node(score_edges, sf_per_msd[k])
                    for k in range(len(sf_per_msd))],
        "flow": "clamp",
    }


def fjtype_node(msd_edges, score_edges, sf_unm_per_msd, sf_mat_per_msd):
    return {
        "nodetype": "category",
        "input": "fjtype",
        "content": [
            {"key": "unmatched", "value": msd_node(msd_edges, score_edges, sf_unm_per_msd)},
            {"key": "matched",   "value": msd_node(msd_edges, score_edges, sf_mat_per_msd)},
        ],
    }


def build_wvsqcd_json(json_content, msd_edges, msd_bins):
    all_systs = ["stat"] + list(SYST_GROUPS.keys())
    syst_desc = " / ".join(f"{s}_up / {s}_down" for s in all_systs)
    msd_desc  = ", ".join(f"[{lo},{hi}]" for lo, hi in msd_bins)
    return {
        "name": "fj_WvsQCD_SF",
        "version": 2,
        "inputs": [
            {"name": "systematic", "type": "string",
             "description": f"Variation key: nominal / {syst_desc}"},
            {"name": "fjtype",     "type": "string",
             "description": "matched or unmatched"},
            {"name": "msoftdrop",  "type": "real",
             "description": f"Soft-drop mass (GeV); msd bins: {msd_desc}"},
            {"name": "WvsQCD",     "type": "real",
             "description": "particleNetWithMass_WvsQCD tagger score"},
        ],
        "output": {"name": "weight", "type": "real",
                   "description": "Per-event scale factor weight"},
        "data": {"nodetype": "category", "input": "systematic",
                 "content": json_content},
    }


# ── tau21 SF helpers ──────────────────────────────────────────────────────────

def _ax_name(h, keyword):
    return next(ax.name for ax in h.axes if keyword in ax.name)


def derive_tau21_sf_2x2(h_mat, h_unm, h_dat, mc_var="nominal", dat_var="nominal"):
    """Derive tau21 SFs for matched and unmatched jets via 2×2 system per coarse bin.
    Integrates all WvsQCD bins; solves per coarse tau21 bin [0,0.45,1.0].
    Returns (SF_unm, unc_unm, SF_mat, unc_mat) each shaped (n_coarse,)."""
    wax_mat = _ax_name(h_mat, "particleNet")
    wax_unm = _ax_name(h_unm, "particleNet")
    wax_dat = _ax_name(h_dat, "particleNet")

    h_mat_1d = h_mat[{wax_mat: sum}]
    h_unm_1d = h_unm[{wax_unm: sum}]
    h_dat_1d = h_dat[{wax_dat: sum}]

    fine_edges_mc  = h_mat.axes[_ax_name(h_mat, "tau21")].edges
    fine_edges_dat = h_dat.axes[_ax_name(h_dat, "tau21")].edges
    n_coarse = len(T21_COARSE_EDGES) - 1

    A_val = np.zeros((2, n_coarse, 2))  # [cat, t21_bin, fjtype(0=unm,1=mat)]
    A_var = np.zeros((2, n_coarse, 2))
    d_val = np.zeros((2, n_coarse))
    d_var = np.zeros((2, n_coarse))

    for ci, cat in enumerate(CATS):
        mat_v, mat_vr = cat_slice(h_mat_1d, cat, mc_var)
        unm_v, unm_vr = cat_slice(h_unm_1d, cat, mc_var)
        dat_v, dat_vr = cat_slice(h_dat_1d, cat, dat_var)
        for b, (lo, hi) in enumerate(zip(T21_COARSE_EDGES[:-1], T21_COARSE_EDGES[1:])):
            mc_m  = (fine_edges_mc[:-1]  >= lo - 1e-9) & (fine_edges_mc[1:]  <= hi + 1e-9)
            dat_m = (fine_edges_dat[:-1] >= lo - 1e-9) & (fine_edges_dat[1:] <= hi + 1e-9)
            A_val[ci, b, 0] = unm_v[mc_m].sum()
            A_var[ci, b, 0] = np.clip(unm_vr, 0, None)[mc_m].sum()
            A_val[ci, b, 1] = mat_v[mc_m].sum()
            A_var[ci, b, 1] = np.clip(mat_vr, 0, None)[mc_m].sum()
            d_val[ci, b]    = dat_v[dat_m].sum()
            d_var[ci, b]    = np.clip(dat_vr, 0, None)[dat_m].sum()

    # Normalise MC to data per category (over all tau21 bins)
    for ci in range(2):
        total_mc = A_val[ci].sum()
        total_dat = d_val[ci].sum()
        if total_mc > 0:
            s = total_dat / total_mc
            A_val[ci] *= s
            A_var[ci] *= s ** 2

    SF_unm = np.zeros(n_coarse); unc_unm = np.zeros(n_coarse)
    SF_mat = np.zeros(n_coarse); unc_mat = np.zeros(n_coarse)
    for b in range(n_coarse):
        SF, sigma = solve_sf(
            A_val[:, b, :], d_val[:, b],
            np.sqrt(A_var[:, b, :]), np.sqrt(d_var[:, b]),
        )
        SF_unm[b], SF_mat[b]   = SF
        unc_unm[b], unc_mat[b] = sigma
    return SF_unm, unc_unm, SF_mat, unc_mat


def tau21_binning_node(coarse_edges, sf_values):
    return {
        "nodetype": "binning",
        "input": "tau21",
        "edges":   [float(e) for e in coarse_edges],
        "content": [float(v) for v in sf_values],
        "flow": "clamp",
    }


def tau21_fjtype_node(coarse_edges, sf_unm, sf_mat):
    return {
        "nodetype": "category",
        "input": "fjtype",
        "content": [
            {"key": "unmatched", "value": tau21_binning_node(coarse_edges, sf_unm)},
            {"key": "matched",   "value": tau21_binning_node(coarse_edges, sf_mat)},
        ],
    }


def build_tau21_json(t21_content):
    all_systs = ["stat"] + list(SYST_GROUPS.keys())
    syst_desc = " / ".join(f"{s}_up / {s}_down" for s in all_systs)
    return {
        "name": "fj_tau21_SF",
        "version": 2,
        "inputs": [
            {"name": "systematic", "type": "string",
             "description": f"Variation key: nominal / {syst_desc}"},
            {"name": "fjtype", "type": "string",
             "description": "matched or unmatched"},
            {"name": "tau21", "type": "real",
             "description": "fatjet tau21; coarse bins [0, 0.45, 1.0]"},
        ],
        "output": {"name": "weight", "type": "real",
                   "description": "Per-event tau21 scale factor weight"},
        "data": {"nodetype": "category", "input": "systematic",
                 "content": t21_content},
    }


# ── WvsQCD SF driver ──────────────────────────────────────────────────────────

def run_wvsqcd_sf(h_mat_full, h_unm_full, h_data_full, tau21_bin_data,
                  tau21_bin_mc=None, label=""):
    """Derive WvsQCD SF in MSD_BINS.
    tau21_bin_data: bin index for data tau21 selection (None = all).
    tau21_bin_mc:   same for MC (None = all); should equal tau21_bin_data for
                    a same-phase-space derivation."""
    h_mat_msd  = [integrate_kinematics(h_mat_full,  "candidate_boost_matched",   lo, hi,
                                        tau21_bin=tau21_bin_mc)
                  for lo, hi in MSD_BINS]
    h_unm_msd  = [integrate_kinematics(h_unm_full,  "candidate_boost_unmatched", lo, hi,
                                        tau21_bin=tau21_bin_mc)
                  for lo, hi in MSD_BINS]
    h_data_msd = [integrate_kinematics(h_data_full, "candidate_boost",           lo, hi,
                                        tau21_bin=tau21_bin_data)
                  for lo, hi in MSD_BINS]

    score_ax = next(a for a in h_mat_msd[0].axes if a.name not in ("cat", "variation"))
    edges = score_ax.edges
    n     = len(edges) - 1
    n_msd = len(MSD_BINS)

    SF_unm_nom  = []; SF_mat_nom  = []
    unc_unm_stat = []; unc_mat_stat = []
    for k, (msd_lo, msd_hi) in enumerate(MSD_BINS):
        tag = f"{label}msd [{msd_lo}, {msd_hi}] GeV"
        print(f"  Solving nominal for {tag}...")
        u, su, m, sm = extract_and_solve(
            h_unm_msd[k], h_mat_msd[k], h_data_msd[k], n, "nominal", "nominal")
        SF_unm_nom.append(u);  unc_unm_stat.append(su)
        SF_mat_nom.append(m);  unc_mat_stat.append(sm)
        print(f"\n    {'Score bin':<18}  {'SF_unm':>8} ± {'stat':>7}    {'SF_mat':>8} ± {'stat':>7}")
        print("    " + "-" * 65)
        for i in range(n):
            lo_, hi_ = edges[i], edges[i+1]
            print(f"    [{lo_:.3f},{hi_:.3f}]  {u[i]:8.4f} ± {su[i]:7.4f}"
                  f"    {m[i]:8.4f} ± {sm[i]:7.4f}")

    json_content = [
        {"key": "nominal",
         "value": fjtype_node(MSD_EDGES, edges, SF_unm_nom, SF_mat_nom)},
        {"key": "stat_up",
         "value": fjtype_node(MSD_EDGES, edges,
                              [SF_unm_nom[k] + unc_unm_stat[k] for k in range(n_msd)],
                              [SF_mat_nom[k] + unc_mat_stat[k] for k in range(n_msd)])},
        {"key": "stat_down",
         "value": fjtype_node(MSD_EDGES, edges,
                              [SF_unm_nom[k] - unc_unm_stat[k] for k in range(n_msd)],
                              [SF_mat_nom[k] - unc_mat_stat[k] for k in range(n_msd)])},
    ]

    for syst_name, (components, data_spec) in SYST_GROUPS.items():
        print(f"  Solving {syst_name}...")
        sigma_sq_unm = [np.zeros(n) for _ in range(n_msd)]
        sigma_sq_mat = [np.zeros(n) for _ in range(n_msd)]
        for mc_up, mc_dn in components:
            data_up = data_spec[0] if isinstance(data_spec, tuple) else data_spec
            data_dn = data_spec[1] if isinstance(data_spec, tuple) else data_spec
            for k in range(n_msd):
                SF_u_unm, _, SF_u_mat, _ = extract_and_solve(
                    h_unm_msd[k], h_mat_msd[k], h_data_msd[k], n, mc_up, data_up)
                SF_d_unm, _, SF_d_mat, _ = extract_and_solve(
                    h_unm_msd[k], h_mat_msd[k], h_data_msd[k], n, mc_dn, data_dn)
                sigma_sq_unm[k] += ((SF_u_unm - SF_d_unm) / 2) ** 2
                sigma_sq_mat[k] += ((SF_u_mat - SF_d_mat) / 2) ** 2
        sigma_unm = [np.sqrt(sigma_sq_unm[k]) for k in range(n_msd)]
        sigma_mat = [np.sqrt(sigma_sq_mat[k]) for k in range(n_msd)]
        json_content += [
            {"key": f"{syst_name}_up",
             "value": fjtype_node(MSD_EDGES, edges,
                                  [SF_unm_nom[k] + sigma_unm[k] for k in range(n_msd)],
                                  [SF_mat_nom[k] + sigma_mat[k] for k in range(n_msd)])},
            {"key": f"{syst_name}_down",
             "value": fjtype_node(MSD_EDGES, edges,
                                  [SF_unm_nom[k] - sigma_unm[k] for k in range(n_msd)],
                                  [SF_mat_nom[k] - sigma_mat[k] for k in range(n_msd)])},
        ]

    return json_content, edges


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mc",              required=True)
    parser.add_argument("--data",            nargs="+", required=True)
    parser.add_argument("--output",          default="output_boosted_control/fj_WvsQCD_SF.json")
    parser.add_argument("--output-tau21cut", default="output_boosted_control/fj_WvsQCD_SF_tau21cut.json")
    parser.add_argument("--output-tau21sf",  default="output_boosted_control/fj_tau21_SF.json")
    args = parser.parse_args()

    print(f"Loading MC: {args.mc}")
    mc = load_coffea(args.mc)

    data_paths = []
    for pat in args.data:
        expanded = glob.glob(pat)
        data_paths.extend(expanded if expanded else [pat])
    print(f"Loading {len(data_paths)} data files...")
    data_coffeas = [load_coffea(p) for p in sorted(data_paths)]

    print("Summing histograms over datasets...")
    h_mat_full  = sum_hist_over_datasets(mc, "fj_matched_WvsQCD_pt_eta")
    h_unm_full  = sum_hist_over_datasets(mc, "fj_unmatched_WvsQCD_pt_eta")
    h_data_full = None
    for dc in data_coffeas:
        h = sum_hist_over_datasets(dc, "fj_WvsQCD_pt_eta")
        h_data_full = h.copy() if h_data_full is None else h_data_full + h

    # ── Part 1: WvsQCD SF, inclusive tau21 ───────────────────────────────────
    print("\n══ Part 1: WvsQCD SF — inclusive tau21 ══")
    json_content_incl, score_edges = run_wvsqcd_sf(
        h_mat_full, h_unm_full, h_data_full,
        tau21_bin_data=None, label="incl. tau21, ")

    correction_incl = build_wvsqcd_json(json_content_incl, MSD_EDGES, MSD_BINS)
    out = {"schema_version": 2, "corrections": [correction_incl]}
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {len(json_content_incl)} keys to {args.output}")

    # ── Part 2: WvsQCD SF, tau21 < 0.45 cut on both data and MC ─────────────
    print("\n══ Part 2: WvsQCD SF — tau21 < 0.45 cut ══")
    # Both MC and data use coarse tau21 [0, 0.45, 1]; bin index 0 = [0, 0.45]
    json_content_cut, _ = run_wvsqcd_sf(
        h_mat_full, h_unm_full, h_data_full,
        tau21_bin_data=0, tau21_bin_mc=0, label="tau21<0.45, ")

    correction_cut = build_wvsqcd_json(json_content_cut, MSD_EDGES, MSD_BINS)
    correction_cut["name"] = "fj_WvsQCD_SF_tau21cut"
    correction_cut["inputs"][0]["description"] += " (tau21 < 0.45 selection applied to data and MC)"
    out2 = {"schema_version": 2, "corrections": [correction_cut]}
    with open(args.output_tau21cut, "w") as f:
        json.dump(out2, f, indent=2)
    print(f"\nSaved {len(json_content_cut)} keys to {args.output_tau21cut}")

    # ── Part 3: tau21 SF — 2×2 per coarse tau21 bin ──────────────────────────
    print("\n══ Part 3: tau21 SF — coarse bins [0, 0.45, 1.0], 2×2 system ══")
    h_mat_t21 = sum_hist_over_datasets(mc, "fj_matched_WvsQCD_tau21")
    h_unm_t21 = sum_hist_over_datasets(mc, "fj_unmatched_WvsQCD_tau21")
    h_dat_t21 = None
    for dc in data_coffeas:
        h = sum_hist_over_datasets(dc, "fj_WvsQCD_tau21")
        h_dat_t21 = h.copy() if h_dat_t21 is None else h_dat_t21 + h

    # Nominal
    SF_unm_nom, unc_unm_stat, SF_mat_nom, unc_mat_stat = \
        derive_tau21_sf_2x2(h_mat_t21, h_unm_t21, h_dat_t21)
    print(f"\n  tau21 bin   | SF_unm ± stat    SF_mat ± stat")
    print("  ------------+-------------------------------")
    for k, (lo, hi) in enumerate(zip(T21_COARSE_EDGES[:-1], T21_COARSE_EDGES[1:])):
        print(f"  [{lo:.2f},{hi:.2f}] | "
              f"{SF_unm_nom[k]:.4f} ± {unc_unm_stat[k]:.4f}    "
              f"{SF_mat_nom[k]:.4f} ± {unc_mat_stat[k]:.4f}")

    t21_content = [
        {"key": "nominal",
         "value": tau21_fjtype_node(T21_COARSE_EDGES, SF_unm_nom, SF_mat_nom)},
        {"key": "stat_up",
         "value": tau21_fjtype_node(T21_COARSE_EDGES,
                                    SF_unm_nom + unc_unm_stat, SF_mat_nom + unc_mat_stat)},
        {"key": "stat_down",
         "value": tau21_fjtype_node(T21_COARSE_EDGES,
                                    SF_unm_nom - unc_unm_stat, SF_mat_nom - unc_mat_stat)},
    ]

    for syst_name, (components, _) in SYST_GROUPS.items():
        print(f"  Solving tau21 {syst_name}...")
        sig_sq_unm = np.zeros(len(T21_COARSE_EDGES) - 1)
        sig_sq_mat = np.zeros(len(T21_COARSE_EDGES) - 1)
        for mc_up, mc_dn in components:
            u_up, _, m_up, _ = derive_tau21_sf_2x2(h_mat_t21, h_unm_t21, h_dat_t21, mc_up)
            u_dn, _, m_dn, _ = derive_tau21_sf_2x2(h_mat_t21, h_unm_t21, h_dat_t21, mc_dn)
            sig_sq_unm += ((u_up - u_dn) / 2) ** 2
            sig_sq_mat += ((m_up - m_dn) / 2) ** 2
        sig_unm = np.sqrt(sig_sq_unm)
        sig_mat = np.sqrt(sig_sq_mat)
        t21_content += [
            {"key": f"{syst_name}_up",
             "value": tau21_fjtype_node(T21_COARSE_EDGES,
                                        SF_unm_nom + sig_unm, SF_mat_nom + sig_mat)},
            {"key": f"{syst_name}_down",
             "value": tau21_fjtype_node(T21_COARSE_EDGES,
                                        SF_unm_nom - sig_unm, SF_mat_nom - sig_mat)},
        ]

    tau21_json = build_tau21_json(t21_content)
    out3 = {"schema_version": 2, "corrections": [tau21_json]}
    with open(args.output_tau21sf, "w") as f:
        json.dump(out3, f, indent=2)
    print(f"\nSaved {len(t21_content)} keys to {args.output_tau21sf}")


if __name__ == "__main__":
    main()
