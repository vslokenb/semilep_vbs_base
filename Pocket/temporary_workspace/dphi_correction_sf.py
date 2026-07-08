"""
Derive per-bin scale factors for the nonpromptDOWN10 contribution.

    SF_i = (data_i - mc_other_i) / nonprompt_i

Scale factors are derived from a reference histogram (default:
dphi_lepton1_DeepMETResolutionTune) in a chosen control region.
Supports folding to |x| and custom bin merging before computing SFs.

Example
-------
    import coffea.util as util
    from nonprompt_scale_factors import derive_nonprompt_scale_factors, plot_scale_factors

    out = util.load("output_merged_debugging_cr_chain_v4_mt_var.coffea")

    centers, sf, sf_err, edges = derive_nonprompt_scale_factors(
        out,
        ref_histogram="dphi_lepton1_DeepMETResolutionTune",
        category="w_cr_mu",
        year="2018",
        abs_x=True,
        custom_bins=[0, 0.5, 1.0, 1.5, 2.0, 3.14159],
    )
    plot_scale_factors(centers, sf, sf_err)
"""

from __future__ import annotations

import json
import numpy as np
import coffea.util as util
from typing import Sequence


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_1d(hists_for_var, sample_key, year, category, variation):
    """
    Slice a histogram to (values, variances, edges) after selecting
    cat=category and variation=variation.  Returns (None, None, None) on miss.

    Sums over ALL sub-keys that contain `year` (e.g. 2018A + 2018B + 2018C + 2018D
    for data, or multiple MC campaign chunks).
    """
    sub = hists_for_var.get(sample_key)
    if sub is None:
        return None, None, None

    year_keys = [k for k in sub if year in k]
    if not year_keys:
        return None, None, None

    total_v = total_var = edges = None
    for year_key in year_keys:
        h = sub[year_key]
        axes_names = [ax.name for ax in h.axes]

        slices = {}
        if "cat" in axes_names:
            slices["cat"] = category
        if "variation" in axes_names:
            available = list(h.axes["variation"])
            slices["variation"] = variation if variation in available else "nominal"

        h = h[slices]
        v, var = h.values(), h.variances()
        e = h.axes[-1].edges

        if total_v is None:
            total_v, total_var, edges = v.copy(), var.copy(), e
        else:
            total_v  += v
            total_var += var

    return total_v, total_var, edges


def _accumulate(hists_for_var, keys, year, category, variation):
    """Sum values and variances over a list of sample keys."""
    total_v = total_var = edges = None
    for key in keys:
        v, var, e = _extract_1d(hists_for_var, key, year, category, variation)
        if v is None:
            continue
        if total_v is None:
            total_v, total_var, edges = v.copy(), var.copy(), e
        else:
            total_v  += v
            total_var += var
    return total_v, total_var, edges


def _fold_abs(vals, variances, edges):
    """
    Fold a symmetric histogram so x -> |x|.

    Bins at -x and +x are summed.  Assumes an even number of bins and
    a range centred on zero (e.g. [-4, 4] with 32 bins).
    """
    n = len(vals)
    if n % 2:
        raise ValueError(
            "abs_x=True requires an even number of bins "
            f"(symmetric range around 0). Got {n} bins."
        )
    half = n // 2
    # positive half: bins[half:]  | negative half: bins[:half]
    # mirror of positive bin i  -> negative bin (half-1-i)
    folded_v   = vals[half:]       + vals[:half][::-1]
    folded_var = variances[half:]  + variances[:half][::-1]
    folded_edges = edges[half:]    # 0 .. max
    return folded_v, folded_var, folded_edges


def _rebin(vals, variances, old_edges, new_edges):
    """
    Merge bins from old_edges into new_edges.

    Assignment is by LEFT edge: old bin i is placed into new bin j when
    old_edges[i] falls in [new_edges[j], new_edges[j+1]).  This is robust
    for new edges that do not coincide with old edges (e.g. np.pi sits
    between old edges 3.0 and 3.25 — the old bin whose left edge is 3.0
    is correctly included, while the old bin whose left edge is 3.25 is not).
    """
    new_v   = np.zeros(len(new_edges) - 1)
    new_var = np.zeros(len(new_edges) - 1)
    old_lo  = old_edges[:-1]          # left edge of every old bin
    for i, (lo, hi) in enumerate(zip(new_edges[:-1], new_edges[1:])):
        mask = (old_lo >= lo - 1e-10) & (old_lo < hi - 1e-10)
        new_v[i]   = vals[mask].sum()
        new_var[i] = variances[mask].sum()
    return new_v, new_var


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def derive_nonprompt_scale_factors(
    out: dict,
    ref_histogram: str = "dphi_lepton1_DeepMETResolutionTune",
    category: str = "w_cr_mu",
    variation: str = "nominal",
    year: str = "2018",
    data_patterns: Sequence[str] = ("SingleMuon", "EGamma"),
    nonprompt_sample: str = "nonpromptDOWN10",
    custom_bins: Sequence[float] | None = None,
    abs_x: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Derive per-bin scale factors for the nonprompt contribution.

        SF_i = (data_i - mc_other_i) / nonprompt_i

    Parameters
    ----------
    out : dict
        Coffea output loaded via coffea.util.load.
    ref_histogram : str
        Key in out['variables'] to use as reference for deriving SFs.
    category : str
        Category label to select on the 'cat' axis (e.g. 'w_cr_mu').
    variation : str
        Systematic variation to use; 'nominal' for central value.
    year : str
        Sub-string used to match per-year dataset keys (e.g. '2018').
    data_patterns : sequence of str
        Sub-strings identifying data sample keys (SingleMuon, EGamma, …).
    nonprompt_sample : str
        Sample key for the nonprompt estimate to be scaled.
    custom_bins : sequence of float or None
        New bin edges for rebinning.  Must coincide with boundaries of the
        original binning.  If abs_x=True, specify edges in |x| space (>= 0).
    abs_x : bool
        Fold the histogram so x -> |x| before computing SFs.
        Assumes the original binning is symmetric around 0.

    Returns
    -------
    bin_centers : np.ndarray, shape (N,)
    scale_factors : np.ndarray, shape (N,)
    sf_uncertainties : np.ndarray, shape (N,)
    bin_edges : np.ndarray, shape (N+1,)
    arrays : dict
        Processed per-bin arrays (after abs_x / rebinning) used for plotting:
        data_v, data_var, mc_v, mc_var, np_v, np_var, clipped.
    """
    hists = out["variables"][ref_histogram]
    all_samples = list(hists.keys())

    data_keys      = [s for s in all_samples if any(p in s for p in data_patterns)]
    mc_other_keys  = [s for s in all_samples
                      if s not in data_keys and s != nonprompt_sample]

    # ---- diagnostic: show every sample and all its sub-keys ----------------
    print(f"Reference histogram  : {ref_histogram}")
    print(f"Category / variation : {category!r} / {variation!r}  year={year}")
    print(f"\nAll samples in histogram (outer key → year sub-keys):")
    for s in all_samples:
        sub_keys = list(hists[s].keys())
        year_sub = [k for k in sub_keys if year in k]
        role = ("DATA" if s in data_keys
                else "NONPROMPT" if s == nonprompt_sample
                else "MC")
        print(f"  [{role:9s}] {s!r:35s}  sub-keys: {sub_keys}")
        if not year_sub:
            print(f"             *** no sub-key containing {year!r} — will be SKIPPED ***")
    print()
    print(f"Data samples ({len(data_keys):2d})   : {data_keys}")
    print(f"Nonprompt sample     : {nonprompt_sample}")
    print(f"MC other ({len(mc_other_keys):2d} samples) : {mc_other_keys}")

    args = (year, category, variation)
    data_v,  data_var,  edges = _accumulate(hists, data_keys,     *args)
    mc_v,    mc_var,    _     = _accumulate(hists, mc_other_keys,  *args)
    np_v,    np_var,    _     = _extract_1d(hists, nonprompt_sample, *args)

    if data_v is None:
        raise ValueError(f"No data samples found matching patterns {data_patterns}")
    if np_v is None:
        raise ValueError(
            f"Nonprompt sample '{nonprompt_sample}' not found "
            f"or has no '{year}' key."
        )

    if mc_v is None:
        mc_v   = np.zeros_like(data_v)
        mc_var = np.zeros_like(data_v)

    # ---- fold to |x| -------------------------------------------------------
    if abs_x:
        orig_edges = edges.copy()
        data_v, data_var, edges = _fold_abs(data_v, data_var, orig_edges)
        mc_v,   mc_var,   _     = _fold_abs(mc_v,   mc_var,   orig_edges)
        np_v,   np_var,   _     = _fold_abs(np_v,   np_var,   orig_edges)

    # ---- custom rebinning --------------------------------------------------
    if custom_bins is not None:
        new_edges = np.asarray(custom_bins, dtype=float)
        data_v, data_var = _rebin(data_v, data_var, edges, new_edges)
        mc_v,   mc_var   = _rebin(mc_v,   mc_var,   edges, new_edges)
        np_v,   np_var   = _rebin(np_v,   np_var,   edges, new_edges)
        edges = new_edges

    # ---- scale factors -----------------------------------------------------
    numerator = data_v - mc_v

    # Mask bins where the SF is ill-defined or unphysical:
    #   • nonprompt <= 0  → zero or negative yield (unphysical), default to 1
    #   • data - mc < 0   → would give negative SF, default to 1
    clipped = (np_v <= 0) | (numerator < 0)

    with np.errstate(invalid="ignore", divide="ignore"):
        sf = np.where(clipped, 1.0, numerator / np_v)

        # σ²_SF = (σ²_data + σ²_mc) / NP²  +  SF² · σ²_NP / NP²
        # Zero uncertainty assigned to clipped (defaulted) bins.
        sf_var = np.where(
            clipped,
            0.0,
            (data_var + mc_var) / np_v**2 + sf**2 * np_var / np_v**2,
        )

    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    sf_err = np.sqrt(sf_var)

    # ---- summary -----------------------------------------------------------
    print("\nPer-bin scale factors:")
    print(f"  {'center':>10}  {'SF':>10}  {'±err':>10}  {'NP yield':>12}  {'data-mc':>12}  note")
    for c, s, e, n, nm, cl in zip(bin_centers, sf, sf_err, np_v, numerator, clipped):
        note = ""
        if n == 0:
            note = "[NP=0 → SF=1]"
        elif n < 0:
            note = "[NP<0 → SF=1]"
        elif nm < 0:
            note = "[data-MC<0 → SF=1]"
        print(f"  {c:10.4f}  {s:10.4f}  {e:10.4f}  {n:12.1f}  {nm:12.1f}  {note}")

    # Bundle processed arrays so callers can plot without re-deriving
    arrays = dict(
        data_v=data_v,   data_var=data_var,
        mc_v=mc_v,       mc_var=mc_var,
        np_v=np_v,       np_var=np_var,
        clipped=clipped,
    )

    return bin_centers, sf, sf_err, edges, arrays


def apply_scale_factors_to_histogram(
    out: dict,
    histogram: str,
    scale_factors: np.ndarray,
    sf_edges: np.ndarray,
    nonprompt_sample: str = "nonpromptDOWN10",
    year: str = "2018",
    abs_x: bool = False,
) -> dict[str, np.ndarray]:
    """
    Apply per-bin SFs (derived from the reference dphi histogram) back to the
    nonprompt contribution of *any* histogram in out['variables'].

    Because the SFs are binned in the reference observable, they are applied
    by matching each bin of the *target* histogram's observable axis to the
    nearest SF bin centre.  This is exact when target == reference histogram.

    Returns a dict  {category: scaled_values}  over all categories,
    for the 'nominal' variation of the given year.
    """
    hists  = out["variables"][histogram]
    sub    = hists.get(nonprompt_sample)
    if sub is None:
        raise KeyError(f"'{nonprompt_sample}' not in out['variables']['{histogram}']")

    year_key = next((k for k in sub if year in k), None)
    if year_key is None:
        raise KeyError(f"No '{year}' key in {list(sub.keys())}")

    h = sub[year_key]
    sf_centers = 0.5 * (sf_edges[:-1] + sf_edges[1:])
    results = {}

    for cat in h.axes["cat"]:
        h_slice = h[{"cat": cat, "variation": "nominal"}]
        obs_edges   = h_slice.axes[-1].edges
        obs_centers = 0.5 * (obs_edges[:-1] + obs_edges[1:])
        lookup      = np.abs(obs_centers) if abs_x else obs_centers

        # nearest-neighbour SF lookup
        idx = np.searchsorted(sf_centers, lookup).clip(0, len(sf_centers) - 1)
        sf_per_bin = scale_factors[idx]

        vals = h_slice.values()
        results[cat] = vals * sf_per_bin

    return results


def plot_scale_factors(
    bin_centers: np.ndarray,
    scale_factors: np.ndarray,
    sf_uncertainties: np.ndarray,
    bin_edges: np.ndarray | None = None,
    title: str = "nonpromptDOWN10 scale factors",
    xlabel: str = r"$|\Delta\phi_{\ell,\,\mathrm{MET}}|$",
    ax=None,
):
    """
    Plot the derived per-bin scale factors with error bars.

    Parameters
    ----------
    ax : matplotlib Axes or None
        If None a new figure is created.

    Returns
    -------
    fig, ax
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure

    xerr = None
    if bin_edges is not None:
        xerr = np.array([
            bin_centers - bin_edges[:-1],
            bin_edges[1:]  - bin_centers,
        ])

    ax.errorbar(
        bin_centers, scale_factors,
        xerr=xerr, yerr=sf_uncertainties,
        fmt="o", color="black", capsize=3, label="SF",
    )
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Scale factor  (data - MC other) / nonprompt")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_comparison(
    bin_centers: np.ndarray,
    sf: np.ndarray,
    bin_edges: np.ndarray,
    arrays: dict,
    title: str = "nonpromptDOWN10 correction",
    xlabel: str = r"$|\Delta\phi_{\ell,\,\mathrm{MET}}|$",
):
    """
    Two-column comparison plot: input histogram (before SF) and output
    histogram (after SF applied to nonprompt), each with a Data/MC ratio panel.

    Parameters
    ----------
    bin_centers, sf, bin_edges
        Returned directly from derive_nonprompt_scale_factors.
    arrays : dict
        The 5th return value of derive_nonprompt_scale_factors.
    title : str
        Overall figure title.
    xlabel : str
        x-axis label for the observable.

    Returns
    -------
    fig, axes  (2-row × 2-col)
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    data_v   = arrays["data_v"]
    data_var = arrays["data_var"]
    mc_v     = arrays["mc_v"]
    mc_var   = arrays["mc_var"]
    np_v     = arrays["np_v"]
    np_var   = arrays["np_var"]
    clipped  = arrays["clipped"]

    widths = np.diff(bin_edges)
    data_err = np.sqrt(data_var)

    # Scaled nonprompt
    np_v_scaled   = np_v   * sf
    np_var_scaled = np_var * sf**2   # variance scales as SF²

    def _draw_panel(ax_top, ax_bot, np_vals, np_errs, panel_title):
        total      = mc_v + np_vals
        total_var  = mc_var + np_errs
        total_err  = np.sqrt(total_var)

        # --- stacked bar chart ---
        ax_top.bar(
            bin_centers, mc_v, width=widths, align="center",
            color="#4C72B0", alpha=0.85, label="MC other",
        )
        ax_top.bar(
            bin_centers, np_vals, width=widths, align="center",
            bottom=mc_v, color="#DD8452", alpha=0.85, label="nonprompt",
        )
        # MC stat uncertainty band
        ax_top.bar(
            bin_centers, 2 * total_err, width=widths, align="center",
            bottom=total - total_err,
            color="none", edgecolor="black", hatch="///", linewidth=0,
            alpha=0.4,
        )
        # Data points
        ax_top.errorbar(
            bin_centers, data_v, yerr=data_err,
            fmt="ko", markersize=5, capsize=3, zorder=5, label="Data",
        )

        # Mark clipped bins with a subtle vertical line
        for xc, cl in zip(bin_centers, clipped):
            if cl:
                ax_top.axvline(xc, color="red", linewidth=0.8,
                               linestyle=":", alpha=0.6)

        ax_top.set_title(panel_title, fontsize=11)
        ax_top.set_ylabel("Events")
        ax_top.set_xlim(bin_edges[0], bin_edges[-1])
        ax_top.set_ylim(bottom=0)

        legend_extras = [Patch(facecolor="none", edgecolor="black",
                               hatch="///", label="MC stat. unc.")]
        handles, labels = ax_top.get_legend_handles_labels()
        ax_top.legend(handles + legend_extras, labels + ["MC stat. unc."],
                      fontsize=8, ncol=2)

        # --- ratio panel ---
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio     = np.where(total > 0, data_v   / total,     np.nan)
            ratio_err = np.where(total > 0, data_err / total,     np.nan)
            mc_rel    = np.where(total > 0, total_err / total,    np.nan)

        # MC uncertainty band in ratio
        ax_bot.bar(
            bin_centers, 2 * mc_rel, width=widths, align="center",
            bottom=1 - mc_rel,
            color="gray", alpha=0.35,
        )
        ax_bot.errorbar(
            bin_centers, ratio, yerr=ratio_err,
            fmt="ko", markersize=5, capsize=3,
        )
        ax_bot.axhline(1.0, color="gray", linestyle="--", linewidth=0.9)
        ax_bot.set_ylabel("Data / MC")
        ax_bot.set_xlabel(xlabel)
        ax_bot.set_xlim(bin_edges[0], bin_edges[-1])
        ax_bot.set_ylim(0.5, 1.5)

    # --- figure layout ---
    fig, axes = plt.subplots(
        2, 2,
        figsize=(13, 7),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08, "wspace": 0.28},
        sharex=False,
    )

    _draw_panel(axes[0, 0], axes[1, 0], np_v,        np_var,        "Before SF")
    _draw_panel(axes[0, 1], axes[1, 1], np_v_scaled, np_var_scaled, "After SF")

    # Remove redundant x-tick labels on top panels
    for ax in axes[0]:
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)

    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    return fig, axes


def dump_correctionlib(
    bin_edges: np.ndarray,
    sf: np.ndarray,
    sf_err: np.ndarray,
    output_path: str = "nonprompt_sf.json",
    name: str = "nonprompt_sf",
    description: str = "",
    obs_name: str = "abs_dphi",
    obs_description: str = r"|Δφ(lepton, MET)|",
    year: str = "2018",
    category: str = "",
    ref_histogram: str = "",
    nonprompt_sample: str = "",
):
    """
    Write (or update) per-bin nonprompt scale factors in a correctionlib v2 JSON.

    Correction inputs (in order):
        year       : string  — "2016" | "2017" | "2018" | …
        systematic : string  — "nominal" | "up" | "down"
        <obs_name> : real    — observable value (e.g. |Δφ|)

    If *output_path* already exists the function loads it and
    adds/replaces only the entry for *year*, leaving other years intact.
    Run once per year and the same file accumulates all of them.

    Up/down = SF ± σ_SF clipped to [0, ∞).
    Clipped bins (SF forced to 1, σ=0) are identical across all variations.
    Out-of-range observable values are clamped to the nearest edge bin.

    Parameters
    ----------
    bin_edges : np.ndarray, shape (N+1,)
    sf        : np.ndarray, shape (N,)   nominal scale factors
    sf_err    : np.ndarray, shape (N,)   absolute uncertainties
    output_path : str   destination .json file
    name        : str   correction name (machine-readable key in the JSON)
    description : str   human-readable description; auto-built if empty
    obs_name    : str   name for the observable input axis
    obs_description : str
    year        : str   data-taking year (becomes a functional input axis)
    category, ref_histogram, nonprompt_sample : str
        Provenance metadata stored in the description string.
    """
    import os

    if not description:
        parts = [f"category={category}"] if category else []
        if ref_histogram:    parts.append(f"ref={ref_histogram}")
        if nonprompt_sample: parts.append(f"sample={nonprompt_sample}")
        description = (
            "Per-bin nonprompt scale factors SF=(data-mc_other)/nonprompt "
            "evaluated per year. " + (", ".join(parts) if parts else "")
        )

    sf_up   = np.maximum(sf + sf_err, 0.0)
    sf_down = np.maximum(sf - sf_err, 0.0)

    def _binning_node(values):
        return {
            "nodetype": "binning",
            "input": obs_name,
            "edges":   [round(float(e), 10) for e in bin_edges],
            "content": [round(float(v), 10) for v in values],
            "flow": "clamp",
        }

    def _systematic_node():
        return {
            "nodetype": "category",
            "input": "systematic",
            "content": [
                {"key": "nominal", "value": _binning_node(sf)},
                {"key": "up",      "value": _binning_node(sf_up)},
                {"key": "down",    "value": _binning_node(sf_down)},
            ],
        }

    # ---- load existing schema or create a fresh one -----------------------
    if os.path.exists(output_path):
        with open(output_path) as fh:
            schema = json.load(fh)

        # Find existing correction by name
        corr = next((c for c in schema["corrections"] if c["name"] == name), None)

        if corr is None:
            # Different correction name — append a brand-new one
            schema["corrections"].append(_new_correction())
        else:
            # Update or insert the year entry
            year_content = corr["data"]["content"]
            existing = next((e for e in year_content if e["key"] == year), None)
            if existing:
                existing["value"] = _systematic_node()
                print(f"  Replaced existing year={year!r} entry in {output_path!r}")
            else:
                year_content.append({"key": year, "value": _systematic_node()})
                print(f"  Added year={year!r} entry to {output_path!r}")
    else:
        corr = None   # signal to build fresh below

    if corr is None:
        # Build a brand-new schema (first run, or new correction name)
        schema = {
            "schema_version": 2,
            "corrections": [
                {
                    "name": name,
                    "description": description,
                    "version": 1,
                    "inputs": [
                        {
                            "name": "year",
                            "type": "string",
                            "description": "Data-taking year (e.g. '2016', '2017', '2018')",
                        },
                        {
                            "name": "systematic",
                            "type": "string",
                            "description": "Systematic variation: nominal / up / down",
                        },
                        {
                            "name": obs_name,
                            "type": "real",
                            "description": obs_description,
                        },
                    ],
                    "output": {
                        "name": "weight",
                        "type": "real",
                        "description": "Multiplicative SF for the nonprompt contribution",
                    },
                    "data": {
                        "nodetype": "category",
                        "input": "year",
                        "content": [
                            {"key": year, "value": _systematic_node()},
                        ],
                    },
                }
            ],
        }

    with open(output_path, "w") as fh:
        json.dump(schema, fh, indent=2)

    # ---- summary -----------------------------------------------------------
    all_years = [e["key"] for e in
                 next(c for c in schema["corrections"] if c["name"] == name)
                 ["data"]["content"]]

    print(f"\nCorrectionlib JSON: {output_path}")
    print(f"  Correction name : {name!r}")
    print(f"  Years in file   : {all_years}")
    print(f"  Written year    : {year!r}  ({len(sf)} bins)")
    print(f"  Edges           : {list(np.round(bin_edges, 4))}")
    print(f"  Nominal SFs     : {list(np.round(sf,      4))}")
    print(f"  Up   SFs        : {list(np.round(sf_up,   4))}")
    print(f"  Down SFs        : {list(np.round(sf_down, 4))}")

    # ---- correctionlib round-trip validation --------------------------------
    try:
        import correctionlib
        cset  = correctionlib.CorrectionSet.from_file(output_path)
        corr_ = cset[name]
        centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        for c, expected in zip(centres, sf):
            result = corr_.evaluate(year, "nominal", float(c))
            assert abs(result - expected) < 1e-6, \
                f"Mismatch at centre={c}: got {result}, expected {expected}"
        print("  correctionlib round-trip validation: PASSED")
    except ImportError:
        print("  (correctionlib not installed — skipping round-trip validation)")
    except Exception as exc:
        print(f"  correctionlib validation WARNING: {exc}")


# ---------------------------------------------------------------------------
# CLI / quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, os

    coffea_file = sys.argv[1] if len(sys.argv) > 1 else \
        "debugging_cr_chain_v5_KILL_LO/output_merged_debugging_cr_chain_v5_KILL_LO.coffea"

    if not os.path.exists(coffea_file):
        print(f"File not found: {coffea_file}")
        sys.exit(1)

    out = util.load(coffea_file)

    # --- example: fold to |dphi|, merge to 6 bins --------------------------
    centers, sf, sf_err, edges, arrays = derive_nonprompt_scale_factors(
        out,
        ref_histogram="dphi_lepton1_DeepMETResolutionTune",
        category="w_cr_sb_lo1_mu",
        year="2018",
        abs_x=True,
        custom_bins=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, np.pi],
    )

    fig_sf, _ = plot_scale_factors(centers, sf, sf_err, edges)
    fig_sf.savefig("nonprompt_sf.png", dpi=150)
    print("Saved nonprompt_sf.png")

    fig_cmp, _ = plot_comparison(centers, sf, edges, arrays)
    fig_cmp.savefig("nonprompt_comparison.png", dpi=150)
    print("Saved nonprompt_comparison.png")

    dump_correctionlib(
        edges, sf, sf_err,
        output_path="nonprompt_sf_lo1_mu.json",   # same file for all years
        name="nonprompt_sf",
        year="2018",                        # change per run
        category="w_cr_sb_lo1_mu",
        ref_histogram="dphi_lepton1_DeepMETResolutionTune",
        nonprompt_sample="nonpromptDOWN10",
    )