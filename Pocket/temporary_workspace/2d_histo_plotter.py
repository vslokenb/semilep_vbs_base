"""
plot_2d_hists.py
----------------
Standalone script to plot 2D histograms from PocketCoffea output files.
Usage:
    python plot_2d_hists.py --input output.coffea --plotdir plots_2d/
    python plot_2d_hists.py --input output.coffea --plotdir plots_2d/ --cat baseline --year 2022
    python plot_2d_hists.py --input output.coffea --plotdir plots_2d/ --log --format pdf
"""

import os
import argparse
import pickle
from collections import defaultdict

import numpy as np
import hist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import mplhep as hep

# ── 2D histograms to plot ────────────────────────────────────────────────────
HISTS_2D = [
    "mT_lep_pt_corr",
    "mT_lep_eta_corr",
    "mT_lep_phi_corr",
    "mT_lep_dphi_corr",
    "mT_MET_corr",
    "mT_METphi_corr",
    "lep_pt_MET_corr",
    "lep_phi_METphi_corr",
]

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_output(path: str) -> dict:
    """Load a PocketCoffea .coffea / .pkl output file."""
    try:
        from coffea.util import load
        return load(path)
    except Exception:
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)


def get_dense_axes(h: hist.Hist):
    """Return only non-categorical axes."""
    return [ax for ax in h.axes
            if not isinstance(ax, (hist.axis.StrCategory, hist.axis.IntCategory))]


def _is_mc(h: hist.Hist) -> bool:
    """Heuristic: MC histograms carry a 'variation' categorical axis; data do not."""
    return any(ax.name == "variation" for ax in h.axes)


def _slice_hist(h: hist.Hist, cat: str, variation: str = "nominal") -> hist.Hist | None:
    """
    Slice a histogram to a single (cat, variation) point, handling both
    MC (has 'variation' axis) and data (has 'era' axis, no 'variation').
    Returns None if the slice is not possible.
    """
    slicing = {"cat": cat}
    if _is_mc(h):
        slicing["variation"] = variation
    else:
        # data: collapse all eras into one
        if any(ax.name == "era" for ax in h.axes):
            slicing["era"] = sum
    try:
        return h[slicing]
    except Exception as e:
        return None


def _axes_match(a: hist.Hist, b: hist.Hist) -> bool:
    """Check that two histograms have identical (name, bins) dense axes."""
    da = [ax for ax in a.axes if not isinstance(ax, (hist.axis.StrCategory, hist.axis.IntCategory))]
    db = [ax for ax in b.axes if not isinstance(ax, (hist.axis.StrCategory, hist.axis.IntCategory))]
    if len(da) != len(db):
        return False
    return all(a.name == b.name and len(a) == len(b) for a, b in zip(da, db))


def sum_samples(h_dict: dict, cat: str, variation: str = "nominal") -> hist.Hist:
    """
    Sum all MC samples in h_dict after slicing to (cat, variation).
    Data samples (no 'variation' axis) are skipped — 2D correlation plots
    are MC-only by design.
    h_dict : {sample: {dataset: hist.Hist}}
    """
    total = None
    for sample, datasets in h_dict.items():
        sample_h = None
        for dataset, h in datasets.items():
            # skip data for 2D MC correlation plots
            if not _is_mc(h):
                continue
            sliced = _slice_hist(h, cat=cat, variation=variation)
            if sliced is None:
                print(f"  [warn] cannot slice {sample}/{dataset} — skipping")
                continue
            if sample_h is None:
                sample_h = sliced
            else:
                if not _axes_match(sample_h, sliced):
                    print(f"  [warn] axis mismatch for {sample}/{dataset} — skipping")
                    continue
                sample_h = sample_h + sliced
        if sample_h is None:
            continue
        if total is None:
            total = sample_h
        else:
            if not _axes_match(total, sample_h):
                print(f"  [warn] axis mismatch summing sample {sample} — skipping")
                continue
            total = total + sample_h
    return total


def available_categories(h_dict: dict) -> list:
    """Infer available 'cat' values from the first histogram found."""
    for sample, datasets in h_dict.items():
        for dataset, h in datasets.items():
            for ax in h.axes:
                if ax.name == "cat":
                    return list(ax)
    return ["baseline"]


def plot_2d(h2d: hist.Hist, var_name: str, cat: str, year: str,
            plot_dir: str, log: bool = False, fmt: str = "png",
            lumi_label: str = "") -> None:
    """
    Render one 2D histogram and save to disk.

    Parameters
    ----------
    h2d       : 2D hist.Hist (categorical axes already sliced away)
    var_name  : histogram key, used for filename and title
    cat       : category string, used for sub-directory and title
    year      : datataking year string
    plot_dir  : base output directory
    log       : use log-z colour scale
    fmt       : file format ('png', 'pdf', …)
    lumi_label: luminosity string shown top-right
    """
    dense = get_dense_axes(h2d)
    if len(dense) != 2:
        print(f"  [skip] {var_name}: expected 2 dense axes, got {len(dense)}")
        return

    xax, yax = dense[0], dense[1]
    values = h2d.values()          # shape (nx, ny)

    # ── Figure ───────────────────────────────────────────────────────────────
    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(8, 7))

    norm = mcolors.LogNorm(vmin=max(values[values > 0].min(), 1e-3),
                           vmax=values.max()) if log else None

    mesh = ax.pcolormesh(
        xax.edges, yax.edges, values.T,
        cmap="viridis",
        norm=norm,
    )

    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label("Events / bin", fontsize=14)

    # ── Labels ───────────────────────────────────────────────────────────────
    ax.set_xlabel(xax.label, fontsize=14)
    ax.set_ylabel(yax.label, fontsize=14)
    ax.tick_params(labelsize=12)

    hep.cms.text("Simulation Preliminary", ax=ax, fontsize=14)
    if lumi_label:
        hep.cms.lumitext(lumi_label, ax=ax, fontsize=13)

    ax.set_title(f"{var_name}  |  cat: {cat}  |  year: {year}",
                 fontsize=11, pad=6)

    # ── Save ─────────────────────────────────────────────────────────────────
    out_dir = os.path.join(plot_dir, year, cat)
    os.makedirs(out_dir, exist_ok=True)
    prefix = "log_" if log else ""
    filepath = os.path.join(out_dir, f"{prefix}{var_name}_{cat}_{year}.{fmt}")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {filepath}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot 2D histograms from PocketCoffea output.")
    parser.add_argument("--input",   required=True,        help="Path to .coffea / .pkl output file")
    parser.add_argument("--plotdir", default="plots_2d",   help="Output directory for plots")
    parser.add_argument("--cat",     nargs="*", default=None, help="Categories to plot (default: all)")
    parser.add_argument("--year",    nargs="*", default=None, help="Years to plot (default: all)")
    parser.add_argument("--log",     action="store_true",  help="Log scale on z-axis")
    parser.add_argument("--format",  default="png",        help="Output format: png, pdf, svg …")
    parser.add_argument("--lumi",    default="",           help="Lumi label e.g. '59.7 fb⁻¹ (13 TeV)'")
    args = parser.parse_args()

    print(f"Loading {args.input} …")
    out = load_output(args.input)

    # PocketCoffea output structure:
    # out["variables"][var_name][sample][dataset] = hist.Hist
    # Some versions store directly out[var_name][sample][dataset]
    variables = out.get("variables", out)

    for var_name in HISTS_2D:
        if var_name not in variables:
            print(f"[skip] {var_name} not found in output")
            continue

        h_dict = variables[var_name]   # {sample: {dataset: hist.Hist}}

        # infer years from the first sample/dataset histogram's axes
        # (PocketCoffea stores a 'year' categorical axis)
        first_h = next(iter(next(iter(h_dict.values())).values()))
        year_axis = [ax for ax in first_h.axes if ax.name == "year"]
        years = (list(year_axis[0]) if year_axis else ["run2"])
        if args.year:
            years = [y for y in years if y in args.year]

        cats = available_categories(h_dict)
        if args.cat:
            cats = [c for c in cats if c in args.cat]

        print(f"\nPlotting {var_name}  |  years={years}  |  cats={cats}")

        for year in years:
            # build per-year h_dict slice if year axis exists
            if year_axis:
                h_dict_year = {
                    s: {d: h[{"year": year}] for d, h in dsets.items()}
                    for s, dsets in h_dict.items()
                }
            else:
                h_dict_year = h_dict

            for cat in cats:
                h2d = sum_samples(h_dict_year, cat=cat, variation="nominal")
                if h2d is None:
                    print(f"  [skip] {var_name}/{cat}/{year}: empty after summing")
                    continue

                dense = get_dense_axes(h2d)
                if len(dense) != 2:
                    print(f"  [skip] {var_name}: not 2D after slicing (got {len(dense)} dense axes)")
                    continue

                plot_2d(
                    h2d,
                    var_name=var_name,
                    cat=cat,
                    year=year,
                    plot_dir=args.plotdir,
                    log=args.log,
                    fmt=args.format,
                    lumi_label=args.lumi,
                )


if __name__ == "__main__":
    main()