"""
plot_2d_hists.py
----------------
Standalone script to plot 2D histograms from PocketCoffea output files.
Produces three sets of plots per histogram / category / year:
  • MC total (sum of all MC samples)
  • Data total (sum of all data eras)
  • Data / MC ratio

Usage:
    python plot_2d_hists.py --input output.coffea --plotdir plots_2d/
    python plot_2d_hists.py --input output.coffea --plotdir plots_2d/ --cat baseline --year 2022
    python plot_2d_hists.py --input output.coffea --plotdir plots_2d/ --log --format pdf
"""

import os
import argparse
import pickle

import numpy as np
import hist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import mplhep as hep

# ── Data sample names ────────────────────────────────────────────────────────
DATA_SAMPLES = {"SingleMuon"}

# ── 2D histograms to plot ────────────────────────────────────────────────────
HISTS_2D = [
    "mT_MET_corr",
    # "mT_lep_eta_corr",
    # "mT_lep_phi_corr",
    # "mT_lep_dphi_corr",
    # "mT_MET_corr",
    # "mT_METphi_corr",
    # "lep_pt_MET_corr",
    # "lep_phi_METphi_corr",
    # "electron_tight_phi_eta",
    "electron_loose_phi_eta",
    "electron_tight_phi_eta"
    "muon_tight_phi_eta",
    "muon_loose_phi_eta",

]

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_output(path: str) -> dict:
    """Load a PocketCoffea .coffea / .pkl output file."""
    try:
        from coffea.util import load
        return load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


def get_dense_axes(h: hist.Hist):
    """Return only non-categorical axes."""
    return [ax for ax in h.axes
            if not isinstance(ax, (hist.axis.StrCategory, hist.axis.IntCategory))]


def _is_data_sample(sample: str) -> bool:
    return sample in DATA_SAMPLES


def _slice_hist(h: hist.Hist, cat: str, is_data: bool, variation: str = "nominal") -> hist.Hist | None:
    """
    Slice a histogram to a single (cat, variation) point.
    For data samples the 'variation' axis is absent; eras are collapsed with sum.
    """
    slicing = {"cat": cat}
    if not is_data:
        slicing["variation"] = variation
    else:
        if any(ax.name == "era" for ax in h.axes):
            slicing["era"] = sum
    try:
        return h[slicing]
    except Exception:
        return None


def _axes_match(a: hist.Hist, b: hist.Hist) -> bool:
    """Check that two histograms have identical (name, bins) dense axes."""
    da = get_dense_axes(a)
    db = get_dense_axes(b)
    if len(da) != len(db):
        return False
    return all(
        ax_a.name == ax_b.name and len(ax_a) == len(ax_b)
        for ax_a, ax_b in zip(da, db)
    )


def _accumulate(total: hist.Hist | None, piece: hist.Hist, label: str) -> hist.Hist | None:
    """Add *piece* into *total*, checking axis compatibility."""
    if total is None:
        return piece
    if not _axes_match(total, piece):
        print(f"  [warn] axis mismatch for {label} — skipping")
        return total
    return total + piece


def sum_mc_samples(h_dict: dict, cat: str, variation: str = "nominal") -> hist.Hist | None:
    """Sum all samples not in DATA_SAMPLES."""
    total = None
    for sample, datasets in h_dict.items():
        if _is_data_sample(sample):
            continue
        sample_h = None
        for dataset, h in datasets.items():
            sliced = _slice_hist(h, cat=cat, is_data=False, variation=variation)
            if sliced is None:
                print(f"  [warn] cannot slice MC {sample}/{dataset} — skipping")
                continue
            sample_h = _accumulate(sample_h, sliced, f"{sample}/{dataset}")
        if sample_h is not None:
            total = _accumulate(total, sample_h, sample)
    return total


def sum_data_samples(h_dict: dict, cat: str) -> hist.Hist | None:
    """Sum all samples in DATA_SAMPLES."""
    total = None
    for sample, datasets in h_dict.items():
        if not _is_data_sample(sample):
            continue
        for dataset, h in datasets.items():
            sliced = _slice_hist(h, cat=cat, is_data=True)
            if sliced is None:
                print(f"  [warn] cannot slice data {sample}/{dataset} — skipping")
                continue
            total = _accumulate(total, sliced, f"{sample}/{dataset}")
    return total


def available_categories(h_dict: dict) -> list:
    """Infer available 'cat' values from the first histogram found."""
    for sample, datasets in h_dict.items():
        for dataset, h in datasets.items():
            for ax in h.axes:
                if ax.name == "cat":
                    return list(ax)
    return ["baseline"]


# ── Plotting ─────────────────────────────────────────────────────────────────

def _save_figure(fig, out_dir: str, filename: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    filepath = os.path.join(out_dir, filename)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {filepath}")


def plot_2d(
    h2d: hist.Hist,
    var_name: str,
    cat: str,
    year: str,
    plot_dir: str,
    kind: str = "mc",
    log: bool = False,
    fmt: str = "png",
    lumi_label: str = "",
) -> None:
    """
    Render one 2D histogram (MC or data) and save to disk.

    Parameters
    ----------
    h2d      : 2D hist.Hist with categorical axes already sliced away
    var_name : histogram key, used for filename and title
    cat      : category string
    year     : data-taking year string
    plot_dir : base output directory
    kind     : 'mc' or 'data', used for sub-directory and title
    log      : log-z colour scale
    fmt      : file format ('png', 'pdf', …)
    lumi_label: luminosity string shown top-right
    """
    dense = get_dense_axes(h2d)
    if len(dense) != 2:
        print(f"  [skip] {var_name} ({kind}): expected 2 dense axes, got {len(dense)}")
        return

    xax, yax = dense[0], dense[1]
    values = h2d.values()

    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(8, 7))

    pos_vals = values[values > 0]
    if pos_vals.size == 0:
        print(f"  [skip] {var_name} ({kind}): all-zero histogram")
        plt.close(fig)
        return

    norm = (
        mcolors.LogNorm(vmin=max(pos_vals.min(), 1e-3), vmax=values.max())
        if log else None
    )

    mesh = ax.pcolormesh(xax.edges, yax.edges, values.T, cmap="viridis", norm=norm)
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label("Events / bin", fontsize=14)

    ax.set_xlabel(xax.label, fontsize=14)
    ax.set_ylabel(yax.label, fontsize=14)
    ax.tick_params(labelsize=12)

    cms_text = "Simulation Preliminary" if kind == "mc" else "Preliminary"
    hep.cms.text(cms_text, ax=ax, fontsize=14)
    if lumi_label:
        hep.cms.lumitext(lumi_label, ax=ax, fontsize=13)

    ax.set_title(
        f"{var_name}  [{kind.upper()}]  |  cat: {cat}  |  year: {year}",
        fontsize=11, pad=6,
    )

    out_dir = os.path.join(plot_dir, year, cat, kind)
    prefix = "log_" if log else ""
    _save_figure(fig, out_dir, f"{prefix}{var_name}_{cat}_{year}_{kind}.{fmt}")


def plot_2d_ratio(
    h_data: hist.Hist,
    h_mc: hist.Hist,
    var_name: str,
    cat: str,
    year: str,
    plot_dir: str,
    fmt: str = "png",
    lumi_label: str = "",
    ratio_range: tuple = (0.5, 1.5),
) -> None:
    """
    Compute and plot the Data / MC ratio for a 2D histogram.

    Bins where MC == 0 are shown as NaN (white).  Bins where data == 0
    but MC > 0 contribute a ratio of 0 (shown in the colour scale).

    Parameters
    ----------
    h_data      : summed data hist.Hist (categorical axes already sliced)
    h_mc        : summed MC   hist.Hist (categorical axes already sliced)
    var_name    : histogram key
    cat         : category string
    year        : data-taking year string
    plot_dir    : base output directory
    fmt         : file format
    lumi_label  : luminosity string shown top-right
    ratio_range : (vmin, vmax) for the colour scale, default (0.5, 1.5)
    """
    if not _axes_match(h_data, h_mc):
        print(f"  [skip] {var_name} ratio: data/MC axis mismatch")
        return

    dense = get_dense_axes(h_mc)
    if len(dense) != 2:
        print(f"  [skip] {var_name} ratio: expected 2 dense axes, got {len(dense)}")
        return

    xax, yax = dense[0], dense[1]
    data_vals = h_data.values()
    mc_vals   = h_mc.values()

    # Ratio: NaN where MC == 0 to avoid misleading colour
    ratio = np.full_like(mc_vals, np.nan, dtype=float)
    mask  = mc_vals > 0
    ratio[mask] = data_vals[mask] / mc_vals[mask]

    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(8, 7))

    vmin, vmax = ratio_range
    mesh = ax.pcolormesh(
        xax.edges, yax.edges, ratio.T,
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
    )

    cbar = fig.colorbar(mesh, ax=ax, pad=0.02, extend="both")
    cbar.set_label("Data / MC", fontsize=14)

    ax.set_xlabel(xax.label, fontsize=14)
    ax.set_ylabel(yax.label, fontsize=14)
    ax.tick_params(labelsize=12)

    hep.cms.text("Preliminary", ax=ax, fontsize=14)
    if lumi_label:
        hep.cms.lumitext(lumi_label, ax=ax, fontsize=13)

    ax.set_title(
        f"{var_name}  [Data/MC]  |  cat: {cat}  |  year: {year}",
        fontsize=11, pad=6,
    )

    out_dir = os.path.join(plot_dir, year, cat, "ratio")
    _save_figure(fig, out_dir, f"ratio_{var_name}_{cat}_{year}.{fmt}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot 2D histograms (MC, data, data/MC ratio) from PocketCoffea output."
    )
    parser.add_argument("--input",       required=True,          help="Path to .coffea / .pkl output file")
    parser.add_argument("--plotdir",     default="plots_2d",     help="Output directory for plots")
    parser.add_argument("--cat",         nargs="*", default=None, help="Categories to plot (default: all)")
    parser.add_argument("--year",        nargs="*", default=None, help="Years to plot (default: all)")
    parser.add_argument("--log",         action="store_true",    help="Log scale on z-axis (MC/data plots)")
    parser.add_argument("--format",      default="png",          help="Output format: png, pdf, svg …")
    parser.add_argument("--lumi",        default="",             help="Lumi label e.g. '59.7 fb⁻¹ (13 TeV)'")
    parser.add_argument("--ratio-range", nargs=2, type=float,
                        default=[0.5, 1.5], metavar=("VMIN", "VMAX"),
                        help="Colour-scale range for the Data/MC ratio plot (default: 0.5 1.5)")
    parser.add_argument("--no-mc",       action="store_true",    help="Skip individual MC plots")
    parser.add_argument("--no-data",     action="store_true",    help="Skip individual data plots")
    parser.add_argument("--no-ratio",    action="store_true",    help="Skip Data/MC ratio plots")
    args = parser.parse_args()

    print(f"Loading {args.input} …")
    out = load_output(args.input)

    # PocketCoffea output structure:
    # out["variables"][var_name][sample][dataset] = hist.Hist
    variables = out.get("variables", out)

    for var_name in HISTS_2D:
        if var_name not in variables:
            print(f"[skip] {var_name} not found in output")
            continue

        h_dict = variables[var_name]   # {sample: {dataset: hist.Hist}}

        # Infer year axis from the first histogram found
        first_h = next(iter(next(iter(h_dict.values())).values()))
        year_axes = [ax for ax in first_h.axes if ax.name == "year"]
        years = list(year_axes[0]) if year_axes else ["run2"]
        if args.year:
            years = [y for y in years if y in args.year]

        cats = available_categories(h_dict)
        if args.cat:
            cats = [c for c in cats if c in args.cat]

        print(f"\nPlotting {var_name}  |  years={years}  |  cats={cats}")

        for year in years:
            # Slice to this year if the axis exists
            if year_axes:
                h_dict_year = {
                    s: {d: h[{"year": year}] for d, h in dsets.items()}
                    for s, dsets in h_dict.items()
                }
            else:
                h_dict_year = h_dict

            for cat in cats:
                h_mc   = sum_mc_samples(h_dict_year, cat=cat, variation="nominal")
                h_data = sum_data_samples(h_dict_year, cat=cat)

                # ── MC plot ───────────────────────────────────────────────
                if not args.no_mc:
                    if h_mc is None:
                        print(f"  [skip] {var_name}/{cat}/{year}: no MC found")
                    else:
                        dense = get_dense_axes(h_mc)
                        if len(dense) != 2:
                            print(f"  [skip] {var_name}: not 2D after slicing (got {len(dense)} dense axes)")
                        else:
                            plot_2d(h_mc, var_name=var_name, cat=cat, year=year,
                                    plot_dir=args.plotdir, kind="mc",
                                    log=args.log, fmt=args.format, lumi_label=args.lumi)

                # ── Data plot ─────────────────────────────────────────────
                if not args.no_data:
                    if h_data is None:
                        print(f"  [skip] {var_name}/{cat}/{year}: no data found")
                    else:
                        dense = get_dense_axes(h_data)
                        if len(dense) != 2:
                            print(f"  [skip] {var_name} (data): not 2D after slicing")
                        else:
                            plot_2d(h_data, var_name=var_name, cat=cat, year=year,
                                    plot_dir=args.plotdir, kind="data",
                                    log=args.log, fmt=args.format, lumi_label=args.lumi)

                # ── Data / MC ratio plot ──────────────────────────────────
                if not args.no_ratio:
                    if h_data is None or h_mc is None:
                        print(f"  [skip] {var_name}/{cat}/{year}: need both data and MC for ratio")
                    else:
                        plot_2d_ratio(
                            h_data, h_mc,
                            var_name=var_name, cat=cat, year=year,
                            plot_dir=args.plotdir,
                            fmt=args.format,
                            lumi_label=args.lumi,
                            ratio_range=tuple(args.ratio_range),
                        )


if __name__ == "__main__":
    main()
