#!/usr/bin/env python
"""
make_plots_multi.py -- `pocket-coffea make-plots` over many .coffea files,
without ever holding the full merged output in memory.

Same behaviour as the standard CLI (same parameters_dump.yaml, same
PlotManager, all scale factors and systematic variations applied), but:

  * takes any number of input files / directories / globs
  * merges and plots a small batch of histograms at a time, dumps those
    plots, frees them, then moves to the next batch

Peak memory is therefore ~(one input file) + (one batch of variables across
all files), not the whole merged accumulator.

By default the inputs are read exactly once: a first pass splits each input
into per-batch shards on scratch, and the plotting loop then reads only the
small shards. Use --no-cache to re-read the inputs for every batch instead
(no scratch space needed, much more I/O).

Each plot is saved once per --format; the default produces both .png and
.pdf (the histograms are read once per batch, only the rendering repeats).

Examples
--------
  ./make_plots_multi.py output_2018_a output_2018_b -o plots

  ./make_plots_multi.py out/*.coffea --cfg out/parameters_dump.yaml \\
      -o plots -b 5 --tmpdir /tmp/$USER/plot_shards

  ./make_plots_multi.py out/ -o plots -oh 'jet.*_pt' -oc vr_mu

  # v28 combine-input layout (V28_COMBINE_HANDOFF.md), one year per run:
  # picks up cr_v28_<year>_g*/output_all.coffea plus the merged per-year
  # nonprompt file, never the raw inverted nonp_*_g* intermediates
  ./make_plots_multi.py --v28 /work/users/vsloken-fnal/semilep_vbs_base/Pocket/temporary_workspace \\
      -y 2017 -o plots_v28_2017
"""

import os
import re
import glob
import shutil
import inspect
import tempfile

import click
import numpy as np
import hist
from omegaconf import OmegaConf

from coffea.util import load, save
from coffea.processor import accumulate

from pocket_coffea.utils.plot_utils import PlotManager
from pocket_coffea.parameters import defaults as parameters_utils


# ----------------------------------------------------------------------
# inputs
# ----------------------------------------------------------------------
def collect_inputs(inputs):
    """Expand files / directories / globs into a sorted list of .coffea files."""
    files = []
    for item in inputs:
        if os.path.isdir(item):
            found = glob.glob(os.path.join(item, "**", "*.coffea"), recursive=True)
            if not found:
                raise click.ClickException(f"No .coffea files found under '{item}'")
            files += found
        elif any(ch in item for ch in "*?["):
            found = glob.glob(item, recursive=True)
            if not found:
                raise click.ClickException(f"Glob '{item}' matched no files")
            files += found
        elif os.path.isfile(item):
            files.append(item)
        else:
            raise click.ClickException(f"Input '{item}' is not a file, directory or glob")

    files = sorted(set(os.path.abspath(f) for f in files))
    if not files:
        raise click.ClickException("No input files resolved")
    return files


def short_name(f):
    """Last two path components -- v28 files are all named output_all.coffea."""
    return os.sep.join(f.rstrip(os.sep).split(os.sep)[-2:])


def collect_v28_inputs(root, year, stage, no_nonprompt):
    """
    Select the v28 inputs for one data-taking period (V28_COMBINE_HANDOFF.md):

      <root>/<stage>_<year>_gNN/output_all.coffea     nominal group passes
      <root>/nonp_<ver>_<year>_*.coffea               merged per-year nonprompt

    Groups are fault-isolation units with no physics meaning; together they
    hold the disjoint set of samples of that year. The raw inverted passes
    (nonp_*_gNN/) only feed the nonprompt merge and are never selected.
    """
    root = os.path.abspath(root)
    pat = re.compile(rf"^{re.escape(stage)}_(.+)_(g\d+)$")
    passes, years_seen, no_output = {}, set(), []
    for d in sorted(glob.glob(os.path.join(root, f"{stage}_*"))):
        m = pat.match(os.path.basename(d))
        if not m or not os.path.isdir(d):
            continue
        y, g = m.groups()
        years_seen.add(y)
        if y != year:
            continue
        f = os.path.join(d, "output_all.coffea")
        if os.path.isfile(f):
            passes[g] = f
        else:
            no_output.append(os.path.basename(d))
    if not years_seen:
        raise click.ClickException(f"No {stage}_<year>_gNN pass directories under {root}")
    if year is None:
        raise click.ClickException(
            "--v28 needs -y/--year (one per run). Years present: "
            + ", ".join(sorted(years_seen)))
    if year not in years_seen:
        raise click.ClickException(
            f"No {stage} passes for '{year}'. Years present: " + ", ".join(sorted(years_seen)))
    for d in no_output:
        print(f"v28: {d}/ has no output_all.coffea (failed pass? see v28_manifest.txt) -- skipped")
    if not passes:
        raise click.ClickException(f"All {stage}_{year}_gNN passes are missing output_all.coffea")
    files = [passes[g] for g in sorted(passes)]
    print(f"v28 {stage} {year}: {len(files)} group pass(es): {', '.join(sorted(passes))}")

    if no_nonprompt:
        return files
    if not stage.startswith("cr_"):
        print(f"v28: stage '{stage}' is not a cr_* stage -- no merged nonprompt file added")
        return files
    # the nonprompt BACKGROUND is the merged per-year file at the workspace
    # root (5 fake-rate variations), not the per-group inverted passes
    nonp_pat = f"nonp_{stage[3:]}_{year}_*.coffea"
    nonp = sorted(f for f in glob.glob(os.path.join(root, nonp_pat)) if os.path.isfile(f))
    if not nonp:
        raise click.ClickException(
            f"No merged nonprompt file matching {nonp_pat} under {root} (the per-group "
            "nonp_*_gNN/ passes are intermediates and are deliberately not used). "
            "Use --no-nonprompt to plot without it, or pass the file as an extra INPUT.")
    if len(nonp) > 1:
        raise click.ClickException(
            f"{len(nonp)} merged nonprompt files match {nonp_pat}: "
            + ", ".join(os.path.basename(f) for f in nonp)
            + ". Remove the stale ones, or use --no-nonprompt and pass the intended "
            "file as an extra INPUT.")
    print(f"v28 nonprompt: {os.path.basename(nonp[0])}")
    return files + nonp


def vet_input(out, src, ds_owner, expect_year):
    """
    v28-mode guards, applied per loaded file:
      * no (sample, dataset) may appear in two inputs -- accumulate would
        silently double count its yields (group passes hold disjoint samples)
      * no file may hold a data-taking period other than the requested year
        (the handoff doc says not to trust directory names)
    """
    periods = set(out.get("datasets_metadata", {}).get("by_datataking_period", {}))
    if periods and periods != {expect_year}:
        raise click.ClickException(
            f"'{short_name(src)}' holds data-taking period(s) {sorted(periods)}, expected "
            f"only '{expect_year}' -- contents disagree with the directory name; "
            "cross-check v28_manifest.csv")
    keys = {(s, d) for hd in out.get("variables", {}).values()
            for s, bd in hd.items() for d in bd}
    clashes = sorted(k for k in keys if k in ds_owner)
    if clashes:
        lines = [f"  {s} / {d}:  {short_name(ds_owner[(s, d)])}  AND  {short_name(src)}"
                 for s, d in clashes[:10]]
        raise click.ClickException(
            f"{len(clashes)} (sample, dataset) pair(s) found in more than one input -- "
            "accumulating them would double count the yields. v28 passes hold disjoint "
            "samples; look for stale or duplicated pass directories / nonprompt files:\n"
            + "\n".join(lines))
    for k in keys:
        ds_owner[k] = src


def filter_variables(variables, only_hist, exclude_hist):
    selected = list(variables)
    if only_hist:
        selected = [v for v in selected if any(re.search(p, v) for p in only_hist)]
    if exclude_hist:
        selected = [v for v in selected if not any(re.search(p, v) for p in exclude_hist)]
    if not selected:
        raise click.ClickException("No variables left after --only-hist / --exclude-hist")
    return sorted(selected)


def batched(seq, n):
    return [seq[i:i + n] for i in range(0, len(seq), n)]


def build_kwargs(func, candidates):
    """
    Keep only the kwargs the installed PocketCoffea actually accepts.

    PlotManager's signature drifts between versions (log vs log_y, presence
    of only_syst / format ...), so filter rather than guess.
    """
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return dict(candidates)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(candidates)
    return {k: v for k, v in candidates.items() if k in params}


def accepts_kwarg(func, name):
    """True if `func` takes `name` as a keyword argument (or has **kwargs)."""
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return True
    return name in params or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


# ----------------------------------------------------------------------
# batch assembly
# ----------------------------------------------------------------------
META_SHARD = "_datasets_metadata.coffea"


def split_to_shards(files, batches, tmpdir, active_ids, verbose=1, vet_v28=None):
    """
    One pass over the inputs: write per-(batch, file) shards to scratch (only
    for the batches in `active_ids`) and accumulate datasets_metadata. Each
    input file is loaded exactly once and freed before the next. The merged
    datasets_metadata is persisted so a later --reuse-shards run needs no
    re-read of the inputs. vet_v28=<year> additionally fails on duplicated
    datasets or year mixing (v28 mode).
    """
    var_to_batch = {v: b for b, batch in enumerate(batches) for v in batch}
    active_ids = set(active_ids)
    meta = None
    seen_extra = set()
    ds_owner = {}

    for i, f in enumerate(files):
        if verbose:
            print(f"  split [{i + 1}/{len(files)}] {short_name(f)} "
                  f"({os.path.getsize(f) / 1e6:.0f} MB)", flush=True)
        out = load(f)
        if vet_v28:
            vet_input(out, f, ds_owner, vet_v28)

        per_batch = {b: {} for b in active_ids}
        for var, obj in out.get("variables", {}).items():
            b = var_to_batch.get(var)
            if b is None:
                seen_extra.add(var)
            elif b in active_ids:
                per_batch[b][var] = obj

        for b, payload in per_batch.items():
            if payload:
                save(payload, os.path.join(tmpdir, f"batch{b:04d}_{i:05d}.coffea"))

        dm = out.get("datasets_metadata")
        if dm is not None:
            meta = dm if meta is None else accumulate([meta, dm])

        del out, per_batch

    if seen_extra and verbose:
        extra = sorted(seen_extra)[:5]
        print(f"  note: {len(seen_extra)} variable(s) present in later files but not the "
              f"first were skipped, e.g. {extra}")
    if meta is None:
        raise click.ClickException("No 'datasets_metadata' found in any input file")
    save(meta, os.path.join(tmpdir, META_SHARD))
    return meta


def load_meta_shard(tmpdir):
    path = os.path.join(tmpdir, META_SHARD)
    return load(path) if os.path.exists(path) else None


def load_batch_from_shards(tmpdir, b):
    hist_objs = None
    for shard in sorted(glob.glob(os.path.join(tmpdir, f"batch{b:04d}_*.coffea"))):
        part = load(shard)
        hist_objs = part if hist_objs is None else accumulate([hist_objs, part])
        del part
    return hist_objs


def load_batch_from_inputs(files, batch, verbose=1, vet_v28=None):
    """Re-read the inputs, keeping only this batch's variables (--no-cache)."""
    hist_objs, meta = None, None
    wanted = set(batch)
    ds_owner = {}
    for i, f in enumerate(files):
        if verbose > 1:
            print(f"    read [{i + 1}/{len(files)}] {short_name(f)}", flush=True)
        out = load(f)
        if vet_v28:
            vet_input(out, f, ds_owner, vet_v28)
        payload = {v: h for v, h in out.get("variables", {}).items() if v in wanted}
        if payload:
            hist_objs = payload if hist_objs is None else accumulate([hist_objs, payload])
        dm = out.get("datasets_metadata")
        if dm is not None:
            meta = dm if meta is None else accumulate([meta, dm])
        del out, payload
    return hist_objs, meta


# ----------------------------------------------------------------------
# variation-axis harmonization
# ----------------------------------------------------------------------
def _axis_key(ax):
    """Comparable identity of an axis: name + categories or bin edges."""
    if isinstance(ax, (hist.axis.StrCategory, hist.axis.IntCategory)):
        return (ax.name, tuple(ax))
    return (ax.name, tuple(np.asarray(ax.edges).tolist()))


def pad_variations(h, target_axis):
    """
    Rebuild `h` with `target_axis` as its variation axis. Rebuilding every
    histogram on the SAME axis object spec makes the axes compare equal
    (labels, order, traits and metadata alike). Variations the input does not
    have are filled with its own nominal (values and variances).
    """
    vi = [i for i, ax in enumerate(h.axes) if ax.name == "variation"][0]
    old = list(h.axes[vi])
    axes = list(h.axes)
    axes[vi] = target_axis
    try:
        storage = h.storage_type()
    except AttributeError:  # older hist/boost-histogram
        storage = h._storage_type()
    new_h = hist.Hist(*axes, storage=storage)
    src = np.moveaxis(h.view(flow=True), vi, 0)
    dst = np.moveaxis(new_h.view(flow=True), vi, 0)
    i_nom = old.index("nominal") if "nominal" in old else 0
    for j, lab in enumerate(target_axis):
        dst[j] = src[old.index(lab) if lab in old else i_nom]
    return new_h


def harmonize_variations(hist_objs, verbose=1):
    """
    Give every histogram of a variable one identical `variation` axis.

    The v28 inputs are inhomogeneous by construction: TT samples carry
    top_pt_reweight (153 entries), other MC has 151, the merged nonprompt only
    its 5 fake-rate entries, data just nominal -- and the StrCategory order can
    differ file to file. hist.Stack.from_dict inside PlotManager requires
    identical axes ("The Histogram axes don't match"). Rebuild every histogram
    on the union of variation labels (nominal first, rest sorted), filling the
    variations a sample does not have with its own nominal -- so it contributes
    its nominal to those variations of the total, the standard convention for
    the systematic band.

    Variables whose OTHER axes (categories, binning) differ across samples
    cannot be harmonized honestly; they are dropped with a warning.
    """
    for var in list(hist_objs):
        entries = [(s, d, h) for s, bd in hist_objs[var].items() for d, h in bd.items()]
        labels, ref, bad = set(), None, None
        for s, d, h in entries:
            if "variation" in [ax.name for ax in h.axes]:
                labels.update(h.axes["variation"])
            key = tuple(_axis_key(ax) for ax in h.axes if ax.name != "variation")
            if ref is None:
                ref = key
            elif key != ref and bad is None:
                bad = (s, d)
        if bad is not None:
            print(f"  WARNING: '{var}' axes differ between samples beyond the variation "
                  f"axis (e.g. {bad[0]}/{bad[1]}) -- cannot be stacked, variable skipped")
            del hist_objs[var]
            continue
        if not labels:
            continue
        target = hist.axis.StrCategory(
            (["nominal"] if "nominal" in labels else []) + sorted(labels - {"nominal"}),
            name="variation", label="variation")
        n_padded = 0
        for s, d, h in entries:
            if "variation" in [ax.name for ax in h.axes]:
                if list(h.axes["variation"]) != list(target):
                    n_padded += 1
                hist_objs[var][s][d] = pad_variations(h, target)
        if verbose > 1:
            print(f"    {var}: variation axis -> {len(target)} entries "
                  f"({n_padded} histogram(s) padded or reordered)")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.argument("inputs", nargs=-1, required=False)
@click.option("--v28", "v28_root", type=str, default=None,
              help="v28 combine-input layout (V28_COMBINE_HANDOFF.md): workspace root "
                   "holding <stage>_<year>_gNN/ pass directories. Selects their "
                   "output_all.coffea plus the merged per-year nonprompt file; the raw "
                   "inverted nonp_*_gNN intermediates are never read. Needs -y; any extra "
                   "INPUTS are added on top of the selection.")
@click.option("-y", "--year", type=str, default=None,
              help="v28 mode: data-taking period to plot (2018, 2017, 2016_PostVFP, "
                   "2016_PreVFP). Exactly one per run -- run once per year.")
@click.option("--stage", type=str, default="cr_v28", show_default=True,
              help="v28 mode: prefix of the nominal pass directories; the merged nonprompt "
                   "prefix is derived from it (cr_X -> nonp_X).")
@click.option("--no-nonprompt", is_flag=True,
              help="v28 mode: do not add the merged per-year nonprompt file.")
@click.option("--cfg", type=str, default=None,
              help="parameters_dump.yaml. Default: found next to the first input.")
@click.option("-op", "--overwrite-parameters", type=str, multiple=True,
              help="Extra YAML file(s) overriding plotting parameters. Repeatable.")
@click.option("-o", "--outputdir", type=str, default="plots", show_default=True,
              help="Output folder for the plots.")
@click.option("-b", "--batch-size", type=int, default=8, show_default=True,
              help="Histograms merged and plotted per batch. Lower it if memory is tight.")
@click.option("--tmpdir", type=str, default=None,
              help="Scratch dir for the per-batch shards. Default: a temp dir, removed at exit.")
@click.option("--no-cache", is_flag=True,
              help="Skip the shard pass and re-read all inputs for every batch "
                   "(no scratch space, much more I/O).")
@click.option("--keep-tmp", is_flag=True, help="Keep the shard directory after finishing.")
@click.option("--start-batch", type=int, default=1, show_default=True,
              help="Resume: first batch to process (1-indexed, matches the [batch N/M] display).")
@click.option("--end-batch", type=int, default=None,
              help="Last batch to process (1-indexed, inclusive). Default: last batch.")
@click.option("--reuse-shards", is_flag=True,
              help="Skip the split pass and reuse shards already in --tmpdir "
                   "(fastest resume; requires a previous run done with --tmpdir).")
@click.option("--list-batches", is_flag=True,
              help="Print the batch -> variables mapping and exit without plotting.")
@click.option("-j", "--workers", type=int, default=8, show_default=True,
              help="Parallel workers used for plotting.")
@click.option("-oc", "--only-cat", type=str, multiple=True,
              help="Only plot these categories. Repeatable.")
@click.option("-os", "--only-syst", type=str, multiple=True,
              help="Restrict the systematic band to these variations. Repeatable. "
                   "Default: all variations in the histograms.")
@click.option("-oh", "--only-hist", type=str, multiple=True,
              help="Only plot variables matching these regexes. Repeatable.")
@click.option("-e", "--exclude-hist", type=str, multiple=True,
              help="Skip variables matching these regexes. Repeatable.")
@click.option("--split-systematics", is_flag=True,
              help="Split the systematics in the ratio panel instead of one combined band.")
@click.option("--partial-unc-band", is_flag=True,
              help="Draw only the band from the variations given with --only-syst.")
@click.option("--no-syst", is_flag=True,
              help="Disable the systematic band (statistical errors only).")
@click.option("--log", "--log-y", "log", is_flag=True, help="Log scale on y.")
@click.option("--log-x", is_flag=True, help="Log scale on x.")
@click.option("--density", is_flag=True, help="Normalise the distributions.")
@click.option("--format", "fmt", type=str, multiple=True, default=("png", "pdf"),
              show_default=True,
              help="Output image format(s); repeatable, each plot is saved once per "
                   "format. Pass a single --format png to skip the pdf copies.")
@click.option("--overwrite", "--over", is_flag=True,
              help="Allow writing into a non-empty output folder.")
@click.option("-v", "--verbose", type=int, default=1, show_default=True,
              help="Verbosity level.")
def main(inputs, v28_root, year, stage, no_nonprompt, cfg, overwrite_parameters,
         outputdir, batch_size, tmpdir, no_cache,
         keep_tmp, start_batch, end_batch, reuse_shards, list_batches, workers,
         only_cat, only_syst, only_hist, exclude_hist, split_systematics,
         partial_unc_band, no_syst, log, log_x, density, fmt, overwrite, verbose):
    """Plot data/MC from .coffea INPUTS (files, dirs or globs) and/or a --v28 workspace."""

    if reuse_shards and no_cache:
        raise click.ClickException("--reuse-shards and --no-cache are mutually exclusive")
    if reuse_shards and not tmpdir:
        raise click.ClickException("--reuse-shards requires --tmpdir pointing at the previous run")
    if not v28_root and (year or no_nonprompt):
        raise click.ClickException("-y/--year and --no-nonprompt only apply to --v28 mode")

    fmts = list(dict.fromkeys(fmt))  # dedupe, keep order
    if len(fmts) > 1 and not (accepts_kwarg(PlotManager.__init__, "format")
                              or accepts_kwarg(PlotManager.plot_datamc_all, "format")):
        print(f"warning: this PocketCoffea version has no 'format' option; "
              f"saving only its default format instead of {fmts}")
        fmts = fmts[:1]

    if v28_root:
        files = collect_v28_inputs(v28_root, year, stage, no_nonprompt)
        if inputs:  # extra files/dirs/globs on top of the v28 selection
            files = sorted(set(files + collect_inputs(inputs)))
    elif inputs:
        files = collect_inputs(inputs)
    else:
        raise click.ClickException("Pass INPUTS, or --v28 <workspace root> with -y <year>")
    print(f"{len(files)} input file(s), "
          f"{sum(os.path.getsize(f) for f in files) / 1e9:.2f} GB total")

    # ---- parameters -----------------------------------------------------
    if cfg is None:
        for cand_dir in (os.path.dirname(files[0]),
                         os.path.dirname(os.path.dirname(files[0]))):
            cand = os.path.join(cand_dir, "parameters_dump.yaml")
            if os.path.exists(cand):
                cfg = cand
                break
    if cfg is None:
        raise click.ClickException(
            "Could not find parameters_dump.yaml automatically; pass it with --cfg")
    if not cfg.endswith((".yaml", ".yml")):
        raise click.ClickException(f"--cfg must be a YAML file, got '{cfg}'")
    print(f"Parameters: {cfg}")

    # keep it as an OmegaConf object: PlotManager does attribute access on
    # style_cfg (e.g. style_cfg.plot_upper_label.by_year[year]), so it must
    # NOT be converted to a plain dict via OmegaConf.to_container.
    parameters = parameters_utils.merge_parameters_from_files(
        OmegaConf.load(cfg), *overwrite_parameters, update=True)
    if "plotting_style" not in parameters:
        raise click.ClickException(f"'plotting_style' missing from {cfg}; supply one with -op")
    style_cfg = parameters["plotting_style"]

    # ---- variable list (from the first input) ---------------------------
    print("Reading variable list from the first input ...")
    first = load(files[0])
    if "variables" not in first:
        raise click.ClickException(
            f"'{files[0]}' has no 'variables' key -- is it a PocketCoffea output?")
    variables = filter_variables(first["variables"].keys(), only_hist, exclude_hist)
    del first

    batches = batched(variables, max(1, batch_size))
    n_batches = len(batches)
    print(f"{len(variables)} variable(s) -> {n_batches} batch(es) of up to {batch_size}")

    # ---- batch range (resume) -------------------------------------------
    if list_batches:
        for b, batch in enumerate(batches):
            print(f"[batch {b + 1}/{n_batches}] {', '.join(batch)}")
        return

    start = max(1, start_batch)
    end = n_batches if end_batch is None else min(end_batch, n_batches)
    if start > n_batches:
        raise click.ClickException(f"--start-batch {start} > number of batches ({n_batches})")
    if end < start:
        raise click.ClickException(f"--end-batch {end} < --start-batch {start}")
    active_ids = list(range(start - 1, end))  # 0-indexed batch indices to process
    if start > 1 or end < n_batches:
        print(f"Processing batches {start}..{end} of {n_batches}")

    # ---- output folder --------------------------------------------------
    if os.path.exists(outputdir) and os.listdir(outputdir) and not overwrite:
        raise click.ClickException(
            f"Output folder '{outputdir}' exists and is not empty. Use --overwrite.")
    os.makedirs(outputdir, exist_ok=True)

    # ---- phase 1: split into shards (single read of the inputs) ---------
    scratch, cleanup = None, False
    meta = None
    if not no_cache:
        if tmpdir:
            scratch = os.path.abspath(tmpdir)
            os.makedirs(scratch, exist_ok=True)
        else:
            scratch = tempfile.mkdtemp(prefix="make_plots_multi_")
            cleanup = not keep_tmp
        if reuse_shards:
            meta = load_meta_shard(scratch)
            if meta is None:
                raise click.ClickException(
                    f"--reuse-shards: no {META_SHARD} in {scratch}. Run once without "
                    "--reuse-shards (using --tmpdir) so the shards + metadata exist.")
            print(f"Reusing existing shards under {scratch} (skipping input read)")
        else:
            print(f"Splitting inputs into per-batch shards under {scratch}")
            meta = split_to_shards(files, batches, scratch, active_ids, verbose=verbose,
                                   vet_v28=year if v28_root else None)

    # never delete shards we're reusing (needed for a possible re-resume)
    prune_shards = (not no_cache) and (not keep_tmp) and (not reuse_shards)

    # ---- phase 2: merge + plot one batch at a time ----------------------
    try:
        for b in active_ids:
            batch = batches[b]
            print(f"\n[batch {b + 1}/{n_batches}] {', '.join(batch)}", flush=True)

            if no_cache:
                hist_objs, meta_b = load_batch_from_inputs(
                    files, batch, verbose=verbose,
                    vet_v28=year if (v28_root and b == active_ids[0]) else None)
                meta = meta_b if meta is None else meta
            else:
                hist_objs = load_batch_from_shards(scratch, b)

            if not hist_objs:
                print("  nothing to plot, skipping")
                continue

            # equalize the variation axes across samples: PlotManager stacks
            # them with hist.Stack, which requires identical axes
            harmonize_variations(hist_objs, verbose=verbose)

            present = [v for v in batch if v in hist_objs]
            if not present:
                print("  nothing left to plot after the axis checks, skipping")
                continue
            # one full plotting pass per format: depending on the PocketCoffea
            # version, 'format' is honoured by PlotManager.__init__ or by
            # plot_datamc_all, so rebuild the manager each time to cover both
            for fmt_i in fmts:
                if verbose and len(fmts) > 1:
                    print(f"  saving as .{fmt_i}", flush=True)
                pm_kwargs = build_kwargs(PlotManager.__init__, dict(
                    variables=present,
                    hist_objs=hist_objs,
                    datasets_metadata=meta,
                    plot_dir=outputdir,
                    style_cfg=style_cfg,
                    only_cat=list(only_cat),
                    only_syst=list(only_syst),
                    workers=workers,
                    log=log,
                    log_y=log,
                    log_x=log_x,
                    density=density,
                    verbose=verbose,
                    format=fmt_i,
                    save=True,
                ))
                plotter = PlotManager(**pm_kwargs)

                plot_kwargs = build_kwargs(plotter.plot_datamc_all, dict(
                    syst=not no_syst,
                    spliteras=False,
                    split_systematics=split_systematics,
                    partial_unc_band=partial_unc_band,
                    format=fmt_i,
                ))
                plotter.plot_datamc_all(**plot_kwargs)
                del plotter

            # free this batch before touching the next one
            del hist_objs
            if prune_shards:
                for shard in glob.glob(os.path.join(scratch, f"batch{b:04d}_*.coffea")):
                    os.remove(shard)
    finally:
        if cleanup and scratch and os.path.isdir(scratch):
            shutil.rmtree(scratch, ignore_errors=True)

    print(f"\nDone. Plots written to {os.path.abspath(outputdir)}")


if __name__ == "__main__":
    main()
