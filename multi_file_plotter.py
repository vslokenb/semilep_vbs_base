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
"""

import os
import re
import glob
import shutil
import inspect
import tempfile

import click
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


def split_to_shards(files, batches, tmpdir, active_ids, verbose=1):
    """
    One pass over the inputs: write per-(batch, file) shards to scratch (only
    for the batches in `active_ids`) and accumulate datasets_metadata. Each
    input file is loaded exactly once and freed before the next. The merged
    datasets_metadata is persisted so a later --reuse-shards run needs no
    re-read of the inputs.
    """
    var_to_batch = {v: b for b, batch in enumerate(batches) for v in batch}
    active_ids = set(active_ids)
    meta = None
    seen_extra = set()

    for i, f in enumerate(files):
        if verbose:
            print(f"  split [{i + 1}/{len(files)}] {os.path.basename(f)} "
                  f"({os.path.getsize(f) / 1e6:.0f} MB)", flush=True)
        out = load(f)

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


def load_batch_from_inputs(files, batch, verbose=1):
    """Re-read the inputs, keeping only this batch's variables (--no-cache)."""
    hist_objs, meta = None, None
    wanted = set(batch)
    for i, f in enumerate(files):
        if verbose > 1:
            print(f"    read [{i + 1}/{len(files)}] {os.path.basename(f)}", flush=True)
        out = load(f)
        payload = {v: h for v, h in out.get("variables", {}).items() if v in wanted}
        if payload:
            hist_objs = payload if hist_objs is None else accumulate([hist_objs, payload])
        dm = out.get("datasets_metadata")
        if dm is not None:
            meta = dm if meta is None else accumulate([meta, dm])
        del out, payload
    return hist_objs, meta


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.argument("inputs", nargs=-1, required=True)
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
def main(inputs, cfg, overwrite_parameters, outputdir, batch_size, tmpdir, no_cache,
         keep_tmp, start_batch, end_batch, reuse_shards, list_batches, workers,
         only_cat, only_syst, only_hist, exclude_hist, split_systematics,
         partial_unc_band, no_syst, log, log_x, density, fmt, overwrite, verbose):
    """Plot data/MC from any number of .coffea INPUTS (files, dirs or globs)."""

    if reuse_shards and no_cache:
        raise click.ClickException("--reuse-shards and --no-cache are mutually exclusive")
    if reuse_shards and not tmpdir:
        raise click.ClickException("--reuse-shards requires --tmpdir pointing at the previous run")

    fmts = list(dict.fromkeys(fmt))  # dedupe, keep order
    if len(fmts) > 1 and not (accepts_kwarg(PlotManager.__init__, "format")
                              or accepts_kwarg(PlotManager.plot_datamc_all, "format")):
        print(f"warning: this PocketCoffea version has no 'format' option; "
              f"saving only its default format instead of {fmts}")
        fmts = fmts[:1]

    files = collect_inputs(inputs)
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
            meta = split_to_shards(files, batches, scratch, active_ids, verbose=verbose)

    # never delete shards we're reusing (needed for a possible re-resume)
    prune_shards = (not no_cache) and (not keep_tmp) and (not reuse_shards)

    # ---- phase 2: merge + plot one batch at a time ----------------------
    try:
        for b in active_ids:
            batch = batches[b]
            print(f"\n[batch {b + 1}/{n_batches}] {', '.join(batch)}", flush=True)

            if no_cache:
                hist_objs, meta_b = load_batch_from_inputs(files, batch, verbose=verbose)
                meta = meta_b if meta is None else meta
            else:
                hist_objs = load_batch_from_shards(scratch, b)

            if not hist_objs:
                print("  nothing to plot, skipping")
                continue

            present = [v for v in batch if v in hist_objs]
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
