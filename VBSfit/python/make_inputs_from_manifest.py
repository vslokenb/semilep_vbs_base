#!/usr/bin/env python3
"""
Generate VBSfit inputs YAML(s) from the v28 manifest.

The v28 PocketCoffea production is split per year x sample-group
(<stage>_<YEAR>_<GROUP>/output_all.coffea, see V28_COMBINE_HANDOFF.md), with
several samples packed into one file and no physics meaning to the groups.
The ONLY reliable sample -> file mapping is v28_manifest.csv ("do not infer
contents from directory names"), so this script turns the manifest into the
inputs config make_histograms.py et al. consume:

    years:
      "<YEAR>":
        background:
          <sample>: {files: [...], datasets: [...], group: ..., limit_files: [...]}
          nonprompt: {files: [...]}
        signal:    { ... }
        data:      { ... }

One YAML per year (inputs_v28_<year>.yaml): the loaders take the FIRST year
in the file, and the handoff's §4.2 pitfall (sample keys silently stacking
across years) never gets a chance to bite.

Nonprompt: the estimate is PRECOMPUTED and DELIVERED as one merged file per
year at the workspace root, nonp_v28_<YEAR>_<tag>.coffea (handoff §2:
sample key 'nonprompt', dataset key 'nonprompt_<YEAR>', 5 variations --
nominal + muon/electron_inverttight_to_fake Up/Down). The manifest indexes
these as stage '--stage-nonprompt' (default 'nonprompt', group 'merged'),
and that is the PRIMARY source: they are collected like any other sample
and land as the 'nonprompt' background entry. Only for a year the manifest
does not cover does the glob fallback (--nonprompt-dir/--nonprompt-pattern)
kick in, with a loud warning when several tags match (an old and a re-made
file would silently sum to a doubled estimate). The 37 raw
nonp_v28_<YEAR>_gNN/ inverted passes carry a different stage and are never
read. The dataset keys carry the year, so the loaders' dataset-year guard
stays ACTIVE for the nonprompt entry (it catches a config pointed at the
wrong year's file).

Classification per sample:
    signal      'dipoleRecoil' in the sample name (EWK VBS signal)
    data        n_variations == 1 in the manifest (data carries only
                'nominal'; MC carries ~153 variations)
    background  everything else

Extra keys (datasets / group / limit_files) are provenance: the loaders read
only 'files' and ignore the rest. limit_files matters downstream -- the MC
stat precision differs between datasets capped at 50 vs 30 vs 20 input files
(handoff §6), which is autoMCStats territory, so it is carried along where a
human will see it.

Paths: 'file' entries from the manifest are kept as-is when absolute, else
joined to --base-dir. The default base dir is the handoff workspace and is a
PLACEHOLDER -- regenerate with --base-dir (or sed the YAMLs) when the files
move to their final cluster location.

Usage:
    python3 make_inputs_from_manifest.py \
        --manifest v28_manifest.csv \
        --outdir config \
        [--base-dir /path/to/workspace] [--check-files]

Stdlib only -- no pyyaml needed.
"""
import argparse
import csv
import glob
import os
import sys
from collections import defaultdict

SIGNAL_TAG = "dipoleRecoil"


def classify(sample, n_variations):
    if SIGNAL_TAG in sample:
        return "signal"
    if n_variations <= 1:
        return "data"
    return "background"


def resolve_path(path, base_dir):
    return path if os.path.isabs(path) else os.path.join(base_dir, path)


def load_manifest(path):
    required = {"dataset", "sample", "year", "stage", "group", "file",
                "n_variations"}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        missing = required - set(reader.fieldnames or [])
        if missing:
            sys.exit(f"ERROR: {path} lacks columns: {sorted(missing)} "
                     f"(regenerate it with make_v28_manifest.py)")
        return list(reader)


def collect(rows, base_dir, stage):
    """(year, section, sample) -> entry dict, for one stage."""
    out = defaultdict(lambda: {"files": set(), "datasets": [], "groups": set(),
                               "limit_files": set()})
    for r in rows:
        if (r.get("stage") or "").strip() != stage:
            continue
        try:
            n_var = int(r["n_variations"])
        except (ValueError, TypeError):
            n_var = 0
        section = classify(r["sample"], n_var)
        e = out[(r["year"], section, r["sample"])]
        e["files"].add(resolve_path(r["file"], base_dir))
        e["datasets"].append(r["dataset"])
        e["groups"].add(r["group"])
        lf = (r.get("limit_files") or "").strip()
        if lf:
            e["limit_files"].add(lf)
    return out


def yaml_quote(s):
    return '"%s"' % str(s).replace("\\", "\\\\").replace('"', '\\"')


def emit_entries(lines, indent, entries, year, section):
    pad = " " * indent
    # Order by (group, sample): consecutive processes then point at the same
    # output_all.coffea, so make_histograms.py's small LRU file cache
    # (load_coffea_cached / V28_CACHE_SIZE) stays hot instead of thrashing.
    keys = sorted((s for (y, sec, s) in entries if y == year and sec == section),
                  key=lambda s: (min(entries[(year, section, s)]["groups"]), s))
    if not keys:
        lines.append(f"{pad}{{}}")
        return 0
    for sample in keys:
        e = entries[(year, section, sample)]
        lines.append(f"{pad}{sample}:")
        lines.append(f"{pad}  files:")
        for f in sorted(e["files"]):
            lines.append(f"{pad}    - {yaml_quote(f)}")
        lines.append(f"{pad}  datasets: [{', '.join(sorted(e['datasets']))}]")
        lines.append(f"{pad}  group: {', '.join(sorted(e['groups']))}")
        if e["limit_files"]:
            lines.append(f"{pad}  limit_files: "
                         f"[{', '.join(sorted(e['limit_files']))}]")
        if e.get("year_guard") is False:
            # the loaders skip their dataset-year filter for this entry
            lines.append(f"{pad}  year_guard: false")
    return len(keys)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="v28_manifest.csv")
    ap.add_argument("--outdir", default="config")
    ap.add_argument("--base-dir",
                    default="/work/users/vsloken-fnal/semilep_vbs_base/"
                            "Pocket/temporary_workspace",
                    help="prepended to relative manifest paths; PLACEHOLDER "
                         "-- point at the real cluster location when known")
    ap.add_argument("--stage-nominal", default="validation",
                    help="value of the manifest's 'stage' column for the "
                         "nominal selection -- the CONFIG name "
                         "('validation', from config_validation.py), not "
                         "the cr_v28 directory prefix. The inverted passes "
                         "carry 'inverted' and are not read (the merged "
                         "nonprompt files are globbed instead).")
    ap.add_argument("--stage-nonprompt", default="nonprompt",
                    help="manifest 'stage' value of the MERGED per-year "
                         "nonprompt products (group 'merged'); the primary "
                         "source for the nonprompt entry")
    ap.add_argument("--nonprompt-dir", default=None,
                    help="FALLBACK only, for years the manifest lacks: "
                         "directory holding the merged nonprompt files "
                         "(default: --base-dir)")
    ap.add_argument("--nonprompt-pattern", default="nonp_v*_{year}_*.coffea",
                    help="fallback glob; {year} is substituted per year. The "
                         "raw nonp_v28_<year>_gNN pass directories do not "
                         "match this (no .coffea suffix).")
    ap.add_argument("--nonprompt-name", default="nonprompt",
                    help="process name for the entry; must equal the sample "
                         "key inside the nonprompt files")
    ap.add_argument("--check-files", action="store_true",
                    help="warn for every referenced output_all.coffea that "
                         "does not exist on this filesystem")
    args = ap.parse_args()

    rows = load_manifest(args.manifest)
    nominal = collect(rows, args.base_dir, args.stage_nominal)

    years = sorted({y for (y, _, _) in nominal})
    if not years:
        stages = sorted({(r.get("stage") or "").strip() for r in rows})
        sys.exit(
            f"ERROR: no rows with stage == '{args.stage_nominal}' in "
            f"{args.manifest}\n"
            f"       stage values actually present: {stages}\n"
            f"       Rerun with --stage-nominal <the nominal-selection one>.")

    # Merged per-year nonprompt products. PRIMARY source: the manifest rows
    # with stage --stage-nonprompt (sample 'nonprompt', dataset
    # 'nonprompt_<YEAR>', 5 variations -> classified as background like any
    # other MC entry) -- merged straight into the nominal tree.
    nonp = collect(rows, args.base_dir, args.stage_nonprompt)
    for (y, sec, s), e in nonp.items():
        tgt = nominal[(y, sec, s)]
        tgt["files"] |= e["files"]
        tgt["datasets"] += e["datasets"]
        tgt["groups"] |= e["groups"]      # 'merged' -- sorts after gNN
        tgt["limit_files"] |= e["limit_files"]
    np_years = {y for (y, _, _) in nonp}

    # FALLBACK for years the manifest does not cover: glob the merged files.
    np_dir = args.nonprompt_dir or args.base_dir
    for year in years:
        if year in np_years:
            continue
        np_glob = os.path.join(np_dir, args.nonprompt_pattern.format(year=year))
        np_files = sorted(os.path.abspath(f)
                          for f in glob.glob(np_glob) if os.path.isfile(f))
        if not np_files:
            print(f"  [warn] {year}: no stage-'{args.stage_nonprompt}' "
                  f"manifest row and no file matches {np_glob} -- the "
                  f"'{args.nonprompt_name}' entry is OMITTED for this year; "
                  f"regenerate the manifest, or point "
                  f"--nonprompt-dir/--nonprompt-pattern at the merged files",
                  file=sys.stderr)
            continue
        if len(np_files) > 1:
            print(f"  [WARN] {year}: {len(np_files)} nonprompt files match "
                  f"{np_glob}:\n          "
                  + "\n          ".join(np_files)
                  + "\n         They will be SUMMED. If these are alternative "
                  f"versions (old and re-made tags), that silently DOUBLES "
                  f"the nonprompt estimate -- tighten --nonprompt-pattern to "
                  f"the intended tag.", file=sys.stderr)
        e = nominal[(year, "background", args.nonprompt_name)]
        e["files"].update(np_files)
        e["datasets"] = [os.path.basename(f) for f in np_files]
        e["groups"].add("nonprompt")   # sorts after gNN -> emitted last
        # dataset keys are nonprompt_<YEAR> (handoff section 2), so the
        # loaders' dataset-year guard stays active for this entry

    os.makedirs(args.outdir, exist_ok=True)
    all_files = set()

    for year in years:
        lines = [
            f"# Generated by {os.path.basename(sys.argv[0])} from "
            f"{os.path.basename(args.manifest)} -- do not hand-edit;",
            f"# regenerate after any change to the v28 outputs or when the",
            f"# files move (--base-dir). Stage '{args.stage_nominal}' = the",
            f"# nominal selection; 'nonprompt' is the merged per-year estimate",
            f"# (nonp_v28_{year}_<tag>.coffea: sample 'nonprompt', dataset",
            f"# 'nonprompt_{year}', 5 variations), NOT the raw gNN passes.",
            f"# One file per year ON PURPOSE: the loaders take the first",
            f"# year, and v28 files never mix years (handoff section 4.2).",
            "years:",
            f"  {yaml_quote(year)}:",
        ]
        counts = {}
        for section in ("background", "signal", "data"):
            lines.append(f"    {section}:")
            counts[section] = emit_entries(lines, 6, nominal, year, section)
        out = os.path.join(args.outdir, f"inputs_v28_{year}.yaml")
        with open(out, "w") as fh:
            fh.write("\n".join(lines) + "\n")

        all_files |= {f for (y, _, _), e in nominal.items()
                      if y == year for f in e["files"]}

        summary = ", ".join(f"{k}={v}" for k, v in counts.items())
        print(f"wrote {out}  ({summary})")

    if args.check_files:
        missing = sorted(f for f in all_files if not os.path.exists(f))
        for f in missing:
            print(f"  [warn] referenced file not found: {f}")
        if missing:
            print(f"  [warn] {len(missing)}/{len(all_files)} referenced files "
                  f"missing -- cross-check the 'PASSES WITH NO USABLE OUTPUT' "
                  f"section of v28_manifest.txt")

    print("\nReminders (handoff doc):")
    print("  - confirm every (process, year) you need is present: the")
    print("    manifest's 'PASSES WITH NO USABLE OUTPUT' list is definitive")
    print("  - limit_files differs per dataset (50/30/20): MC stats are not")
    print("    uniform -- relevant for autoMCStats")
    print("  - variation axes differ across samples (top_pt_reweight is TT")
    print("    only): select variations explicitly before adding histograms")


if __name__ == "__main__":
    main()
