#!/usr/bin/env python3
"""
update_datasets.py — Refresh dataset JSON files using dataset-discovery-cli.

For each JSON file in DATASET_FILES:
  1. Reads existing das_names and preserves xsec values.
  2. For MC, derives DAS wildcard patterns for any of the 4 Run2 UL eras that
     are missing from the JSON (substitutes campaign/conditions tags).
  3. Drives dataset-discovery-cli interactively via pexpect for each DAS name.
  4. Merges all era results into the original JSON file with xsec re-injected.

Requirements:
    pip install pexpect
    dataset-discovery-cli must be on PATH (activate the pocket_coffea venv first)

Usage:
    python update_datasets.py [--site T3_US_FNALLPC] [--dry-run] [--only FILENAME]
"""

import os
import ast
import re
import json
import argparse
import tempfile
from pathlib import Path

try:
    import pexpect
    HAS_PEXPECT = True
except ImportError:
    HAS_PEXPECT = False

WORKSPACE = Path(__file__).parent
DATASETS_DIR = WORKSPACE / "datasets"

# Run2 UL MC campaign → (NanoAOD campaign tag, conditions tag)
MC_CAMPAIGNS = {
    "2016_PreVFP":  ("RunIISummer20UL16NanoAODAPVv9", "106X_mcRun2_asymptotic_preVFP_v11"),
    "2016_PostVFP": ("RunIISummer20UL16NanoAODv9",    "106X_mcRun2_asymptotic_v17"),
    "2017":         ("RunIISummer20UL17NanoAODv9",     "106X_mc2017_realistic_v9"),
    "2018":         ("RunIISummer20UL18NanoAODv9",     "106X_upgrade2018_realistic_v16_L1v1"),
}

# All dataset JSON files to update (DY: NLO-only; all signal and minor processes included)
DATASET_FILES = [
    # ── DY (NLO only) ──────────────────────────────────────────────────────────
    "DYJetsToLL_M-10to50_TuneCP5_13TeV-amcatnloFXFX-pythia8.json",
    "DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8.json",

    # ── W+jets (HT-binned LO) ──────────────────────────────────────────────────
    "WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8.json",
    "WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8.json",
    "WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8.json",
    "WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8.json",
    "WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8.json",
    "WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8.json",
    "WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8.json",
    "WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8.json",


    "WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8.json",
    "WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8.json",

    # ── TTbar ──────────────────────────────────────────────────────────────────
    "TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8.json",
    "TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8.json",

    # ── Single top ─────────────────────────────────────────────────────────────
    "ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8.json",
    "ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8.json",
    "ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8.json",
    "ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8.json",
    "ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8.json",

    # ── tt+V ───────────────────────────────────────────────────────────────────
    "ttWJets_TuneCP5_13TeV_madgraphMLM_pythia8.json",
    "ttZJets_TuneCP5_13TeV_madgraphMLM_pythia8.json",

    # ── Diboson / triboson / other minor backgrounds ───────────────────────────
    "GluGluWWToLNuQQ_TuneCP5_13TeV_madgraph-pythia8.json",
    "WGToLNuG_TuneCP5_13TeV-madgraphMLM-pythia8.json",
    "WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8.json",
    "WWZ_4F_TuneCP5_13TeV-amcatnlo-pythia8.json",
    "WZTo3LNu_mllmin01_NNPDF31_TuneCP5_13TeV_powheg_pythia8.json",
    "WZZ_TuneCP5_13TeV-amcatnlo-pythia8.json",
    "ZGToLLG_01J_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8.json",
    "ZZZ_TuneCP5_13TeV-amcatnlo-pythia8.json",

    # ── EWK VBS signal ─────────────────────────────────────────────────────────
    "WplusTo2JWminusToLNuJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
    "WplusToLNuWminusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
    "WminusToLNuWminusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
    "WplusToLNuWplusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
    "WminusToLNuZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
    "WplusToLNuZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
    "WminusTo2JZTo2LJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
    "WplusTo2JZTo2LJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
    "ZTo2LZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",

    # ── QCD VBS backgrounds ────────────────────────────────────────────────────
    "WminusTo2JZTo2LJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
    "WminusToLNuWminusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
    "WminusToLNuZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
    "WplusTo2JWminusToLNuJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
    "WplusTo2JZTo2LJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
    "WplusToLNuWminusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
    "WplusToLNuWplusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
    "WplusToLNuZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",
    "ZTo2LZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8.json",

    # ── Data ───────────────────────────────────────────────────────────────────
    # SingleMuon: 2016preVFP + 2016postVFP + 2017 + 2018
    "SingleMuon.json",
    # SingleElectron: 2016preVFP + 2016postVFP + 2017  (replaced by EGamma in 2018)
    "SingleElectron.json",
    # EGamma: 2018 only
    "EGamma.json",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_das_names(raw):
    """Return a list from the stringified-list das_names metadata field."""
    if not raw:
        return []
    raw = raw.strip()
    if raw.startswith("["):
        try:
            return ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            pass
    return [raw.strip("'\"")] if raw else []


def detect_mc_year(das_name):
    """Return the year string embedded in a MC DAS name, or None for data."""
    for year, (campaign, _) in MC_CAMPAIGNS.items():
        if campaign in das_name:
            return year
    return None


def detect_year_from_das(das_name):
    """
    Return the pocket_coffea year string for any DAS path (MC or data).
    MC:   matched via campaign tag in MC_CAMPAIGNS.
    Data: matched via Run20XX era string; HIPM suffix → 2016_PreVFP,
          plain Run2016 → 2016_PostVFP.
    """
    year = detect_mc_year(das_name)
    if year:
        return year
    if "Run2016" in das_name:
        return "2016_PreVFP" if "HIPM" in das_name else "2016_PostVFP"
    if "Run2017" in das_name:
        return "2017"
    if "Run2018" in das_name:
        return "2018"
    return None


def populate_year(discovery):
    """
    Set metadata['year'] for every file entry in a discovery/replicas JSON
    by reading the year from the entry's das_names field.
    Checks both file_entry["das_names"] and file_entry["metadata"]["das_names"].
    """
    for entry in discovery.values():
        if not isinstance(entry, dict):
            continue
        for file_entry in entry.get("files", []):
            if not isinstance(file_entry, dict) or "metadata" not in file_entry:
                continue
            metadata = file_entry["metadata"]
            # das_names can live at the file_entry level or inside metadata
            raw = file_entry.get("das_names") or metadata.get("das_names", [])
            das_list = parse_das_names(raw) if isinstance(raw, str) else (raw or [])
            for das in das_list:
                year = detect_year_from_das(das)
                if year:
                    metadata["year"] = year
                    break


def derive_missing_das(existing_das, source_year, target_year):
    """
    Build a DAS wildcard pattern for target_year from an existing source_year path.
    Returns None if the existing path doesn't contain the expected campaign tag.
    The version suffix (-v1, -v2 …) is replaced with * so DAS finds any version.
    """
    src_campaign, src_cond = MC_CAMPAIGNS[source_year]
    tgt_campaign, tgt_cond = MC_CAMPAIGNS[target_year]
    if src_campaign not in existing_das:
        return None
    das = existing_das.replace(src_campaign, tgt_campaign)
    das = das.replace(src_cond, tgt_cond)
    # Wildcard the version number so small version differences don't block results
    das = re.sub(r"-v\d+/", "-*/", das)
    return das


# ── dataset-discovery-cli driver ──────────────────────────────────────────────

CLI_COMMAND = "dataset-discovery-cli"
PEXPECT_TIMEOUT = 180  # seconds per DAS query

BLOCKED_SITES = [
    "T2_KR_KRISTI",
]

# The REPL shows "> " on its own line, then the command list on the next line.
# Match the simple "> " prompt — it's the first thing printed and appears after
# every command completes. The command list line is decorative and may contain
# ANSI codes that make regex matching unreliable.
REPL_PROMPT   = r"> "
# Sub-prompts inside each command
P_QUERY_FOR   = r"Query for:\s*"
P_SELECT_IDX  = r"Select datasets indices.*\(all\):\s*"
P_SELECT_SITE = r"Select sites \[round-robin.*\].*:\s*"
P_SAVE        = r"(?i)(save to|output file|filename).*:\s*"
P_BLOCK_SITES = r"Exclude the sites.*:\s*"


def _send(child, text, debug=False):
    if debug:
        print(f"  >>> SEND: {text!r}")
    child.sendline(text)


def _wait(child, pattern, label, debug=False):
    if debug:
        print(f"  ... EXPECT [{label}]: {pattern!r}")
    child.expect(pattern)
    if debug:
        # Show both the decoded string and raw hex so ANSI codes are visible
        before_hex = child.before.encode("utf-8", errors="replace").hex()
        after_hex  = child.after.encode("utf-8", errors="replace").hex()
        print(f"  <<< BEFORE: {child.before!r}")
        print(f"  <<< BEFORE (hex): {before_hex}")
        print(f"  <<< MATCH:  {child.after!r}")
        print(f"  <<< MATCH  (hex): {after_hex}")


def _query_one(child, das_pattern, debug=False):
    """
    Within an already-open CLI session, accumulate one era via
    query → select → replicas.  Does NOT save.
    Returns True if datasets were found.
    """
    print(f"    Querying: {das_pattern}")

    _send(child, "query", debug)
    _wait(child, P_QUERY_FOR, "Query for:", debug)

    _send(child, das_pattern, debug)
    _wait(child, REPL_PROMPT, "REPL after query", debug)

    if re.search(r"(?i)no (dataset|result|match)", child.before):
        print(f"    WARNING: no datasets found for {das_pattern}")
        return False

    _send(child, "select", debug)
    _wait(child, P_SELECT_IDX, "select indices", debug)
    _send(child, "all", debug)
    _wait(child, REPL_PROMPT, "REPL after select", debug)

    _send(child, "replicas", debug)
    _wait(child, P_SELECT_IDX, "replicas indices", debug)
    _send(child, "all", debug)
    _wait(child, P_SELECT_SITE, "select sites", debug)
    _send(child, "round-robin", debug)
    _wait(child, REPL_PROMPT, "REPL after replicas", debug)

    return True


def run_session(das_patterns, output_json, dry_run=False, debug=False):
    """
    Open a single dataset-discovery-cli session, accumulate all das_patterns
    via query→select→replicas, then one save, then quit.
    Returns True if output_json was written.
    """
    if dry_run:
        for das in das_patterns:
            print(f"    [dry-run] would query: {das}")
        return False

    if not HAS_PEXPECT:
        print("ERROR: pexpect is not installed.  Run: pip install pexpect")
        return False

    import sys
    # Force plain-text output: prompt_toolkit/rich inject ANSI codes between
    # characters (e.g. "Qu\x1b[1mery for:") which break regex matching.
    plain_env = {**os.environ, "TERM": "dumb", "NO_COLOR": "1", "FORCE_COLOR": "0"}
    child = pexpect.spawn(CLI_COMMAND, timeout=PEXPECT_TIMEOUT, encoding="utf-8",
                          env=plain_env)
    if debug:
        child.logfile_read = sys.stdout
        print(f"  [debug] spawned: {CLI_COMMAND}  (pid {child.pid})")

    try:
        if debug:
            print(f"  ... EXPECT [initial REPL]: {REPL_PROMPT!r}")
        child.expect(REPL_PROMPT)
        if debug:
            print(f"  <<< MATCH: {child.after!r}")

        if BLOCKED_SITES:
            _send(child, "block-sites", debug)
            _wait(child, P_BLOCK_SITES, "Exclude the sites:", debug)
            _send(child, ",".join(BLOCKED_SITES), debug)
            _wait(child, REPL_PROMPT, "REPL after block-sites", debug)

        any_ok = False
        for das_pattern in das_patterns:
            try:
                ok = _query_one(child, das_pattern, debug)
                any_ok = any_ok or ok
            except pexpect.TIMEOUT:
                print(f"    ERROR: timed out on {das_pattern}")
            except pexpect.EOF:
                print(f"    ERROR: CLI exited unexpectedly on {das_pattern}")
                child.close()
                return False

        if not any_ok:
            _send(child, "quit", debug)
            child.expect(pexpect.EOF, timeout=30)
            child.close()
            return False

        # ── single save after all eras accumulated ─────────────────────────
        _send(child, "save", debug)
        _wait(child, P_SAVE, "save filename", debug)
        _send(child, str(output_json), debug)
        # CLI asks whether to clear the selected samples list after saving
        _wait(child, r"empty your selected samples.*\[y/n\].*:", "clear samples?", debug)
        _send(child, "n", debug)
        _wait(child, REPL_PROMPT, "REPL after save", debug)

        _send(child, "quit", debug)
        child.expect(pexpect.EOF, timeout=30)
        child.close()

    except pexpect.TIMEOUT:
        print("    ERROR: timed out waiting for initial CLI prompt")
        if debug:
            print(f"  [debug] buffer at timeout: {child.before!r}")
        try:
            child.close(force=True)
        except Exception:
            pass
        return False
    except pexpect.EOF:
        if debug:
            print(f"  [debug] EOF — buffer: {child.before!r}")
        child.close()

    return output_json.exists()


# ── per-file update logic ─────────────────────────────────────────────────────

def build_das_tasks(json_path):
    """
    Read an existing dataset JSON and return:
      - xsec_map: {entry_key: xsec_string}
      - tasks:    [(das_name_or_wildcard, year_string)]  one per era to query
    For MC, derives DAS patterns for any Run2 era not yet present.
    For data, re-queries each existing das_name as-is.
    """
    with open(json_path) as f:
        data = json.load(f)

    xsec_map = {
        k: v["metadata"]["xsec"]
        for k, v in data.items()
        if v["metadata"].get("xsec") not in (None, "N/A")
    }

    # Collect existing (das_name, year) pairs
    existing_by_year: dict[str, str] = {}  # year → das_name
    for entry in data.values():
        is_mc = entry["metadata"].get("isMC", "False") == "True"
        if not is_mc:
            continue
        for das in parse_das_names(entry["metadata"].get("das_names", "")):
            year = detect_mc_year(das)
            if year and year not in existing_by_year:
                existing_by_year[year] = das

    tasks: list[tuple[str, str]] = []

    if existing_by_year:
        # MC sample — build tasks for all 4 eras
        src_year, src_das = next(iter(existing_by_year.items()))
        for target_year in MC_CAMPAIGNS:
            if target_year in existing_by_year:
                tasks.append((existing_by_year[target_year], target_year))
            else:
                derived = derive_missing_das(src_das, src_year, target_year)
                if derived:
                    tasks.append((derived, target_year))
                else:
                    print(f"  WARNING: could not derive DAS for {target_year} from {src_das}")
    else:
        # Data sample — re-query each existing das_name individually
        for entry in data.values():
            year = entry["metadata"].get("year", "unknown")
            for das in parse_das_names(entry["metadata"].get("das_names", "")):
                tasks.append((das, year))

    return xsec_map, tasks


def inject_xsec(discovery, xsec_map):
    """
    Inject correct xsec values into the dataset-discovery-cli output format:

        {
          "SampleName": {
            "sample": "...",
            "json_output": "...",
            "files": [
              { "das_names": [...], "metadata": { "xsec": 1.0, ... } },
              ...
            ]
          }
        }

    Matches sample names from xsec_map (stripped of trailing _YYYY suffix)
    against the top-level keys / "sample" fields in the discovery JSON.
    """
    xsec_by_sample: dict[str, float] = {}
    for orig_key, xsec in xsec_map.items():
        sample = re.sub(r"_20\d{2}.*$", "", orig_key)
        try:
            xsec_by_sample[sample] = float(xsec)
        except (TypeError, ValueError):
            pass

    for sample_name, entry in discovery.items():
        if not isinstance(entry, dict):
            continue

        xsec = xsec_by_sample.get(sample_name)
        if xsec is None:
            for s, x in xsec_by_sample.items():
                if s in sample_name or sample_name in s:
                    xsec = x
                    break

        if xsec is None:
            print(f"  WARNING: no xsec found for {sample_name!r} — leaving placeholder")
            continue

        for file_entry in entry.get("files", []):
            if isinstance(file_entry, dict) and "metadata" in file_entry:
                file_entry["metadata"]["xsec"] = xsec


def update_json(json_path, dry_run=False, debug=False):
    print(f"\n{'='*60}")
    print(f"Processing: {json_path.name}")

    if not json_path.exists():
        print(f"  SKIP: file not found")
        return

    xsec_map, das_tasks = build_das_tasks(json_path)
    print(f"  {len(das_tasks)} DAS queries to run")
    if xsec_map:
        sample0 = next(iter(xsec_map))
        print(f"  Preserved xsec from original (e.g. {sample0}: {xsec_map[sample0]})")

    # One temp file receives all eras in a single save at the end of the session.
    with tempfile.NamedTemporaryFile(
        suffix=".json", prefix="_tmp_session_",
        dir=DATASETS_DIR, delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)

    das_patterns = [das for das, _year in das_tasks]

    # The CLI produces two files when saving to foo.json:
    #   foo.json          — discovery metadata (sample info, json_output, …)
    #   foo_replicas.json — the actual xrootd file list
    tmp_replicas_path = tmp_path.parent / (tmp_path.stem + "_replicas.json")

    try:
        ok = run_session(das_patterns, tmp_path, dry_run=dry_run, debug=debug)

        discovery: dict = {}
        replicas: dict = {}
        if ok:
            if tmp_path.exists():
                with open(tmp_path) as f:
                    discovery = json.load(f)
                print(f"    Loaded {len(discovery)} sample(s) from discovery output")
            if tmp_replicas_path.exists():
                with open(tmp_replicas_path) as f:
                    replicas = json.load(f)
                print(f"    Loaded {len(replicas)} sample(s) from replicas output")
        elif not dry_run:
            print("    WARNING: session produced no output")
    finally:
        for p in (tmp_path, tmp_replicas_path):
            if p.exists():
                os.unlink(p)

    if dry_run or (not discovery and not replicas):
        if not dry_run:
            print("  No new data retrieved; skipping.")
        return

    inject_xsec(discovery, xsec_map)
    inject_xsec(replicas, xsec_map)
    populate_year(discovery)
    populate_year(replicas)

    # Save both files to datasets/discovery/
    discovery_dir = DATASETS_DIR / "discovery"
    discovery_dir.mkdir(exist_ok=True)

    if discovery:
        out = discovery_dir / json_path.name
        with open(out, "w") as f:
            json.dump(discovery, f, indent=4)
        print(f"  Saved → {out.relative_to(WORKSPACE)}")

    if replicas:
        out_rep = discovery_dir / (json_path.stem + "_replicas.json")
        with open(out_rep, "w") as f:
            json.dump(replicas, f, indent=4)
        print(f"  Saved → {out_rep.relative_to(WORKSPACE)}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--site", default="T3_US_FNALLPC",
                        help="(unused — replicas uses round-robin; kept for future use)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without calling the CLI")
    parser.add_argument("--debug", action="store_true",
                        help="Print every send/expect exchange and raw CLI output")
    parser.add_argument("--only", metavar="FILENAME",
                        help="Process only this JSON filename (e.g. TTTo2L2Nu....json)")
    args = parser.parse_args()

    if not HAS_PEXPECT and not args.dry_run:
        print("ERROR: pexpect is required.  Install with:  pip install pexpect")
        raise SystemExit(1)

    files = DATASET_FILES
    if args.only:
        files = [f for f in files if args.only in f]
        if not files:
            print(f"ERROR: --only '{args.only}' did not match any file in DATASET_FILES")
            raise SystemExit(1)

    for fname in files:
        update_json(DATASETS_DIR / fname, dry_run=args.dry_run, debug=args.debug)

    print("\nDone.")


if __name__ == "__main__":
    main()
