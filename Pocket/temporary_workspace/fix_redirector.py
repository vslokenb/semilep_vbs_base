#!/usr/bin/env python3
"""
fix_redirector.py — Replace site-specific xrootd URLs in dataset JSONs
with the CMS global redirector so failed skim jobs can be resubmitted
without hitting bad sites.

Usage:
    python fix_redirector.py [--datasets-dir datasets/] [--dry-run]
"""

import argparse
import json
import re
import sys
from pathlib import Path

# Sites that are currently unreachable / timing out
BAD_SITES = [
    "cmsdcache-kit-disk.gridka.de",   # T2_DE_KIT
    "xrootd-cms.infn.it",             # T2_IT_*
    "cms-xrdr.sdfarm.kr",             # T3_KR_KISTI
    "cmsdcadisk.fnal.gov",            # FNAL dcache (non-EOS)
]

GLOBAL_REDIRECTOR = "root://cmsxrootd.fnal.gov/"


def store_path_from_url(url: str) -> str | None:
    """
    Extract the logical /store/... path from any site-specific xrootd URL.
    Handles path prefixes injected by various storage systems:
      root://server:port//store/...
      root://server///store/...              (triple-slash)
      root://server:port//xrd/store/...      (sdfarm.kr)
      root://server//dcache/uscmsdisk/store/ (FNAL dcache)
    """
    m = re.search(r'//+(?:xrd/|dcache/[^/]+/)?(store/.+)', url)
    return ('/' + m.group(1)) if m else None


def rewrite_url(url: str) -> tuple[str, bool]:
    """Return (new_url, was_changed)."""
    for site in BAD_SITES:
        if site in url:
            path = store_path_from_url(url)
            if path:
                return GLOBAL_REDIRECTOR + path.lstrip('/'), True
    return url, False


def fix_json(json_path: Path, dry_run: bool = False) -> int:
    """Rewrite all bad-site URLs in a dataset JSON. Returns number of URLs changed."""
    with open(json_path) as f:
        data = json.load(f)

    changed = 0
    for sample, entry in data.items():
        if not isinstance(entry, dict):
            continue
        files = entry.get("files", [])
        if not isinstance(files, list):
            continue
        new_files = []
        for url in files:
            if not isinstance(url, str):
                new_files.append(url)
                continue
            new_url, was_changed = rewrite_url(url)
            if was_changed:
                changed += 1
            new_files.append(new_url)
        entry["files"] = new_files

    if changed and not dry_run:
        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)

    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets-dir", default="datasets",
                        help="Directory containing dataset JSON files")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change without writing files")
    args = parser.parse_args()

    datasets_dir = Path(args.datasets_dir)
    jsons = sorted(p for p in datasets_dir.glob("*.json")
                   if "discovery" not in str(p))

    if not jsons:
        print(f"No JSON files found in {datasets_dir}")
        sys.exit(1)

    total = 0
    for jpath in jsons:
        n = fix_json(jpath, dry_run=args.dry_run)
        if n:
            tag = "[dry-run] " if args.dry_run else ""
            print(f"  {tag}{n:4d} URLs rewritten  {jpath.name}")
            total += n

    action = "would rewrite" if args.dry_run else "rewrote"
    print(f"\nDone — {action} {total} URLs across {len(jsons)} files.")
    if args.dry_run and total:
        print("Rerun without --dry-run to apply changes.")


if __name__ == "__main__":
    main()
