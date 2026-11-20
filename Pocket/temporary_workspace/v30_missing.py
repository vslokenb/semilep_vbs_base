#!/usr/bin/env python3
"""Print the comma-separated dataset keys for <config> / year <Y> that do not
yet have an output_<key>.coffea in <outdir>.  Used for per-dataset resume:
data datasets carry era suffixes so sample names are not 1:1 with datasets.
Usage: v30_missing.py <config.py> <outdir> <year>"""
import json, re, sys, os, glob

cfg_path, outdir, year = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = open(cfg_path).read()
blk = cfg[cfg.index('"filter"'):]
samples = set(re.findall(r'^\s*"([^"]+)",', blk[:blk.index(']')], re.M))
jsons = re.findall(r'f"\{localdir\}/datasets/([^"]+)"', cfg)
jsons = [j for j in jsons if not re.search(r'^\s*#', j)]
# only the uncommented json lines
active = [l.split('datasets/')[1].split('"')[0]
          for l in cfg.splitlines()
          if 'f"{localdir}/datasets/' in l and not l.strip().startswith('#')]

expected = []
for j in active:
    p = os.path.join('datasets', j)
    try: d = json.load(open(p))
    except Exception: continue
    for k, v in d.items():
        m = v.get('metadata', {})
        if m.get('year') == year and m.get('sample') in samples:
            expected.append(k)
expected = sorted(set(expected))
missing = [k for k in expected
           if not os.path.exists(os.path.join(outdir, f"output_{k}.coffea"))]
if len(sys.argv) > 4 and sys.argv[4] == '--count':
    print(f"{len(expected) - len(missing)}/{len(expected)}")
else:
    print(",".join(missing))
