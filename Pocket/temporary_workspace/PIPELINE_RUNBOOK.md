# PocketCoffea VBS Semileptonic — Operator Runbook

Refreshed 2026-10-22. Supersedes the 2026-09-03 version (kept as
`PIPELINE_RUNBOOK.md.bak_*`). Everything here is measured from the v28/v29
campaigns, not assumed.

Workspace: `/work/users/vsloken-fnal/semilep_vbs_base/Pocket/temporary_workspace`
Platform rules (storage, quotas, Dask, proxies) live in `~/.claude/CLAUDE.md` —
read that first; this document only covers what is specific to this pipeline.

---

## 1. What exists

| config | purpose | categories | output |
|---|---|---|---|
| `config_validation.py` | nominal selection, CRs + SR | 17 | histograms |
| `config_inverted.py` | inverted-lepton (feeds nonprompt) | 17 | histograms |
| `config_nonprompt.py` | nonprompt estimate | — | histograms |
| `config_semilep.py` | **SR only, columnar** | 5 | histograms **+ `ColOut` arrays** |
| `config_boosted_control*.py` | fat-jet SF derivation | 2 | histograms |

All use `workflow_v9.py` (`VBSSemileptonicProcessor`).

Completed campaigns:
- **v28** — `cr_v28_<YEAR>_<GID>/`, `nonp_v28_<YEAR>_<GID>/`, 40+40 passes, 4 years.
  Catalogued in `v28_manifest.csv`; consumer spec in `V28_COMBINE_HANDOFF.md`.
- **v29** — `v29_semilep_2017_<GID>/`, 5 passes, 2017 only, SR+columnar. In progress.

---

## 2. The grouped execution model (and why)

**Do not run one pass per sample, and do not run everything in one pass.**

Three facts, all measured:

1. **`--process-separately` destroys parallelism.** With it, each dataset runs
   alone: measured **4 tasks, 2 busy cores of 400, ~1% utilisation**, and a
   dataset took 70–85 min regardless of cluster size. Dropping it submits a whole
   group as one fileset: **4,338 tasks, 70+ busy cores, ~70% utilisation**.
2. **But one giant pass is all-or-nothing.** Without `-ps` there are no
   incremental saves: a single bad chunk discards the entire pass. A 7-hour 2018
   validation pass was lost to one `KilledWorker`.
3. **So: group samples, run each group without `-ps`.** A failure costs one
   group-year. This is the hybrid model and it is what the run scripts implement.

**Grouping rule** (see the generator inline in this runbook's history, or
`v28_groups.txt` / `v29_groups.txt` for the results):

- A pass's wall-clock is set by its **largest** sample, because tasks run
  concurrently — so packing small samples alongside a big one costs ~nothing.
- Each pass pays one cluster startup (2–7 min), so fewer groups is faster.
- Isolate the heaviest samples (ttbar) so a `KilledWorker` can't take out a
  whole year's worth of other samples.

v29/2017 grouping: 5 groups for 45 samples (vs 45 passes if run per-sample):

| group | events | contents |
|---|---|---|
| g01 | 86.6 M | TTToSemiLeptonic alone |
| g02 | 47.2 M | TTTo2L2Nu alone |
| g03 | 30.0 M | 4 samples |
| g04 | 28.6 M | 9 samples |
| g05 | 15.3 M | 30 samples |

---

## 3. Cluster sizing — measured, not guessed

**Cores per worker is the setting that matters. More cores is worse.**

| shape | GB/core | outcome |
|---|---|---|
| 125w × 8c (1000 cores) | 4 | `scheduler-connection-lost`, died in ~1.5 min |
| 100w × 8c | 8 | died ~3 min |
| 50w × 8c | 8 | ttbar: **0 of 4 years succeeded** (`KilledWorker`) |
| **50w × 2c** | **12** | ttbar: **3 of 4** first try, 4/4 after retry; **70–76% CPU** |

Why: at 8 tasks per worker, one pathological chunk takes down its seven
neighbours. At 2 it can only strand one. Memory was **never** the binding
constraint — peak worker usage was 6.3 of 24 GiB (26%), cluster-wide 8–29%.

**Do not request huge memory.** A 50w × 8c × 64 GB (3,200 GiB) ask sat `PENDING`
and was never scheduled; the same job at 40 cores / 80 GiB scheduled in 2 min.

Current option files:
- `custom_semilep_heavy.yaml` — 50w × 2c × 24 GB (heavy MC: g01, g02)
- `custom_semilep_light.yaml` — 50w × 4c × 32 GB (everything else)
- `custom_data_small.yaml` — 10w × 4c × 8 GB (data passes: 1 variation, trivial)

**chunksize is NOT a useful lever.** Measured on one dataset:

| chunksize | chunks | wall-clock |
|---|---|---|
| 5000 | 541 | 3.5–6 h (thrash) |
| 25000 | 7 | ~65 min |
| 50000 | 11 | ~71 min |

25000 and 50000 are within noise; 5000 is catastrophic because per-chunk fixed
cost (40 `model.load_model()` calls in `workflow_v9.py`, plus ~150 variations)
dominates. Leave it at 50000.

---

## 4. Launch procedure

```bash
cd /work/users/vsloken-fnal/semilep_vbs_base/Pocket/temporary_workspace

# 1. proxy MUST be at the worker-visible path (see §6)
export X509_USER_PROXY=/work/users/vsloken-fnal/x509up_u2070155
voms-proxy-info | grep timeleft          # must be > run duration

# 2. no leftover cluster (one per user)
#    use the MCP tool, not the gateway client (see §6)

# 3. launch
TS=$(date -u +%Y%m%d_%H%M%S)
LOG="logs/run_v29_semilep_${TS}.log"; echo "$LOG" > .current_log_v29
nohup ./run_v29_semilep.sh > "$LOG" 2>&1 &
```

All runners **skip any pass whose `output_all.coffea` already exists**, so a
relaunch resumes rather than redoing work. There is **no checkpointing inside a
pass** — a pass killed at 49% restarts from 0.

---

## 5. Failure modes — detection and remedy

| # | symptom | detection | remedy |
|---|---|---|---|
| 1 | **Launch hang** | cluster reaches RUNNING with all workers, **zero tasks ever submitted**, log mtime frozen, client ~1% CPU | kill + stop cluster + relaunch. Clears every time. |
| 2 | **`KilledWorker`** | in `<out>/error/*.err` | lower cores/worker (8 → 2). Retry ladder also tries fewer files. |
| 3 | **`scheduler-connection-lost`** | in `.err` | transient — plain re-run at same settings succeeds |
| 4 | **Transient XRootD** (`Bad address`, `Operation expired`) | `FileNotFoundError` / `OSError` naming a file that opens fine later | plain re-run; for chronically slow sites use XCache (§7) |
| 5 | **Hung worker** | task counts **frozen across checks** while CPU collapses to ~1 core on **one** worker; **no error file** | kill + relaunch at 2 cores/worker. Retry ladders do NOT catch this (no error to match on). |
| 6 | **Cluster stuck PENDING** | `list_dask_clusters` shows PENDING, `query_dask_logs` returns **nothing** (scheduler pod never started) | facility-side. Check `get_facility_health`. Shrink the request; if it persists it needs admin action. |

**The single most important monitoring signal is log mtime age.** Modes 1 and 5
look healthy on every other surface — processes alive, cluster RUNNING, a
progress bar on screen. Include mtime age in every status check.

---

## 6. Gotchas that have cost real time

**Proxy location.** `voms-proxy-init` writes to `/tmp/x509up_u<uid>`, which
**workers cannot see**. Either export first:

```bash
export X509_USER_PROXY=/work/users/vsloken-fnal/x509up_u2070155
voms-proxy-init -voms cms -valid 192:00
```

or renew normally and copy: `cp /tmp/x509up_u2070155 /work/users/vsloken-fnal/`.
A stale proxy at the `/work` path fails with `Auth failed: No protocols left to
try`, not an obvious expiry message.

**Use MCP tools, not the gateway client, when the gateway misbehaves.** Four
`Gateway().connect(name).shutdown()` calls timed out on a stuck cluster;
`stop_dask_cluster` (MCP) killed it instantly. Always wrap gateway-client calls
in `timeout`.

**`str(cluster.status)` returns the enum VALUE** (`"2"` for RUNNING), not the
name. A watchdog comparing against `"RUNNING"` silently never matches — this
killed a healthy run and deleted its output.

**Never `rm -rf` a pass directory without checking for `output_all.coffea`.**
A teardown that wipes `<prefix>_*_g*` unconditionally destroys completed work.

**`--limit-files` may be a no-op.** It caps files *per dataset*. If every dataset
in a group has 1 file, 50/30/20 all read the same data — reducing it cannot fix
a chunk-level problem. Check file counts before using it as a remedy.

---

## 7. Corrections validation (run before every campaign)

A correction-name mismatch fails on the **first chunk of every group**, so it
looks like a total outage, not a resource problem. Validate first.

```python
# JEC/JER tags must exist in the CVMFS snapshot the config resolves to
import correctionlib
cs = correctionlib.CorrectionSet.from_file(
  "/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run2-2017-UL-NanoAODv9/latest/jet_jerc.json.gz")
[k for k in cs.keys() if "Summer19UL17_JRV4_MC" in k]   # must be non-empty
```

Checklist (all verified green for 2017 on 2026-10-20):

- JEC `Summer19UL17_V5_MC`, JER `Summer19UL17_JRV4_MC` — present in CVMFS
- JES Regrouped: 11 of 12 sources used (`Regrouped_Total` excluded — double-counts)
- `params/fj_taggers.yaml` — the `.json.gz` files must **exist and evaluate**
- `params/qgtagging.yaml`, `params/classifiers.yaml` (40 BDT models), `params/pileup.yaml`
- Lepton/b-tag SF yaml 2017 entries

**Two name-collision traps, both of which have caused total failures:**

1. **`sf_ele_trigger` vs `sf_ele_trigger_run2`.** Importing `SF_ele_trigger`
   from `weights_run2_UL` registers it as **`sf_ele_trigger_run2`**; the generic
   `sf_ele_trigger` from `common_weights` is a *different, Run3-style* weight.
   Run2 UL configs must request `sf_ele_trigger_run2`.
2. **`_FJ_SOURCES` entries are correctionlib category keys, NOT weight names.**
   They are baked into the fat-jet SF json at derivation time. Renaming the
   pileup *weight* to `PileupWeight` must **not** be propagated into
   `_FJ_SOURCES` — the SF file has no such key and every evaluation throws
   `IndexError: Index not available in Category ... val: PileupWeight_up`.
   For 2017 the custom pileup correction comes from `params/pileup.yaml`
   (`puWeights_17.json`) via the `PileupWeight` weight, and `_FJ_SOURCES`
   omits pileup entirely so it is varied exactly once, not twice.

Verify tokens actually evaluate before launching:

```python
corr.evaluate(f"{token}_up", "matched", 80.0, 0.9)   # must not raise
```

---

## 8. Monitoring

Every status check should report: **log mtime age**, procs, pass outcomes,
outputs on disk, and **CPU/core usage from the scheduler**.

```python
# scheduler probe — the only reliable view of real work
def probe(dask_scheduler=None):
    s = dask_scheduler; ws = list(s.workers.values())
    return {"workers": len(ws),
            "cores": sum(w.nthreads for w in ws),
            "busy": sum(len(w.processing) for w in ws),
            "cpu": sum((w.metrics or {}).get("cpu", 0) for w in ws),
            "tasks": len(s.tasks)}
```

Reading it:

- **CPU ≈ cores busy × 100%** → healthy
- **busy tasks concentrated on ONE worker + CPU ~1 core** → mode 5, kill it
- **tasks 0/0 with workers up** → mode 1, kill it
- **bar at 0% with tasks retiring** → normal; most chunks run concurrently so
  the bar jumps late. Judge by task counts, not the bar.

Do not extrapolate ETA from the first 15 minutes — the preprocessing phase is
much faster than processing and will make any projection 3–10× too optimistic.

---

## 9. Outputs

```
<prefix>_<YEAR>_<GID>/output_all.coffea
```

One file per group-year, containing all that group's datasets. Years are never
mixed (`--filter-years` enforces it; verify via
`datasets_metadata.by_datataking_period`).

Structure, normalization, systematics and the extraction recipe are documented
in **`V28_COMBINE_HANDOFF.md`** — read that before consuming outputs. Key points:

- `variables[hist][SAMPLE][DATASET]` — note the two different key types
- histograms are **absolute expected yields** (lumi × XS already applied)
- `--limit-files` does **not** bias yields (`sum_genweights` is accumulated from
  processed events, `base.py:806`) — it costs statistical precision only
- regenerate the catalogue with `python make_v28_manifest.py`

---

## 10. Rules of engagement

- **Known code bugs: stop and report, do not fix.** Config and workflow files
  are user-owned.
- **Never delete completed outputs.** Move aside within the user's own space.
- Kill the process tree **before** stopping the Dask cluster.
- One Dask cluster per user — a stale one blocks every new launch.
- Check `query_storage_usage` (not `du`) before large writes; `/work` has a
  100 GB quota and v28 alone consumed 25 GB.
