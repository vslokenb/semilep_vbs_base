# Handoff bundle: VBS semileptonic PocketCoffea pipeline on Purdue AF

Snapshot taken 2026-11-18. These are **copies**; the live originals are in
`/work/users/vsloken-fnal/semilep_vbs_base/Pocket/temporary_workspace/`.
Re-copy them if the originals have changed since this date.

## Read in this order

| # | File | What it is |
|---|---|---|
| 1 | `AGENT_SETUP_HANDOFF.md` | **Start here.** What the agent does, its tools and its current status, plus step-by-step setup for a new user (accounts, apptainer, conda env, Dask workers, patches, path rewrites, smoke test) |
| 2 | `claude_config/platform_CLAUDE.md` | Purdue AF platform rules (copy of `~/.claude/CLAUDE.md`): quotas, storage visibility, Dask Gateway, proxies |
| 3 | `PIPELINE_RUNBOOK.md` | Operator runbook the agent follows: grouping, cluster sizing, launch, failure modes, monitoring, rules of engagement |
| 4 | `GRID_WORKFLOW_NOTES.txt` | Lessons measured over the v29/v30 campaigns: `-ps`, the GIL, retries, XRootD limits, gateway hostname drift |
| 5 | `SHIPPING_MANIFEST.md` | Every param, SF, model and dataset file the production configs depend on |
| 6 | `V28_COMBINE_HANDOFF.md` | Output `.coffea` structure and normalization for downstream/Combine |
| 7 | `MANIFEST_V29.md` | Example per-dataset output catalogue (v29, 2017 columnar) |

## Setup inputs

| File | Use |
|---|---|
| `environment.yml` | Conda env spec. **Also install `xrootd=5.9.2` from conda-forge**; the spec omits it (see AGENT_SETUP_HANDOFF §3.2). |
| `patches/pocketcoffea_purdue_af.patch` | Local, unpushed fixes on top of fork `vslokenb/PocketCoffea@nonprompt` commit `a4704f09`: the worker thread-variable allowlist and the adapted-chunksize fix. Apply with `git apply` in the checkout. |
| `patches/workspace_v31_uncommitted.patch` | Uncommitted v31 changes to `workflow_v9.py`, `config_validation.py`, `config_inverted.py` and `params/classifiers.yaml`, against `semilep_vbs_base@main` (`f4c7753`). It does **not** include the untracked `bdt_nov13/` models (12 MB), which must be copied separately. |
| `claude_config/settings_permissions.json` | Claude Code permission allowlist (read-only shell and Dask MCP tools) to merge into the new user's `~/.claude/settings.json` |

## Quick start for the receiving agent
1. Read `AGENT_SETUP_HANDOFF.md` §3 and do the steps in order. Replace `<user>` and `<uid>` throughout.
2. Run the environment check (§3.7), then the smoke test (§4).
3. Only then launch a campaign, following `PIPELINE_RUNBOOK.md` §4.
