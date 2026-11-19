# Handoff: the "workbook manager" pipeline agent, and how to rebuild its working area for a new user

Written 2026-11-18 from the live state of user `vsloken-fnal`'s Purdue AF session.
**Audience: a Claude Code agent** setting up an identical area for a different user.
Placeholders: `<user>` = the new user's Purdue username (`whoami`), `<uid>` = `id -u`.

Read these first, in this order. They hold the measured details; this file only links them together:
1. `~/.claude/CLAUDE.md`: Purdue AF platform rules (quotas, storage visibility, Dask Gateway, proxies).
2. `PIPELINE_RUNBOOK.md`: the operator runbook the agent runs (grouping, cluster sizing, failure modes, rules of engagement).
3. `GRID_WORKFLOW_NOTES.txt`: lessons measured over the v29/v30 campaigns (the `-ps` verdict, the GIL, retries, XRootD limits, gateway hostname drift).
4. `SHIPPING_MANIFEST.md`: every param, SF and dataset file the two production configs depend on.
5. `V28_COMBINE_HANDOFF.md`: output format for downstream consumers.

All are in `/work/users/vsloken-fnal/semilep_vbs_base/Pocket/temporary_workspace`.

---

## 1. What the agent is

| | |
|---|---|
| Job name | "Launch agent for pipeline runbook" (Claude Code background job `aa588e05`) |
| Started | 2026-09-03, cwd `/home/vsloken-fnal`, permission mode `auto`, effort `high` |
| Mandate | Run the process in `PIPELINE_RUNBOOK.md`: launch PocketCoffea campaigns on Dask Gateway, monitor them, diagnose failures, relaunch, keep the notes and runbook up to date |
| Current campaign | **v31**: full Run 2 × (`config_validation.py`, `config_inverted.py`), with the `bdt_nov13` 10-fold BDTs and the `workflow_v9` jetcentral1..4 patch |
| Hard rules | Known code bugs: stop and report, never fix (configs and workflow are user-owned). Never delete completed outputs. Kill the process tree before stopping the cluster. One Dask cluster per user. Use `query_storage_usage` before large writes. |

### Tools it uses (counted from the session transcript)

| Tool | Calls | Used for |
|---|---|---|
| `Bash` | ~1400 | launching runners with `nohup`, tailing logs, log mtime age, counting outputs, `pgrep`, `voms-proxy-info`, scheduler probes via `client.run_on_scheduler` |
| `mcp__purdue-af-agentic-interface__list_dask_clusters` | 55 | the single-cluster check before every launch; spotting PENDING clusters |
| `…__stop_dask_cluster` | 28 | killing stuck or stale clusters. **Use this before trying `Gateway().connect().shutdown()`, which hangs on a stuck cluster.** |
| `…__get_dask_worker_count` | 19 | confirming the cluster actually scaled |
| `…__query_dask_logs` | 16 | worker/scheduler errors. An empty result on a PENDING cluster means the scheduler pod never started (failure mode 6). |
| `…__get_facility_health` | 13 | separating facility-side problems from our own |
| `…__get_dask_cluster_usage` / `…__get_dask_cluster_info` | 5 / 2 | CPU/memory utilisation |
| `…__list_dask_cluster_options` | 4 | gateway limits (≤64 cores, ≤64 GiB per worker) |
| `…__query_storage_usage` / `…__scale_dask_cluster` | 1 / 1 | quota check; manual scaling |
| `Read` / `Edit` / `Write` | 36 / 32 / 13 | runbook and notes upkeep; new run scripts and option YAMLs |
| `Monitor`, `TaskStop`, `AskUserQuestion`, `Agent` | few | watchers and occasional user decisions |

Global allowlist (`~/.claude/settings.json`) pre-approves `ps`, `tail`, `find`, `ls` and the read-only Dask/health MCP tools. Copy it so a monitoring agent doesn't hit a permission prompt on every status check.

### Workspace scripts it drives

| Script | Role |
|---|---|
| `run_v31_all.sh` | Top-level chain: loops `2016_PreVFP → 2016_PostVFP → 2017 → 2018`, calls `run_v30_year.sh` with `PREFIX=v31`, logs to `logs/run_v31_<YEAR>_<ts>.log`, writes the current log path to `.current_log_v30` |
| `run_v30_year.sh` | One era: validation pass, then inverted pass. Each pass runs up to 3 attempts. Non-ttbar datasets run as one invocation; the two ttbar datasets run as a separate invocation capped at `TT_CAP` files. Stops clusters between invocations. Runs inside the apptainer image. |
| `run_inside_v30.sh` | The `pocket-coffea run … -e dask@purdue-af --process-separately` call, inside the container; activates `my-env` and sets `PYTHONPATH`, `PYTHONUNBUFFERED` and `OMP_*=1` |
| `v30_missing.py` | Lists dataset keys that have no `output_<key>.coffea` yet. Drives per-dataset resume (data samples expand into per-era keys). |
| `watch_v30_year.sh` | Read-only watcher (never kills or deletes). Alerts when a dataset is written or fails, and when the chain exits. Expires after 24 h by default and must be re-armed. Its first "ALERT" on startup is cosmetic. |
| `custom_v30_2018.yaml` | Run options in use: 50 workers × 1 core × 6 GiB, chunksize 20000, tree-reduction 4, retries 3 |

## 2. Status at 2026-11-18 17:22 UTC

Chain `run_v31_all.sh` (PID 743631) is **running**. Watcher `watch_v30_year.sh` is armed for 2017.

| Era | validation | inverted |
|---|---|---|
| 2016_PreVFP | ✅ 55/55 (5.5 GB) | ✅ 55/55 (128 MB) |
| 2016_PostVFP | ✅ 51/51 (5.4 GB) | ✅ 51/51 (112 MB) |
| 2017 | ⏳ **52/55**, attempt 2 (`ttWJets` in flight, then the 2 capped ttbar) | pending, 0/55 |
| 2018 | pending, 0/53 | pending, 0/53 |

Overall: 264 of 428 dataset outputs, about 17 GB. VOMS proxy at `/work/users/vsloken-fnal/x509up_u2070155` has ~53 h left (192464 s at the time of writing). 2018 will likely outlast it, so renew it before 2018 starts.

There has been one real fault since launch. On 17 Nov a single `scheduler-connection-lost` killed the Dask client. Under `-ps` the loop then failed the remaining 50 datasets in seconds with `ClosedClientError`, which inflated the cumulative fail count to 54. Attempt 2 recovered them. The suggested improvement, not yet implemented: abort an invocation as soon as a `ClosedClientError` appears.

Previous agent jobs in this area (all finished): config-validation file tracing (`environment.yml`, `SHIPPING_MANIFEST.md`), cross-section fixes, v30 working-point audit, `bdt_nov13` classifier swap, the JES patch trace, CR outlier scan.

---

## 3. Prerequisites to rebuild the exact area for a new user

### 3.0 Accounts and credentials (the user must do these; an agent cannot)
- Purdue AF account, plus a CMS VO membership with a grid certificate installed in `~/.globus/`.
- Read access to `root://cmseos.fnal.gov//eos/uscms/store/group/lnujj/semilep_vbs_skim/`. All 2407 dataset file entries point there. **XCache does not serve these group skims.**
- GitHub access to `vslokenb/semilep_vbs_base` (private, SSH) and `vslokenb/PocketCoffea` (branch `nonprompt`).
- An AF session sized for a client process (a few cores is fine; the work runs on gateway workers). The Purdue AF MCP server is preconfigured in AF sessions.

### 3.1 Directory layout (everything under `/work`, never `/home`)
```
/work/users/<user>/
├── envs/my-env/                     # conda env (Python 3.11)
├── latest_poco/PocketCoffea/        # fork checkout, on PYTHONPATH (shadows the pip install)
├── semilep_vbs_base/Pocket/temporary_workspace/   # analysis workspace = cwd for all runs
└── x509up_u<uid>                    # worker-visible proxy copy
```

### 3.2 Runtime environment: host → apptainer → conda env → Dask workers
Every run goes through four layers. Each one supplies something the others don't. All of the following was verified on 2026-11-18.

**Layer 1: the AF session (host).** AlmaLinux 8 JupyterHub pod.
- `apptainer` 1.5.4 is at `/usr/bin/apptainer` (`singularity` is an alias).
- There is **no conda hook on the host**: `/usr/local/etc/profile.d/conda.sh` exists only inside the container. Run the conda steps below inside the container.
- Session env vars the run depends on, all preset by the AF (don't set them yourself):
  - `JUPYTERHUB_API_TOKEN`: Dask Gateway auth (`DASK_GATEWAY__AUTH__TYPE=jupyterhub`)
  - `DASK_GATEWAY__ADDRESS`, `DASK_GATEWAY__PROXY_ADDRESS`
  - `NB_UID`, `NB_GID`, `NB_USER`: the executor forwards these to workers
  - `X509_CERT_DIR=/cvmfs/cms.cern.ch/grid/etc/grid-security/certificates`
- Apptainer passes the host environment through by default, and the token is visible inside the container. **Never add `--cleanenv` or `-e`**; that strips the token and the gateway refuses the cluster.
- `rsync` does not exist anywhere (host, container or env). Use `cp -a` or `scp`.

**Layer 2: the container.**
- Image: `/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/pocketcoffea:lxplus-el9-latest`
  - A cvmfs symlink to an unpacked AlmaLinux 9.6 image. `latest` floats; it currently resolves to `.flat/29/29390d73…`.
  - To freeze the setup, record `readlink -f <image>` at setup time.
- What the container provides:
  - the EL9 userland
  - conda 25.7.0 with its hook at `/usr/local/etc/profile.d/conda.sh`
  - `voms-proxy-*`
  - `/etc/grid-security/certificates`
  - its own Python 3.11.13 at `/usr/local/bin/python`. **This interpreter is not used**: the conda env replaces it.
- Bind mounts (exactly as in `run_v30_year.sh`):
  ```
  -B /work/users/<user>/ -B /cvmfs -B /cvmfs/cms-griddata.cern.ch -B /cvmfs/cms.cern.ch -B /tmp -B /etc/sysconfig/
  ```
  - `/work/users/<user>` carries the env, code, workspace and proxy.
  - The cvmfs binds carry the JERC/jsonpog corrections and the CA certificates.
  - `/tmp` and `/etc/sysconfig` are kept as the original setup has them.
- Invocation pattern: `apptainer exec "${BINDS[@]}" "$IMG" bash <workspace>/run_inside_v30.sh <args>`

**Layer 3: the conda env `my-env`** (`/work/users/<user>/envs/my-env`, about 3.2 GB). Create it **inside the container**:
```bash
IMG=/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/pocketcoffea:lxplus-el9-latest
apptainer exec -B /work/users/<user>/ -B /cvmfs -B /tmp "$IMG" bash -lc '
  source /usr/local/etc/profile.d/conda.sh
  conda env create -p /work/users/<user>/envs/my-env -f <workspace>/environment.yml
  conda activate /work/users/<user>/envs/my-env
  conda install -y -c conda-forge xrootd=5.9.2        # NOT in environment.yml, but required
  pip install -e /work/users/<user>/latest_poco/PocketCoffea
'
```
- **The `environment.yml` gap.** `xrootd` (the `XRootD` Python bindings, plus the `xrdfs`/`xrdcp` binaries the env puts on `PATH`) is conda-installed in the running env but missing from the spec. Without it, uproot cannot open the `root://cmseos.fnal.gov` inputs. Add it to `environment.yml` as well.
- Versions in the running env:

  | package | version |
  |---|---|
  | python | 3.11 |
  | coffea | 0.7.29 |
  | awkward | 1.10.5 |
  | uproot | 4.3.7 |
  | dask / distributed | 2026.3.0 |
  | dask-gateway | 2025.4.0 |
  | **numpy** | **1.26.4** (must stay < 2) |
  | pandas | 3.0.2 |
  | numba | 0.65.0 |
  | hist | 2.10.1 |
  | correctionlib | 2.7.0 |
  | xgboost | 3.2.0 |
  | onnxruntime | 1.24.4 |
  | scikit-learn | 1.8.0 |
  | omegaconf | 2.3.0 |
  | vector | 1.4.3 |
  | cloudpickle | 3.1.2 |
  | xrootd | 5.9.2 |
  | pocket-coffea | 0.9.9 |
- Merge `.coffea` outputs with this interpreter. The global pixi env's numpy 2 writes pickles the downstream scripts cannot read.
- This is a conda env by deliberate choice, so the platform's "use pixi" guidance does not apply here. It must sit on `/work`, because the Dask workers activate it.
- **Which PocketCoffea runs** is decided by `PYTHONPATH`, not by the pip install. In the original env the editable install points at an *older* checkout (`/work/users/vsloken-fnal/PocketCoffea`), and `PYTHONPATH=/work/users/<user>/latest_poco/PocketCoffea` shadows it.
  - For a new user, install from `latest_poco` (as above) so the two agree, and keep exporting `PYTHONPATH` anyway.
  - Check: inside the container, `python -c "import pocket_coffea; print(pocket_coffea.__file__)"` must print the `latest_poco` path.

**What `run_inside_v30.sh` sets inside the container (all required):**
```bash
source /usr/local/etc/profile.d/conda.sh
conda activate /work/users/<user>/envs/my-env
export PYTHONPATH=/work/users/<user>/latest_poco/PocketCoffea   # patched fork
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH       # env libs (xrootd, xgboost) before the image's
export PYTHONUNBUFFERED=1                                        # otherwise a stalled launch logs nothing
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
```
`run_v30_year.sh` exports `X509_USER_PROXY=/work/users/<user>/x509up_u<uid>` on the host. Apptainer passes it into the container.

**Layer 4: the Dask Gateway workers.**
- Workers do **not** run in the apptainer image, and they don't inherit the session environment.
- `executors_purdue_af.py` creates the cluster with `conda_env=<my-env path>` and forwards only an allowlist of variables:
  - `NB_UID`, `NB_GID`, `PYTHONPATH`, `LD_LIBRARY_PATH`, `X509_CERT_DIR`, `PATH`, `X509_USER_PROXY`
  - plus the `OMP_*` thread variables, which come from the local patch in §3.3
- So everything a worker needs must be on `/work` (or `/depot`/`/cvmfs`): the env, the `latest_poco` code, the workspace (`workflow_v9` is shipped by value with cloudpickle, but params, SF and BDT files are read from disk) and the proxy.
- `custom-setup-commands` in the YAMLs is **ignored** by `dask@purdue-af`.
- Gateway limits: at most 64 cores and 64 GiB per worker, one cluster per user.

### 3.3 PocketCoffea fork, with two patches that are not pushed
```bash
cd /work/users/<user>/latest_poco
git clone -b nonprompt https://github.com/vslokenb/PocketCoffea.git
cd PocketCoffea && git checkout a4704f09      # validated commit
pip install -e .                              # into my-env
```
The running checkout has **uncommitted** changes that the pinned commit does not include. Apply them by hand, or copy them from `/work/users/vsloken-fnal/latest_poco/PocketCoffea` (`git diff > patch`):
1. `pocket_coffea/executors/executors_purdue_af.py`: add `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS` and `VECLIB_MAXIMUM_THREADS` to the `propagate_env` allowlist. Without them, BLAS takes about 22 threads per task.
2. `pocket_coffea/scripts/runner.py` (`-ps` branch): `adapted_chunksize = max(adapted_chunksize, 5000)` and pass `chunksize=adapted_chunksize` to `get_runner`. Upstream computes the adapted value but then passes the unadapted one.

A cleaner option is to ask the original user to commit and push these to the fork first. Verify which code actually executes:
`python -c "import pocket_coffea.scripts.runner as m; print(m.__file__)"` should print the `latest_poco` path.

### 3.4 Analysis workspace (the working tree, not a git clone)
`git clone git@github.com:vslokenb/semilep_vbs_base.git` **does not reproduce the running state.** The repo has about 900 uncommitted or untracked paths. Those that matter for v31:
- modified: `workflow_v9.py` (jetcentral1..4 patch), `config_validation.py`, `config_inverted.py`, `params/classifiers.yaml`
- untracked: `bdt_nov13/` (12 MB, the 40 BDT models that `classifiers.yaml` points to)

Two ways to get the exact state:
- **(a) preferred:** ask the original user to commit and push the v31 state, then clone it. Or
- **(b)** copy the tree: `rsync` is not installed on the AF, so use `cp -a` and exclude outputs. Copy the code, `params/`, `datasets/`, `bdt_nov13/`, the SF files listed in `SHIPPING_MANIFEST.md`, the run, watch and missing scripts, and the option YAMLs. **Do not copy** `v2*_*/`, `v3*_*/`, `*.coffea`, `*.root` outputs or `logs/`. The workspace holds tens of GB of past outputs, and `/work` has a 100 GB quota. This copies from another user's directory, which is read-only for you; get the owner's consent first.

The param YAMLs use `${config_dir:}`, so the tree can be moved anywhere without editing paths.

**Known open issue:** `params/pileup.yaml` points 2017 at `${config_dir:}/puWeights_17.json`, which **does not exist**. Candidates are `params/puWeights_2017.json` and `puWeights.json.gz`. The v31 2017 pass is producing outputs anyway, so check whether that entry is actually read before relying on it, and ask the user which file is intended.

### 3.5 Hardcoded paths to rewrite for `<user>`
These scripts embed `/work/users/vsloken-fnal` and `x509up_u2070155`:
`run_v31_all.sh`, `run_v30_year.sh` (`WORKDIR`, `BINDS`, `PY`, `X509_USER_PROXY`), `run_inside_v30.sh` (conda env, `PYTHONPATH`, `cd`), `watch_v30_year.sh`, and the `custom-setup-commands` block in each `custom_*.yaml` (inert for `dask@purdue-af`, but keep it consistent). Find them all with:
```bash
grep -rl "vsloken-fnal\|2070155" --include='*.sh' --include='*.yaml' --include='*.py' .
```

### 3.6 Run options: required gateway override
PocketCoffea's defaults still name `dask-gateway-k8s…`, which no longer resolves, and the executor ignores the `DASK_GATEWAY__*` env vars. Every run-options YAML must contain:
```yaml
gateway-address: "http://dask-gateway.geddes.rcac.purdue.edu"
gateway-proxy-address: "traefik-dask-gateway.cms.geddes.rcac.purdue.edu:8786"
```
`custom_v30_2018.yaml` (used by v31) already has them; older option YAMLs may not. Check any YAML you copy before using it, and confirm the name resolves with `getent hosts dask-gateway.geddes.rcac.purdue.edu`. The failure signature is a `ClientConnectorDNSError` about 2 minutes after launch.

### 3.7 Environment check (run once after setup, before any launch)
```bash
IMG=/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/pocketcoffea:lxplus-el9-latest
apptainer exec -B /work/users/<user>/ -B /cvmfs -B /cvmfs/cms-griddata.cern.ch -B /cvmfs/cms.cern.ch -B /tmp -B /etc/sysconfig/ "$IMG" bash -c '
  source /usr/local/etc/profile.d/conda.sh
  conda activate /work/users/<user>/envs/my-env
  export PYTHONPATH=/work/users/<user>/latest_poco/PocketCoffea LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
  which python pocket-coffea xrdfs voms-proxy-info
  python -c "import XRootD, numpy, coffea, dask_gateway, xgboost, pocket_coffea; print(numpy.__version__, pocket_coffea.__file__)"
  [ -n "$JUPYTERHUB_API_TOKEN" ] && echo "gateway token visible"
  ls $X509_CERT_DIR >/dev/null && echo "CA certs OK"
  voms-proxy-info -file $X509_USER_PROXY -timeleft
  xrdfs cmseos.fnal.gov ls /eos/uscms/store/group/lnujj/semilep_vbs_skim | head -3
'
```
Expected results:
- `python`, `pocket-coffea` and `xrdfs` resolve to `my-env`.
- numpy is 1.26.x.
- `pocket_coffea` resolves to `latest_poco`.
- The token is visible and the CA certs are found.
- The proxy shows more time left than the run will take.
- The skim directory lists.

### 3.8 Proxy (every campaign)
```bash
export X509_USER_PROXY=/work/users/<user>/x509up_u$(id -u)
voms-proxy-init -voms cms -valid 192:00
voms-proxy-info -file $X509_USER_PROXY -timeleft     # must exceed run duration
```
The executor refuses a proxy outside `/work` or `/depot`. A stale proxy fails with `Auth failed: No protocols left to try`.

### 3.9 Agent configuration for the new user
- Copy `~/.claude/settings.json`'s `permissions.allow` list (§1).
- Launch the agent as a background job with `--permission-mode auto --effort high`, cwd set to the workspace. Point it at `PIPELINE_RUNBOOK.md` and this file.
- Have it validate corrections before the first launch (runbook §7). A name mismatch fails on the first chunk of every group and looks like a total outage.

## 4. Smoke test before a full campaign
1. `query_storage_usage`: confirm `/work` headroom. A full v31 campaign is about 23 GB of histograms; columnar configs are far larger (2.2 KB/event).
2. `list_dask_clusters`: confirm no cluster exists.
3. In `my-env` inside the container: `python -c "import config_validation"` from the workspace. This checks imports and `${config_dir:}` resolution.
4. One small dataset, one year:
   `bash run_inside_v30.sh config_validation.py smoke_test <one_dataset_key> custom_v30_2018.yaml 5 "" 2018`.
   Then confirm `smoke_test/output_<key>.coffea` exists. Don't use `-t/--test` timings to estimate throughput; they understate it about 5×.
5. Launch the chain with `nohup`, start the watcher, and include **log mtime age** in every status check. Launch hangs and hung workers look healthy on every other signal.
