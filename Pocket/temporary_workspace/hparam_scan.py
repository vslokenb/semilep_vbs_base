"""
hparam_scan.py  —  Optuna hyperparameter scan for the VBS homogeneous transformer

Parallelisation strategy
------------------------
Each batch job runs scan_worker.py, which calls run_worker() here.
All workers share the same study via JournalFileStorage — an append-only
log file that is safe on EOS/AFS/NFS (unlike SQLite which requires POSIX
file locking that network filesystems don't support).

Typical CERN batch usage (HTCondor):
  # submit N_WORKERS jobs, each running:
  python scan_worker.py --parquet /eos/.../data.parquet
                        --category whad_withbveto_mu
                        --journal  /eos/.../scan.log
                        --n_trials 10

  # after all jobs finish, collect results:
  python scan_worker.py --collect
                        --journal  /eos/.../scan.log
                        --outdir   /eos/.../scans/

Single-machine usage (sequential, with EOS persistence):
  best_params = run_scan(data_list, journal_path="/eos/.../scan.log", ...)
"""

import os
import copy
import json
import numpy as np
import torch
from torch.nn import BCELoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage, JournalFileStorage

from homogenous_transformer import (
    PhysicsHomoGraph,
    balance_signal_background_weights_graphs,
)


# ──────────────────────────────────────────────────────────────────────────────
# Storage helper
# ──────────────────────────────────────────────────────────────────────────────

def _make_storage(journal_path: str) -> JournalStorage:
    """
    JournalFileStorage writes append-only to a plain text file.
    Safe on EOS/AFS/NFS. Multiple processes can write simultaneously.
    """
    os.makedirs(os.path.dirname(os.path.abspath(journal_path)), exist_ok=True)
    return JournalStorage(JournalFileStorage(journal_path))


# ──────────────────────────────────────────────────────────────────────────────
# Parameter space
# ──────────────────────────────────────────────────────────────────────────────

def _suggest_params(trial: optuna.Trial) -> dict:
    """
    Single source of truth for the search space.
    Add or remove parameters here; the rest of the code adapts automatically.
    """
    return dict(
        # ── Architecture ──────────────────────────────────────────────────────
        channels   = trial.suggest_categorical("channels",   [16, 32]),
        num_layers = trial.suggest_int(        "num_layers",  2, 3),
        heads      = trial.suggest_categorical("heads",      [2, 3, 4]),
        dropout    = trial.suggest_float(      "dropout",     0.20, 0.40, step=0.04),

        # ── Optimiser ─────────────────────────────────────────────────────────
        lr           = trial.suggest_float("lr",           5e-5, 1e-3, log=True),
        weight_decay = trial.suggest_float("weight_decay", 5e-7, 1e-5, log=True),

        # ── LR scheduler ──────────────────────────────────────────────────────
        sched_patience = trial.suggest_int(  "sched_patience", 10, 15),
        sched_factor   = trial.suggest_float("sched_factor",    0.3, 0.6, step=0.1),

        # ── Early stopping ────────────────────────────────────────────────────
        es_patience = trial.suggest_int("es_patience", 15, 20),

        # ── Data pipeline ─────────────────────────────────────────────────────
        batch_size   = trial.suggest_categorical("batch_size",  [16,32,64]),
        balance_to   = trial.suggest_categorical("balance_to",  ["unity"]),
        random_state = trial.suggest_int(        "random_state", 1, 50),

        # ── Physics attention additions ────────────────────────────────────────
        use_dr_near_bias = trial.suggest_categorical("use_dr_near_bias", [True, False]),
        use_dr_far_bias  = trial.suggest_categorical("use_dr_far_bias",  [True, False]),
        use_type_bias    = trial.suggest_categorical("use_type_bias",    [True, False]),
        use_edge_gate    = trial.suggest_categorical("use_edge_gate",    [True, False]),
        use_raw_inject   = trial.suggest_categorical("use_raw_inject",   [True, False]),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Single trial
# ──────────────────────────────────────────────────────────────────────────────

def _run_trial(
    trial     : optuna.Trial,
    data_list : list,
    device    : torch.device,
    max_epochs: int,
    u_dim     : int,
) -> float:
    p = _suggest_params(trial)

    train_val, val_raw = train_test_split(
        data_list, test_size=0.25, random_state=p["random_state"]
    )
    train_raw, _ = train_test_split(
        train_val, test_size=0.2, random_state=p["random_state"]
    )

    train_list = balance_signal_background_weights_graphs(
        copy.deepcopy(train_raw), balance_to=p["balance_to"]
    )
    val_list = balance_signal_background_weights_graphs(
        copy.deepcopy(val_raw), balance_to=p["balance_to"]
    )

    # num_workers + pin_memory keeps GPU fed between batches
    train_loader = DataLoader(train_list, batch_size=p["batch_size"],
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_list,   batch_size=p["batch_size"] * 2,
                              shuffle=False, num_workers=4, pin_memory=True)

    model = PhysicsHomoGraph(
        channels         = p["channels"],
        num_layers       = p["num_layers"],
        u_dim            = u_dim,
        heads            = p["heads"],
        dropout          = p["dropout"],
        use_dr_near_bias = p["use_dr_near_bias"],
        use_dr_far_bias  = p["use_dr_far_bias"],
        use_type_bias    = p["use_type_bias"],
        use_edge_gate    = p["use_edge_gate"],
        use_raw_inject   = p["use_raw_inject"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"]
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max",
        factor=p["sched_factor"], patience=p["sched_patience"], min_lr=1e-7,
    )
    criterion = BCELoss(reduction="none")

    best_auc = -1.0
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):

        model.train()
        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            out  = model(batch).squeeze(-1)
            y    = batch.y.float()
            loss = (criterion(out, y) * batch.weight.float()).sum() \
                   / batch.weight.float().sum()
            loss.backward()
            optimizer.step()

        model.eval()
        logits_list, labels_list, weights_list = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device, non_blocking=True)
                logits_list.append(model(batch).squeeze(-1).cpu())
                labels_list.append(batch.y.float().cpu())
                weights_list.append(batch.weight.float().cpu())

        logits  = torch.cat(logits_list).numpy()
        labels  = torch.cat(labels_list).numpy()
        weights = torch.cat(weights_list).numpy()

        if len(np.unique(labels)) < 2:
            raise optuna.exceptions.TrialPruned()

        val_auc = roc_auc_score(labels, logits, sample_weight=weights)
        scheduler.step(val_auc)

        trial.report(val_auc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if val_auc > best_auc:
            best_auc = val_auc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= p["es_patience"]:
                break

    return best_auc


# ──────────────────────────────────────────────────────────────────────────────
# Public: run_worker  —  called by each parallel batch job
# ──────────────────────────────────────────────────────────────────────────────

def run_worker(
    data_list    : list,
    n_trials     : int,
    journal_path : str,
    study_name   : str = "vbs_gnn_scan",
    max_epochs   : int = 50,
    n_startup    : int = 10,
) -> None:
    """
    Run n_trials trials, writing results to the shared journal file.
    Safe to call from multiple concurrent batch jobs simultaneously.

    n_startup should be >= total number of workers so TPE has enough
    random trials before it starts modelling parameter correlations.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[worker] device={device}  trials={n_trials}  journal={journal_path}")

    u_dim   = data_list[0].u.size(1)
    storage = _make_storage(journal_path)

    sampler = TPESampler(
        n_startup_trials = n_startup,
        multivariate     = True,   # models correlations between params
        seed             = None,   # None = different seed per worker
    )
    pruner = MedianPruner(
        n_startup_trials = n_startup,
        n_warmup_steps   = 8,
        interval_steps   = 2,
    )

    # load_if_exists=True: first worker creates the study,
    # all subsequent workers join the same one automatically.
    study = optuna.create_study(
        study_name     = study_name,
        storage        = storage,
        direction      = "maximize",
        sampler        = sampler,
        pruner         = pruner,
        load_if_exists = True,
    )

    study.optimize(
        lambda trial: _run_trial(trial, data_list, device, max_epochs, u_dim),
        n_trials       = n_trials,
        gc_after_trial = True,
    )

    print(f"[worker] done. Best so far: {study.best_value:.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# Public: collect_results  —  run once after all workers finish
# ──────────────────────────────────────────────────────────────────────────────

def collect_results(
    journal_path : str,
    study_name   : str = "vbs_gnn_scan",
    outdir       : str = ".",
) -> dict:
    """
    Load the completed study from the journal, save CSVs, return best_params.
    Call this once after all parallel workers have finished.
    """
    storage = _make_storage(journal_path)
    study   = optuna.load_study(study_name=study_name, storage=storage)

    n_complete = len([t for t in study.trials if t.value is not None])
    print("\n" + "=" * 60)
    print(f"Completed trials : {n_complete}")
    print(f"Best trial       : #{study.best_trial.number}")
    print(f"Best AUC         : {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k:22s} = {v}")
    print("=" * 60)

    os.makedirs(outdir, exist_ok=True)
    _save_scan_report(study, outdir)

    json_path = os.path.join(outdir, f"{study_name}_best_params.json")
    with open(json_path, "w") as f:
        json.dump({"best_auc": study.best_value, **study.best_params}, f, indent=2)
    print(f"[collect] Best params -> {json_path}")

    return study.best_params


# ──────────────────────────────────────────────────────────────────────────────
# Public: run_scan  —  single-machine convenience wrapper, unchanged call site
# ──────────────────────────────────────────────────────────────────────────────

def run_scan(
    data_list    : list,
    n_trials     : int  = 50,
    max_epochs   : int  = 50,
    study_name   : str  = "vbs_gnn_scan",
    journal_path : str  = None,
    outdir       : str  = "/eos/home-v/vslokenb/vbs_semilep/scan",
    n_startup    : int  = 10,
) -> dict:
    """
    Sequential single-machine scan with EOS-safe persistence.
    For parallel execution, use run_worker() from multiple batch jobs,
    then call collect_results() afterwards.

    The call site in main() changes only one argument name:
        storage="sqlite:///..."  ->  journal_path="/eos/.../scan.log"
    """
    if journal_path is None:
        import tempfile
        journal_path = os.path.join(tempfile.gettempdir(), f"{study_name}.log")
        print(f"[run_scan] No journal_path given, using {journal_path}")

    run_worker(
        data_list    = data_list,
        n_trials     = n_trials,
        journal_path = journal_path,
        study_name   = study_name,
        max_epochs   = max_epochs,
        n_startup    = n_startup,
    )
    return collect_results(journal_path, study_name, outdir)


# ──────────────────────────────────────────────────────────────────────────────
# Public: load_best_trial
# ──────────────────────────────────────────────────────────────────────────────

def load_best_trial(study_name: str, outdir: str) -> dict:
    """Load best params from a previously saved JSON without re-running."""
    json_path = os.path.join(outdir, f"{study_name}_best_params.json")
    with open(json_path) as f:
        params = json.load(f)
    params.pop("best_auc", None)
    return params


# ──────────────────────────────────────────────────────────────────────────────
# Reporting helpers
# ──────────────────────────────────────────────────────────────────────────────

def _save_scan_report(study: optuna.Study, outdir: str = ".") -> None:
    import pandas as pd

    rows = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        row = {"trial": t.number, "val_auc": t.value}
        row.update(t.params)
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows).sort_values("val_auc", ascending=False)
        path = os.path.join(outdir, "scan_results.csv")
        df.to_csv(path, index=False)
        print(f"[scan] Trial table    -> {path}")

    try:
        importances = optuna.importance.get_param_importances(study)
        df_imp = pd.DataFrame(
            importances.items(), columns=["parameter", "importance"]
        ).sort_values("importance", ascending=False)
        path = os.path.join(outdir, "scan_importance.csv")
        df_imp.to_csv(path, index=False)
        print(f"[scan] Importances    -> {path}")
        print(df_imp.to_string(index=False))
    except Exception as e:
        print(f"[scan] Could not compute importances: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# build_model_from_params
# ──────────────────────────────────────────────────────────────────────────────

def build_model_from_params(params: dict, u_dim: int, device: torch.device):
    """
    Reconstruct PhysicsHomoGraph + optimiser + scheduler from best_params dict.
    """
    model = PhysicsHomoGraph(
        channels         = params["channels"],
        num_layers       = params["num_layers"],
        u_dim            = u_dim,
        heads            = params["heads"],
        dropout          = params["dropout"],
        use_dr_near_bias = params["use_dr_near_bias"],
        use_dr_far_bias  = params["use_dr_far_bias"],
        use_type_bias    = params["use_type_bias"],
        use_edge_gate    = params["use_edge_gate"],
        use_raw_inject   = params["use_raw_inject"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr           = params["lr"],
        weight_decay = params["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode     = "max",
        factor   = params["sched_factor"],
        patience = params["sched_patience"],
        min_lr   = 1e-7,
    )
    return model, optimizer, scheduler