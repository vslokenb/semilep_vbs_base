"""
hparam_scan.py  —  Optuna hyperparameter scan for the VBS homogeneous transformer

Varies:
  Training dynamics : lr, weight_decay, early-stop patience, scheduler patience/factor
  Architecture      : channels, num_layers, heads, dropout
  Data pipeline     : random_state (split seed), batch_size, balance_to strategy

Usage (from your existing main, after data_list is built):

    from hparam_scan import run_scan, load_best_trial

    best_params = run_scan(
        data_list,
        n_trials=60,
        study_name="vbs_scan",
        storage="sqlite:///vbs_scan.db",   # omit for in-memory
    )

    # Re-train final model with best params on the full train+val set
    model = load_best_trial("vbs_scan", storage="sqlite:///vbs_scan.db")
"""

import os
import copy
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

# ── Import from your existing module ─────────────────────────────────────────
# Adjust the import path to wherever HomoGraph and related helpers live.
from homogenous_transformer import (
    HomoGraph,
    balance_signal_background_weights_graphs,
    collect_graph_metadata,
    PhysicsHomoGraph
)


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
        channels   = trial.suggest_categorical("channels",   [32, 64, 128]),
        num_layers = trial.suggest_int(        "num_layers",  2, 5),
        heads      = trial.suggest_categorical("heads",      [2, 3, 4]),
        dropout    = trial.suggest_float(      "dropout",     0.0, 0.35, step=0.05),

        # ── Optimiser ─────────────────────────────────────────────────────────
        lr           = trial.suggest_float("lr",           1e-5, 5e-3, log=True),
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),

        # ── LR scheduler ──────────────────────────────────────────────────────
        sched_patience = trial.suggest_int(  "sched_patience",  3, 10),
        sched_factor   = trial.suggest_float("sched_factor",    0.2, 0.8, step=0.1),

        # ── Early stopping ────────────────────────────────────────────────────
        es_patience = trial.suggest_int("es_patience", 14, 15),

        # ── Data pipeline ─────────────────────────────────────────────────────
        batch_size  = trial.suggest_categorical("batch_size", [32]),
        balance_to  = trial.suggest_categorical("balance_to", ["signal", "background", "unity"]),

        # ── Split seed ────────────────────────────────────────────────────────
        # Varying this distinguishes models that genuinely generalise from those
        # that are simply lucky about which events land in the test set.
        random_state = trial.suggest_int("random_state", 40, 45),
        # -- Attention biases -----------------------
        use_dr_near_bias   = trial.suggest_categorical("use_dr_bias",   [True, False]),
        use_dr_far_bias   = trial.suggest_categorical("use_dr_bias",   [True, False]),
        use_type_bias = trial.suggest_categorical("use_type_bias", [True, False]),
        use_edge_gate = trial.suggest_categorical("use_edge_gate", [True, False]),
        use_raw_inject = trial.suggest_categorical("use_raw_inject", [True, False]),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Single trial
# ──────────────────────────────────────────────────────────────────────────────

def _run_trial(trial: optuna.Trial, data_list: list, device: torch.device,
               max_epochs: int, u_dim: int) -> float:
    """
    Build, train, and evaluate one configuration.
    Returns weighted validation AUC (maximised by Optuna).
    Raises TrialPruned when Optuna decides the trial is unlikely to win.
    """
    p = _suggest_params(trial)

    # ── Split ─────────────────────────────────────────────────────────────────
    # deep-copy so weight mutations in balance_signal_background_weights_graphs
    # don't bleed across trials.
    train_val, val_raw = train_test_split(
        data_list, test_size=0.25, random_state=p["random_state"]
    )
    train_raw, _ = train_test_split(          # _ = hold-out, not used in scan
        train_val, test_size=0.2, random_state=p["random_state"]
    )

    train_list = balance_signal_background_weights_graphs(
        copy.deepcopy(train_raw), balance_to=p["balance_to"]
    )
    val_list = balance_signal_background_weights_graphs(
        copy.deepcopy(val_raw), balance_to=p["balance_to"]
    )

    train_loader = DataLoader(train_list, batch_size=p["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_list,   batch_size=p["batch_size"] * 2, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = PhysicsHomoGraph(
        channels      = p["channels"],
        num_layers    = p["num_layers"],
        u_dim         = u_dim,
        heads         = p["heads"],
        dropout       = p["dropout"],
        use_dr_near_bias   = p["use_dr_near_bias"],
        use_dr_far_bias   = p["use_dr_far_bias"],
        use_type_bias = p["use_type_bias"],
        use_edge_gate = p["use_edge_gate"],
        use_raw_inject = p["use_raw_inject"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"]
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode      = "max",          # maximise AUC, not minimise loss
        factor    = p["sched_factor"],
        patience  = p["sched_patience"],
        min_lr    = 1e-7,
    )
    criterion = BCELoss(reduction="none")

    best_auc          = -1.0
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):

        # ── Train step ────────────────────────────────────────────────────────
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out  = model(batch).squeeze()
            y    = batch.y.float()
            loss = (criterion(out, y) * batch.weight.float()).sum() \
                   / batch.weight.float().sum()
            loss.backward()
            optimizer.step()

        # ── Validation AUC ────────────────────────────────────────────────────
        model.eval()
        logits_list, labels_list, weights_list = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits_list.append(model(batch).squeeze().cpu())
                labels_list.append(batch.y.float().cpu())
                weights_list.append(batch.weight.float().cpu())

        logits  = torch.cat(logits_list).numpy()
        labels  = torch.cat(labels_list).numpy()
        weights = torch.cat(weights_list).numpy()

        # Guard against degenerate splits (single class present)
        if len(np.unique(labels)) < 2:
            raise optuna.exceptions.TrialPruned()

        val_auc = roc_auc_score(labels, logits, sample_weight=weights)
        scheduler.step(val_auc)

        # ── Pruning ───────────────────────────────────────────────────────────
        trial.report(val_auc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        # ── Early stopping ────────────────────────────────────────────────────
        if val_auc > best_auc:
            best_auc          = val_auc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= p["es_patience"]:
                break

    return best_auc


# ──────────────────────────────────────────────────────────────────────────────
# Public: run_scan
# ──────────────────────────────────────────────────────────────────────────────

def run_scan(
    data_list  : list,
    n_trials   : int  = 30,
    max_epochs : int  = 50,
    study_name : str  = "vbs_gnn_scan",
    outdir     : str  = ".",        # <-- replaces storage= param, write to EOS directly
    n_startup  : int  = 10,
    show_progress_bar: bool = True,
) -> dict:
    import json

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    u_dim  = data_list[0].u.size(1)

    sampler = TPESampler(n_startup_trials=n_startup, seed=0)
    pruner  = MedianPruner(
        n_startup_trials = n_startup,
        n_warmup_steps   = 10,
        interval_steps   = 2,
    )

    # Always in-memory — no SQLite, no filesystem locking issues
    study = optuna.create_study(
        study_name = study_name,
        storage    = None,
        direction  = "maximize",
        sampler    = sampler,
        pruner     = pruner,
    )

    study.optimize(
        lambda trial: _run_trial(trial, data_list, device, max_epochs, u_dim),
        n_trials          = n_trials,
        show_progress_bar = show_progress_bar,
        gc_after_trial    = True,
    )

    # ── Report to stdout ──────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print(f"Best trial : #{study.best_trial.number}")
    print(f"Best AUC   : {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k:20s} = {v}")
    print("═" * 60)

    # ── Persist to EOS as plain files (no locking needed) ────────────────────
    os.makedirs(outdir, exist_ok=True)
    _save_scan_report(study, outdir=outdir)

    # Best params as JSON so you can reload without re-running
    json_path = os.path.join(outdir, f"{study_name}_best_params.json")
    with open(json_path, "w") as f:
        json.dump({"best_auc": study.best_value, **study.best_params}, f, indent=2)
    print(f"[hparam_scan] Best params → {json_path}")

    return study.best_params
# ──────────────────────────────────────────────────────────────────────────────
# Public: load_best_trial
# ──────────────────────────────────────────────────────────────────────────────

def load_best_trial(study_name: str, outdir: str) -> dict:
    import json
    json_path = os.path.join(outdir, f"{study_name}_best_params.json")
    with open(json_path) as f:
        params = json.load(f)
    params.pop("best_auc", None)   # remove the metadata field before passing to model
    return params


# ──────────────────────────────────────────────────────────────────────────────
# Reporting helpers
# ──────────────────────────────────────────────────────────────────────────────

def _save_scan_report(study: optuna.Study, outdir: str = ".") -> None:
    """
    Save two artefacts:
      scan_results.csv   — all completed trials, sorted by AUC
      scan_importance.csv — hyperparameter importances (FAnova)
    """
    import pandas as pd

    # ── All trials table ──────────────────────────────────────────────────────
    rows = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        row = {"trial": t.number, "val_auc": t.value}
        row.update(t.params)
        rows.append(row)

    if rows:
        df_trials = pd.DataFrame(rows).sort_values("val_auc", ascending=False)
        path = os.path.join(outdir, "scan_results.csv")
        df_trials.to_csv(path, index=False)
        print(f"[hparam_scan] Trial table → {path}")

    # ── Parameter importances ─────────────────────────────────────────────────
    try:
        importances = optuna.importance.get_param_importances(study)
        df_imp = pd.DataFrame(
            importances.items(), columns=["parameter", "importance"]
        ).sort_values("importance", ascending=False)
        path = os.path.join(outdir, "scan_importance.csv")
        df_imp.to_csv(path, index=False)
        print(f"[hparam_scan] Importances  → {path}")
        print(df_imp.to_string(index=False))
    except Exception as e:
        print(f"[hparam_scan] Could not compute importances: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Integration: drop-in replacement for neural_net_initialization() in main()
# ──────────────────────────────────────────────────────────────────────────────

def build_model_from_params(params: dict, u_dim: int, device: torch.device):
    """
    Reconstruct model + optimiser + scheduler from a best_params dict
    (as returned by run_scan or load_best_trial).

    Replaces the hardcoded neural_net_initialization() call for final training.
    """
    model = PhysicsHomoGraph(
        channels      = 64,
        num_layers    = 3,
        u_dim         = u_dim,
        heads         = 3,
        use_dr_near_bias   = p["use_dr_near_bias"],
        # use_dr_far_bias   = p["use_dr_far_bias"],
        use_type_bias = True,
        use_edge_gate = True,
        use_raw_inject = True
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


# ──────────────────────────────────────────────────────────────────────────────
# How to wire this into your existing main()
# ──────────────────────────────────────────────────────────────────────────────
#
#   # After: data_list = get_hographical_fast(df, cache_path=cache)
#
#   from hparam_scan import run_scan, build_model_from_params
#
#   best_params = run_scan(
#       data_list,
#       n_trials   = 60,
#       max_epochs = 80,
#       study_name = f"vbs_{args.category}",
#       storage    = f"sqlite:///vbs_{args.category}_scan.db",
#   )
#
#   # Use the best random_state for the final split
#   rs = best_params["random_state"]
#   train_val_list, test_list  = train_test_split(data_list, test_size=0.2,  random_state=rs)
#   train_list,     val_list   = train_test_split(train_val_list, test_size=0.25, random_state=rs)
#
#   train_loader = DataLoader(
#       balance_signal_background_weights_graphs(train_list, best_params["balance_to"]),
#       batch_size=best_params["batch_size"], shuffle=True)
#   val_loader   = DataLoader(
#       balance_signal_background_weights_graphs(val_list,   best_params["balance_to"]),
#       batch_size=best_params["batch_size"] * 2, shuffle=False)
#   test_loader  = DataLoader(
#       balance_signal_background_weights_graphs(test_list,  best_params["balance_to"]),
#       batch_size=best_params["batch_size"] * 2, shuffle=False)
#
#   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#   u_dim  = data_list[0].u.size(1)
#   model, optimizer, scheduler = build_model_from_params(best_params, u_dim, device)
#
#   # Continue with your existing train/test loop, substituting:
#   #   patience  → best_params["es_patience"]
#   #   scheduler → already configured above