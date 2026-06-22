"""Phase 3/4/9 — Train exactly ONE config and write a full report into ``runs/<run_id>/``.

Callable from the CLI (``python -m src.train --hidden 128 ...``) and from the run worker
(``runner.py``). Writes ``card.json`` + ``history.csv`` + plots + ``model.pt`` /
``scaler.pkl`` / ``climatology.pkl``.

Two-phase discipline (BUILD_SPEC 7.2):
  - Tuning runs (``is_final=false``, default): early-stop on validation; ``val_metrics``
    populated; ``test_metrics``/``skill_vs`` stay ``null``. The test set is never touched.
  - Final run (``is_final=true``, set once): retrain from scratch on train+val merged, no
    peeking, then evaluate the test set exactly once and fill ``test_metrics``/``skill_vs``.

Data: Open-Meteo Historical Weather API (ECMWF ERA5), CC BY 4.0.
"""

from __future__ import annotations

import argparse
import pickle
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from . import config, evaluate, store, windowing
from .models import baselines
from .models.lstm import build_from_config


def _load_dataframe() -> pd.DataFrame:
    if not config.CACHE_PARQUET.exists():
        raise FileNotFoundError(
            f"No cached data at {config.CACHE_PARQUET}. Run `python -m src.fetch` first."
        )
    return pd.read_parquet(config.CACHE_PARQUET)


def _make_loader(prepared: windowing.Prepared, starts, batch: int, *, shuffle: bool) -> DataLoader:
    ds = windowing.WindowDataset(prepared, starts)
    use_workers = config.NUM_WORKERS > 0
    return DataLoader(
        ds,
        batch_size=batch,
        shuffle=shuffle,
        pin_memory=(config.DEVICE == "cuda"),
        num_workers=config.NUM_WORKERS,
        persistent_workers=use_workers,
        drop_last=False,
    )


@torch.no_grad()
def predict_temps(
    model: nn.Module,
    prepared: windowing.Prepared,
    starts: np.ndarray,
    *,
    batch: int,
    use_amp: bool,
) -> np.ndarray:
    """Run the model over a split's windows -> temperature predictions in deg C, ``(n, H)``."""
    model.eval()
    loader = _make_loader(prepared, starts, batch, shuffle=False)
    chunks = []
    amp = use_amp and config.DEVICE == "cuda"
    for xb, _ in loader:
        xb = xb.to(config.DEVICE, non_blocking=True)
        with torch.autocast("cuda", enabled=amp):
            out = model(xb)
        chunks.append(out.float().cpu().numpy())
    y_scaled = np.concatenate(chunks, axis=0)
    return windowing.inverse_target(prepared, y_scaled, starts)


def _evaluate_split(
    model, prepared, split: str, *, batch: int, use_amp: bool, with_mape: bool
):
    """Compute LSTM + baseline metrics for a split. Returns (metrics, skill_vs, horizon)."""
    starts = prepared.starts[split]
    lstm_pred = predict_temps(model, prepared, starts, batch=batch, use_amp=use_amp)
    actual = baselines.actual_targets(
        prepared.target_raw, starts, prepared.lookback, prepared.horizon
    )
    base = baselines.all_baselines(
        prepared.target_raw, prepared.clim_values, starts, prepared.lookback, prepared.horizon
    )
    metrics, skill_vs, horizon = evaluate.assemble_metrics(lstm_pred, actual, base)
    return metrics, skill_vs, horizon, lstm_pred, actual, starts


def train_one(cfg: dict, run_id: str | None = None) -> str:
    """Train a single config end-to-end and return its run_id."""
    config.ensure_dirs()
    full_cfg = config.default_config()
    full_cfg.update(cfg or {})
    config.set_seed(int(full_cfg["seed"]))

    if run_id is None:
        run_id = store.make_run_id(full_cfg)
        store.new_card(run_id, full_cfg)

    store.set_status(run_id, "running")
    store.init_history(run_id)

    is_final = bool(full_cfg["is_final"])
    batch = int(full_cfg["batch"])
    epochs = int(full_cfg["epochs"])
    use_amp = bool(full_cfg["use_amp"]) and config.DEVICE == "cuda"
    lr = float(full_cfg["lr"])

    # --- Data ---------------------------------------------------------------
    df = _load_dataframe()
    prepared = windowing.prepare(df, full_cfg)

    # Record the split (years + window counts) up-front so the UI shows it during training,
    # not just when the run finishes.
    store.patch_card(
        run_id,
        data={
            "train_years": prepared.years["train_years"],
            "val_years": prepared.years["val_years"],
            "test_years": prepared.years["test_years"],
            "n_train_windows": int(len(prepared.starts["train"])),
            "n_val_windows": int(len(prepared.starts.get("val", []))),
            "n_test_windows": int(len(prepared.starts.get("test", []))),
            "features": prepared.feature_names,
        },
    )

    train_loader = _make_loader(prepared, prepared.starts["train"], batch, shuffle=True)
    monitor_split = None if is_final else "val"

    # --- Model / optim ------------------------------------------------------
    model = build_from_config(prepared.n_features, full_cfg).to(config.DEVICE)
    print(f"[train] {run_id} on {next(model.parameters()).device} | features={prepared.n_features}")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()
    grad_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_rmse = float("inf")
    best_state = None
    early_stop_epoch = None
    patience = config.EARLY_STOP_PATIENCE
    epochs_without_improve = 0
    started = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running, n_batches = 0.0, 0
        for xb, yb in train_loader:
            xb = xb.to(config.DEVICE, non_blocking=True)
            yb = yb.to(config.DEVICE, non_blocking=True)
            opt.zero_grad()
            with torch.autocast("cuda", enabled=use_amp):
                loss = loss_fn(model(xb), yb)
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_scaler.step(opt)
            grad_scaler.update()
            running += float(loss.item())
            n_batches += 1
        train_loss = running / max(n_batches, 1)

        # Validation (tuning only). Final run trains for fixed epochs, no peeking.
        val_loss = None
        val_rmse = None
        if monitor_split is not None:
            val_pred = predict_temps(
                model, prepared, prepared.starts[monitor_split], batch=batch, use_amp=use_amp
            )
            val_actual = baselines.actual_targets(
                prepared.target_raw,
                prepared.starts[monitor_split],
                prepared.lookback,
                prepared.horizon,
            )
            val_rmse = evaluate.rmse(val_pred, val_actual)
            val_loss = float(np.mean(np.abs(val_pred - val_actual)))  # MAE proxy for the curve

        store.append_history(run_id, epoch, train_loss, val_loss, val_rmse)
        store.update_progress(run_id, epoch, epochs)
        dt = time.time() - t0
        print(
            f"[train] {run_id} epoch {epoch}/{epochs} "
            f"train_loss={train_loss:.4f} "
            + (f"val_rmse={val_rmse:.3f}C " if val_rmse is not None else "")
            + f"({dt:.1f}s)"
        )

        if monitor_split is not None:
            if val_rmse < best_val_rmse - 1e-4:
                best_val_rmse = val_rmse
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                early_stop_epoch = epoch
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1
                if epochs_without_improve >= patience:
                    print(f"[train] {run_id} early stop at epoch {epoch}")
                    break

    epochs_run = epoch
    if best_state is not None:
        model.load_state_dict(best_state)
    train_time = time.time() - started

    # --- Final evaluation ---------------------------------------------------
    card = store.read_card(run_id)
    card["data"] = {
        "train_years": prepared.years["train_years"],
        "val_years": prepared.years["val_years"],
        "test_years": prepared.years["test_years"],
        "n_train_windows": int(len(prepared.starts["train"])),
        "n_val_windows": int(len(prepared.starts.get("val", []))),
        "n_test_windows": int(len(prepared.starts.get("test", []))),
        "features": prepared.feature_names,
    }

    if is_final:
        metrics, skill_vs, horizon, lstm_pred, actual, starts = _evaluate_split(
            model, prepared, "test", batch=batch, use_amp=use_amp, with_mape=True
        )
        card["test_metrics"] = metrics
        card["skill_vs"] = skill_vs
        card["test_horizon"] = horizon
        card["val_metrics"] = None
        best_val_rmse_out = None
        # Pred-vs-actual plot on the test slice (next-hour series).
        test_index = prepared.index[starts + prepared.lookback]
        evaluate.plot_pred_vs_actual(
            np.asarray(test_index), actual[:, 0], lstm_pred[:, 0],
            store.plots_dir(run_id) / "pred_vs_actual.png",
        )
        evaluate.plot_rmse_vs_horizon(
            horizon, store.plots_dir(run_id) / "rmse_vs_horizon.png", split="test"
        )
    else:
        metrics, _skill, horizon, lstm_pred, actual, starts = _evaluate_split(
            model, prepared, "val", batch=batch, use_amp=use_amp, with_mape=False
        )
        card["val_metrics"] = {
            "lstm": {k: metrics["lstm"][k] for k in ("mae_C", "rmse_C", "bias_C", "r2")}
        }
        card["val_horizon"] = horizon
        card["test_metrics"] = None
        card["skill_vs"] = None
        best_val_rmse_out = round(float(best_val_rmse), 4) if np.isfinite(best_val_rmse) else None
        if prepared.horizon > 1:
            evaluate.plot_rmse_vs_horizon(
                horizon, store.plots_dir(run_id) / "rmse_vs_horizon.png", split="val"
            )

    card["training"] = {
        "epochs_planned": epochs,
        "epochs_run": int(epochs_run),
        "early_stop_epoch": int(early_stop_epoch) if early_stop_epoch else None,
        "best_val_rmse_C": best_val_rmse_out,
        "train_time_seconds": round(train_time, 1),
    }
    card["progress"] = {"current_epoch": int(epochs_run), "total_epochs": epochs}
    store.write_card(run_id, card)

    # Training curve plot from history.csv.
    try:
        evaluate.plot_training_curve(
            store.history_path(run_id), store.plots_dir(run_id) / "training_curve.png"
        )
    except Exception as exc:  # plotting must never fail the run
        print(f"[train] {run_id} plot warning: {exc}")

    # --- Persist artifacts --------------------------------------------------
    torch.save(
        {"state_dict": model.state_dict(), "config": full_cfg, "feature_names": prepared.feature_names},
        store.model_path(run_id),
    )
    with open(store.scaler_path(run_id), "wb") as fh:
        pickle.dump(
            {
                "feature_scaler": prepared.feature_scaler,
                "target_scaler": prepared.target_scaler,
                "feature_names": prepared.feature_names,
                "use_anomaly": prepared.use_anomaly,
            },
            fh,
        )
    with open(store.climatology_path(run_id), "wb") as fh:
        pickle.dump(prepared.clim_series, fh)

    store.set_status(run_id, "done")
    print(f"[train] {run_id} done in {train_time:.0f}s")
    return run_id


def main() -> None:
    p = argparse.ArgumentParser(description="Train one LSTM config and write a run report.")
    p.add_argument("--hidden", type=int, default=config.HIDDEN)
    p.add_argument("--layers", type=int, default=config.NUM_LAYERS)
    p.add_argument("--lookback", type=int, default=config.LOOKBACK)
    p.add_argument("--horizon", type=int, default=config.HORIZON)
    p.add_argument("--stride", type=int, default=config.STRIDE)
    p.add_argument("--batch", type=int, default=config.BATCH)
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--lr", type=float, default=config.LR)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--anomaly", action="store_true")
    p.add_argument("--final", action="store_true", help="Train final model on train+val, eval test once.")
    args = p.parse_args()

    cfg = config.default_config()
    cfg.update(
        {
            "hidden_size": args.hidden,
            "num_layers": args.layers,
            "lookback": args.lookback,
            "horizon": args.horizon,
            "stride": args.stride,
            "batch": args.batch,
            "epochs": args.epochs,
            "lr": args.lr,
            "use_amp": not args.no_amp,
            "use_anomaly": args.anomaly,
            "is_final": args.final,
        }
    )
    run_id = train_one(cfg)
    print(f"Run complete: {run_id}")


if __name__ == "__main__":
    main()
