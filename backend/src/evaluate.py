"""Phase 2/4 — Evaluation harness: metrics, skill scores, and report plots.

Metrics are always computed in real units (deg C) after any inverse-transform; the caller
passes prediction/actual arrays of shape ``(n_windows, horizon)``. Plots are written with a
non-interactive backend so this runs head-less inside the training worker thread.

Data: Open-Meteo Historical Weather API (ECMWF ERA5), CC BY 4.0.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # head-less; no display needed inside the worker thread
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_EPS = 1e-6


# --- Core metrics -------------------------------------------------------------
def mae(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - actual)))


def rmse(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - actual) ** 2)))


def mape(pred: np.ndarray, actual: np.ndarray) -> float:
    """Mean absolute percentage error (%). Near-zero actuals are masked out.

    Temperature in deg C crosses zero, so MAPE is reported with that caveat; it is only
    surfaced for the LSTM on the final test set (per the report schema).
    """
    mask = np.abs(actual) > _EPS
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((pred[mask] - actual[mask]) / actual[mask])) * 100.0)


def rmse_per_horizon(pred: np.ndarray, actual: np.ndarray) -> list[float]:
    """RMSE at each horizon hour -> list of length ``horizon``."""
    return [rmse(pred[:, k], actual[:, k]) for k in range(pred.shape[1])]


def skill_score(rmse_model: float, rmse_baseline: float) -> float:
    """``1 - rmse_model / rmse_baseline``; > 0 means the model beats the baseline."""
    if rmse_baseline < _EPS:
        return float("nan")
    return float(1.0 - rmse_model / rmse_baseline)


def basic_metrics(pred: np.ndarray, actual: np.ndarray, *, with_mape: bool = False) -> dict:
    out = {"mae_C": round(mae(pred, actual), 4), "rmse_C": round(rmse(pred, actual), 4)}
    if with_mape:
        out["mape_pct"] = round(mape(pred, actual), 4)
    return out


def assemble_metrics(
    lstm_pred: np.ndarray,
    actual: np.ndarray,
    baseline_preds: dict[str, np.ndarray],
) -> tuple[dict, dict, dict]:
    """Build the report's metrics blocks for one split.

    Returns ``(metrics, skill_vs, horizon)`` where:
      - ``metrics`` maps model -> {mae_C, rmse_C[, mape_pct]} (lstm + every baseline)
      - ``skill_vs`` maps baseline -> skill score of the LSTM against it
      - ``horizon`` holds per-horizon RMSE arrays for the RMSE-vs-horizon chart
    """
    metrics: dict[str, dict] = {"lstm": basic_metrics(lstm_pred, actual, with_mape=True)}
    for name, pred in baseline_preds.items():
        metrics[name] = basic_metrics(pred, actual)

    skill_vs = {
        name: round(skill_score(metrics["lstm"]["rmse_C"], metrics[name]["rmse_C"]), 4)
        for name in baseline_preds
    }

    hours = list(range(1, actual.shape[1] + 1))
    horizon = {"hours": hours, "lstm": rmse_per_horizon(lstm_pred, actual)}
    for name, pred in baseline_preds.items():
        horizon[name] = rmse_per_horizon(pred, actual)

    return metrics, skill_vs, horizon


# --- Plots --------------------------------------------------------------------
def plot_training_curve(history_csv: Path, out_path: Path) -> None:
    """Train/val loss + val RMSE per epoch, from ``history.csv``."""
    df = pd.read_csv(history_csv)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(df["epoch"], df["train_loss"], label="train loss", color="tab:blue")
    ax1.plot(df["epoch"], df["val_loss"], label="val loss", color="tab:orange")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax2 = ax1.twinx()
    ax2.plot(df["epoch"], df["val_rmse_C"], label="val RMSE (C)", color="tab:green", ls="--")
    ax2.set_ylabel("val RMSE (C)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.set_title("Training curve")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_rmse_vs_horizon(horizon: dict, out_path: Path, *, split: str = "val") -> None:
    """RMSE vs horizon hour for the LSTM and every baseline (the key exam chart)."""
    hours = horizon["hours"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in ("lstm", "persistence", "diurnal", "climatology"):
        if name in horizon:
            ax.plot(hours, horizon[name], marker="o", ms=3, label=name)
    ax.set_xlabel("horizon hour")
    ax.set_ylabel("RMSE (C)")
    ax.set_title(f"RMSE vs horizon ({split})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_pred_vs_actual(
    times: np.ndarray,
    actual: np.ndarray,
    lstm: np.ndarray,
    out_path: Path,
    *,
    n_hours: int = 240,
) -> None:
    """Predicted (next-hour) vs actual over a multi-day slice of the split."""
    n = min(n_hours, len(actual))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(times[:n], actual[:n], label="actual", color="black", lw=1.5)
    ax.plot(times[:n], lstm[:n], label="lstm (+1h)", color="tab:red", lw=1.0)
    ax.set_xlabel("time")
    ax.set_ylabel("temperature (C)")
    ax.set_title("Predicted vs actual (test slice)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
