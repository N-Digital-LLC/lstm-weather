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


def bias(pred: np.ndarray, actual: np.ndarray) -> float:
    """Mean error (deg C). Positive means the model runs warm; negative means cold.

    Unlike MAE/RMSE (magnitude only), this exposes a systematic over- or under-prediction.
    """
    return float(np.mean(pred - actual))


def r2(pred: np.ndarray, actual: np.ndarray) -> float:
    """Coefficient of determination: fraction of variance explained vs predicting the mean.

    ``1 - SS_res / SS_tot``; 1.0 is perfect, 0.0 matches a constant-mean predictor, and it can
    go negative when the model is worse than the mean. NaN if the actuals have no variance.
    """
    ss_res = float(np.sum((actual - pred) ** 2))
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
    if ss_tot < _EPS:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


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


def mae_per_horizon(pred: np.ndarray, actual: np.ndarray) -> list[float]:
    """MAE at each horizon hour -> list of length ``horizon``."""
    return [mae(pred[:, k], actual[:, k]) for k in range(pred.shape[1])]


def bias_per_horizon(pred: np.ndarray, actual: np.ndarray) -> list[float]:
    """Bias (mean error) at each horizon hour -> list of length ``horizon``."""
    return [bias(pred[:, k], actual[:, k]) for k in range(pred.shape[1])]


def skill_score(rmse_model: float, rmse_baseline: float) -> float:
    """``1 - rmse_model / rmse_baseline``; > 0 means the model beats the baseline."""
    if rmse_baseline < _EPS:
        return float("nan")
    return float(1.0 - rmse_model / rmse_baseline)


def basic_metrics(pred: np.ndarray, actual: np.ndarray, *, with_mape: bool = False) -> dict:
    out = {
        "mae_C": round(mae(pred, actual), 4),
        "rmse_C": round(rmse(pred, actual), 4),
        "bias_C": round(bias(pred, actual), 4),
        "r2": round(r2(pred, actual), 4),
    }
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
      - ``horizon`` holds per-horizon RMSE arrays (flat, per model) plus nested ``mae`` and
        ``bias`` blocks ({model -> per-horizon array}) for the extra charts
    """
    metrics: dict[str, dict] = {"lstm": basic_metrics(lstm_pred, actual, with_mape=True)}
    for name, pred in baseline_preds.items():
        metrics[name] = basic_metrics(pred, actual)

    skill_vs = {
        name: round(skill_score(metrics["lstm"]["rmse_C"], metrics[name]["rmse_C"]), 4)
        for name in baseline_preds
    }

    hours = list(range(1, actual.shape[1] + 1))
    # Flat per-model RMSE arrays are kept at the top level for backward compatibility with the
    # existing RMSE-vs-horizon charts; MAE and bias are added as nested {model -> array} blocks.
    horizon: dict = {"hours": hours, "lstm": rmse_per_horizon(lstm_pred, actual)}
    mae_block = {"lstm": mae_per_horizon(lstm_pred, actual)}
    bias_block = {"lstm": bias_per_horizon(lstm_pred, actual)}
    for name, pred in baseline_preds.items():
        horizon[name] = rmse_per_horizon(pred, actual)
        mae_block[name] = mae_per_horizon(pred, actual)
        bias_block[name] = bias_per_horizon(pred, actual)
    horizon["mae"] = mae_block
    horizon["bias"] = bias_block

    return metrics, skill_vs, horizon


# --- Plots --------------------------------------------------------------------
def plot_training_curve(history_csv: Path, out_path: Path) -> None:
    """Train/val loss + val RMSE per epoch, from ``history.csv``.

    Final runs train on merged train+val with no validation monitor, so the
    ``val_loss`` / ``val_rmse_C`` columns are empty. Only series that actually
    have data are drawn (and the right-hand RMSE axis is added only when needed),
    so the legend never lists empty curves.
    """
    df = pd.read_csv(history_csv)

    def has_data(col: str) -> bool:
        return col in df.columns and df[col].notna().any()

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(df["epoch"], df["train_loss"], label="train loss", color="tab:blue")
    if has_data("val_loss"):
        ax1.plot(df["epoch"], df["val_loss"], label="val loss", color="tab:orange")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")

    lines, labels = ax1.get_legend_handles_labels()
    if has_data("val_rmse_C"):
        ax2 = ax1.twinx()
        ax2.plot(
            df["epoch"], df["val_rmse_C"], label="val RMSE (C)", color="tab:green", ls="--"
        )
        ax2.set_ylabel("val RMSE (C)")
        l2, lab2 = ax2.get_legend_handles_labels()
        lines += l2
        labels += lab2

    ax1.legend(lines, labels, loc="upper right")
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
