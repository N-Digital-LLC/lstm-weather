"""Phase 3 — Lazy sliding-window dataset + strict chronological split (BUILD_SPEC.md 5.3).

The cardinal rule for hourly data: **do not pre-materialize** the ``(n_windows, L, F)`` array —
it is multiple GB and OOMs. Instead we scale the base series **once** and the ``Dataset``
slices each window on the fly in ``__getitem__``. Memory stays flat regardless of window count.

Other invariants enforced here:
- **Strictly chronological split**, never shuffled. Train = earliest ~70%, val = next ~15%,
  test = most recent ~15%.
- **Scaler + climatology fit on the training pool only.** For tuning runs the pool is the train
  years; for the final run (``is_final``) the pool is train+val merged (test still held out).
- Each window is fully contained in its split (lookback does not reach across a boundary), so the
  three splits are independent.
- ``STRIDE`` thins window *starts* for fast sweeps; resolution inside each window is unchanged.

Data: Open-Meteo Historical Weather API (ECMWF ERA5), CC BY 4.0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from . import config
from .features import (
    Climatology,
    add_calendar_features,
    chronological_split_bounds,
    feature_columns,
    fit_climatology,
)


@dataclass
class Prepared:
    """Everything train/evaluate/serve need, derived from the cached parquet for one config."""

    index: pd.DatetimeIndex
    feature_names: list[str]
    n_features: int
    features_scaled: np.ndarray         # (T, F) float32, scaled with the train-pool feature scaler
    target_scaled: np.ndarray           # (T,) float32, model-space target (anomaly- or raw-aware)
    target_raw: np.ndarray              # (T,) deg C actual temperature
    clim_values: np.ndarray             # (T,) deg C climatology per timestamp
    feature_scaler: StandardScaler
    target_scaler: StandardScaler
    clim_series: Climatology            # picklable climatology object (saved per run)
    use_anomaly: bool
    lookback: int
    horizon: int
    starts: dict[str, np.ndarray]       # "train" / "val" / "test" -> absolute window starts
    bounds: tuple[int, int]             # (train_end, val_end) row indices
    is_final: bool
    years: dict[str, str]


class WindowDataset(Dataset):
    """Lazy sliding-window dataset over a :class:`Prepared`'s scaled base series.

    Holds references to the already-scaled feature matrix and target vector (no per-window
    materialization). ``__getitem__`` slices an ``(L, F)`` input and its ``(H,)`` target.
    """

    def __init__(self, prepared: Prepared, starts: np.ndarray) -> None:
        self.X = prepared.features_scaled                # np.float32 (T, F)
        self.y = prepared.target_scaled                  # np.float32 (T,)
        self.starts = np.asarray(starts, dtype=np.int64)
        self.L = prepared.lookback
        self.H = prepared.horizon

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, k: int):
        i = int(self.starts[k])
        x = torch.from_numpy(self.X[i : i + self.L])                     # (L, F)
        y = torch.from_numpy(self.y[i + self.L : i + self.L + self.H])   # (H,)
        return x, y


def inverse_target(prepared: Prepared, y_scaled: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Invert model-space predictions ``(N, H)`` back to deg C.

    Undo the target scaler; if the run is in anomaly mode, add the per-timestamp climatology back
    at each predicted hour (``p + k`` for window start ``i``, ``p = i + L``).
    """
    mean = float(prepared.target_scaler.mean_[0])
    scale = float(prepared.target_scaler.scale_[0])
    deg = y_scaled * scale + mean
    if prepared.use_anomaly:
        p = np.asarray(starts, dtype=np.int64) + prepared.lookback
        out = np.empty_like(deg)
        for k in range(deg.shape[1]):
            out[:, k] = deg[:, k] + prepared.clim_values[p + k]
        return out
    return deg


def _window_starts(split_start: int, split_end: int, lookback: int, horizon: int, stride: int) -> np.ndarray:
    """Absolute start indices ``i`` for windows fully contained in ``[split_start, split_end)``."""
    first = split_start
    last = split_end - lookback - horizon                # inclusive last valid start
    if last < first:
        return np.empty(0, dtype=np.int64)
    return np.arange(first, last + 1, stride, dtype=np.int64)


def _years(index: pd.DatetimeIndex, start: int, end: int) -> str:
    """Human-readable inclusive year range for the rows ``[start, end)``."""
    if end <= start:
        return ""
    return f"{index[start].year}-{index[end - 1].year}"


def prepare(df: pd.DataFrame, cfg: dict) -> Prepared:
    """Build scalers, climatology, and the lazy window starts for one run config.

    No leakage: the feature scaler, target scaler, and climatology are all fit on the **training
    pool** only. For ``is_final`` runs the pool is train+val merged and test is the held-out set;
    otherwise the pool is the train years and validation is held out for model selection.
    """
    lookback = int(cfg["lookback"])
    horizon = int(cfg["horizon"])
    stride = int(cfg["stride"])
    use_anomaly = bool(cfg["use_anomaly"])
    is_final = bool(cfg["is_final"])

    df = df.sort_index()
    df = add_calendar_features(df, with_year=True)
    feat_names = feature_columns(config.HOURLY_VARS, with_year=True)

    n = len(df)
    train_end, val_end = chronological_split_bounds(n, config.TRAIN_FRAC, config.VAL_FRAC)
    pool_end = val_end if is_final else train_end        # rows the scaler/climatology may see

    target_raw = df[config.TARGET].to_numpy(dtype=np.float64)

    # Climatology fit on the training pool only.
    clim = fit_climatology(df.iloc[:pool_end], config.TARGET)
    clim_values = clim.predict_index(df.index)

    # Feature scaler fit on the training pool rows only.
    feat_matrix = df[feat_names].to_numpy(dtype=np.float64)
    feature_scaler = StandardScaler().fit(feat_matrix[:pool_end])
    features_scaled = feature_scaler.transform(feat_matrix).astype(np.float32)
    features_scaled = np.ascontiguousarray(features_scaled)

    # Target (model space): anomaly or raw, scaled with a 1-D scaler fit on the pool only.
    target_model_raw = (target_raw - clim_values) if use_anomaly else target_raw
    target_scaler = StandardScaler().fit(target_model_raw[:pool_end].reshape(-1, 1))
    target_scaled = (
        target_scaler.transform(target_model_raw.reshape(-1, 1)).astype(np.float32).ravel()
    )
    target_scaled = np.ascontiguousarray(target_scaled)

    # Window starts per split.
    if is_final:
        train_starts = _window_starts(0, val_end, lookback, horizon, stride)
        val_starts = np.empty(0, dtype=np.int64)         # merged into the training pool
    else:
        train_starts = _window_starts(0, train_end, lookback, horizon, stride)
        val_starts = _window_starts(train_end, val_end, lookback, horizon, stride)
    test_starts = _window_starts(val_end, n, lookback, horizon, stride)

    years = {
        "train_years": _years(df.index, 0, pool_end),
        "val_years": "" if is_final else _years(df.index, train_end, val_end),
        "test_years": _years(df.index, val_end, n),
    }

    return Prepared(
        index=df.index,
        feature_names=feat_names,
        n_features=len(feat_names),
        features_scaled=features_scaled,
        target_scaled=target_scaled,
        target_raw=target_raw,
        clim_values=clim_values,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        clim_series=clim,
        use_anomaly=use_anomaly,
        lookback=lookback,
        horizon=horizon,
        starts={"train": train_starts, "val": val_starts, "test": test_starts},
        bounds=(train_end, val_end),
        is_final=is_final,
        years=years,
    )


def load_dataframe() -> pd.DataFrame:
    """Load the cached parquet (raise a clear error if Phase 1 hasn't been run)."""
    if not config.CACHE_PARQUET.exists():
        raise FileNotFoundError(
            f"Cache not found at {config.CACHE_PARQUET}. Run `python -m src.fetch` first."
        )
    return pd.read_parquet(config.CACHE_PARQUET).sort_index()
