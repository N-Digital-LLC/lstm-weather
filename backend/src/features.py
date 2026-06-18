"""Phase 2 — Feature engineering: cyclical calendar features, climatology, anomaly.

Three responsibilities (BUILD_SPEC.md 5.2):

1. Cyclical calendar features: hour-of-day sin/cos, day-of-year sin/cos, plus an optional
   normalized year for long-term trend. These let the LSTM lock onto the diurnal and annual
   cycles directly.
2. Climatology: the mean target per ``(day-of-year, hour)`` slot computed over the **training
   period only**, lightly smoothed along the year. It powers both the climatology baseline and
   the optional anomaly target.
3. Anomaly mode: ``target = value - climatology``; predict the anomaly and add climatology back
   at inference.

All "learned" statistics (climatology) are fit on the **training rows only**; the rest of the
pipeline passes in a training-restricted slice so we never leak the future. Importing this
module has no side effects.

Data: Open-Meteo Historical Weather API (ECMWF ERA5), CC BY 4.0.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Names of the engineered calendar columns appended by ``add_calendar_features``.
CALENDAR_FEATURES = ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "year_norm"]


def add_calendar_features(df: pd.DataFrame, *, with_year: bool = True) -> pd.DataFrame:
    """Append cyclical hour-of-day / day-of-year features (and an optional year trend).

    Returns a new DataFrame; the input is not mutated.
    """
    out = df.copy()
    idx = out.index
    hour = idx.hour.to_numpy()
    doy = idx.dayofyear.to_numpy()
    # Account for leap years so the annual cycle stays phase-aligned.
    days_in_year = np.where(idx.is_leap_year, 366.0, 365.0)

    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["doy_sin"] = np.sin(2 * np.pi * (doy - 1) / days_in_year)
    out["doy_cos"] = np.cos(2 * np.pi * (doy - 1) / days_in_year)

    if with_year:
        years = idx.year.to_numpy().astype(np.float64)
        span = max(years.max() - years.min(), 1.0)
        out["year_norm"] = (years - years.min()) / span
    else:
        out["year_norm"] = 0.0
    return out


def feature_columns(base_vars: list[str], *, with_year: bool = True) -> list[str]:
    """Full ordered model-input column list: raw vars + calendar features."""
    cal = CALENDAR_FEATURES if with_year else [c for c in CALENDAR_FEATURES if c != "year_norm"]
    return list(base_vars) + cal


@dataclass
class Climatology:
    """Mean target per ``(day-of-year, hour)`` slot, smoothed along the year.

    ``grid`` has shape ``(366, 24)`` indexed by ``(doy - 1, hour)``. Day-of-year is a 1:1 proxy
    for ``(month, day)`` within a year and makes circular smoothing along the calendar trivial.
    Fit on the training rows only (see :func:`fit_climatology`).
    """

    grid: np.ndarray          # (366, 24) deg C
    fallback: float           # global training mean, used if a slot is ever empty
    target: str

    def predict_index(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Climatological value for every timestamp in ``index`` (deg C)."""
        doy = np.clip(index.dayofyear.to_numpy(), 1, 366)
        hour = index.hour.to_numpy()
        return self.grid[doy - 1, hour]

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: str | Path) -> "Climatology":
        with open(path, "rb") as fh:
            return pickle.load(fh)


def _circular_smooth(grid: np.ndarray, window: int) -> np.ndarray:
    """Circular moving average along axis 0 (day-of-year) with the given window."""
    if window <= 1:
        return grid
    pad = window // 2
    padded = np.concatenate([grid[-pad:], grid, grid[:pad]], axis=0)
    kernel = np.ones(window) / window
    out = np.empty_like(grid)
    for h in range(grid.shape[1]):
        smoothed = np.convolve(padded[:, h], kernel, mode="same")
        out[:, h] = smoothed[pad:-pad]
    return out


def fit_climatology(
    train_df: pd.DataFrame,
    target: str,
    *,
    smooth_window: int = 15,
) -> Climatology:
    """Fit climatology on the **training slice only** (no leakage).

    Builds a ``(366, 24)`` grid of mean target by ``(day-of-year, hour)``, fills any empty slots
    by interpolation along the calendar, then lightly smooths along the year to remove sampling
    noise without erasing the seasonal/diurnal shape.
    """
    doy = train_df.index.dayofyear.to_numpy()
    hour = train_df.index.hour.to_numpy()
    values = train_df[target].to_numpy(dtype=float)

    grid = np.full((366, 24), np.nan, dtype=float)
    sums = np.zeros((366, 24))
    counts = np.zeros((366, 24))
    np.add.at(sums, (doy - 1, hour), values)
    np.add.at(counts, (doy - 1, hour), 1.0)
    nonzero = counts > 0
    grid[nonzero] = sums[nonzero] / counts[nonzero]

    fallback = float(np.nanmean(values))

    # Fill empty slots per hour-column by linear interpolation along day-of-year.
    for h in range(24):
        col = grid[:, h]
        mask = np.isnan(col)
        if mask.all():
            col[:] = fallback
        elif mask.any():
            idx = np.arange(366)
            col[mask] = np.interp(idx[mask], idx[~mask], col[~mask])
        grid[:, h] = col

    grid = _circular_smooth(grid, smooth_window)
    return Climatology(grid=grid, fallback=fallback, target=target)


def to_anomaly(values: np.ndarray, climatology_values: np.ndarray) -> np.ndarray:
    """Convert raw target to anomaly: ``value - climatology``."""
    return values - climatology_values


def from_anomaly(anomaly: np.ndarray, climatology_values: np.ndarray) -> np.ndarray:
    """Invert the anomaly transform: ``anomaly + climatology``."""
    return anomaly + climatology_values


def chronological_split_bounds(
    n: int, train_frac: float, val_frac: float
) -> tuple[int, int]:
    """Return ``(train_end, val_end)`` row indices for a strict chronological split.

    Rows ``[0:train_end)`` are train, ``[train_end:val_end)`` are validation, and
    ``[val_end:n)`` are test. No shuffling — the test set is always the most recent period.
    """
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return train_end, val_end
