"""Phase 2 — naive baselines (BUILD_SPEC.md 6.1).

All baselines predict ``H`` hours ahead for a set of windows. A window is identified by its
absolute start row ``i`` in the underlying hourly series; with lookback ``L`` the input is
``series[i:i+L]`` and the **first target hour** is at position ``p = i + L`` (targets are
``series[p : p + H]``). The last observed hour is ``p - 1``.

Every function takes the raw target series (deg C), the per-timestamp climatology array (deg C),
the window starts, ``lookback`` and ``horizon``; each returns an ``(N, H)`` array in real units,
evaluated on the exact same windows as the LSTM so comparisons are fair.

Data: Open-Meteo Historical Weather API (ECMWF ERA5), CC BY 4.0.
"""

from __future__ import annotations

import numpy as np


def _target_positions(starts: np.ndarray, lookback: int) -> np.ndarray:
    """First-target position ``p = i + L`` for each window start ``i``."""
    return np.asarray(starts, dtype=np.int64) + lookback


def actual_targets(
    target_raw: np.ndarray, starts: np.ndarray, lookback: int, horizon: int
) -> np.ndarray:
    """Ground-truth temperature (deg C) for each window's H target hours -> ``(N, H)``."""
    p = _target_positions(starts, lookback)
    out = np.empty((len(p), horizon), dtype=float)
    for k in range(horizon):
        out[:, k] = target_raw[p + k]
    return out


def persistence(
    target_raw: np.ndarray, starts: np.ndarray, lookback: int, horizon: int
) -> np.ndarray:
    """Persistence: every future hour equals the last observed hour (``series[p-1]``)."""
    p = _target_positions(starts, lookback)
    last = target_raw[p - 1]                              # (N,)
    return np.repeat(last[:, None], horizon, axis=1)      # (N, H)


def diurnal_persistence(
    target_raw: np.ndarray, starts: np.ndarray, lookback: int, horizon: int
) -> np.ndarray:
    """Diurnal persistence: hour ``t+k`` equals the same hour the previous day (``t+k-24``)."""
    p = _target_positions(starts, lookback)
    out = np.empty((len(p), horizon), dtype=float)
    for k in range(horizon):
        out[:, k] = target_raw[p + k - 24]
    return out


def climatology(
    clim_values: np.ndarray, starts: np.ndarray, lookback: int, horizon: int
) -> np.ndarray:
    """Climatology: the (day-of-year, hour) climatological value at each target timestamp."""
    p = _target_positions(starts, lookback)
    out = np.empty((len(p), horizon), dtype=float)
    for k in range(horizon):
        out[:, k] = clim_values[p + k]
    return out


def all_baselines(
    target_raw: np.ndarray,
    clim_values: np.ndarray,
    starts: np.ndarray,
    lookback: int,
    horizon: int,
) -> dict[str, np.ndarray]:
    """Compute persistence, diurnal persistence, and climatology for the given windows."""
    return {
        "persistence": persistence(target_raw, starts, lookback, horizon),
        "diurnal": diurnal_persistence(target_raw, starts, lookback, horizon),
        "climatology": climatology(clim_values, starts, lookback, horizon),
    }
