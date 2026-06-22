"""Phase 6 — Forecast/history serving helpers used by ``api.py``.

Loads the cached parquet once and, per chosen run, the saved ``model.pt`` / ``scaler.pkl`` /
``climatology.pkl``. A forecast uses the ``L`` hours ending at the requested datetime as the model
input and returns, for each of the next ``H`` hours, the LSTM prediction alongside all three
baselines (and the actual value when it falls inside the cached archive).

Everything here mirrors training exactly: the same calendar features, the same saved scalers, and
the same anomaly handling, so served numbers match the run's reported metrics.

Data: Open-Meteo Historical Weather API (ECMWF ERA5), CC BY 4.0.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from . import config, store
from .features import Climatology, add_calendar_features, chronological_split_bounds
from .models.lstm import build_from_config

_df: pd.DataFrame | None = None
_models: dict[str, "LoadedRun"] = {}


@dataclass
class LoadedRun:
    run_id: str
    model: torch.nn.Module
    feature_scaler: object
    target_scaler: object
    climatology: Climatology
    feature_names: list[str]
    use_anomaly: bool
    lookback: int
    horizon: int


def get_dataframe() -> pd.DataFrame:
    """Cached hourly archive (raw HOURLY_VARS columns), loaded once."""
    global _df
    if _df is None:
        if not config.CACHE_PARQUET.exists():
            raise FileNotFoundError(
                f"No cached data at {config.CACHE_PARQUET}. Run `python -m src.fetch` first."
            )
        _df = pd.read_parquet(config.CACHE_PARQUET).sort_index()
    return _df


def _done_cards() -> list[dict]:
    return [c for c in store.list_cards() if c.get("status") == "done"]


def default_run_id() -> str | None:
    """Best done run (lowest ``best_val_rmse_C``); fall back to the most recent done run."""
    done = _done_cards()
    if not done:
        return None
    scored = [
        c for c in done
        if (c.get("training") or {}).get("best_val_rmse_C") is not None
    ]
    if scored:
        best = min(scored, key=lambda c: c["training"]["best_val_rmse_C"])
        return best["run_id"]
    return done[0]["run_id"]                              # list_cards is sorted newest-first


def load_run(run_id: str) -> LoadedRun:
    """Load (and cache) a run's model + scalers + climatology for inference."""
    if run_id in _models:
        return _models[run_id]

    ckpt = torch.load(store.model_path(run_id), map_location=config.DEVICE)
    feature_names = ckpt["feature_names"]
    model = build_from_config(len(feature_names), ckpt["config"]).to(config.DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    with open(store.scaler_path(run_id), "rb") as fh:
        scalers = pickle.load(fh)
    with open(store.climatology_path(run_id), "rb") as fh:
        clim = pickle.load(fh)

    loaded = LoadedRun(
        run_id=run_id,
        model=model,
        feature_scaler=scalers["feature_scaler"],
        target_scaler=scalers["target_scaler"],
        climatology=clim,
        feature_names=feature_names,
        use_anomaly=bool(scalers.get("use_anomaly", False)),
        lookback=int(ckpt["config"]["lookback"]),
        horizon=int(ckpt["config"]["horizon"]),
    )
    _models[run_id] = loaded
    return loaded


def data_split() -> dict:
    """Canonical chronological train/val/test split of the cached archive.

    Deterministic (70/15/15 by row count, never shuffled) and identical for every tuning run,
    so the UI can show "which years are train/val/test" without needing a run to exist. The
    ``final`` note reflects that final runs merge train+val and evaluate the held-out test years.
    """
    df = get_dataframe()
    n = len(df)
    train_end, val_end = chronological_split_bounds(n, config.TRAIN_FRAC, config.VAL_FRAC)
    idx = df.index

    def segment(start: int, end: int) -> dict:
        if end <= start:
            return {"years": "", "start": None, "end": None, "rows": 0}
        return {
            "years": f"{idx[start].year}-{idx[end - 1].year}",
            "start": idx[start].isoformat(),
            "end": idx[end - 1].isoformat(),
            "rows": int(end - start),
        }

    return {
        "location": "Varna",
        "total_rows": int(n),
        "fractions": {
            "train": config.TRAIN_FRAC,
            "val": config.VAL_FRAC,
            "test": round(1.0 - config.TRAIN_FRAC - config.VAL_FRAC, 4),
        },
        "train": segment(0, train_end),
        "val": segment(train_end, val_end),
        "test": segment(val_end, n),
        "note": (
            "Strict chronological split (earliest→latest, never shuffled). Tuning runs train on "
            "the train years and are judged on validation; the test years stay untouched. A final "
            "run merges train+val and evaluates the test years exactly once."
        ),
    }


def history(start: str | None, end: str | None, var: str) -> dict:
    """Cached hourly series for ``var`` between ``start`` and ``end`` (inclusive)."""
    df = get_dataframe()
    if var not in df.columns:
        raise KeyError(f"Unknown variable {var!r}. Available: {list(df.columns)}")
    s = df[var]
    if start is not None:
        s = s[s.index >= pd.Timestamp(start)]
    if end is not None:
        s = s[s.index <= pd.Timestamp(end)]
    return {
        "var": var,
        "series": [
            {"datetime": ts.isoformat(), "value": _maybe_float(v)}
            for ts, v in s.items()
        ],
    }


def _maybe_float(v) -> float | None:
    f = float(v)
    return None if np.isnan(f) else f


@torch.no_grad()
def forecast(datetime_str: str, horizon: int, run_id: str | None) -> dict:
    """Forecast the next ``horizon`` hours after ``datetime`` using the chosen (or default) run."""
    if run_id is None:
        run_id = default_run_id()
    if run_id is None:
        raise FileNotFoundError("No completed runs available to forecast with.")

    run = load_run(run_id)
    df = get_dataframe()
    feat_df = add_calendar_features(df, with_year=True)

    t = pd.Timestamp(datetime_str)
    if t not in df.index:
        # snap to the most recent available hour at or before t
        pos = df.index.searchsorted(t, side="right") - 1
        if pos < 0:
            raise ValueError(f"datetime {datetime_str} is before the available archive.")
    else:
        pos = int(df.index.get_loc(t))

    L = run.lookback
    if pos - L + 1 < 0:
        raise ValueError(f"Not enough history before {datetime_str} for lookback {L}.")

    # Effective horizon is bounded by what the model was trained to emit (direct multi-step).
    H = min(int(horizon), run.horizon)

    # Build the (1, L, F) scaled input from the L hours ending at pos.
    window = feat_df.iloc[pos - L + 1 : pos + 1][run.feature_names].to_numpy(dtype=np.float64)
    x_scaled = run.feature_scaler.transform(window).astype(np.float32)
    xb = torch.from_numpy(x_scaled[None, :, :]).to(config.DEVICE)
    out = run.model(xb).float().cpu().numpy().ravel()    # (run.horizon,)

    # Invert target scaler; add climatology back at each target hour if anomaly mode.
    mean = float(run.target_scaler.mean_[0])
    scale = float(run.target_scaler.scale_[0])
    deg = out * scale + mean

    last_time = df.index[pos]
    target_times = [last_time + pd.Timedelta(hours=k + 1) for k in range(H)]
    clim_at_targets = run.climatology.predict_index(pd.DatetimeIndex(target_times))

    if run.use_anomaly:
        lstm_vals = [float(deg[k] + clim_at_targets[k]) for k in range(H)]
    else:
        lstm_vals = [float(deg[k]) for k in range(H)]

    temp = df[config.TARGET]
    last_observed = float(temp.iloc[pos])

    rows = []
    for k in range(H):
        tt = target_times[k]
        prev_day = tt - pd.Timedelta(hours=24)
        diurnal = float(temp.loc[prev_day]) if prev_day in temp.index else None
        actual = float(temp.loc[tt]) if tt in temp.index else None
        rows.append(
            {
                "datetime": tt.isoformat(),
                "lstm": round(lstm_vals[k], 3),
                "persistence": round(last_observed, 3),
                "diurnal": None if diurnal is None else round(diurnal, 3),
                "climatology": round(float(clim_at_targets[k]), 3),
                "actual": None if actual is None else round(actual, 3),
            }
        )

    return {
        "location": "Varna",
        "issued_for": last_time.isoformat(),
        "horizon_hours": H,
        "unit": "C",
        "run_id": run_id,
        "model": "lstm",
        "forecast": rows,
    }
