"""Central configuration: coordinates, variables, hyperparameters, device, seed.

Matches the contract in BUILD_SPEC.md section 13. Importing this module also wires up
reproducible seeding and exposes the paths used across the pipeline. These are the
*defaults*; every value under "LSTM hyperparameters" / "Windowing" is overridable per
run via the run config dict (see ``runner.py`` / ``train.py``).
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

# --- Reproducibility ---------------------------------------------------------
SEED = 42

# --- Location (Varna, Bulgaria) ---------------------------------------------
LAT, LON, TZ = 43.21, 27.91, "Europe/Sofia"

# --- Data ---------------------------------------------------------------------
START_DATE = "1940-01-01"
HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "surface_pressure",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "cloud_cover",
    "shortwave_radiation",
]
TARGET = "temperature_2m"

# --- Windowing ----------------------------------------------------------------
LOOKBACK = 168          # L (one week of hours)
HORIZON = 1             # H (set 24 for the multi-step phase)
STRIDE = 1              # 12-24 for fast sweeps, 1 for the final run

# --- LSTM hyperparameters -----------------------------------------------------
HIDDEN = 128
NUM_LAYERS = 2
DROPOUT = 0.2
BATCH = 256
EPOCHS = 30
LR = 1e-3
EARLY_STOP_PATIENCE = 5   # epochs without val-RMSE improvement before stopping
USE_AMP = True
USE_ANOMALY = False

# Tuning run (default): evaluate on validation only, test untouched.
# Set True for exactly ONE run: retrains on train+val merged and evaluates test once.
IS_FINAL = False

# Chronological split fractions (train / val / test), in timeline order.
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# test = remainder (~0.15)

# --- Runtime ------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# ERA5 has a ~5-7 day update lag; pull up to today minus this many days.
ERA5_LAG_DAYS = 7
# Fetch in chunks of this many years to keep each request small/robust.
FETCH_CHUNK_YEARS = 5

# DataLoader workers. The spec suggests 4 on a Linux GPU box; on Windows the worker is
# launched from inside a uvicorn background thread, where multiprocessing spawn is flaky,
# and our windows are cheap in-memory slices, so 0 is the safe, fast default.
NUM_WORKERS = 0

# --- Paths --------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_DIR / "data"
RUNS_DIR = BACKEND_DIR / "runs"
CACHE_PARQUET = DATA_DIR / "varna_hourly.parquet"


def default_config() -> dict:
    """Return a fresh dict of the per-run, overridable defaults.

    The run system copies this and overrides individual keys (hidden_size, etc.).
    """
    return {
        "hidden_size": HIDDEN,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "lookback": LOOKBACK,
        "horizon": HORIZON,
        "stride": STRIDE,
        "batch": BATCH,
        "epochs": EPOCHS,
        "lr": LR,
        "use_amp": USE_AMP,
        "use_anomaly": USE_ANOMALY,
        "is_final": IS_FINAL,
        "seed": SEED,
    }


def set_seed(seed: int = SEED) -> None:
    """Seed python, numpy and torch (incl. CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs() -> None:
    """Create the gitignored output directories if they don't exist."""
    for d in (DATA_DIR, RUNS_DIR):
        d.mkdir(parents=True, exist_ok=True)


# Seed on import so every entrypoint is reproducible; log it.
set_seed(SEED)
print(f"[config] seed={SEED} device={DEVICE}")
