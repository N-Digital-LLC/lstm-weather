"""Run storage helpers: the on-disk layout for ``runs/<run_id>/``.

No database — a run is just a folder with a ``card.json`` report, a ``history.csv`` epoch log,
the saved ``model.pt`` / ``scaler.pkl`` / ``climatology.pkl``, and a ``plots/`` directory.
Listing/reading runs simply scans ``runs/`` and parses each ``card.json``.

Both ``train.py`` (writes progress + final report) and ``runner.py`` (queue/worker) use these
helpers; keeping them here avoids a circular import between those two modules.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from . import config

_TZ = ZoneInfo(config.TZ)
HISTORY_COLUMNS = ["epoch", "train_loss", "val_loss", "val_rmse_C"]


# --- Paths --------------------------------------------------------------------
def runs_root() -> Path:
    config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    return config.RUNS_DIR


def run_dir(run_id: str) -> Path:
    return runs_root() / run_id


def card_path(run_id: str) -> Path:
    return run_dir(run_id) / "card.json"


def history_path(run_id: str) -> Path:
    return run_dir(run_id) / "history.csv"


def plots_dir(run_id: str) -> Path:
    d = run_dir(run_id) / "plots"
    d.mkdir(parents=True, exist_ok=True)
    return d


def model_path(run_id: str) -> Path:
    return run_dir(run_id) / "model.pt"


def scaler_path(run_id: str) -> Path:
    return run_dir(run_id) / "scaler.pkl"


def climatology_path(run_id: str) -> Path:
    return run_dir(run_id) / "climatology.pkl"


def now_iso() -> str:
    return datetime.now(_TZ).isoformat(timespec="seconds")


def make_run_id(cfg: dict) -> str:
    """``<YYYY-MM-DD_HH-MM-SS>__h{hidden}_l{layers}_L{lookback}_lr{lr}``.

    Encodes the swept hyperparameters so per-param/matrix sweep runs are distinguishable on disk
    (otherwise lr/lookback-only sweeps would share a stem). ``runner._unique_run_id`` still
    de-duplicates any remaining same-second collisions.
    """
    ts = datetime.now(_TZ).strftime("%Y-%m-%d_%H-%M-%S")
    return (
        f"{ts}__h{int(cfg['hidden_size'])}_l{int(cfg['num_layers'])}"
        f"_L{int(cfg['lookback'])}_lr{float(cfg['lr']):g}"
    )


# --- card.json read/write -----------------------------------------------------
def read_card(run_id: str) -> dict:
    with open(card_path(run_id), "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_card(run_id: str, card: dict) -> None:
    """Atomic write so a polling UI never reads a half-written report."""
    path = card_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(card, fh, indent=2, default=str)
    os.replace(tmp, path)


def new_card(run_id: str, cfg: dict) -> dict:
    """Initial ``queued`` report, written the moment a run is enqueued."""
    card = {
        "run_id": run_id,
        "status": "queued",
        "error": None,
        "started_at": None,
        "finished_at": None,
        "device": config.DEVICE,
        "config": {
            "hidden_size": int(cfg["hidden_size"]),
            "num_layers": int(cfg["num_layers"]),
            "lookback": int(cfg["lookback"]),
            "horizon": int(cfg["horizon"]),
            "batch": int(cfg["batch"]),
            "lr": float(cfg["lr"]),
            "stride": int(cfg["stride"]),
            "use_amp": bool(cfg["use_amp"]),
            "use_anomaly": bool(cfg["use_anomaly"]),
            "is_final": bool(cfg["is_final"]),
            "seed": int(cfg["seed"]),
        },
        "data": None,
        "training": None,
        "progress": {"current_epoch": 0, "total_epochs": int(cfg["epochs"])},
        "val_metrics": None,
        "test_metrics": None,
        "skill_vs": None,
        "val_horizon": None,
        "test_horizon": None,
    }
    write_card(run_id, card)
    return card


def patch_card(run_id: str, **fields) -> dict:
    card = read_card(run_id)
    card.update(fields)
    write_card(run_id, card)
    return card


def set_status(run_id: str, status: str, *, error: str | None = None) -> dict:
    card = read_card(run_id)
    card["status"] = status
    if error is not None:
        card["error"] = error
    if status == "running" and not card.get("started_at"):
        card["started_at"] = now_iso()
    if status in ("done", "failed"):
        card["finished_at"] = now_iso()
    write_card(run_id, card)
    return card


def update_progress(run_id: str, current_epoch: int, total_epochs: int) -> None:
    card = read_card(run_id)
    card["progress"] = {"current_epoch": current_epoch, "total_epochs": total_epochs}
    write_card(run_id, card)


# --- history.csv --------------------------------------------------------------
def init_history(run_id: str) -> None:
    with open(history_path(run_id), "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(HISTORY_COLUMNS)


def append_history(run_id: str, epoch: int, train_loss: float, val_loss, val_rmse_C) -> None:
    with open(history_path(run_id), "a", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(
            [
                epoch,
                f"{train_loss:.6f}",
                "" if val_loss is None else f"{val_loss:.6f}",
                "" if val_rmse_C is None else f"{val_rmse_C:.6f}",
            ]
        )


def read_history(run_id: str) -> list[dict]:
    path = history_path(run_id)
    if not path.exists():
        return []
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append(
                {
                    "epoch": int(row["epoch"]),
                    "train_loss": _to_float(row.get("train_loss")),
                    "val_loss": _to_float(row.get("val_loss")),
                    "val_rmse_C": _to_float(row.get("val_rmse_C")),
                }
            )
    return rows


def _to_float(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# --- listing / deletion -------------------------------------------------------
def list_run_ids() -> list[str]:
    root = runs_root()
    ids = [p.name for p in root.iterdir() if p.is_dir() and (p / "card.json").exists()]
    return sorted(ids)


def list_cards() -> list[dict]:
    cards = []
    for run_id in list_run_ids():
        try:
            cards.append(read_card(run_id))
        except (OSError, json.JSONDecodeError):
            continue
    cards.sort(key=lambda c: c.get("run_id", ""), reverse=True)
    return cards


def delete_run(run_id: str) -> bool:
    d = run_dir(run_id)
    if d.exists():
        shutil.rmtree(d)
        return True
    return False
