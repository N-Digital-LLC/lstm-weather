"""Backfill the bias / R^2 / per-horizon MAE+bias metrics onto already-finished runs.

The newer metrics can't be derived from the scalar values stored in ``card.json`` (bias and R^2
need the raw predictions), so this re-evaluates each finished run from its saved artifacts and
patches ``card.json`` in place. Reproduction is exact: ``windowing.prepare`` is deterministic and
every run saved its ``model.pt`` / ``scaler.pkl`` / ``climatology.pkl``.

Usage::

    python -m src.backfill_metrics              # every finished run missing the new metrics
    python -m src.backfill_metrics --force      # recompute even runs that already have them
    python -m src.backfill_metrics --run-id <id>  # a single run

Data: Open-Meteo Historical Weather API (ECMWF ERA5), CC BY 4.0.
"""

from __future__ import annotations

import argparse

import pandas as pd

from . import config, evaluate, serve, store, train, windowing
from .models import baselines


def _already_done(card: dict, split: str) -> bool:
    """True if this card's LSTM metrics for ``split`` already carry the new ``bias_C`` field."""
    block = card.get("test_metrics") if split == "test" else card.get("val_metrics")
    lstm = (block or {}).get("lstm") or {}
    return "bias_C" in lstm


def backfill_run(run_id: str, df: pd.DataFrame, *, force: bool = False) -> str:
    """Re-evaluate one finished run and patch its ``card.json``. Returns a short status string."""
    card = store.read_card(run_id)
    if card.get("status") != "done":
        return f"skip ({card.get('status')})"

    cfg = card.get("config") or {}
    split = "test" if cfg.get("is_final") else "val"
    if _already_done(card, split) and not force:
        return "skip (already has metrics)"

    full_cfg = config.default_config()
    full_cfg.update(cfg)

    prepared = windowing.prepare(df, full_cfg)
    starts = prepared.starts[split]
    if len(starts) == 0:
        return f"skip (no {split} windows)"

    model = serve.load_run(run_id).model
    lstm_pred = train.predict_temps(
        model, prepared, starts, batch=int(full_cfg["batch"]), use_amp=False
    )
    actual = baselines.actual_targets(
        prepared.target_raw, starts, prepared.lookback, prepared.horizon
    )
    base = baselines.all_baselines(
        prepared.target_raw, prepared.clim_values, starts, prepared.lookback, prepared.horizon
    )
    metrics, skill_vs, horizon = evaluate.assemble_metrics(lstm_pred, actual, base)

    old_rmse = ((card.get("test_metrics" if split == "test" else "val_metrics") or {}).get("lstm") or {}).get("rmse_C")

    if split == "test":
        card["test_metrics"] = metrics
        card["skill_vs"] = skill_vs
        card["test_horizon"] = horizon
        card["val_metrics"] = None
    else:
        card["val_metrics"] = {
            "lstm": {k: metrics["lstm"][k] for k in ("mae_C", "rmse_C", "bias_C", "r2")}
        }
        card["val_horizon"] = horizon
        card["test_metrics"] = None
        card["skill_vs"] = None

    store.write_card(run_id, card)
    new = metrics["lstm"]
    return (
        f"ok [{split}] rmse {old_rmse}->{new['rmse_C']} "
        f"bias={new['bias_C']} r2={new['r2']}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Backfill bias/R^2/per-horizon metrics onto finished runs.")
    p.add_argument("--run-id", help="Backfill only this run (default: all finished runs).")
    p.add_argument("--force", action="store_true", help="Recompute even if the metrics already exist.")
    args = p.parse_args()

    df = windowing.load_dataframe()
    run_ids = [args.run_id] if args.run_id else store.list_run_ids()

    n_ok = 0
    for run_id in run_ids:
        try:
            status = backfill_run(run_id, df, force=args.force)
        except Exception as exc:  # one bad run must not abort the whole batch
            status = f"FAILED ({type(exc).__name__}: {exc})"
        if status.startswith("ok"):
            n_ok += 1
        print(f"[backfill] {run_id}: {status}")

    print(f"[backfill] done — {n_ok}/{len(run_ids)} run(s) updated.")


if __name__ == "__main__":
    main()
