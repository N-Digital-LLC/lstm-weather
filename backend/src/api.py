"""Phases 5/6 — FastAPI app: forecast, history, and the run/experiment endpoints.

Run with::

    uvicorn src.api:app --reload --port 8000

The single training worker is started at app startup (one GPU -> one job at a time). Run state is
read straight from ``runs/<run_id>/card.json`` via ``store.py`` — no database.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import config, runner, serve, store

app = FastAPI(title="Varna Hourly Weather LSTM", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    config.ensure_dirs()
    runner.start_worker()


# --- Request models -----------------------------------------------------------
class RunRequest(BaseModel):
    hidden_size: int | None = None
    num_layers: int | None = None
    lookback: int | None = None
    horizon: int | None = None
    stride: int | None = None
    batch: int | None = None
    epochs: int | None = None
    lr: float | None = None
    dropout: float | None = None
    use_amp: bool | None = None
    use_anomaly: bool | None = None
    is_final: bool | None = None
    seed: int | None = None

    def overrides(self) -> dict:
        return {k: v for k, v in self.model_dump().items() if v is not None}


class SweepRequest(RunRequest):
    hidden_sizes: list[int] = [64, 128, 256]

    def shared(self) -> dict:
        ov = self.overrides()
        ov.pop("hidden_sizes", None)
        ov.pop("hidden_size", None)              # hidden varies across the sweep
        return ov


# --- Health -------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    return {"status": "ok", "device": config.DEVICE}


# --- Forecast / history -------------------------------------------------------
@app.get("/history")
def history(
    start: str | None = None,
    end: str | None = None,
    var: str = "temperature_2m",
) -> dict:
    try:
        return serve.history(start, end, var)
    except (FileNotFoundError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/forecast")
def forecast(
    datetime: str = Query(..., description="YYYY-MM-DDTHH(:MM) — last observed hour"),
    horizon: int = 24,
    run_id: str | None = None,
) -> dict:
    try:
        return serve.forecast(datetime, horizon, run_id)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# --- Runs / experiments -------------------------------------------------------
@app.post("/runs")
def create_run(req: RunRequest) -> dict:
    run_id = runner.enqueue(req.overrides())
    return {"run_id": run_id, "status": "queued"}


@app.post("/runs/sweep")
def create_sweep(req: SweepRequest) -> dict:
    run_ids = runner.enqueue_sweep(req.hidden_sizes, req.shared())
    return {"run_ids": run_ids}


@app.get("/runs")
def list_runs() -> dict:
    summaries = []
    for card in store.list_cards():
        training = card.get("training") or {}
        test = card.get("test_metrics") or {}
        summaries.append(
            {
                "run_id": card.get("run_id"),
                "status": card.get("status"),
                "started_at": card.get("started_at"),
                "finished_at": card.get("finished_at"),
                "config": card.get("config"),
                "progress": card.get("progress"),
                "is_final": (card.get("config") or {}).get("is_final", False),
                "best_val_rmse_C": training.get("best_val_rmse_C"),
                "test_rmse_C": (test.get("lstm") or {}).get("rmse_C") if test else None,
                "error": card.get("error"),
            }
        )
    return {"runs": summaries}


@app.get("/runs/compare")
def compare_runs(ids: str = Query(..., description="comma-separated run_ids")) -> dict:
    run_ids = [r.strip() for r in ids.split(",") if r.strip()]
    runs = []
    for run_id in run_ids:
        try:
            card = store.read_card(run_id)
        except (OSError, ValueError):
            continue
        training = card.get("training") or {}
        runs.append(
            {
                "run_id": run_id,
                "config": card.get("config"),
                "status": card.get("status"),
                "is_final": (card.get("config") or {}).get("is_final", False),
                "best_val_rmse_C": training.get("best_val_rmse_C"),
                "val_metrics": card.get("val_metrics"),
                "test_metrics": card.get("test_metrics"),
                "skill_vs": card.get("skill_vs"),
                "val_horizon": card.get("val_horizon"),
                "test_horizon": card.get("test_horizon"),
                "history": store.read_history(run_id),
            }
        )
    return {"runs": runs}


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict:
    try:
        return store.read_card(run_id)
    except (OSError, ValueError):
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")


@app.get("/runs/{run_id}/history")
def get_run_history(run_id: str) -> dict:
    return {"run_id": run_id, "history": store.read_history(run_id)}


@app.delete("/runs/{run_id}")
def delete_run(run_id: str) -> dict:
    ok = store.delete_run(run_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")
    return {"deleted": run_id}
