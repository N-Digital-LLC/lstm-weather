# -*- coding: utf-8 -*-
"""Snapshot the read-only API responses into static JSON for the frontend export.

This drives the real FastAPI app in-process (Starlette TestClient), so the static
snapshots are byte-for-byte what the live backend would return — including
precomputed forecasts (real model inference) for a handful of test-year dates.

Output goes to ``frontend/public/snapshots/``. The static frontend build reads
these files instead of calling ``http://localhost:8000``.

Run from ``backend/`` with the venv python:

    .venv\\Scripts\\python.exe make_snapshots.py
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from src.api import app

BACKEND_DIR = Path(__file__).resolve().parent
SNAP_DIR = BACKEND_DIR.parent / "frontend" / "public" / "snapshots"

# Representative last-observed timestamps across seasons, inside the held-out test
# years (2013-2026) so the model never trained on them and actuals exist.
FORECAST_DATETIMES = [
    "2024-01-15T12:00",
    "2024-04-15T12:00",
    "2024-07-15T12:00",
    "2024-10-15T12:00",
    "2020-07-01T00:00",
    "2016-12-20T06:00",
]
FORECAST_HORIZONS = [1, 6, 12, 24]


def _write(rel: str, obj) -> None:
    path = SNAP_DIR / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    client = TestClient(app)

    # --- health -------------------------------------------------------------
    _write("health.json", {"status": "ok", "device": "snapshot (static export)"})

    # --- data split ---------------------------------------------------------
    r = client.get("/data/split")
    r.raise_for_status()
    _write("data_split.json", r.json())
    print("data_split.json")

    # --- runs list ----------------------------------------------------------
    r = client.get("/runs")
    r.raise_for_status()
    runs_payload = r.json()
    _write("runs.json", runs_payload)
    run_ids = [run["run_id"] for run in runs_payload.get("runs", [])]
    print(f"runs.json ({len(run_ids)} runs)")

    # --- per-run card + history --------------------------------------------
    for i, rid in enumerate(run_ids, 1):
        rc = client.get(f"/runs/{rid}")
        if rc.status_code == 200:
            _write(f"runs/{rid}.json", rc.json())
        rh = client.get(f"/runs/{rid}/history")
        if rh.status_code == 200:
            _write(f"history/{rid}.json", rh.json())
        if i % 25 == 0:
            print(f"  ...{i}/{len(run_ids)} run snapshots")
    print(f"runs/ + history/ ({len(run_ids)} each)")

    # --- precomputed forecasts (real inference, final model) ----------------
    final_id = next(
        (run["run_id"] for run in runs_payload["runs"]
         if run.get("is_final") and run.get("status") == "done"),
        None,
    )
    index = []
    n_ok = 0
    for dt in FORECAST_DATETIMES:
        for h in FORECAST_HORIZONS:
            params = {"datetime": dt, "horizon": h}
            if final_id:
                params["run_id"] = final_id
            fr = client.get("/forecast", params=params)
            if fr.status_code != 200:
                print(f"  forecast {dt} h{h} -> {fr.status_code} (skipped)")
                continue
            fname = f"forecasts/{dt.replace(':', '-')}__h{h}.json"
            _write(fname, fr.json())
            index.append({"datetime": dt, "horizon": h, "run_id": final_id, "file": fname})
            n_ok += 1
    _write("forecasts/index.json", {"final_run_id": final_id, "items": index})
    print(f"forecasts/ ({n_ok} precomputed, final_run_id={final_id})")

    print("\nSnapshots written to", SNAP_DIR)


if __name__ == "__main__":
    main()
