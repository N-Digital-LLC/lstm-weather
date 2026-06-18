# Hourly Weather Forecasting with an LSTM (Varna) — Build Spec

> **For the human:** Save as `BUILD_SPEC.md` in an empty repo root, open the folder in Cursor, and say:
> *"Read BUILD_SPEC.md and implement it. Work through the phases in order. After each phase, stop and let me run/verify it before continuing."*
> Each phase in section 10 ends in something runnable — don't let the agent build it all at once.

---

## 1. Overview

A local tool that predicts **hourly** weather for **Varna, Bulgaria** using an **LSTM** trained on ERA5 reanalysis from the **Open-Meteo Historical Weather API** — and lets you **launch training runs from the web UI**, each producing a **timestamped report**, and **compare runs** (e.g. hidden size 64 vs 128 vs 256) with charts.

- **Data source:** `https://archive-api.open-meteo.com/v1/archive` — ERA5, **hourly**, January 1940 to present, gap-free, no API key, CC BY 4.0 (attribute Open-Meteo / ECMWF ERA5).
- **Location:** Varna — `latitude=43.21`, `longitude=27.91`, `timezone=Europe/Sofia`.
- **Scale:** ~750,000 hourly timesteps. Hourly captures the diurnal cycle that daily aggregates destroy.
- **Primary target:** next-hour `temperature_2m`, then the next 24 hours.
- **Inputs:** temperature, relative humidity, dew point, surface pressure, precipitation, wind speed/direction, cloud cover, shortwave radiation, plus cyclical calendar features (hour-of-day AND day-of-year).

Compared head to head:
- **Naive baselines** — persistence, diurnal persistence (same hour yesterday), climatology.
- **LSTM** (PyTorch, GPU), with a **run/experiment system** so multiple configs can be trained and compared.

**Academic context:** PhD exam on LSTMs. Required deliverables: chronological evaluation, baseline comparison, RMSE-vs-horizon curve, and a **hidden-size comparison (64/128/256)** with per-run reports — the evidence for discussing capacity vs overfitting and LSTM strengths/weaknesses.

## 2. Goals and non-goals

**Goals**
- End-to-end pipeline: fetch -> cache -> features -> lazy windowing -> GPU training -> chronological eval -> serve -> visualize.
- **Experiment system:** launch training from the UI, queue jobs, store a timestamped report per run, compare runs with charts.
- Honest evaluation vs persistence, diurnal persistence, climatology, for next-hour and next-24h.

**Non-goals**
- Not trying to beat operational numerical weather prediction (say why).
- No cloud deploy, no auth, no multi-user. Single user, single GPU, local.
- No heavy job-queue infrastructure (no Celery/Redis) — a simple in-process worker thread is enough.

## 3. Tech stack

**Backend (Python 3.11)**
- `torch` (CUDA build) — LSTM on GPU
- `pandas`, `numpy`, `pyarrow` — data + parquet cache
- `scikit-learn` — scalers, metrics
- `requests` — Open-Meteo fetch
- `fastapi`, `uvicorn[standard]`, `pydantic` — serving + run API
- `matplotlib` — report plots

**Frontend**
- Next.js (App Router) + TypeScript + Tailwind
- `recharts` or `chart.js` for hourly forecast charts AND run-comparison charts

**Constraints**
- Pin versions in `requirements.txt`. Global seed `42` (numpy, torch, cuda). Log it.
- Auto-select device: `cuda` if available else `cpu`; print which at startup.
- Cache fetched data to `backend/data/varna_hourly.parquet`; skip network unless `--refresh`.

## 4. Repository structure

```
weather-lstm/
├── BUILD_SPEC.md
├── README.md                   # generated: run instructions + Open-Meteo attribution
├── backend/
│   ├── requirements.txt
│   ├── data/                   # gitignored: cached parquet
│   ├── runs/                   # gitignored: one folder per training run (reports, models, plots)
│   ├── src/
│   │   ├── config.py           # coords, vars, default hyperparams, device, seed, paths
│   │   ├── fetch.py            # chunked hourly download + cache
│   │   ├── features.py         # hour/day cyclical features, climatology, anomaly
│   │   ├── windowing.py        # LAZY sliding-window Dataset; chronological split
│   │   ├── models/
│   │   │   ├── baselines.py    # persistence, diurnal persistence, climatology
│   │   │   └── lstm.py         # LSTM forecaster (single- and multi-step)
│   │   ├── train.py            # train ONE config; GPU/AMP; writes report (card.json) + history + plots
│   │   ├── evaluate.py         # MAE/RMSE/MAPE per horizon, skill scores, plots
│   │   ├── runner.py           # in-process job queue + worker thread; run storage helpers
│   │   └── api.py              # FastAPI: forecast, history, AND run endpoints
│   └── (artifacts live inside each runs/<run_id>/ folder)
└── frontend/
    └── (Next.js app: Forecast page + Training page + Comparison page)
```

`.gitignore` excludes `backend/data/`, `backend/runs/`, `node_modules/`.

## 5. Data

### 5.1 Fetch (`fetch.py`)

GET the archive endpoint with `hourly=`. Example:
```
https://archive-api.open-meteo.com/v1/archive
  ?latitude=43.21&longitude=27.91
  &start_date=1940-01-01&end_date=<today-7d>
  &hourly=temperature_2m,relative_humidity_2m,dew_point_2m,surface_pressure,precipitation,wind_speed_10m,wind_direction_10m,cloud_cover,shortwave_radiation
  &timezone=Europe/Sofia
```
- **Chunk by ~5-year windows and concatenate** (one 85-year hourly request is fragile); sort by timestamp.
- `end_date` ~ today - 7 days (ERA5 lag, else tail is null).
- Cache to `data/varna_hourly.parquet`; skip network if cache exists unless `--refresh`.
- Validate: continuous hourly index, no gaps; forward-fill stray nulls with a logged warning; print row count + date range.

### 5.2 Features (`features.py`)

- Cyclical: hour-of-day sin/cos, day-of-year sin/cos; optional normalized year for trend.
- Climatology: mean target per `(month, day, hour)` over **training years only**, lightly smoothed; saved per run.
- Anomaly mode (flag): target = value - climatology; predict anomaly, add climatology back at inference.

### 5.3 Windowing + split (`windowing.py`)

- Lookback `L = 168` hours -> predict next `H` hours (`H=1`, then `H=24`).
- **LAZY windowing — do not pre-materialize.** Store the scaled base series once; `Dataset.__getitem__(i)` slices on the fly. (Pre-building all windows is multiple GB and will OOM.)
- `STRIDE` config: default `1`; use `12`/`24` for fast sweeps. **Use the same stride/seed/data for all swept runs so the comparison is fair.**
- **Chronological split — never shuffle.** Train = earliest ~70%, val = next ~15%, test = most recent ~15%.
- Scaler fit on **training rows only**, saved per run.

## 6. Models

### 6.1 Baselines (`baselines.py`) — required
- **Persistence:** next hour = this hour.
- **Diurnal persistence:** hour t+k = same hour previous day (t+k-24). The strong baseline at multi-hour horizons.
- **Climatology:** the `(month, day, hour)` climatological value (training years only).

### 6.2 LSTM (`lstm.py`)
```
input: (B, L, F)
  -> LSTM(input_size=F, hidden=HIDDEN, num_layers=NUM_LAYERS, batch_first=True, dropout=DROPOUT)
  -> last timestep hidden state
  -> Dropout -> Linear(HIDDEN -> H)
output: (B, H)
```
- HIDDEN and NUM_LAYERS come from the run config (so the UI can vary them). Single-step (H=1) first, then direct multi-step (H=24).
- Loss SmoothL1/Huber or MSE; Adam, lr 1e-3, grad clip 1.0, early stopping on val RMSE, batch 256.

## 7. Training one run (`train.py`)

`train.py` trains **exactly one config** and writes everything into `runs/<run_id>/`. It is callable both from the CLI and from the runner.

- `run_id = "<YYYY-MM-DD_HH-MM-SS>__h{hidden}_l{layers}"`.
- GPU: move model + batches to device; `DataLoader(pin_memory=True, num_workers=4, persistent_workers=True)`; mixed precision via `torch.autocast("cuda")` + `GradScaler` (flag).
- After **each epoch**, append to `runs/<run_id>/history.csv` (epoch, train_loss, val_loss, val_rmse_C) AND update `card.json` `progress` (current_epoch) so the UI shows live progress.
- On finish: write `card.json`, save `model.pt`, `scaler.pkl`, `climatology.pkl`, and plots to `runs/<run_id>/plots/`.
- **Test evaluation is gated by `is_final`** (see 7.2). For tuning runs (`is_final=false`) `evaluate.py` runs **only on validation**; `test_metrics` and `skill_vs` are left `null`. Only the final run touches the test period.

### 7.2 Two-phase workflow & the `is_final` flag — do this exactly

This is the discipline an examiner will probe. Two distinct phases:

**Phase A — tuning (`is_final=false`, the default).**
- Trains on the **train** years, judged on the **validation** years.
- `evaluate.py` computes validation metrics only. `test_metrics`/`skill_vs` stay `null` in the report.
- All sweep runs (64/128/256, lookback, etc.) are tuning runs. You pick the winner by `best_val_rmse_C` — never by test.
- Result: every tuning report is *provably* untouched by the test set. The reports themselves are the proof of no leakage — stronger than any timestamp.

**Phase B — final model (`is_final=true`, set exactly once).**
- Re-trains the chosen config **from scratch** (new weights) on **train + validation merged** (e.g. 1940–2014), because tuning is done and more data makes a better final model.
- `evaluate.py` runs on the **test** period **once**, populating `test_metrics` and `skill_vs`.
- This is the only run whose report contains test numbers.

Rule of thumb to state at the defense: *"Tuned on validation, trained the final model on train+validation merged, touched the test set exactly once. Tuning reports have no test numbers at all — by construction, not by promise."*

When `is_final=true`, `windowing.py` merges train+val into the training pool and keeps test as the held-out set; `data.train_years` in the report becomes the merged range (e.g. `"1940-2014"`) and the report records `"is_final": true`.

### 7.1 Report schema — `runs/<run_id>/card.json`
This is the document you read from during the defense.
```json
{
  "run_id": "2026-06-17_22-30-05__h128_l2",
  "status": "done",                       // queued | running | done | failed
  "error": null,
  "started_at": "2026-06-17T22:30:05+03:00",
  "finished_at": "2026-06-17T23:14:48+03:00",
  "device": "cuda",
  "config": { "hidden_size":128, "num_layers":2, "lookback":168, "horizon":1,
              "batch":256, "lr":0.001, "stride":1, "use_amp":true,
              "use_anomaly":false, "is_final":false, "seed":42 },
  "data": { "train_years":"1940-2002", "val_years":"2003-2014", "test_years":"2015-2026",
            "n_train_windows":431520, "n_val_windows":89280, "n_test_windows":92000,
            "features":["temperature_2m","relative_humidity_2m","..."] },
  "training": { "epochs_planned":30, "epochs_run":19, "early_stop_epoch":19,
                "best_val_rmse_C":1.42, "train_time_seconds":2683 },
  "progress": { "current_epoch":19, "total_epochs":30 },
  "val_metrics": { "lstm": {"mae_C":1.11,"rmse_C":1.49} },
  "test_metrics": null,
  "skill_vs": null
}
```
- For **tuning runs** (`is_final=false`): `val_metrics` is populated, `test_metrics` and `skill_vs` are `null`.
- For the **final run** (`is_final=true`): `test_metrics` and `skill_vs` are filled in (same shape as below) and `data.train_years` shows the merged train+val range.

Final-run `test_metrics` shape:
```json
  "test_metrics": {
     "lstm":        {"mae_C":1.05,"rmse_C":1.42,"mape_pct":7.1},
     "persistence": {"mae_C":1.31,"rmse_C":1.78},
     "diurnal":     {"mae_C":1.19,"rmse_C":1.63},
     "climatology": {"mae_C":2.40,"rmse_C":3.05} },
  "skill_vs": { "persistence":0.20, "diurnal":0.13, "climatology":0.53 }
}
```

## 8. Run system (`runner.py`) — launch & queue training jobs

A dependency-free, single-GPU job system.

- An in-process `queue.Queue` and **one** worker thread started at FastAPI startup (one GPU -> one job at a time).
- `enqueue(config) -> run_id`: create `runs/<run_id>/`, write `card.json` with `status:"queued"`, push to the queue, return run_id immediately.
- Worker loop: pop config -> set `status:"running"` -> call `train.py` (which updates progress per epoch) -> on success `status:"done"` -> on exception `status:"failed"` + `error` message. Always continue to the next job.
- Run listing/reading just scans the `runs/` directory and parses each `card.json` (no database needed).
- **Sweep helper:** `enqueue_sweep(hidden_sizes, shared_config)` enqueues one run per hidden size with identical data/stride/seed and only `hidden_size` varying — the fair comparison.

## 9. Backend API (`api.py`)

Forecast/history (model loaded from a chosen run):
- `GET /health` -> `{"status":"ok","device":"cuda"}`
- `GET /history?start=...&end=...&var=temperature_2m` -> cached hourly series.
- `GET /forecast?datetime=YYYY-MM-DDTHH&horizon=24&run_id=<id>` -> uses the L hours ending at `datetime`; returns per-hour `{datetime, lstm, persistence, diurnal, climatology, actual?}`. Defaults to the best/most-recent done run if `run_id` omitted.

Runs/experiments:
- `POST /runs` body=`{hidden_size,num_layers,lookback,horizon,epochs,...}` -> `{run_id,status:"queued"}`
- `POST /runs/sweep` body=`{hidden_sizes:[64,128,256], ...shared}` -> `{run_ids:[...]}`
- `GET /runs` -> list of summaries (run_id, started_at, status, progress, config, key metrics) — UI polls this.
- `GET /runs/{run_id}` -> full `card.json`.
- `GET /runs/{run_id}/history` -> per-epoch curve (epoch, train_loss, val_loss, val_rmse_C).
- `GET /runs/compare?ids=a,b,c` -> assembled comparison: each run's config, `best_val_rmse_C`, val/test metrics (test only where `is_final`), and epoch curves, ready to chart together. Comparison across tuning runs uses **validation** metrics (that's how you pick).
- `DELETE /runs/{run_id}` -> remove a run folder (optional).
- CORS for `http://localhost:3000`.

## 10. Frontend (Next.js) — three pages

**Forecast page**
- History line chart (`GET /history`); datetime + horizon picker -> `GET /forecast`; overlay LSTM / persistence / diurnal / climatology + actuals; per-model MAE for the window; a run selector so you can forecast with any trained run.

**Training page**
- **Launch form:** hidden size, layers, lookback, horizon, epochs, anomaly toggle -> `POST /runs` (these are **tuning** runs, `is_final=false`).
- **"Compare hidden sizes" button:** one click -> `POST /runs/sweep {hidden_sizes:[64,128,256]}`.
- **"Train final model" action:** pick the winning run's config -> `POST /runs {...config, is_final:true}`. This retrains on train+val merged and is the only run that evaluates on test. Show a clear confirm dialog ("this touches the test set — do it once").
- **Runs table:** started date/time, config, an **`is_final` badge**, **live status** (queued / running with epoch progress bar / done / failed), `best_val_rmse_C`. A **test RMSE** column that is shown only for final runs (blank/"—" for tuning runs, by design — visible proof the test set was untouched). Auto-poll `GET /runs` every ~3s.
- Each row links to a **run detail view** rendering the full `card.json` (the timestamped report) and that run's training-curve + RMSE-vs-horizon plots.

**Comparison page** (the exam artifact)
- Select 2+ completed runs (or open straight from a sweep) -> `GET /runs/compare`.
- Charts:
  1. **Best val RMSE vs hidden size** (bar) — the headline capacity/overfitting story. (Uses **validation** metrics — tuning runs have no test numbers.)
  2. **Val-loss-per-epoch, overlaid** for each run — shows larger models diverging/overfitting (train falls, val doesn't).
  3. **Test RMSE vs horizon hour** — only available once a final model exists.
  4. A **comparison table** with date/time stamps and metrics, exportable/printable for the report.

Tailwind; fetch client-side; no SSR backend calls.

## 11. Build phases (in order)

**Phase 1 — Fetch + cache.** `config.py`, `fetch.py`. *Check: chunked hourly download writes parquet; ~750k rows, no gaps.*

**Phase 2 — Features + baselines + eval harness.** `features.py`, `baselines.py`, `evaluate.py`. *Check: baseline metrics for next-hour temperature on a chronological split.*

**Phase 3 — Lazy windowing + single-step LSTM on GPU.** `windowing.py`, `lstm.py` (H=1), `train.py` writing `card.json` + `history.csv` + plots into `runs/<run_id>/`. *Check: trains on cuda (prints device + epoch time), val loss drops, memory flat, report written.*

**Phase 4 — Multi-step (H=24, direct).** Add RMSE-vs-horizon plot to the report. *Check: 24h metrics + skill scores per horizon hour in `card.json`.*

**Phase 5 — Run system backend.** `runner.py` (queue + worker) and the `/runs*` endpoints in `api.py`. *Check: `POST /runs` returns a run_id; the job trains in the background; `GET /runs` shows it move queued -> running (with epoch progress) -> done.*

**Phase 6 — Forecast serving.** `/forecast`, `/history` using a chosen run. *Check: `curl /forecast?...&run_id=<id>` returns the per-hour JSON.*

**Phase 7 — Frontend Forecast + Training pages.** Launch form, runs table with live status/progress, run detail view. *Check: start a run from the browser; watch it progress live; open its report.*

**Phase 8 — Hidden-size sweep + Comparison page.** `POST /runs/sweep [64,128,256]` (tuning runs), `/runs/compare`, the comparison charts on **validation** metrics. *Check: one click queues 3 runs; they train sequentially; the comparison page shows val-RMSE-vs-hidden-size and overlaid training curves.* **This is the deliverable for your "why 128 not 256" answer.**

**Phase 9 — Final model.** Implement the `is_final=true` path: `windowing.py` merges train+val, `train.py`/`evaluate.py` evaluate on test once, the "Train final model" UI action, the test column + `is_final` badge in the runs table. *Check: exactly one run has test numbers; tuning runs show "—" for test.*

**Phase 10 (optional) — Precipitation classification / recursive vs direct multi-step.** Stretch goals + strong exam material.

## 12. Deliverables / acceptance criteria

- [ ] Hourly data fetched once and cached; reproducible from cache offline.
- [ ] Strictly chronological split; scaler + climatology fit on training period only (no leakage).
- [ ] Lazy windowing; memory flat during training; training confirmed on GPU (device + epoch time printed).
- [ ] Each run writes a timestamped `card.json` report (years, epochs, time, metrics, skill scores).
- [ ] Training can be launched from the web UI; runs queue on one GPU; live progress shown.
- [ ] One-click 64/128/256 sweep with identical data/seed/stride (only hidden size differs).
- [ ] **Tuning runs (`is_final=false`) never compute test metrics** — `test_metrics` is `null` in their reports; model selection uses validation only.
- [ ] **Exactly one final run** (`is_final=true`) retrains on train+val merged and evaluates the test set once; it is the only report with test numbers.
- [ ] Comparison page: val-RMSE-vs-hidden-size, overlaid training curves, RMSE-vs-horizon, comparison table with timestamps.
- [ ] `README.md` with setup, run commands, GPU notes, Open-Meteo / ERA5 attribution.

## 13. Key contracts

**`config.py` (defaults; overridable per run)**
```python
import torch
SEED = 42
LAT, LON, TZ = 43.21, 27.91, "Europe/Sofia"
START_DATE = "1940-01-01"
HOURLY_VARS = ["temperature_2m","relative_humidity_2m","dew_point_2m","surface_pressure",
               "precipitation","wind_speed_10m","wind_direction_10m","cloud_cover","shortwave_radiation"]
TARGET = "temperature_2m"
LOOKBACK = 168
HORIZON = 1
STRIDE = 1
HIDDEN = 128
NUM_LAYERS = 2
DROPOUT = 0.2
BATCH = 256
EPOCHS = 30
USE_AMP = True
USE_ANOMALY = False
IS_FINAL = False     # tuning run -> evaluate on validation only, test untouched.
                     # Set True for exactly ONE run: retrains on train+val merged, evaluates test once.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DATA_PATH = "data/varna_hourly.parquet"
RUNS_DIR = "runs"
```

**GPU training core**
```python
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
for xb, yb in loader:                       # pin_memory=True, num_workers=4
    xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
    opt.zero_grad()
    with torch.autocast("cuda", enabled=USE_AMP):
        loss = loss_fn(model(xb), yb)
    scaler.scale(loss).backward()
    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt); scaler.update()
```

**Run queue core (`runner.py`)**
```python
import queue, threading
_q = queue.Queue()
def _worker():
    while True:
        cfg = _q.get()
        try:
            set_status(cfg["run_id"], "running")
            train_one(cfg)                      # updates progress per epoch, writes card.json
            set_status(cfg["run_id"], "done")
        except Exception as e:
            set_status(cfg["run_id"], "failed", error=str(e))
        finally:
            _q.task_done()
threading.Thread(target=_worker, daemon=True).start()   # ONE worker = one GPU job at a time
```

**Skill score**
```python
skill = 1.0 - rmse_model / rmse_baseline    # >0 means model beats baseline
```

## 14. Gotchas (read before coding)

- **One GPU -> one worker thread.** Never train multiple runs at once; they queue. Running them concurrently will OOM or thrash.
- **Fair sweep:** identical data, seed, stride across 64/128/256 — only `hidden_size` differs. Otherwise the comparison is meaningless.
- **No shuffling, ever** — chronological split; random splitting leaks the future.
- **Never tune on the test set.** Pick hyperparameters by validation RMSE. Tuning runs must not even compute test metrics (`is_final=false` -> `test_metrics=null`). Touch the test set once, in the single final run. The empty `test_metrics` on tuning reports is your proof of no leakage — stronger than the timestamps.
- **Lazy windowing** — never build the full (windows x L x F) array; slice in `__getitem__`. #1 OOM cause for hourly.
- **Fit scaler/climatology on the training period only.**
- **ERA5 update lag** — `end_date` ~ today - 7 days.
- **Inverse-transform before reporting** — metrics in deg C.
- **Confirm GPU** — print `next(model.parameters()).device` and epoch time; an accidental CPU run is the overnight scenario.
- **Live progress** — write current epoch to `card.json` each epoch so the UI bar moves; the frontend polls `/runs`.
- **Reproducibility** — seed numpy + torch (+cuda); the report records seed, split, and full config.
- **Attribution** — data is CC BY 4.0; credit Open-Meteo and ECMWF ERA5 in the README.
