# lstm-weather — Hourly Weather Forecasting for Varna with an LSTM

A local tool that predicts **hourly** weather for **Varna, Bulgaria** using an **LSTM**
(PyTorch, GPU) trained on ERA5 reanalysis from the
[Open-Meteo Historical Weather API](https://open-meteo.com/). It compares the LSTM head-to-head
against naive baselines (persistence, diurnal persistence, climatology) on a strictly
chronological split, and ships an **experiment system**: launch training runs from the web UI,
watch them queue and progress live, store a timestamped report per run, and **compare runs**
(e.g. hidden size 64 vs 128 vs 256) with charts.

> PhD exam project on LSTMs. The deliverables are the chronological evaluation, the baseline
> comparison, the RMSE-vs-horizon curve, and the **hidden-size comparison with per-run reports**.
> See [`BUILD_SPEC.md`](BUILD_SPEC.md) for the full specification.

## Project layout

```
lstm-weather/
├── BUILD_SPEC.md             # full spec
├── backend/
│   ├── requirements.txt
│   ├── data/                 # gitignored: cached parquet (varna_hourly.parquet)
│   ├── runs/                 # gitignored: one folder per run (card.json, model.pt, plots, …)
│   └── src/
│       ├── config.py         # coords, vars, default hyperparams, device, seed, paths
│       ├── fetch.py          # chunked hourly download + parquet cache
│       ├── features.py       # cyclical calendar features, climatology, anomaly
│       ├── windowing.py      # LAZY sliding-window Dataset; chronological split
│       ├── models/
│       │   ├── baselines.py  # persistence, diurnal persistence, climatology
│       │   └── lstm.py       # LSTM forecaster (single- and multi-step)
│       ├── train.py          # train ONE config; GPU/AMP; writes card.json + history + plots
│       ├── evaluate.py       # MAE/RMSE/MAPE per horizon, skill scores, plots
│       ├── store.py          # runs/<id>/ storage helpers (no database)
│       ├── runner.py         # in-process job queue + single worker thread
│       ├── serve.py          # forecast/history inference helpers
│       └── api.py            # FastAPI: forecast, history, AND run endpoints
└── frontend/                 # Next.js app: Forecast + Training + Comparison pages
```

Each run is just a folder under `backend/runs/<run_id>/` containing `card.json` (the timestamped
report), `history.csv` (per-epoch curve), `model.pt`, `scaler.pkl`, `climatology.pkl`, and
`plots/`. Listing runs simply scans that directory — no database.

## Setup

### Backend (Python 3.11)

```bash
cd backend
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

**GPU notes.** `requirements.txt` pins the default `torch` wheel. For CUDA acceleration install the
matching build, e.g.:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

The code auto-selects the device (`cuda` if available, else `cpu`) and prints it at startup. Each
run also prints `next(model.parameters()).device` and per-epoch time so an accidental CPU run is
obvious. Training ~500k windows on CPU is very slow — use a CUDA GPU for reported runs.

### Frontend (Next.js)

```bash
cd frontend
npm install
npm run dev          # http://localhost:3000
```

The UI talks to the backend at `http://localhost:8000` by default; override with
`NEXT_PUBLIC_API_BASE` if needed.

## Usage

### 1. Fetch + cache the data (once)

From `backend/`:

```bash
python -m src.fetch            # uses cache if present
python -m src.fetch --refresh  # force re-download from Open-Meteo
```

Downloads ~85 years of hourly ERA5 data for Varna in ~5-year chunks, validates a gap-free hourly
index, and caches to `backend/data/varna_hourly.parquet` (~750k rows). The network is hit only
once unless `--refresh`.

### 2. Start the backend API

```bash
cd backend
uvicorn src.api:app --reload --port 8000
```

This starts the single training worker (one GPU → one job at a time). Key endpoints:

- `GET /health` → `{"status":"ok","device":"cuda"}`
- `GET /history?start=…&end=…&var=temperature_2m`
- `GET /forecast?datetime=YYYY-MM-DDTHH&horizon=24&run_id=<id>`
- `POST /runs` `{hidden_size,num_layers,lookback,horizon,epochs,…}` → `{run_id,status:"queued"}`
- `POST /runs/sweep` `{hidden_sizes:[64,128,256], …shared}` → `{run_ids:[…]}`
- `GET /runs`, `GET /runs/{id}`, `GET /runs/{id}/history`, `GET /runs/compare?ids=a,b,c`
- `DELETE /runs/{id}`

### 3. Train from the CLI (optional)

```bash
python -m src.train --hidden 128 --layers 2 --horizon 1 --epochs 30
python -m src.train --hidden 128 --horizon 24            # direct multi-step
python -m src.train --hidden 128 --final                 # FINAL run (see below)
```

### 4. Train and compare from the UI

- **Training page** — launch tuning runs, click **Compare hidden sizes** to queue a fair
  64/128/256 sweep (identical data/seed/stride, only `hidden_size` varies), and watch the runs
  table update live (queued → running with an epoch progress bar → done). Open any run for its full
  report and interactive training-curve / RMSE-vs-horizon charts.
- **Comparison page** — select 2+ runs to see best-val-RMSE-vs-hidden-size, overlaid val-RMSE
  curves, test-RMSE-vs-horizon (final only), and a printable comparison table.
- **Forecast page** — pick a run, a datetime, and a horizon to overlay the LSTM against the
  baselines and the actual values, with per-model MAE.

## The two-phase discipline (`is_final`)

This is the anti-leakage discipline an examiner will probe:

- **Tuning runs (`is_final=false`, default).** Trained on the train years, judged on validation.
  `evaluate.py` computes validation metrics only — `test_metrics`/`skill_vs` stay `null` in the
  report. All sweep runs are tuning runs; pick the winner by `best_val_rmse_C`, never by test.
- **Final run (`is_final=true`, set exactly once).** Retrains the chosen config from scratch on
  **train + validation merged**, then evaluates the **test** set once, filling `test_metrics` and
  `skill_vs`. It is the only report with test numbers.

In the runs table the **Test RMSE** column is blank ("—") for tuning runs — visible, by-construction
proof that the test set was untouched, stronger than any timestamp. The UI's "Make final" action
shows a confirm dialog because it touches the test set.

## Reproducibility

Global seed `42` (numpy, torch, torch.cuda), logged at startup. Each `card.json` records the seed,
the split year ranges, window counts, the full config, epochs/time, and metrics — so any run is
reproducible from the cached parquet with no network access.

## Data attribution

Weather data is provided by the [Open-Meteo](https://open-meteo.com/) Historical Weather API, based
on **ECMWF ERA5 / ERA5-Land** reanalysis, licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

- Open-Meteo: https://open-meteo.com/
- ERA5: Hersbach, H. et al. (2018): ERA5 hourly data, Copernicus Climate Change Service (C3S)
  Climate Data Store (CDS), ECMWF.
