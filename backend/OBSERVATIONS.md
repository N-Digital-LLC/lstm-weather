# Observations

A running log of things I noticed while training and evaluating the LSTM weather models.
Useful for talking through findings at the end.

---

## Early stopping kicked in at epoch 10 (run `2026-06-18_16-41-30__h64_l2`)

**What happened:** Training was planned for 30 epochs (`EPOCHS=30`) but stopped early at
epoch 10.

**Why:** The trainer tracks the best validation RMSE and counts consecutive epochs that fail
to beat it. With `EARLY_STOP_PATIENCE = 5`, once 5 epochs in a row show no improvement,
training stops. An epoch only counts as an improvement if it beats the previous best by more
than `1e-4` °C.

**The run's validation curve:**

| epoch | val_rmse (C) | best so far | improved? | epochs_without_improve |
|------:|-------------:|------------:|:---------:|----------------------:|
| 1 | 1.679 | 1.679 | yes | 0 |
| 2 | 1.617 | 1.617 | yes | 0 |
| 3 | 1.587 | 1.587 | yes | 0 |
| 4 | 1.572 | 1.572 | yes | 0 |
| **5** | **1.553** | **1.553** | **yes (best)** | 0 |
| 6 | 1.586 | 1.553 | no | 1 |
| 7 | 1.590 | 1.553 | no | 2 |
| 8 | 1.559 | 1.553 | no | 3 |
| 9 | 1.574 | 1.553 | no | 4 |
| 10 | 1.560 | 1.553 | no | 5 -> stop |

**Takeaways:**
- Best validation model was at **epoch 5** (val RMSE 1.553 °C). The saved weights
  (`best_state`) correspond to that epoch, not epoch 10.
- Meanwhile `train_loss` kept dropping (0.0209 -> 0.0193) while val RMSE plateaued/bounced
  around 1.55-1.59 -> classic sign of **overfitting** starting after epoch 5.
- Early stopping saved ~20 epochs of wasted compute and kept the best-generalizing model.
- To let it ride out the noisy plateau longer, increase `EARLY_STOP_PATIENCE` in
  `src/config.py`.

**Relevant code:** `src/train.py` (lines ~196-205), `src/config.py` (`EARLY_STOP_PATIENCE`).

---

## The hyperparameter sweep is a near-tie — the "best" run is within seed noise

**What happened:** After 108 tuning runs (a 54-config grid × {anomaly on, anomaly off}), I ranked
them all by best validation RMSE in the Comparison tab's new *Model selection* panel. The top run
was `h128_l1_L168_lr0.001` at **1.522 °C**, but:

| measure | value |
|---|---|
| best val RMSE (#1) | 1.522 °C |
| gap from #1 to #2 | **0.011 °C** |
| gap from #1 to #5 | **0.016 °C** |
| runs within 0.05 °C of the best | **77 of 108** |

**Why it matters:** A gap of ~0.01 °C between the top configs is smaller than the run-to-run
variation a different random seed would produce. So picking the single lowest number is essentially
arbitrary — a re-seed would likely reshuffle the top ~20. The panel flags this as
**"Hairline — likely seed noise"** (threshold `HAIRLINE_C = 0.05`).

**Takeaways:**
- The model is **robust and insensitive** to these hyperparameters across the swept range
  (hidden 32-128, layers 1-3, lookback 72-336, lr 1e-3/5e-4). That is itself a clean finding worth
  reporting, not a problem.
- Do **not** crown #1 on the raw minimum. The defensible choice is **Occam's razor**: among the
  ~77 tied configs, take the *smallest / cheapest* one (fewest params, shortest lookback) since
  accuracy is equivalent but it trains faster and overfits less.
- Before finalizing, confirm seed stability: re-run the top 5 configs with fresh seeds and compare
  the **mean** RMSE per config (the "Seed-stability re-run" control queues top-5 × N seeds via
  `POST /runs`). Whichever config has the lowest average — not the luckiest seed — is the winner.
- Only then set `IS_FINAL=True` (or `--final`) for that one config to retrain on train+val and
  score the held-out test set once.

## The top candidate is healthy, not just low

**What happened:** The panel's *Candidate health* check on the #1 run (mean over the 24 horizon
hours of the validation set) showed:

| model | mean val RMSE |
|---|---|
| **LSTM** | **1.448 °C** |
| persistence | 3.501 °C (LSTM −2.05) |
| diurnal | 2.529 °C (LSTM −1.08) |
| climatology | 3.606 °C (LSTM −2.16) |

- **Beats every baseline** by a wide margin (≈1-2 °C) → "all clear".
- **Convergence:** best epoch **5 of 10** run (30 planned) → early-stopped on a plateau, so it is
  neither a lucky early epoch (best epoch > 2) nor cut off at the cap → "healthy convergence".

**Note:** the health card's *mean-over-horizon* RMSE (1.448) is naturally a bit lower than the
aggregate `best_val_rmse_C` (1.522) shown in the leaderboard — they average errors differently —
but both comfortably clear the baselines.

**Relevant code:** `frontend/app/comparison/page.tsx` (`ModelSelectionPanel`, `HAIRLINE_C`,
`LUCKY_EPOCH`), backend `src/api.py` (`/runs`, `/runs/{id}`), `src/evaluate.py` (baseline metrics).

---

## Seed re-runs confirmed the tie — promoting `h128_l1_L168_lr0.001` as final

**What happened:** Using the panel's *Seed-stability re-run* control I re-queued the top-5 configs
with fresh seeds (1…N; seed 42 was the original) and re-ranked once they finished.

**Result:**
- The **same config came back on top**: `h128_l1_L168_lr0.001` (Hidden 128 · Layers 1 ·
  Lookback 168 · lr 1e-3).
- Its **seed-to-seed spread was only +0.008 °C**.

**Interpretation:**
- +0.008 °C spread means the winner is **reproducible, not a lucky seed** — it survives re-seeding.
- But that spread (~0.008) is about the same size as the gap from #1 to the runners-up (~0.011), so
  the config is **not *significantly* better** than the rest of the top cluster. The honest framing
  is "a stable, top-tier, baseline-beating choice", not "the single best model".
- Decision: **promote it** as the final model. Equally defensible would have been an Occam pick of
  the cheapest config within ~0.016 °C (e.g. the Hidden-64 variant), but the reproducible #1 is a
  clean choice.

**How it was promoted:** the *Candidate health* card's **"Promote to final run"** button posts the
candidate's real card config (so `use_anomaly`, absent from the run id, is carried over correctly)
with `is_final=true` and `stride=1`. That retrains on **train+val merged** and scores the held-out
**test set exactly once**; the test metrics + skill-vs-baselines then populate the lower charts of
the Comparison tab. The button is disabled until the candidate beats all baselines.

**Discipline:** the test number from this final run is the reported result — do **not** re-pick a
different config after seeing the test score (that would turn the test set into a second validation
set).

**Relevant code:** `frontend/app/comparison/page.tsx` (`queueSeedRepeats`, `promoteCandidate`),
backend `src/train.py` (`is_final` path), `src/windowing.py` (train+val merge for final),
`src/api.py` (`POST /runs`).

---
---

# Experiment setup & data notes

_Backfilled from earlier sessions and from the run cards — the experimental context behind the
numbers above, kept here so the writeup has the "what are we actually predicting" details in one
place._

## The dataset: 86 years of hourly Varna weather (ERA5)

**What we're training on:** hourly reanalysis weather for **Varna, Bulgaria** (43.21 N, 27.91 E)
from the Open-Meteo Historical Weather API (ECMWF **ERA5**, CC BY 4.0).

- **757,776 hourly rows**, spanning **1940-01-01 → 2026-06-11**, gap-free hourly.
- Fetched in ~5-year chunks and cached to `backend/data/varna_hourly.parquet`; everything after the
  one-time fetch runs offline.
- A handful (**41**) of stray nulls from the API were forward-filled; otherwise the series is
  continuous.

**Model inputs (14 features):** 9 raw weather variables — `temperature_2m` (the target),
`relative_humidity_2m`, `dew_point_2m`, `surface_pressure`, `precipitation`, `wind_speed_10m`,
`wind_direction_10m`, `cloud_cover`, `shortwave_radiation` — plus 5 engineered **cyclical calendar
features**: `hour_sin/cos`, `doy_sin/cos` (day-of-year, leap-year aware), and a normalized
`year_norm` long-term-trend term. The sin/cos encodings let the LSTM lock onto the daily and annual
cycles directly instead of having to learn that hour 23 is adjacent to hour 0.

**Relevant code:** `src/fetch.py`, `src/features.py` (`add_calendar_features`, `CALENDAR_FEATURES`),
`src/config.py` (`LAT/LON/TZ`, `HOURLY_VARS`).

## Strictly chronological 70 / 15 / 15 split — never shuffled

**What:** the series is cut in timeline order into train / validation / test, by row count:

| split | years | windows |
|---|---|---|
| train | **1940 – 2000** | ~530,252 |
| validation | **2000 – 2013** | ~113,475 |
| test (held out) | **2013 – 2026** | ~113,476 |

**Why it matters (experimentally):**
- **No shuffling.** A random split would let the model peek at hours adjacent in time to the test
  hours (temporal leakage) and massively overstate skill. Chronological split means the model is
  always predicting a genuinely *future*, unseen period — the realistic forecasting setting.
- The **test set is the most recent 13 years (2013-2026)**, so the final score also doubles as a
  mild distribution-shift / climate-trend check.
- All "learned" statistics (feature scaler, **climatology**) are fit on the **training rows only**;
  the val/test slices are transformed with those fixed stats so nothing from the future leaks back.

**Relevant code:** `src/features.py` (`chronological_split_bounds`, `fit_climatology`),
`src/windowing.py`, `src/config.py` (`TRAIN_FRAC=0.70`, `VAL_FRAC=0.15`).

## The task: one week in → next 24 hours out (and the anomaly option)

**Framing:** a sliding window of **lookback `L = 168` hours (one week)** of all 14 features is fed
in to predict the **next `H = 24` hours** of temperature (multi-step). `stride` controls how densely
windows are sampled (larger stride = faster sweeps; `stride=1` for the full final run).

**Anomaly mode (`use_anomaly`)** — the main experimental variable in the 108-run sweep: instead of
predicting absolute temperature, the model can predict the **anomaly = value − climatology** and add
the climatology back at inference. The idea is to hand the seasonal/diurnal shape to the model for
free so it only has to learn the *departure* from normal. The sweep ran the full 54-config grid both
ways (anomaly on / off) precisely to test whether this helps — the headline finding above is that it
barely moves validation RMSE either way, i.e. the plain model already captures the cycles well via
the calendar features.

**Relevant code:** `src/features.py` (`to_anomaly`/`from_anomaly`), `src/windowing.py`,
`src/config.py` (`LOOKBACK`, `HORIZON`, `STRIDE`, `USE_ANOMALY`).

## The three baselines and what beating them proves

Every LSTM is scored against the same three naive forecasts, on the **exact same windows**, so the
skill score is fair:

- **Persistence** — every future hour = the last observed hour. Strong at very short lead times,
  falls apart as the day turns over.
- **Diurnal persistence** — hour `t+k` = the same clock hour *yesterday* (`t+k−24`). Captures the
  daily cycle, so it's roughly flat across the 24-hour horizon.
- **Climatology** — the per-`(day-of-year, hour)` training-period mean (lightly smoothed along the
  year). Captures season + time-of-day but nothing about current conditions.

Beating persistence shows the model learned more than "tomorrow ≈ now"; beating diurnal/climatology
shows it learned more than "the average day for this date." A real forecaster has to clear all three.

**Relevant code:** `src/models/baselines.py`, `src/features.py` (`Climatology`).

## How error grows with the forecast horizon (and how the baselines behave)

**What happened:** reading the per-horizon validation RMSE from a representative tuning run
(`h64_l2_L168`, anomaly on) shows the shape of the problem clearly:

| horizon | LSTM | persistence | diurnal | climatology |
|---:|---:|---:|---:|---:|
| +1 h | **0.60** | 0.76 | 2.53 | 3.61 |
| +6 h | **1.15** | 3.52 | 2.53 | 3.61 |
| +12 h | **1.57** | ~4.71 (peak) | 2.53 | 3.61 |
| +24 h | **2.12** | 2.53 | 2.53 | 3.61 |

**Takeaways:**
- **The LSTM beats every baseline at every horizon** — even at +1 h it edges out persistence
  (0.60 vs 0.76), and at +24 h it's still well under the best baseline (2.12 vs diurnal 2.53).
- **Error grows monotonically with lead time** (0.6 → 2.1 °C over 24 h) — expected; further-out
  hours are harder.
- **Persistence is the trap baseline:** great at +1 h, but it peaks around **+12-13 h (~4.7 °C)**
  when "now" is maximally stale (predicting tomorrow's afternoon from tonight), then recovers near
  +24 h as the daily cycle comes back around.
- **Diurnal and climatology are flat** across the horizon by construction (they don't depend on lead
  time), so they're the bar the LSTM must clear at long range — and it does.
- The LSTM carries a small **cold bias (~−0.15 °C)** at most horizons (it slightly under-predicts);
  worth a sentence in the writeup but minor next to the ~1-2 °C it gains over the baselines.

**Relevant code:** `src/evaluate.py` (`rmse_per_horizon`, `bias_per_horizon`), per-run
`card.json` (`val_horizon`), Comparison tab "RMSE vs horizon" chart.

---
---

# The final model — held-out test results

## `2026-06-19_11-00-30__h128_l1_L168_lr0.001` (is_final, test 2013-2026)

**Final config:** **Hidden 128 · Layers 1 · Lookback 168 (1 week) · Horizon 24 · lr 1e-3 · anomaly
OFF · seed 42.** Trained on **train+val merged (1940-2013, 643,918 windows)** for the full 30 epochs
and evaluated **exactly once** on the held-out **test years 2013-2026 (113,476 windows)**. Ran in
240.6 s on CUDA.

**Headline test metrics (LSTM, real °C):**

| metric | LSTM | persistence | diurnal | climatology |
|---|---:|---:|---:|---:|
| **RMSE** | **1.943** | 3.700 | 2.604 | 3.563 |
| MAE | 1.379 | 2.859 | 1.878 | 2.738 |
| bias | **−0.158** | ~0 | ~0 | −1.155 |
| R² | **0.945** | 0.801 | 0.902 | 0.816 |

**Skill scores (RMSE reduction vs baseline):** persistence **+47.5%**, climatology **+45.5%**,
diurnal **+25.4%**.

**Takeaways:**
- **The model generalizes and clears every baseline on data it never saw.** R² = 0.945 (explains
  94.5% of test-period variance over 13 unseen years), and it beats even the toughest baseline
  (diurnal) by 25%.
- **Val → test gap is real and expected.** The tuning leaderboard sat around **1.5 °C val RMSE**;
  the test comes in at **1.94 °C**. That ~0.4 °C rise is the honest cost of scoring a genuinely
  future, unseen 13-year span (2013-2026, with some climate drift). This is a *finding to report*,
  not a bug — and per protocol we do **not** re-pick after seeing it.
- **Error grows smoothly with lead time:** **0.59 °C @ +1 h → 2.49 °C @ +24 h.** Further-out hours
  are harder, as expected.
- **Diurnal is the baseline to respect, especially at long range.** It's flat (~2.60 °C across all
  horizons), so as the LSTM degrades toward +24 h its edge over "same hour yesterday" **shrinks to
  ~0.11 °C by +24 h** (2.49 vs 2.60). Most of the model's value is in the first ~12 hours; by a full
  day out it has nearly converged to diurnal. The +25% skill-vs-diurnal headline is carried by the
  short horizons.
- **Persistence peaks mid-horizon** (~4.70 °C around +12-13 h, when "now" is most stale) and recovers
  by +24 h — the same shape seen on validation, confirming it's a property of the problem, not the
  split.
- **Small cold bias of −0.16 °C** (the model slightly under-predicts), consistent with validation.
  A fixed +0.16 °C bias-correction at inference would shave a little MAE essentially for free — a
  clean, cheap improvement to mention.
- **MAPE = 30% looks alarming but is a known artifact:** temperature crosses 0 °C, so near-zero
  actuals blow up the percentage error. RMSE/MAE in °C are the trustworthy headline metrics here;
  MAPE is reported only with that caveat.
- **No early stopping on the final run** (ran all 30 epochs, `best_val_rmse_C = null`) — by design,
  since the final run merges away the validation set and can't monitor it. We rely on the
  hyperparameters chosen during tuning being safe; the healthy test R² confirms no gross overfit.

**Bottom line:** a single-layer, 128-unit LSTM with a one-week context window forecasts Varna's
hourly temperature 24 h out at **1.94 °C RMSE / 1.38 °C MAE (R² 0.95)** on a fully held-out
2013-2026 test set, beating persistence/climatology by ~46-48% and the strong diurnal baseline by
25%.

**Relevant code:** `src/train.py` (`is_final` path, no val monitor), `src/windowing.py` (train+val
merge), `src/evaluate.py` (`mape`, skill scores), run `card.json` (`test_metrics`, `skill_vs`,
`test_horizon`).

---
---

# Using the Forecast tab — how to check predictions against real dates

## What it actually does

The Forecast tab (the home page) is an **inference + sanity-check tool**: you pick a run, a "last
observed hour", and a horizon, and it:

1. Feeds the model the **`L = 168` hours (one week) ending at the chosen timestamp** as input,
2. Predicts the **next `H` hours** (capped to the model's trained `horizon = 24`),
3. Overlays the **LSTM** against **persistence / diurnal / climatology**, and — when those future
   hours exist in the cached archive — the **actual** temperature too,
4. Shows the **MAE** for each model over the window (only where actuals exist).

It mirrors training exactly (same calendar features, the run's saved scaler, same anomaly setting),
so what you see here is consistent with the run's reported metrics.

## How to "check a future date"

There are two distinct modes depending on the datetime you choose:

- **Backtest mode (recommended for checking accuracy):** pick a datetime **inside the data we have**
  — ideally in the **test years 2013-2026** so the model never trained on it. The future hours then
  have **actuals**, so the chart shows LSTM-vs-actual and you get a real MAE. This is how you "check"
  the model: e.g. issue for `2024-01-15T12:00`, horizon 24, and watch the blue LSTM line track the
  white actual line.
- **Pure-forecast mode:** pick the **latest available hour**. There are no actuals after it, so the
  `actual` line is empty and you're seeing a genuine forward forecast with nothing to grade it
  against.

## The important caveat: "future" is bounded by the cached archive

This is **not a live forecast of the real calendar future.** It can only forecast hours for which it
has the preceding week of inputs, and our cache is **ERA5 reanalysis with a ~7-day lag**, so the
archive ends about a week before today. Consequences:

- The furthest "future" you can forecast is ~24 h past the **end of the cached data** (roughly a week
  ago in real time).
- Asking for a datetime past the archive **snaps back** to the most recent available hour.
- A datetime **before 1940 + 168 h** is rejected (not enough history for the lookback window).
- To move the window closer to *today*, refresh the cache first: `python -m src.fetch --refresh`,
  then pick the new latest hour.

## Picking the right run (gotcha)

The dropdown now **auto-selects the final model** and marks it `★ FINAL — …`. The fallback option
"Auto — best validation run" picks the lowest *validation* RMSE — which is a **tuning** run, not the
final (final runs have no validation score). For reporting/serving, always forecast with the
**★ FINAL** run.

**Relevant code:** `src/serve.py` (`forecast`, lookback window build, baseline + actual assembly,
ERA5 lag via `ERA5_LAG_DAYS`), `src/api.py` (`GET /forecast`), `frontend/app/page.tsx` (run
auto-select + ★ FINAL labels).
