// Typed client for the FastAPI backend. All calls are client-side (no SSR backend calls).
//
// Two modes:
//  - Live mode (default): calls the FastAPI backend at API_BASE.
//  - Static mode (NEXT_PUBLIC_STATIC === "1"): reads precomputed JSON snapshots from
//    /snapshots/* (produced by backend/make_snapshots.py) so the results app runs as a
//    fully static, offline website with no backend. Training/forecast write actions are
//    disabled in static mode.

export const IS_STATIC = process.env.NEXT_PUBLIC_STATIC === "1";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, "") || "http://localhost:8000";

// Snapshots live in /public/snapshots. Prefix with the deploy base path so the
// app works both at the domain root and under a subpath (e.g. GitHub Pages).
export const BASE_PATH = (process.env.NEXT_PUBLIC_BASE_PATH || "").replace(/\/$/, "");
const SNAP_BASE = `${BASE_PATH}/snapshots`;

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!res.ok) {
    const detail = await res.text().catch(() => res.statusText);
    throw new Error(`GET ${path} failed (${res.status}): ${detail}`);
  }
  return res.json() as Promise<T>;
}

async function snap<T>(rel: string): Promise<T> {
  const res = await fetch(`${SNAP_BASE}/${rel}`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`Snapshot not found: ${rel} (${res.status})`);
  }
  return res.json() as Promise<T>;
}

function readonlyReject<T>(): Promise<T> {
  return Promise.reject(
    new Error(
      "Приложение с резултати само за четене: стартирането/изтриването на пускове е изключено в " +
        "този статичен пакет. Стартирайте пълния сървър, за да обучавате.",
    ),
  );
}

async function sendJSON<T>(path: string, method: string, body?: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => res.statusText);
    throw new Error(`${method} ${path} failed (${res.status}): ${detail}`);
  }
  return res.json() as Promise<T>;
}

// --- Types -------------------------------------------------------------------
export interface RunConfig {
  hidden_size: number;
  num_layers: number;
  lookback: number;
  horizon: number;
  batch: number;
  lr: number;
  stride: number;
  use_amp: boolean;
  use_anomaly: boolean;
  is_final: boolean;
  seed: number;
}

export interface Progress {
  current_epoch: number;
  total_epochs: number;
}

export interface RunSummary {
  run_id: string;
  status: "queued" | "running" | "done" | "failed";
  started_at: string | null;
  finished_at: string | null;
  config: RunConfig;
  progress: Progress;
  is_final: boolean;
  best_val_rmse_C: number | null;
  val_mae_C: number | null;
  val_bias_C: number | null;
  val_r2: number | null;
  test_rmse_C: number | null;
  test_mae_C: number | null;
  test_bias_C: number | null;
  test_r2: number | null;
  error: string | null;
}

export interface Metric {
  mae_C: number;
  rmse_C: number;
  bias_C: number;
  r2: number;
  mape_pct?: number;
}

export interface HorizonCurve {
  hours: number[];
  lstm: number[];
  persistence?: number[];
  diurnal?: number[];
  climatology?: number[];
  mae?: Record<string, number[]>;
  bias?: Record<string, number[]>;
}

export interface RunCard {
  run_id: string;
  status: string;
  error: string | null;
  started_at: string | null;
  finished_at: string | null;
  device: string;
  config: RunConfig;
  data: {
    train_years: string;
    val_years: string;
    test_years: string;
    n_train_windows: number;
    n_val_windows: number;
    n_test_windows: number;
    features: string[];
  } | null;
  training: {
    epochs_planned: number;
    epochs_run: number;
    early_stop_epoch: number | null;
    best_val_rmse_C: number | null;
    train_time_seconds: number;
  } | null;
  progress: Progress;
  val_metrics: { lstm: Metric } | null;
  test_metrics: Record<string, Metric> | null;
  skill_vs: Record<string, number> | null;
  val_horizon?: HorizonCurve | null;
  test_horizon?: HorizonCurve | null;
}

export interface EpochRow {
  epoch: number;
  train_loss: number | null;
  val_loss: number | null;
  val_rmse_C: number | null;
}

export interface CompareRun {
  run_id: string;
  config: RunConfig;
  status: string;
  is_final: boolean;
  best_val_rmse_C: number | null;
  training: {
    epochs_planned: number;
    epochs_run: number;
    early_stop_epoch: number | null;
    best_val_rmse_C: number | null;
    train_time_seconds: number;
  } | null;
  val_metrics: { lstm: Metric } | null;
  test_metrics: Record<string, Metric> | null;
  skill_vs: Record<string, number> | null;
  val_horizon?: HorizonCurve | null;
  test_horizon?: HorizonCurve | null;
  history: EpochRow[];
}

export interface SweepBody {
  hidden_sizes?: number[];
  num_layers_list?: number[];
  lookback_list?: number[];
  lr_list?: number[];
  mode?: "one_factor" | "grid";
  // shared, held-constant config (stride/batch/horizon/epochs/anomaly/...)
  [key: string]: unknown;
}

export interface ForecastRow {
  datetime: string;
  lstm: number;
  persistence: number;
  diurnal: number | null;
  climatology: number;
  actual: number | null;
}

export interface ForecastResponse {
  location: string;
  issued_for: string;
  horizon_hours: number;
  unit: string;
  run_id: string;
  model: string;
  forecast: ForecastRow[];
}

export interface HistoryResponse {
  var: string;
  series: { datetime: string; value: number | null }[];
}

export interface SplitSegment {
  years: string;
  start: string | null;
  end: string | null;
  rows: number;
}

export interface DataSplit {
  location: string;
  total_rows: number;
  fractions: { train: number; val: number; test: number };
  train: SplitSegment;
  val: SplitSegment;
  test: SplitSegment;
  note: string;
}

export interface ForecastExample {
  datetime: string;
  horizon: number;
  run_id: string | null;
  file: string;
}

interface ForecastIndex {
  final_run_id: string | null;
  items: ForecastExample[];
}

// Assemble a CompareRun from the per-run card + history snapshots (mirrors the
// backend /runs/compare response shape) so the Comparison page works offline.
async function compareFromSnapshots(ids: string[]): Promise<{ runs: CompareRun[] }> {
  const runs = await Promise.all(
    ids.map(async (id) => {
      const card = await snap<RunCard>(`runs/${id}.json`);
      const hist = await snap<{ history: EpochRow[] }>(`history/${id}.json`).catch(() => ({
        history: [] as EpochRow[],
      }));
      return {
        run_id: card.run_id,
        config: card.config,
        status: card.status,
        is_final: card.config.is_final,
        best_val_rmse_C: card.training?.best_val_rmse_C ?? null,
        training: card.training,
        val_metrics: card.val_metrics,
        test_metrics: card.test_metrics,
        skill_vs: card.skill_vs,
        val_horizon: card.val_horizon ?? null,
        test_horizon: card.test_horizon ?? null,
        history: hist.history,
      } as CompareRun;
    }),
  );
  return { runs };
}

// Resolve a forecast request to the nearest precomputed snapshot (matching the
// requested horizon when possible, then the closest issued-for datetime).
async function forecastFromSnapshots(
  datetime: string,
  horizon: number,
): Promise<ForecastResponse> {
  const idx = await snap<ForecastIndex>("forecasts/index.json");
  if (!idx.items.length) throw new Error("В този пакет няма предварително изчислени прогнози.");
  const sameHorizon = idx.items.filter((it) => it.horizon === horizon);
  const pool = sameHorizon.length ? sameHorizon : idx.items;
  const target = new Date(datetime).getTime();
  let best = pool[0];
  let bestDist = Infinity;
  for (const it of pool) {
    const dist = Math.abs(new Date(it.datetime).getTime() - target);
    if (dist < bestDist) {
      bestDist = dist;
      best = it;
    }
  }
  return snap<ForecastResponse>(best.file);
}

// --- Calls -------------------------------------------------------------------
export const api = {
  health: () =>
    IS_STATIC
      ? snap<{ status: string; device: string }>("health.json")
      : getJSON<{ status: string; device: string }>("/health"),

  listRuns: () =>
    IS_STATIC
      ? snap<{ runs: RunSummary[] }>("runs.json")
      : getJSON<{ runs: RunSummary[] }>("/runs"),

  dataSplit: () =>
    IS_STATIC ? snap<DataSplit>("data_split.json") : getJSON<DataSplit>("/data/split"),

  getRun: (runId: string) =>
    IS_STATIC
      ? snap<RunCard>(`runs/${runId}.json`)
      : getJSON<RunCard>(`/runs/${encodeURIComponent(runId)}`),

  getRunHistory: (runId: string) =>
    IS_STATIC
      ? snap<{ run_id: string; history: EpochRow[] }>(`history/${runId}.json`)
      : getJSON<{ run_id: string; history: EpochRow[] }>(
          `/runs/${encodeURIComponent(runId)}/history`,
        ),

  compare: (ids: string[]) =>
    IS_STATIC
      ? compareFromSnapshots(ids)
      : getJSON<{ runs: CompareRun[] }>(`/runs/compare?ids=${encodeURIComponent(ids.join(","))}`),

  createRun: (overrides: Record<string, unknown>) =>
    IS_STATIC
      ? readonlyReject<{ run_id: string; status: string }>()
      : sendJSON<{ run_id: string; status: string }>("/runs", "POST", overrides),

  createSweep: (body: SweepBody) =>
    IS_STATIC
      ? readonlyReject<{ run_ids: string[] }>()
      : sendJSON<{ run_ids: string[] }>("/runs/sweep", "POST", body),

  deleteRun: (runId: string) =>
    IS_STATIC
      ? readonlyReject<{ deleted: string }>()
      : sendJSON<{ deleted: string }>(`/runs/${encodeURIComponent(runId)}`, "DELETE"),

  history: (start: string | null, end: string | null, varName = "temperature_2m") => {
    if (IS_STATIC) return Promise.resolve<HistoryResponse>({ var: varName, series: [] });
    const params = new URLSearchParams();
    if (start) params.set("start", start);
    if (end) params.set("end", end);
    params.set("var", varName);
    return getJSON<HistoryResponse>(`/history?${params.toString()}`);
  },

  forecast: (datetime: string, horizon: number, runId?: string) => {
    if (IS_STATIC) return forecastFromSnapshots(datetime, horizon);
    const params = new URLSearchParams({ datetime, horizon: String(horizon) });
    if (runId) params.set("run_id", runId);
    return getJSON<ForecastResponse>(`/forecast?${params.toString()}`);
  },

  // Static package only: the fixed set of precomputed backtests available to pick from.
  forecastExamples: (): Promise<{ finalRunId: string | null; items: ForecastExample[] }> =>
    IS_STATIC
      ? snap<ForecastIndex>("forecasts/index.json").then((i) => ({
          finalRunId: i.final_run_id,
          items: i.items,
        }))
      : Promise.resolve({ finalRunId: null, items: [] }),
};
