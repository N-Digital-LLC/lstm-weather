// Typed client for the FastAPI backend. All calls are client-side (no SSR backend calls).

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, "") || "http://localhost:8000";

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!res.ok) {
    const detail = await res.text().catch(() => res.statusText);
    throw new Error(`GET ${path} failed (${res.status}): ${detail}`);
  }
  return res.json() as Promise<T>;
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
  test_rmse_C: number | null;
  error: string | null;
}

export interface Metric {
  mae_C: number;
  rmse_C: number;
  mape_pct?: number;
}

export interface HorizonCurve {
  hours: number[];
  lstm: number[];
  persistence?: number[];
  diurnal?: number[];
  climatology?: number[];
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
  val_metrics: { lstm: Metric } | null;
  test_metrics: Record<string, Metric> | null;
  skill_vs: Record<string, number> | null;
  val_horizon?: HorizonCurve | null;
  test_horizon?: HorizonCurve | null;
  history: EpochRow[];
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

// --- Calls -------------------------------------------------------------------
export const api = {
  health: () => getJSON<{ status: string; device: string }>("/health"),

  listRuns: () => getJSON<{ runs: RunSummary[] }>("/runs"),

  getRun: (runId: string) => getJSON<RunCard>(`/runs/${encodeURIComponent(runId)}`),

  getRunHistory: (runId: string) =>
    getJSON<{ run_id: string; history: EpochRow[] }>(
      `/runs/${encodeURIComponent(runId)}/history`,
    ),

  compare: (ids: string[]) =>
    getJSON<{ runs: CompareRun[] }>(`/runs/compare?ids=${encodeURIComponent(ids.join(","))}`),

  createRun: (overrides: Record<string, unknown>) =>
    sendJSON<{ run_id: string; status: string }>("/runs", "POST", overrides),

  createSweep: (body: Record<string, unknown>) =>
    sendJSON<{ run_ids: string[] }>("/runs/sweep", "POST", body),

  deleteRun: (runId: string) =>
    sendJSON<{ deleted: string }>(`/runs/${encodeURIComponent(runId)}`, "DELETE"),

  history: (start: string | null, end: string | null, varName = "temperature_2m") => {
    const params = new URLSearchParams();
    if (start) params.set("start", start);
    if (end) params.set("end", end);
    params.set("var", varName);
    return getJSON<HistoryResponse>(`/history?${params.toString()}`);
  },

  forecast: (datetime: string, horizon: number, runId?: string) => {
    const params = new URLSearchParams({ datetime, horizon: String(horizon) });
    if (runId) params.set("run_id", runId);
    return getJSON<ForecastResponse>(`/forecast?${params.toString()}`);
  },
};
