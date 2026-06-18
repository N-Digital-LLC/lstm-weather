"use client";

import { useEffect, useMemo, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, type ForecastResponse, type RunSummary } from "@/lib/api";
import { fmtNum } from "@/lib/format";

const MODELS = [
  { key: "actual", label: "Actual", color: "#e2e8f0", dash: false },
  { key: "lstm", label: "LSTM", color: "#38bdf8", dash: false },
  { key: "persistence", label: "Persistence", color: "#f97316", dash: true },
  { key: "diurnal", label: "Diurnal", color: "#a78bfa", dash: true },
  { key: "climatology", label: "Climatology", color: "#34d399", dash: true },
] as const;

function mae(rows: ForecastResponse["forecast"], key: keyof ForecastResponse["forecast"][number]) {
  const pairs = rows.filter((r) => r.actual !== null && r[key] !== null);
  if (pairs.length === 0) return null;
  const sum = pairs.reduce((acc, r) => acc + Math.abs((r[key] as number) - (r.actual as number)), 0);
  return sum / pairs.length;
}

export default function ForecastPage() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [runId, setRunId] = useState<string>("");
  const [datetime, setDatetime] = useState<string>("2024-01-15T12:00");
  const [horizon, setHorizon] = useState<number>(24);
  const [data, setData] = useState<ForecastResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .listRuns()
      .then((r) => setRuns(r.runs.filter((x) => x.status === "done")))
      .catch((e) => setError(String(e)));
  }, []);

  async function runForecast() {
    setLoading(true);
    setError(null);
    try {
      const res = await api.forecast(datetime, horizon, runId || undefined);
      setData(res);
    } catch (e) {
      setError(String(e));
      setData(null);
    } finally {
      setLoading(false);
    }
  }

  const chartData = useMemo(
    () =>
      (data?.forecast ?? []).map((r) => ({
        t: new Date(r.datetime).toLocaleString(undefined, {
          month: "short",
          day: "2-digit",
          hour: "2-digit",
        }),
        actual: r.actual,
        lstm: r.lstm,
        persistence: r.persistence,
        diurnal: r.diurnal,
        climatology: r.climatology,
      })),
    [data],
  );

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Forecast</h1>
        <p className="text-sm text-slate-400">
          Predict the next hours of temperature for Varna and overlay the LSTM against the naive
          baselines and the actual values.
        </p>
      </div>

      <div className="card grid gap-4 sm:grid-cols-4">
        <div>
          <label className="label">Run</label>
          <select className="input" value={runId} onChange={(e) => setRunId(e.target.value)}>
            <option value="">Best / most recent</option>
            {runs.map((r) => (
              <option key={r.run_id} value={r.run_id}>
                {r.run_id}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className="label">Issued for (last observed hour)</label>
          <input
            type="datetime-local"
            className="input"
            value={datetime}
            onChange={(e) => setDatetime(e.target.value)}
          />
        </div>
        <div>
          <label className="label">Horizon (hours)</label>
          <select
            className="input"
            value={horizon}
            onChange={(e) => setHorizon(Number(e.target.value))}
          >
            {[1, 6, 12, 24].map((h) => (
              <option key={h} value={h}>
                {h}
              </option>
            ))}
          </select>
        </div>
        <div className="flex items-end">
          <button className="btn-primary w-full" onClick={runForecast} disabled={loading}>
            {loading ? "Forecasting…" : "Forecast"}
          </button>
        </div>
      </div>

      {error && (
        <div className="card border-rose-800 bg-rose-950/40 text-sm text-rose-200">{error}</div>
      )}

      {data && (
        <>
          <div className="card">
            <div className="mb-2 flex items-center justify-between">
              <h2 className="font-semibold">
                Temperature forecast (°C) · run <span className="text-sky-300">{data.run_id}</span>
              </h2>
              <span className="text-xs text-slate-400">
                issued for {new Date(data.issued_for).toLocaleString()}
              </span>
            </div>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
                  <CartesianGrid stroke="#1e293b" />
                  <XAxis dataKey="t" stroke="#64748b" fontSize={11} />
                  <YAxis stroke="#64748b" fontSize={11} unit="°" />
                  <Tooltip
                    contentStyle={{ background: "#0f172a", border: "1px solid #334155" }}
                    labelStyle={{ color: "#cbd5e1" }}
                  />
                  <Legend />
                  {MODELS.map((m) => (
                    <Line
                      key={m.key}
                      type="monotone"
                      dataKey={m.key}
                      name={m.label}
                      stroke={m.color}
                      strokeWidth={m.key === "actual" || m.key === "lstm" ? 2.5 : 1.5}
                      strokeDasharray={m.dash ? "5 4" : undefined}
                      dot={false}
                      connectNulls
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="card">
            <h2 className="mb-3 font-semibold">MAE over this window (°C, where actuals exist)</h2>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              {(["lstm", "persistence", "diurnal", "climatology"] as const).map((k) => (
                <div key={k} className="rounded-lg border border-slate-800 bg-slate-900 p-3">
                  <div className="text-xs uppercase tracking-wide text-slate-400">{k}</div>
                  <div className="text-xl font-semibold">{fmtNum(mae(data.forecast, k))}</div>
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
