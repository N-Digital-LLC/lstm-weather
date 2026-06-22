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
import {
  api,
  IS_STATIC,
  type ForecastExample,
  type ForecastResponse,
  type RunSummary,
} from "@/lib/api";
import { fmtNum } from "@/lib/format";

// "2024-07-15T12:00" -> "15 Jul 2024, 12:00" (these are naive local timestamps).
function fmtIssued(dt: string): string {
  const d = new Date(dt);
  if (Number.isNaN(d.getTime())) return dt;
  return d.toLocaleString("bg-BG", {
    day: "2-digit",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

const MODELS = [
  { key: "actual", label: "Реално", color: "#e2e8f0", dash: false },
  { key: "lstm", label: "LSTM", color: "#38bdf8", dash: false },
  { key: "persistence", label: "Персистентност", color: "#f97316", dash: true },
  { key: "diurnal", label: "Денонощен", color: "#a78bfa", dash: true },
  { key: "climatology", label: "Климатология", color: "#34d399", dash: true },
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

  // Static package only: the fixed set of precomputed backtests.
  const [examples, setExamples] = useState<ForecastExample[]>([]);
  const [finalRunId, setFinalRunId] = useState<string | null>(null);

  useEffect(() => {
    api
      .listRuns()
      .then((r) => {
        const done = r.runs.filter((x) => x.status === "done");
        setRuns(done);
        // Auto-select the final model (newest first) so it's the default forecaster.
        const finalRun = done.find((x) => x.is_final);
        if (finalRun) setRunId(finalRun.run_id);
      })
      .catch((e) => setError(String(e)));
  }, []);

  // In the static package, load the available examples and default to the first.
  useEffect(() => {
    if (!IS_STATIC) return;
    api
      .forecastExamples()
      .then(({ finalRunId, items }) => {
        setFinalRunId(finalRunId);
        setExamples(items);
        if (items.length) {
          setDatetime(items[0].datetime);
          const horizons = items
            .filter((it) => it.datetime === items[0].datetime)
            .map((it) => it.horizon);
          setHorizon(horizons.includes(24) ? 24 : horizons[horizons.length - 1]);
        }
      })
      .catch((e) => setError(String(e)));
  }, []);

  // Final models first so the (usually single) final run is trivial to pick.
  const sortedRuns = useMemo(
    () => [...runs].sort((a, b) => Number(b.is_final) - Number(a.is_final)),
    [runs],
  );

  // Distinct issued-for dates and the horizons available for the chosen date.
  const exampleDates = useMemo(
    () => Array.from(new Set(examples.map((e) => e.datetime))),
    [examples],
  );
  const horizonsForDate = useMemo(
    () =>
      Array.from(new Set(examples.filter((e) => e.datetime === datetime).map((e) => e.horizon))).sort(
        (a, b) => a - b,
      ),
    [examples, datetime],
  );

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

  // Static package: auto-load whenever the (constrained) selection changes.
  useEffect(() => {
    if (!IS_STATIC || !examples.length) return;
    runForecast();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datetime, horizon, examples.length]);

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
        <h1 className="text-2xl font-bold">Прогноза</h1>
        <p className="text-sm text-slate-400">
          Прогнозирайте следващите часове на температурата за Варна и сравнете LSTM с наивните
          базови модели и реалните стойности.
        </p>
      </div>

      {IS_STATIC ? (
        <>
          <div className="card border-sky-800 bg-sky-950/40 text-sm text-sky-100">
            <strong>Предварително изчислени бектестове.</strong> Този офлайн пакет няма работещ
            модел, затова съдържа фиксиран набор от реални бектестове от финалния модел. Изберете
            някоя от наличните дати и хоризонти по-долу — графиката се обновява автоматично.
          </div>

          <div className="card grid gap-4 sm:grid-cols-3">
            <div>
              <label className="label">Модел</label>
              <input
                className="input cursor-not-allowed opacity-80"
                value={finalRunId ? `★ ФИНАЛЕН — ${finalRunId}` : "финален модел"}
                readOnly
              />
            </div>
            <div>
              <label className="label">Изготвена към (последен наблюдаван час)</label>
              <select
                className="input"
                value={datetime}
                onChange={(e) => setDatetime(e.target.value)}
              >
                {exampleDates.map((d) => (
                  <option key={d} value={d}>
                    {fmtIssued(d)}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="label">Хоризонт (часове)</label>
              <select
                className="input"
                value={horizon}
                onChange={(e) => setHorizon(Number(e.target.value))}
              >
                {horizonsForDate.map((h) => (
                  <option key={h} value={h}>
                    +{h} ч
                  </option>
                ))}
              </select>
            </div>
          </div>
        </>
      ) : (
        <div className="card grid gap-4 sm:grid-cols-4">
          <div>
            <label className="label">Пуск</label>
            <select className="input" value={runId} onChange={(e) => setRunId(e.target.value)}>
              <option value="">Автоматично — най-добрият валидационен пуск</option>
              {sortedRuns.map((r) => (
                <option key={r.run_id} value={r.run_id}>
                  {r.is_final ? "★ ФИНАЛЕН — " : ""}
                  {r.run_id}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="label">Изготвена към (последен наблюдаван час)</label>
            <input
              type="datetime-local"
              className="input"
              value={datetime}
              onChange={(e) => setDatetime(e.target.value)}
            />
          </div>
          <div>
            <label className="label">Хоризонт (часове)</label>
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
              {loading ? "Прогнозиране…" : "Прогноза"}
            </button>
          </div>
        </div>
      )}

      {error && (
        <div className="card border-rose-800 bg-rose-950/40 text-sm text-rose-200">{error}</div>
      )}

      {data && (
        <>
          <div className="card">
            <div className="mb-2 flex items-center justify-between">
              <h2 className="font-semibold">
                Прогноза за температура (°C) · пуск{" "}
                <span className="text-sky-300">{data.run_id}</span>
              </h2>
              <span className="text-xs text-slate-400">
                изготвена към {new Date(data.issued_for).toLocaleString("bg-BG")}
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
            <h2 className="mb-3 font-semibold">MAE за този прозорец (°C, където има реални данни)</h2>
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
