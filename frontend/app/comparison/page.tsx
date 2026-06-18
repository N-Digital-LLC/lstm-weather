"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, type CompareRun, type RunSummary } from "@/lib/api";
import { colorFor, fmtNum } from "@/lib/format";

export default function ComparisonPage() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [compared, setCompared] = useState<CompareRun[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .listRuns()
      .then((r) => setRuns(r.runs.filter((x) => x.status === "done")))
      .catch((e) => setError(String(e)));
  }, []);

  function toggle(id: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  async function compare() {
    setError(null);
    try {
      const res = await api.compare([...selected]);
      setCompared(res.runs);
    } catch (e) {
      setError(String(e));
    }
  }

  const barData = useMemo(
    () =>
      compared
        .filter((r) => r.best_val_rmse_C !== null)
        .map((r) => ({
          name: `h${r.config.hidden_size}`,
          run_id: r.run_id,
          rmse: r.best_val_rmse_C as number,
        })),
    [compared],
  );

  const curveData = useMemo(() => {
    const maxEpoch = Math.max(0, ...compared.map((r) => r.history.length));
    const rows: Record<string, number | null>[] = [];
    for (let e = 1; e <= maxEpoch; e++) {
      const row: Record<string, number | null> = { epoch: e };
      compared.forEach((r) => {
        const h = r.history.find((x) => x.epoch === e);
        row[r.run_id] = h ? h.val_rmse_C : null;
      });
      rows.push(row);
    }
    return rows;
  }, [compared]);

  const finalRuns = compared.filter((r) => r.test_horizon && r.test_horizon.hours.length > 1);
  const horizonData = useMemo(() => {
    if (finalRuns.length === 0) return [];
    const hours = finalRuns[0].test_horizon!.hours;
    return hours.map((hr, i) => {
      const row: Record<string, number | null> = { hour: hr };
      finalRuns.forEach((r) => {
        row[r.run_id] = r.test_horizon!.lstm?.[i] ?? null;
      });
      return row;
    });
  }, [finalRuns]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Comparison</h1>
        <p className="text-sm text-slate-400">
          The exam artifact: pick the swept runs and compare capacity vs overfitting on validation
          metrics. Test numbers appear only once a final model exists.
        </p>
      </div>

      <div className="card">
        <h2 className="mb-3 font-semibold">Select runs ({selected.size})</h2>
        {runs.length === 0 ? (
          <p className="text-slate-500">No completed runs yet.</p>
        ) : (
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {runs.map((r) => (
              <label
                key={r.run_id}
                className="flex items-center gap-2 rounded-lg border border-slate-800 bg-slate-900 px-3 py-2 text-sm"
              >
                <input
                  type="checkbox"
                  checked={selected.has(r.run_id)}
                  onChange={() => toggle(r.run_id)}
                />
                <span className="truncate">
                  {r.run_id}
                  {r.is_final && <span className="ml-2 text-fuchsia-300">final</span>}
                </span>
              </label>
            ))}
          </div>
        )}
        <button
          className="btn-primary mt-4"
          onClick={compare}
          disabled={selected.size < 2}
          title={selected.size < 2 ? "Select at least 2 runs" : undefined}
        >
          Compare {selected.size} runs
        </button>
        {error && <p className="mt-3 text-sm text-rose-300">{error}</p>}
      </div>

      {compared.length > 0 && (
        <>
          <div className="grid gap-6 lg:grid-cols-2">
            <div className="card">
              <h2 className="mb-3 font-semibold">Best val RMSE vs hidden size</h2>
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={barData} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
                    <CartesianGrid stroke="#1e293b" />
                    <XAxis dataKey="name" stroke="#64748b" fontSize={11} />
                    <YAxis stroke="#64748b" fontSize={11} unit="°" />
                    <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155" }} />
                    <Bar dataKey="rmse" name="best val RMSE °C">
                      {barData.map((_, i) => (
                        <Cell key={i} fill={colorFor(i)} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="card">
              <h2 className="mb-3 font-semibold">Val RMSE per epoch (overlaid)</h2>
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={curveData} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
                    <CartesianGrid stroke="#1e293b" />
                    <XAxis dataKey="epoch" stroke="#64748b" fontSize={11} />
                    <YAxis stroke="#64748b" fontSize={11} unit="°" />
                    <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155" }} />
                    <Legend />
                    {compared.map((r, i) => (
                      <Line
                        key={r.run_id}
                        type="monotone"
                        dataKey={r.run_id}
                        name={`h${r.config.hidden_size}`}
                        stroke={colorFor(i)}
                        dot={false}
                        connectNulls
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {horizonData.length > 0 && (
            <div className="card">
              <h2 className="mb-3 font-semibold">Test RMSE vs horizon hour (final models)</h2>
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={horizonData} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
                    <CartesianGrid stroke="#1e293b" />
                    <XAxis dataKey="hour" stroke="#64748b" fontSize={11} unit="h" />
                    <YAxis stroke="#64748b" fontSize={11} unit="°" />
                    <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155" }} />
                    <Legend />
                    {finalRuns.map((r, i) => (
                      <Line
                        key={r.run_id}
                        type="monotone"
                        dataKey={r.run_id}
                        name={`h${r.config.hidden_size}`}
                        stroke={colorFor(i)}
                        strokeWidth={2}
                        dot={false}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          <div className="card overflow-x-auto">
            <h2 className="mb-3 font-semibold">Comparison table</h2>
            <table className="w-full text-left text-sm">
              <thead className="text-xs uppercase tracking-wide text-slate-400">
                <tr className="border-b border-slate-800">
                  <th className="py-2 pr-4">Run (timestamp)</th>
                  <th className="py-2 pr-4">Hidden</th>
                  <th className="py-2 pr-4">Layers</th>
                  <th className="py-2 pr-4">Final</th>
                  <th className="py-2 pr-4">Best val RMSE</th>
                  <th className="py-2 pr-4">Test RMSE</th>
                  <th className="py-2 pr-4">Skill vs diurnal</th>
                </tr>
              </thead>
              <tbody>
                {compared.map((r) => (
                  <tr key={r.run_id} className="border-b border-slate-800/60">
                    <td className="py-2 pr-4 font-mono text-xs text-slate-300">{r.run_id}</td>
                    <td className="py-2 pr-4">{r.config.hidden_size}</td>
                    <td className="py-2 pr-4">{r.config.num_layers}</td>
                    <td className="py-2 pr-4">{r.is_final ? "yes" : "—"}</td>
                    <td className="py-2 pr-4">{fmtNum(r.best_val_rmse_C)}</td>
                    <td className="py-2 pr-4">
                      {r.test_metrics?.lstm ? fmtNum(r.test_metrics.lstm.rmse_C) : "—"}
                    </td>
                    <td className="py-2 pr-4">
                      {r.skill_vs?.diurnal !== undefined ? fmtNum(r.skill_vs.diurnal) : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <button className="btn-secondary mt-4" onClick={() => window.print()}>
              Print / export
            </button>
          </div>
        </>
      )}
    </div>
  );
}
