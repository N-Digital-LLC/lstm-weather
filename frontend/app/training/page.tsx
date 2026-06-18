"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { api, type RunSummary } from "@/lib/api";
import { FinalBadge, ProgressBar } from "@/components/RunStatus";
import { fmtDateTime, fmtNum } from "@/lib/format";

const DEFAULT_FORM = {
  hidden_size: 128,
  num_layers: 2,
  lookback: 168,
  horizon: 1,
  stride: 1,
  batch: 256,
  epochs: 30,
  use_anomaly: false,
};

export default function TrainingPage() {
  const [form, setForm] = useState(DEFAULT_FORM);
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  async function refresh() {
    try {
      const r = await api.listRuns();
      setRuns(r.runs);
    } catch (e) {
      setError(String(e));
    }
  }

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 3000); // live polling
    return () => clearInterval(id);
  }, []);

  function setField<K extends keyof typeof form>(key: K, value: (typeof form)[K]) {
    setForm((f) => ({ ...f, [key]: value }));
  }

  async function launch() {
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const res = await api.createRun({ ...form, is_final: false });
      setNotice(`Queued tuning run ${res.run_id}`);
      refresh();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  async function sweep() {
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const { hidden_size: _hs, ...shared } = form;
      const res = await api.createSweep({ ...shared, is_final: false, hidden_sizes: [64, 128, 256] });
      setNotice(`Queued sweep: ${res.run_ids.join(", ")}`);
      refresh();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  async function trainFinal(run: RunSummary) {
    const ok = window.confirm(
      `Train the FINAL model from ${run.run_id}'s config?\n\n` +
        "This retrains on train+validation merged and evaluates the TEST set — do this exactly " +
        "once, on your chosen winner. All tuning reports stay test-free by design.",
    );
    if (!ok) return;
    setBusy(true);
    setError(null);
    try {
      const c = run.config;
      const res = await api.createRun({
        hidden_size: c.hidden_size,
        num_layers: c.num_layers,
        lookback: c.lookback,
        horizon: c.horizon,
        stride: c.stride,
        batch: c.batch,
        lr: c.lr,
        use_anomaly: c.use_anomaly,
        epochs: form.epochs,
        is_final: true,
      });
      setNotice(`Queued FINAL run ${res.run_id}`);
      refresh();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  async function remove(run: RunSummary) {
    if (!window.confirm(`Delete run ${run.run_id}?`)) return;
    try {
      await api.deleteRun(run.run_id);
      refresh();
    } catch (e) {
      setError(String(e));
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Training</h1>
        <p className="text-sm text-slate-400">
          Launch tuning runs (judged on validation), sweep hidden sizes for the fair 64/128/256
          comparison, then promote the winner to the single test-touching final model.
        </p>
      </div>

      <div className="card">
        <h2 className="mb-4 font-semibold">Launch a tuning run</h2>
        <div className="grid gap-4 sm:grid-cols-3 lg:grid-cols-4">
          <Field label="Hidden size">
            <input
              type="number"
              className="input"
              value={form.hidden_size}
              onChange={(e) => setField("hidden_size", Number(e.target.value))}
            />
          </Field>
          <Field label="Layers">
            <input
              type="number"
              className="input"
              value={form.num_layers}
              onChange={(e) => setField("num_layers", Number(e.target.value))}
            />
          </Field>
          <Field label="Lookback (h)">
            <input
              type="number"
              className="input"
              value={form.lookback}
              onChange={(e) => setField("lookback", Number(e.target.value))}
            />
          </Field>
          <Field label="Horizon (h)">
            <input
              type="number"
              className="input"
              value={form.horizon}
              onChange={(e) => setField("horizon", Number(e.target.value))}
            />
          </Field>
          <Field label="Stride">
            <input
              type="number"
              className="input"
              value={form.stride}
              onChange={(e) => setField("stride", Number(e.target.value))}
            />
          </Field>
          <Field label="Batch">
            <input
              type="number"
              className="input"
              value={form.batch}
              onChange={(e) => setField("batch", Number(e.target.value))}
            />
          </Field>
          <Field label="Epochs">
            <input
              type="number"
              className="input"
              value={form.epochs}
              onChange={(e) => setField("epochs", Number(e.target.value))}
            />
          </Field>
          <Field label="Anomaly target">
            <label className="mt-2 inline-flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={form.use_anomaly}
                onChange={(e) => setField("use_anomaly", e.target.checked)}
              />
              predict value − climatology
            </label>
          </Field>
        </div>
        <div className="mt-4 flex flex-wrap gap-3">
          <button className="btn-primary" onClick={launch} disabled={busy}>
            Launch run
          </button>
          <button className="btn-secondary" onClick={sweep} disabled={busy}>
            Compare hidden sizes (64 / 128 / 256)
          </button>
        </div>
        {notice && <p className="mt-3 text-sm text-emerald-300">{notice}</p>}
        {error && <p className="mt-3 text-sm text-rose-300">{error}</p>}
      </div>

      <div className="card overflow-x-auto">
        <h2 className="mb-4 font-semibold">Runs</h2>
        <table className="w-full text-left text-sm">
          <thead className="text-xs uppercase tracking-wide text-slate-400">
            <tr className="border-b border-slate-800">
              <th className="py-2 pr-4">Run</th>
              <th className="py-2 pr-4">Started</th>
              <th className="py-2 pr-4">Config</th>
              <th className="py-2 pr-4">Final</th>
              <th className="py-2 pr-4">Status</th>
              <th className="py-2 pr-4">Best val RMSE</th>
              <th className="py-2 pr-4">Test RMSE</th>
              <th className="py-2 pr-4">Actions</th>
            </tr>
          </thead>
          <tbody>
            {runs.length === 0 && (
              <tr>
                <td colSpan={8} className="py-6 text-center text-slate-500">
                  No runs yet — launch one above.
                </td>
              </tr>
            )}
            {runs.map((r) => (
              <tr key={r.run_id} className="border-b border-slate-800/60 hover:bg-slate-800/30">
                <td className="py-2 pr-4">
                  <Link className="text-sky-300 hover:underline" href={`/runs/${encodeURIComponent(r.run_id)}`}>
                    {r.run_id}
                  </Link>
                </td>
                <td className="py-2 pr-4 text-slate-400">{fmtDateTime(r.started_at)}</td>
                <td className="py-2 pr-4 text-slate-300">
                  h{r.config?.hidden_size} · l{r.config?.num_layers} · L{r.config?.lookback} · H
                  {r.config?.horizon} · s{r.config?.stride}
                  {r.config?.use_anomaly ? " · anom" : ""}
                </td>
                <td className="py-2 pr-4">
                  <FinalBadge isFinal={r.is_final} />
                </td>
                <td className="py-2 pr-4">
                  <ProgressBar status={r.status} progress={r.progress} />
                  {r.status === "failed" && r.error && (
                    <div className="mt-1 max-w-xs truncate text-[11px] text-rose-300" title={r.error}>
                      {r.error}
                    </div>
                  )}
                </td>
                <td className="py-2 pr-4">{fmtNum(r.best_val_rmse_C)}</td>
                <td className="py-2 pr-4">
                  {r.is_final ? fmtNum(r.test_rmse_C) : <span className="text-slate-600">—</span>}
                </td>
                <td className="py-2 pr-4">
                  <div className="flex gap-2">
                    {r.status === "done" && !r.is_final && (
                      <button
                        className="btn-secondary px-2 py-1 text-xs"
                        onClick={() => trainFinal(r)}
                        disabled={busy}
                      >
                        Make final
                      </button>
                    )}
                    <button className="btn-danger px-2 py-1 text-xs" onClick={() => remove(r)}>
                      Delete
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="label">{label}</label>
      {children}
    </div>
  );
}
