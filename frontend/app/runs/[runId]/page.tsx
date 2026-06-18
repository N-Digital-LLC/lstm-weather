"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
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
import { api, type EpochRow, type HorizonCurve, type RunCard } from "@/lib/api";
import { StatusBadge } from "@/components/RunStatus";
import { fmtDuration, fmtNum } from "@/lib/format";

export default function RunDetailPage({ params }: { params: { runId: string } }) {
  const runId = decodeURIComponent(params.runId);
  const [card, setCard] = useState<RunCard | null>(null);
  const [history, setHistory] = useState<EpochRow[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let timer: ReturnType<typeof setInterval>;
    async function load() {
      try {
        const [c, h] = await Promise.all([api.getRun(runId), api.getRunHistory(runId)]);
        setCard(c);
        setHistory(h.history);
        if (c.status === "done" || c.status === "failed") clearInterval(timer);
      } catch (e) {
        setError(String(e));
      }
    }
    load();
    timer = setInterval(load, 3000);
    return () => clearInterval(timer);
  }, [runId]);

  if (error) return <div className="card text-rose-300">{error}</div>;
  if (!card) return <div className="card text-slate-400">Loading {runId}…</div>;

  const horizon = card.test_horizon ?? card.val_horizon ?? null;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <Link href="/training" className="text-sm text-sky-300 hover:underline">
            ← Back to training
          </Link>
          <h1 className="mt-1 text-2xl font-bold">{card.run_id}</h1>
        </div>
        <StatusBadge status={card.status} />
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <ReportCard title="Config">
          <KV k="hidden_size" v={card.config.hidden_size} />
          <KV k="num_layers" v={card.config.num_layers} />
          <KV k="lookback" v={card.config.lookback} />
          <KV k="horizon" v={card.config.horizon} />
          <KV k="stride" v={card.config.stride} />
          <KV k="batch" v={card.config.batch} />
          <KV k="lr" v={card.config.lr} />
          <KV k="use_amp" v={String(card.config.use_amp)} />
          <KV k="use_anomaly" v={String(card.config.use_anomaly)} />
          <KV k="is_final" v={String(card.config.is_final)} />
          <KV k="seed" v={card.config.seed} />
          <KV k="device" v={card.device} />
        </ReportCard>

        <ReportCard title="Data">
          {card.data ? (
            <>
              <KV k="train_years" v={card.data.train_years} />
              <KV k="val_years" v={card.data.val_years || "—"} />
              <KV k="test_years" v={card.data.test_years} />
              <KV k="n_train_windows" v={card.data.n_train_windows.toLocaleString()} />
              <KV k="n_val_windows" v={card.data.n_val_windows.toLocaleString()} />
              <KV k="n_test_windows" v={card.data.n_test_windows.toLocaleString()} />
              <KV k="features" v={`${card.data.features.length} cols`} />
            </>
          ) : (
            <p className="text-slate-500">Pending…</p>
          )}
        </ReportCard>

        <ReportCard title="Training">
          {card.training ? (
            <>
              <KV k="epochs_planned" v={card.training.epochs_planned} />
              <KV k="epochs_run" v={card.training.epochs_run} />
              <KV k="early_stop_epoch" v={card.training.early_stop_epoch ?? "—"} />
              <KV k="best_val_rmse_C" v={fmtNum(card.training.best_val_rmse_C)} />
              <KV k="train_time" v={fmtDuration(card.training.train_time_seconds)} />
            </>
          ) : (
            <p className="text-slate-500">
              epoch {card.progress.current_epoch}/{card.progress.total_epochs}
            </p>
          )}
        </ReportCard>
      </div>

      <MetricsBlock card={card} />

      <div className="card">
        <h2 className="mb-3 font-semibold">Training curve</h2>
        {history.length > 0 ? (
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={history} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
                <CartesianGrid stroke="#1e293b" />
                <XAxis dataKey="epoch" stroke="#64748b" fontSize={11} />
                <YAxis yAxisId="loss" stroke="#64748b" fontSize={11} />
                <YAxis yAxisId="rmse" orientation="right" stroke="#34d399" fontSize={11} unit="°" />
                <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155" }} />
                <Legend />
                <Line yAxisId="loss" type="monotone" dataKey="train_loss" name="train loss" stroke="#38bdf8" dot={false} connectNulls />
                <Line yAxisId="loss" type="monotone" dataKey="val_loss" name="val loss (MAE)" stroke="#f97316" dot={false} connectNulls />
                <Line yAxisId="rmse" type="monotone" dataKey="val_rmse_C" name="val RMSE °C" stroke="#34d399" strokeDasharray="5 4" dot={false} connectNulls />
              </LineChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <p className="text-slate-500">No epochs logged yet.</p>
        )}
      </div>

      {horizon && horizon.hours.length > 1 && (
        <div className="card">
          <h2 className="mb-3 font-semibold">
            RMSE vs horizon ({card.test_horizon ? "test" : "validation"})
          </h2>
          <HorizonChart horizon={horizon} />
        </div>
      )}

      <details className="card">
        <summary className="cursor-pointer font-semibold">Raw card.json</summary>
        <pre className="mt-3 overflow-x-auto text-xs text-slate-300">
          {JSON.stringify(card, null, 2)}
        </pre>
      </details>
    </div>
  );
}

function MetricsBlock({ card }: { card: RunCard }) {
  if (card.test_metrics) {
    return (
      <div className="card">
        <h2 className="mb-3 font-semibold">Test metrics (final model)</h2>
        <div className="grid gap-3 sm:grid-cols-4">
          {Object.entries(card.test_metrics).map(([model, m]) => (
            <div key={model} className="rounded-lg border border-slate-800 bg-slate-900 p-3">
              <div className="text-xs uppercase tracking-wide text-slate-400">{model}</div>
              <div className="text-lg font-semibold">RMSE {fmtNum(m.rmse_C)}</div>
              <div className="text-sm text-slate-400">
                MAE {fmtNum(m.mae_C)}
                {m.mape_pct !== undefined ? ` · MAPE ${fmtNum(m.mape_pct)}%` : ""}
              </div>
            </div>
          ))}
        </div>
        {card.skill_vs && (
          <div className="mt-3 text-sm text-slate-300">
            Skill vs:{" "}
            {Object.entries(card.skill_vs).map(([b, s]) => (
              <span key={b} className="mr-3">
                {b} <span className={s > 0 ? "text-emerald-300" : "text-rose-300"}>{fmtNum(s)}</span>
              </span>
            ))}
          </div>
        )}
      </div>
    );
  }
  if (card.val_metrics) {
    return (
      <div className="card">
        <h2 className="mb-3 font-semibold">Validation metrics (tuning run)</h2>
        <div className="flex gap-6">
          <div>
            <div className="text-xs uppercase tracking-wide text-slate-400">LSTM RMSE</div>
            <div className="text-2xl font-semibold">{fmtNum(card.val_metrics.lstm.rmse_C)}</div>
          </div>
          <div>
            <div className="text-xs uppercase tracking-wide text-slate-400">LSTM MAE</div>
            <div className="text-2xl font-semibold">{fmtNum(card.val_metrics.lstm.mae_C)}</div>
          </div>
        </div>
        <p className="mt-3 text-xs text-slate-500">
          Test metrics are intentionally absent — tuning runs never touch the test set.
        </p>
      </div>
    );
  }
  return null;
}

function HorizonChart({ horizon }: { horizon: HorizonCurve }) {
  const data = horizon.hours.map((h, i) => ({
    hour: h,
    lstm: horizon.lstm?.[i],
    persistence: horizon.persistence?.[i],
    diurnal: horizon.diurnal?.[i],
    climatology: horizon.climatology?.[i],
  }));
  return (
    <div className="h-80">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
          <CartesianGrid stroke="#1e293b" />
          <XAxis dataKey="hour" stroke="#64748b" fontSize={11} unit="h" />
          <YAxis stroke="#64748b" fontSize={11} unit="°" />
          <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155" }} />
          <Legend />
          <Line type="monotone" dataKey="lstm" name="LSTM" stroke="#38bdf8" strokeWidth={2.5} dot={false} />
          <Line type="monotone" dataKey="persistence" name="Persistence" stroke="#f97316" strokeDasharray="5 4" dot={false} />
          <Line type="monotone" dataKey="diurnal" name="Diurnal" stroke="#a78bfa" strokeDasharray="5 4" dot={false} />
          <Line type="monotone" dataKey="climatology" name="Climatology" stroke="#34d399" strokeDasharray="5 4" dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function ReportCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="card">
      <h2 className="mb-3 font-semibold">{title}</h2>
      <dl className="space-y-1 text-sm">{children}</dl>
    </div>
  );
}

function KV({ k, v }: { k: string; v: string | number }) {
  return (
    <div className="flex justify-between gap-4">
      <dt className="text-slate-400">{k}</dt>
      <dd className="font-mono text-slate-200">{v}</dd>
    </div>
  );
}
