"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, IS_STATIC, type EpochRow, type RunCard } from "@/lib/api";
import { StatusBadge } from "@/components/RunStatus";
import { fmtDuration, fmtNum } from "@/lib/format";

export default function RunDetailClient({ runId }: { runId: string }) {
  const [card, setCard] = useState<RunCard | null>(null);
  const [history, setHistory] = useState<EpochRow[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let timer: ReturnType<typeof setInterval> | undefined;
    async function load() {
      try {
        const [c, h] = await Promise.all([api.getRun(runId), api.getRunHistory(runId)]);
        setCard(c);
        setHistory(h.history);
        if (IS_STATIC || c.status === "done" || c.status === "failed") {
          if (timer) clearInterval(timer);
        }
      } catch (e) {
        setError(String(e));
      }
    }
    load();
    // No live polling in the static results package.
    if (!IS_STATIC) timer = setInterval(load, 3000);
    return () => {
      if (timer) clearInterval(timer);
    };
  }, [runId]);

  if (error) return <div className="card text-rose-300">{error}</div>;
  if (!card) return <div className="card text-slate-400">Зареждане на {runId}…</div>;

  const horizon = card.test_horizon ?? card.val_horizon ?? null;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <Link href="/training" className="text-sm text-sky-300 hover:underline">
            ← Назад към обучението
          </Link>
          <h1 className="mt-1 text-2xl font-bold">{card.run_id}</h1>
        </div>
        <StatusBadge status={card.status} />
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <ReportCard title="Конфигурация">
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

        <ReportCard title="Разделяне на данните (този пуск)">
          {card.data ? (
            <>
              <KV k="Години за обучение" v={card.data.train_years || "—"} />
              <KV
                k="Години за валидация"
                v={card.data.val_years || (card.config.is_final ? "слети с обучението" : "—")}
              />
              <KV k="Тестови години" v={card.data.test_years || "—"} />
              <KV k="Прозорци за обучение" v={card.data.n_train_windows.toLocaleString()} />
              <KV k="Прозорци за валидация" v={card.data.n_val_windows.toLocaleString()} />
              <KV k="Тестови прозорци" v={card.data.n_test_windows.toLocaleString()} />
              <KV k="Признаци" v={`${card.data.features.length} колони`} />
              {card.config.is_final && (
                <p className="pt-1 text-xs text-slate-500">
                  Финален пуск: обучение+валидация слети за обучението; тестовите години се оценяват веднъж.
                </p>
              )}
            </>
          ) : (
            <p className="text-slate-500">Изчаква…</p>
          )}
        </ReportCard>

        <ReportCard title="Обучение">
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
              епоха {card.progress.current_epoch}/{card.progress.total_epochs}
            </p>
          )}
        </ReportCard>
      </div>

      <MetricsBlock card={card} />

      <div className="card">
        <h2 className="mb-3 font-semibold">Крива на обучение</h2>
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
                <Line yAxisId="loss" type="monotone" dataKey="train_loss" name="загуба обучение" stroke="#38bdf8" dot={false} connectNulls />
                <Line yAxisId="loss" type="monotone" dataKey="val_loss" name="загуба валидация (MAE)" stroke="#f97316" dot={false} connectNulls />
                <Line yAxisId="rmse" type="monotone" dataKey="val_rmse_C" name="вал. RMSE °C" stroke="#34d399" strokeDasharray="5 4" dot={false} connectNulls />
              </LineChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <p className="text-slate-500">Все още няма записани епохи.</p>
        )}
      </div>

      {horizon && horizon.hours.length > 1 && (
        <>
          <div className="card">
            <h2 className="mb-3 font-semibold">
              RMSE спрямо хоризонт ({card.test_horizon ? "тест" : "валидация"})
            </h2>
            <HorizonChart
              hours={horizon.hours}
              series={{
                lstm: horizon.lstm,
                persistence: horizon.persistence,
                diurnal: horizon.diurnal,
                climatology: horizon.climatology,
              }}
              unit="°"
            />
          </div>
          {horizon.mae && (
            <div className="card">
              <h2 className="mb-3 font-semibold">
                MAE спрямо хоризонт ({card.test_horizon ? "тест" : "валидация"})
              </h2>
              <HorizonChart hours={horizon.hours} series={horizon.mae} unit="°" />
            </div>
          )}
          {horizon.bias && (
            <div className="card">
              <h2 className="mb-3 font-semibold">
                Отклонение спрямо хоризонт ({card.test_horizon ? "тест" : "валидация"})
              </h2>
              <p className="mb-3 text-xs text-slate-500">
                Средна грешка за всеки час напред — над нулата моделът завишава, под нея занижава.
              </p>
              <HorizonChart hours={horizon.hours} series={horizon.bias} unit="°" zeroLine />
            </div>
          )}
        </>
      )}

      <details className="card">
        <summary className="cursor-pointer font-semibold">Суров card.json</summary>
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
        <h2 className="mb-3 font-semibold">Тестови метрики (финален модел)</h2>
        <div className="grid gap-3 sm:grid-cols-4">
          {Object.entries(card.test_metrics).map(([model, m]) => (
            <div key={model} className="rounded-lg border border-slate-800 bg-slate-900 p-3">
              <div className="text-xs uppercase tracking-wide text-slate-400">{model}</div>
              <div className="text-lg font-semibold">RMSE {fmtNum(m.rmse_C)}</div>
              <div className="text-sm text-slate-400">
                MAE {fmtNum(m.mae_C)}
                {m.mape_pct !== undefined ? ` · MAPE ${fmtNum(m.mape_pct)}%` : ""}
              </div>
              <div className="text-sm text-slate-400">
                Bias {fmtNum(m.bias_C)}
                {m.r2 !== undefined ? ` · R² ${fmtNum(m.r2)}` : ""}
              </div>
            </div>
          ))}
        </div>
        {card.skill_vs && (
          <div className="mt-3 text-sm text-slate-300">
            Умение спрямо:{" "}
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
        <h2 className="mb-3 font-semibold">Валидационни метрики (настройващ пуск)</h2>
        <div className="flex flex-wrap gap-6">
          <div>
            <div className="text-xs uppercase tracking-wide text-slate-400">LSTM RMSE</div>
            <div className="text-2xl font-semibold">{fmtNum(card.val_metrics.lstm.rmse_C)}</div>
          </div>
          <div>
            <div className="text-xs uppercase tracking-wide text-slate-400">LSTM MAE</div>
            <div className="text-2xl font-semibold">{fmtNum(card.val_metrics.lstm.mae_C)}</div>
          </div>
          <div>
            <div className="text-xs uppercase tracking-wide text-slate-400">LSTM Bias</div>
            <div className="text-2xl font-semibold">{fmtNum(card.val_metrics.lstm.bias_C)}</div>
          </div>
          <div>
            <div className="text-xs uppercase tracking-wide text-slate-400">LSTM R²</div>
            <div className="text-2xl font-semibold">{fmtNum(card.val_metrics.lstm.r2)}</div>
          </div>
        </div>
        <p className="mt-3 text-xs text-slate-500">
          Тестовите метрики умишлено липсват — настройващите пускове никога не докосват тестовия набор.
        </p>
      </div>
    );
  }
  return null;
}

const HORIZON_SERIES = [
  { key: "lstm", name: "LSTM", stroke: "#38bdf8", width: 2.5, dashed: false },
  { key: "persistence", name: "Персистентност", stroke: "#f97316", width: 1, dashed: true },
  { key: "diurnal", name: "Денонощен", stroke: "#a78bfa", width: 1, dashed: true },
  { key: "climatology", name: "Климатология", stroke: "#34d399", width: 1, dashed: true },
] as const;

function HorizonChart({
  hours,
  series,
  unit,
  zeroLine = false,
}: {
  hours: number[];
  series: Record<string, number[] | undefined>;
  unit?: string;
  zeroLine?: boolean;
}) {
  const present = HORIZON_SERIES.filter((s) => Array.isArray(series[s.key]));
  const data = hours.map((h, i) => {
    const row: Record<string, number | undefined> = { hour: h };
    present.forEach((s) => {
      row[s.key] = series[s.key]?.[i];
    });
    return row;
  });
  return (
    <div className="h-80">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
          <CartesianGrid stroke="#1e293b" />
          <XAxis dataKey="hour" stroke="#64748b" fontSize={11} unit="h" />
          <YAxis stroke="#64748b" fontSize={11} unit={unit} />
          <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155" }} />
          <Legend />
          {zeroLine && <ReferenceLine y={0} stroke="#64748b" strokeDasharray="3 3" />}
          {present.map((s) => (
            <Line
              key={s.key}
              type="monotone"
              dataKey={s.key}
              name={s.name}
              stroke={s.stroke}
              strokeWidth={s.width}
              strokeDasharray={s.dashed ? "5 4" : undefined}
              dot={false}
            />
          ))}
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
