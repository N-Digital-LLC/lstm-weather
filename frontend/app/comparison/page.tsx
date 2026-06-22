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
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import { api, IS_STATIC, type CompareRun, type Metric, type RunCard, type RunSummary } from "@/lib/api";
import { DataSplitBanner } from "@/components/DataSplit";
import { colorFor, fmtNum, runLabel } from "@/lib/format";

const BASELINES = ["persistence", "diurnal", "climatology"] as const;

// Below this val-RMSE gap (°C) between #1 and #2, the "win" is within seed noise.
const HAIRLINE_C = 0.05;
// A best epoch this early suggests the run may have stopped on a lucky early epoch
// rather than genuinely converging (underfit risk).
const LUCKY_EPOCH = 2;
const BASELINE_COLORS: Record<string, string> = {
  lstm: "#38bdf8",
  persistence: "#f97316",
  diurnal: "#a78bfa",
  climatology: "#34d399",
};
// Bulgarian display names for the baseline keys (keys stay English data keys).
const BASELINE_LABELS: Record<string, string> = {
  lstm: "LSTM",
  persistence: "персистентност",
  diurnal: "денонощен",
  climatology: "климатология",
};

// Prefer test metrics (final runs) but fall back to validation metrics (tuning runs).
function primaryLstm(r: CompareRun): Metric | null {
  return r.test_metrics?.lstm ?? r.val_metrics?.lstm ?? null;
}

export default function ComparisonPage() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [compared, setCompared] = useState<CompareRun[]>([]);
  const [error, setError] = useState<string | null>(null);

  // --- Model-selection panel state ---
  const [selKind, setSelKind] = useState<"all" | "anom" | "plain">("all");
  const [candidateId, setCandidateId] = useState<string | null>(null);
  const [candidateCard, setCandidateCard] = useState<RunCard | null>(null);
  const [candidateErr, setCandidateErr] = useState<string | null>(null);
  const [seedCount, setSeedCount] = useState(3);
  const [seedBusy, setSeedBusy] = useState(false);
  const [seedMsg, setSeedMsg] = useState<string | null>(null);
  const [promoteBusy, setPromoteBusy] = useState(false);
  const [promoteMsg, setPromoteMsg] = useState<string | null>(null);

  useEffect(() => {
    api
      .listRuns()
      .then((r) => setRuns(r.runs.filter((x) => x.status === "done")))
      .catch((e) => setError(String(e)));
  }, []);

  // All completed runs ranked by best validation RMSE, optionally split by anomaly.
  const leaderboard = useMemo(() => {
    const pool = runs.filter(
      (r) =>
        r.best_val_rmse_C !== null &&
        (selKind === "all"
          ? true
          : selKind === "anom"
            ? r.config.use_anomaly
            : !r.config.use_anomaly),
    );
    return [...pool].sort(
      (a, b) => (a.best_val_rmse_C as number) - (b.best_val_rmse_C as number),
    );
  }, [runs, selKind]);

  // "Real win vs hairline": gap from #1 to #2 / #5, and how many runs sit within noise.
  const margin = useMemo(() => {
    if (leaderboard.length < 2) return null;
    const best = leaderboard[0].best_val_rmse_C as number;
    const second = leaderboard[1].best_val_rmse_C as number;
    const fifth = leaderboard[Math.min(4, leaderboard.length - 1)].best_val_rmse_C as number;
    const nWithin = leaderboard.filter(
      (r) => (r.best_val_rmse_C as number) - best < HAIRLINE_C,
    ).length;
    return {
      best,
      gap2: second - best,
      gap5: fifth - best,
      hairline: second - best < HAIRLINE_C,
      nWithin,
    };
  }, [leaderboard]);

  // Default the inspected candidate to the current #1 whenever the ranking changes.
  useEffect(() => {
    setCandidateId(leaderboard[0]?.run_id ?? null);
  }, [leaderboard]);

  // Load the full card for the inspected candidate (val_horizon + early-stop info).
  useEffect(() => {
    if (!candidateId) {
      setCandidateCard(null);
      return;
    }
    let active = true;
    setCandidateErr(null);
    setCandidateCard(null);
    api
      .getRun(candidateId)
      .then((c) => active && setCandidateCard(c))
      .catch((e) => active && setCandidateErr(String(e)));
    return () => {
      active = false;
    };
  }, [candidateId]);

  // "Healthy, not just low": beats every baseline on val + sane early-stop behaviour.
  const health = useMemo(() => {
    if (!candidateCard) return null;
    const vh = candidateCard.val_horizon;
    const tr = candidateCard.training;
    const mean = (a?: number[] | null) =>
      a && a.length ? a.reduce((s, x) => s + x, 0) / a.length : null;
    const lstm = mean(vh?.lstm);
    const baselines = BASELINES.map((b) => {
      const m = mean(vh?.[b]);
      return {
        name: b,
        rmse: m,
        beats: lstm !== null && m !== null ? lstm < m : null,
        margin: lstm !== null && m !== null ? m - lstm : null,
      };
    });
    const bestEpoch = tr?.early_stop_epoch ?? tr?.epochs_run ?? null;
    const epochsRun = tr?.epochs_run ?? null;
    const epochsPlanned = tr?.epochs_planned ?? null;
    return {
      lstm,
      baselines,
      beatsAll: baselines.every((b) => b.beats === true),
      bestEpoch,
      epochsRun,
      epochsPlanned,
      luckyEarly: bestEpoch !== null && bestEpoch <= LUCKY_EPOCH,
      noEarlyStop:
        epochsRun !== null && epochsPlanned !== null && epochsRun >= epochsPlanned,
    };
  }, [candidateCard]);

  async function compareTop(n: number) {
    const ids = leaderboard.slice(0, n).map((r) => r.run_id);
    if (ids.length < 2) return;
    setSelected(new Set(ids));
    setError(null);
    try {
      const res = await api.compare(ids);
      setCompared(res.runs);
      document.getElementById("manual-compare")?.scrollIntoView({ behavior: "smooth" });
    } catch (e) {
      setError(String(e));
    }
  }

  // Re-queue the top-N configs each with fresh seeds, so a stable winner can be told
  // apart from a lucky seed. Identical config (incl. stride) keeps it a fair re-run.
  async function queueSeedRepeats(n: number) {
    const top = leaderboard.slice(0, n);
    if (top.length === 0 || seedBusy) return;
    const seeds = Array.from({ length: seedCount }, (_, i) => i + 1); // 1..seedCount (42 already done)
    setSeedBusy(true);
    setSeedMsg(null);
    try {
      let queued = 0;
      for (const r of top) {
        const c = r.config;
        for (const seed of seeds) {
          await api.createRun({
            hidden_size: c.hidden_size,
            num_layers: c.num_layers,
            lookback: c.lookback,
            horizon: c.horizon,
            batch: c.batch,
            lr: c.lr,
            stride: c.stride,
            use_amp: c.use_amp,
            use_anomaly: c.use_anomaly,
            is_final: false,
            seed,
          });
          queued++;
        }
      }
      setSeedMsg(
        `Добавени ${queued} пуска (${top.length} конфигурации × ${seeds.length} зърна). ` +
          "Следете раздела Обучение; презаредете този раздел, когато приключат, после сравнете средното RMSE за всяка конфигурация.",
      );
    } catch (e) {
      setSeedMsg(`Грешка: ${String(e)}`);
    } finally {
      setSeedBusy(false);
    }
  }

  // Promote the inspected candidate to THE final run: retrain on train+val merged and score
  // the held-out test set exactly once. Uses the card's real config so use_anomaly (absent from
  // the run id) can't be set wrong by hand; stride forced to 1 for the full-data final run.
  async function promoteCandidate() {
    if (!candidateCard || promoteBusy) return;
    const c = candidateCard.config;
    const ok = window.confirm(
      `Да повиша ли ${candidateCard.run_id} до ФИНАЛЕН пуск?\n\n` +
        `${runLabel(c)}\n\n` +
        "Това преобучава върху слети обучение+валидация и оценява заделения тестов набор точно веднъж. " +
        "Направете го само за една конфигурация.",
    );
    if (!ok) return;
    setPromoteBusy(true);
    setPromoteMsg(null);
    try {
      const res = await api.createRun({
        hidden_size: c.hidden_size,
        num_layers: c.num_layers,
        lookback: c.lookback,
        horizon: c.horizon,
        batch: c.batch,
        lr: c.lr,
        stride: 1,
        use_amp: c.use_amp,
        use_anomaly: c.use_anomaly,
        is_final: true,
        seed: c.seed,
      });
      setPromoteMsg(
        `Финален пуск ${res.run_id} е добавен в опашката. Той се обучава върху обучение+валидация и оценява теста веднъж — ` +
          "следете раздела Обучение, после тестовите му метрики/умение се появяват тук.",
      );
    } catch (e) {
      setPromoteMsg(`Грешка: ${String(e)}`);
    } finally {
      setPromoteBusy(false);
    }
  }

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
          name: runLabel(r.config),
          run_id: r.run_id,
          rmse: r.best_val_rmse_C as number,
          mae: r.val_metrics?.lstm.mae_C ?? null,
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

  // Train vs val loss per epoch. Two series per run: "<id>__train" and "<id>__val".
  const lossRuns = useMemo(
    () => compared.filter((r) => r.history.some((h) => h.train_loss !== null)),
    [compared],
  );
  const lossCurveData = useMemo(() => {
    const maxEpoch = Math.max(0, ...lossRuns.map((r) => r.history.length));
    const rows: Record<string, number | null>[] = [];
    for (let e = 1; e <= maxEpoch; e++) {
      const row: Record<string, number | null> = { epoch: e };
      lossRuns.forEach((r) => {
        const h = r.history.find((x) => x.epoch === e);
        row[`${r.run_id}__train`] = h ? h.train_loss : null;
        row[`${r.run_id}__val`] = h ? h.val_loss : null;
      });
      rows.push(row);
    }
    return rows;
  }, [lossRuns]);

  // Generalization gap = (final val_loss) - (final train_loss) at the last epoch where both exist.
  const gapData = useMemo(
    () =>
      lossRuns
        .map((r) => {
          const withBoth = r.history.filter(
            (h) => h.train_loss !== null && h.val_loss !== null,
          );
          if (withBoth.length === 0) return null;
          const last = withBoth[withBoth.length - 1];
          return {
            run_id: r.run_id,
            name: runLabel(r.config),
            gap: (last.val_loss as number) - (last.train_loss as number),
          };
        })
        .filter((x): x is { run_id: string; name: string; gap: number } => x !== null),
    [lossRuns],
  );

  const finalRuns = useMemo(
    () => compared.filter((r) => r.test_horizon && r.test_horizon.hours.length > 1),
    [compared],
  );

  // Test RMSE vs horizon, LSTM only, one line per final run.
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

  const maeRuns = useMemo(
    () => finalRuns.filter((r) => r.test_horizon!.mae?.lstm),
    [finalRuns],
  );
  const maeHorizonData = useMemo(() => {
    if (maeRuns.length === 0) return [];
    const hours = maeRuns[0].test_horizon!.hours;
    return hours.map((hr, i) => {
      const row: Record<string, number | null> = { hour: hr };
      maeRuns.forEach((r) => {
        row[r.run_id] = r.test_horizon!.mae!.lstm?.[i] ?? null;
      });
      return row;
    });
  }, [maeRuns]);

  // LSTM vs baselines: RMSE vs horizon for the first final run.
  const baselineHorizonRun = finalRuns[0] ?? null;
  const baselineHorizonData = useMemo(() => {
    if (!baselineHorizonRun) return [];
    const h = baselineHorizonRun.test_horizon!;
    return h.hours.map((hr, i) => ({
      hour: hr,
      lstm: h.lstm?.[i] ?? null,
      persistence: h.persistence?.[i] ?? null,
      diurnal: h.diurnal?.[i] ?? null,
      climatology: h.climatology?.[i] ?? null,
    }));
  }, [baselineHorizonRun]);

  // Aggregate test RMSE per model (LSTM + baselines), grouped per final run.
  const baselineBarRuns = useMemo(() => compared.filter((r) => r.test_metrics), [compared]);
  const baselineBarData = useMemo(
    () =>
      baselineBarRuns.map((r) => ({
        name: runLabel(r.config),
        run_id: r.run_id,
        lstm: r.test_metrics?.lstm?.rmse_C ?? null,
        persistence: r.test_metrics?.persistence?.rmse_C ?? null,
        diurnal: r.test_metrics?.diurnal?.rmse_C ?? null,
        climatology: r.test_metrics?.climatology?.rmse_C ?? null,
      })),
    [baselineBarRuns],
  );

  // Skill vs baselines (% improvement), grouped per run.
  const skillRuns = useMemo(() => compared.filter((r) => r.skill_vs), [compared]);
  const skillData = useMemo(
    () =>
      skillRuns.map((r) => ({
        name: runLabel(r.config),
        run_id: r.run_id,
        persistence: r.skill_vs?.persistence ?? null,
        diurnal: r.skill_vs?.diurnal ?? null,
        climatology: r.skill_vs?.climatology ?? null,
      })),
    [skillRuns],
  );

  // Quality metric bars: bias, R2, MAPE.
  const biasData = useMemo(
    () =>
      compared
        .map((r) => {
          const m = primaryLstm(r);
          return m ? { run_id: r.run_id, name: runLabel(r.config), bias: m.bias_C } : null;
        })
        .filter((x): x is { run_id: string; name: string; bias: number } => x !== null),
    [compared],
  );
  const r2Data = useMemo(
    () =>
      compared
        .map((r) => {
          const m = primaryLstm(r);
          return m ? { run_id: r.run_id, name: runLabel(r.config), r2: m.r2 } : null;
        })
        .filter((x): x is { run_id: string; name: string; r2: number } => x !== null),
    [compared],
  );
  const mapeData = useMemo(
    () =>
      compared
        .map((r) => {
          const mape = r.test_metrics?.lstm?.mape_pct;
          return mape !== undefined && mape !== null
            ? { run_id: r.run_id, name: runLabel(r.config), mape }
            : null;
        })
        .filter((x): x is { run_id: string; name: string; mape: number } => x !== null),
    [compared],
  );

  // Accuracy vs cost (Pareto) scatter data.
  const timeScatter = useMemo(
    () =>
      compared
        .map((r, i) =>
          r.training && r.best_val_rmse_C !== null
            ? {
                run_id: r.run_id,
                name: runLabel(r.config),
                x: r.training.train_time_seconds,
                y: r.best_val_rmse_C,
                color: colorFor(i),
              }
            : null,
        )
        .filter(
          (x): x is { run_id: string; name: string; x: number; y: number; color: string } =>
            x !== null,
        ),
    [compared],
  );
  const epochScatter = useMemo(
    () =>
      compared
        .map((r, i) => {
          const epochs = r.training?.early_stop_epoch ?? r.training?.epochs_run ?? null;
          return epochs !== null && r.best_val_rmse_C !== null
            ? {
                run_id: r.run_id,
                name: runLabel(r.config),
                x: epochs,
                y: r.best_val_rmse_C,
                color: colorFor(i),
              }
            : null;
        })
        .filter(
          (x): x is { run_id: string; name: string; x: number; y: number; color: string } =>
            x !== null,
        ),
    [compared],
  );

  // Hyperparameter sensitivity: best val RMSE vs each swept hyperparameter.
  const sensitivity = useMemo(() => {
    const pts = compared
      .filter((r) => r.best_val_rmse_C !== null)
      .map((r, i) => ({
        run_id: r.run_id,
        name: runLabel(r.config),
        rmse: r.best_val_rmse_C as number,
        hidden_size: r.config.hidden_size,
        num_layers: r.config.num_layers,
        lookback: r.config.lookback,
        lr: r.config.lr,
        color: colorFor(i),
      }));
    return pts;
  }, [compared]);

  // Zoom the Y axis to the actual RMSE range so small differences (and the lowest
  // bar) are visibly distinguishable instead of all sitting near the top.
  const sensYDomain = useMemo<[number, number]>(() => {
    const vals = sensitivity.map((p) => p.rmse);
    if (vals.length === 0) return [0, 1];
    const min = Math.min(...vals);
    const max = Math.max(...vals);
    const pad = Math.max((max - min) * 0.25, 0.02);
    const lo = Math.max(0, +(min - pad).toFixed(3));
    const hi = +(max + pad).toFixed(3);
    return [lo, hi];
  }, [sensitivity]);

  const SENSITIVITY_PARAMS: { key: "lookback" | "num_layers" | "hidden_size" | "lr"; label: string }[] = [
    { key: "lookback", label: "прозорец назад (часове)" },
    { key: "num_layers", label: "брой слоеве" },
    { key: "hidden_size", label: "скрит размер" },
    { key: "lr", label: "скорост на обучение" },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Сравнение</h1>
        <p className="text-sm text-slate-400">
          Артефактът за защитата: изберете пусковете от сериите и сравнете капацитет спрямо
          преобучение по валидационни метрики. Тестовите числа, базовите модели и умението се
          появяват едва когато съществува финален модел.
        </p>
      </div>

      <DataSplitBanner />

      <ModelSelectionPanel
        leaderboard={leaderboard}
        margin={margin}
        selKind={selKind}
        setSelKind={setSelKind}
        candidateId={candidateId}
        setCandidateId={setCandidateId}
        candidateCard={candidateCard}
        candidateErr={candidateErr}
        health={health}
        onCompareTop={compareTop}
        seedCount={seedCount}
        setSeedCount={setSeedCount}
        seedBusy={seedBusy}
        seedMsg={seedMsg}
        onQueueSeeds={queueSeedRepeats}
        promoteBusy={promoteBusy}
        promoteMsg={promoteMsg}
        onPromote={promoteCandidate}
      />

      <div className="card" id="manual-compare">
        <h2 className="mb-3 font-semibold">Изберете пускове ({selected.size})</h2>
        {runs.length === 0 ? (
          <p className="text-slate-500">Все още няма завършени пускове.</p>
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
                  {r.is_final && <span className="ml-2 text-fuchsia-300">финален</span>}
                </span>
              </label>
            ))}
          </div>
        )}
        <button
          className="btn-primary mt-4"
          onClick={compare}
          disabled={selected.size < 2}
          title={selected.size < 2 ? "Изберете поне 2 пуска" : undefined}
        >
          Сравни {selected.size} пуска
        </button>
        {error && <p className="mt-3 text-sm text-rose-300">{error}</p>}
      </div>

      {compared.length > 0 && (
        <>
          <SectionHeader title="Капацитет и сходимост" />
          <div className="grid gap-6 lg:grid-cols-2">
            <div className="card">
              <h2 className="mb-3 font-semibold">Най-добро вал. RMSE по пуск</h2>
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={barData} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
                    <CartesianGrid stroke="#1e293b" />
                    <XAxis dataKey="name" stroke="#64748b" fontSize={11} />
                    <YAxis stroke="#64748b" fontSize={11} unit="°" />
                    <Tooltip cursor={{ fill: "#1e293b55" }} content={<RmseMaeTooltip />} />
                    <Bar dataKey="rmse" name="най-добро вал. RMSE °C">
                      {barData.map((_, i) => (
                        <Cell key={i} fill={colorFor(i)} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="card">
              <h2 className="mb-3 font-semibold">Вал. RMSE по епоха (наслоено)</h2>
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
                        name={runLabel(r.config)}
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

          {lossRuns.length > 0 && (
            <SectionHeader title="Преобучение и обобщаване" />
          )}
          {lossRuns.length > 0 && (
            <div className="grid gap-6 lg:grid-cols-2">
              <div className="card">
                <h2 className="mb-3 font-semibold">Загуба обучение спрямо валидация по епоха</h2>
                <p className="mb-2 text-xs text-slate-500">
                  Плътна линия = загуба при обучение, прекъсната = загуба при валидация (същите цветове
                  на пусковете като в графиката горе). Посочете за имена на пусковете. Разширяваща се
                  разлика подсказва преобучение.
                </p>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={lossCurveData}
                      margin={{ top: 8, right: 16, bottom: 8, left: 0 }}
                    >
                      <CartesianGrid stroke="#1e293b" />
                      <XAxis dataKey="epoch" stroke="#64748b" fontSize={11} />
                      <YAxis stroke="#64748b" fontSize={11} domain={["auto", "auto"]} />
                      <Tooltip
                        contentStyle={{ background: "#0f172a", border: "1px solid #334155" }}
                      />
                      {lossRuns.map((r, i) => [
                        <Line
                          key={`${r.run_id}__train`}
                          type="monotone"
                          dataKey={`${r.run_id}__train`}
                          name={`${runLabel(r.config)} · обучение`}
                          stroke={colorFor(i)}
                          dot={false}
                          connectNulls
                        />,
                        <Line
                          key={`${r.run_id}__val`}
                          type="monotone"
                          dataKey={`${r.run_id}__val`}
                          name={`${runLabel(r.config)} · валидация`}
                          stroke={colorFor(i)}
                          strokeDasharray="4 3"
                          dot={false}
                          connectNulls
                        />,
                      ])}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="card">
                <h2 className="mb-3 font-semibold">Разлика в обобщаването (вал. − обуч. загуба)</h2>
                <p className="mb-2 text-xs text-slate-500">
                  Разлика на последната епоха. По-високи стълбове означават, че моделът пасва на
                  данните за обучение далеч по-добре, отколкото на валидацията.
                </p>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={gapData} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
                      <CartesianGrid stroke="#1e293b" />
                      <XAxis dataKey="name" stroke="#64748b" fontSize={11} />
                      <YAxis stroke="#64748b" fontSize={11} />
                      <Tooltip
                        cursor={{ fill: "#1e293b55" }}
                        contentStyle={{ background: "#0f172a", border: "1px solid #334155" }}
                      />
                      <Bar dataKey="gap" name="вал. − обуч. загуба">
                        {gapData.map((_, i) => (
                          <Cell key={i} fill={colorFor(i)} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          )}

          {(baselineBarData.length > 0 || baselineHorizonData.length > 0) && (
            <SectionHeader title="LSTM спрямо базови модели" />
          )}
          <div className="grid gap-6 lg:grid-cols-2">
            {baselineBarData.length > 0 && (
              <div className="card">
                <h2 className="mb-3 font-semibold">Тест RMSE: LSTM спрямо базови модели</h2>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={baselineBarData}
                      margin={{ top: 8, right: 16, bottom: 8, left: 0 }}
                    >
                      <CartesianGrid stroke="#1e293b" />
                      <XAxis dataKey="name" stroke="#64748b" fontSize={11} />
                      <YAxis stroke="#64748b" fontSize={11} unit="°" />
                      <Tooltip
                        cursor={{ fill: "#1e293b55" }}
                        contentStyle={{ background: "#0f172a", border: "1px solid #334155" }}
                      />
                      <Legend />
                      <Bar dataKey="lstm" name="LSTM" fill={BASELINE_COLORS.lstm} />
                      <Bar
                        dataKey="persistence"
                        name="персистентност"
                        fill={BASELINE_COLORS.persistence}
                      />
                      <Bar dataKey="diurnal" name="денонощен" fill={BASELINE_COLORS.diurnal} />
                      <Bar
                        dataKey="climatology"
                        name="климатология"
                        fill={BASELINE_COLORS.climatology}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {baselineHorizonData.length > 0 && (
              <div className="card">
                <h2 className="mb-3 font-semibold">
                  RMSE спрямо хоризонт: LSTM спрямо базови модели
                </h2>
                <p className="mb-2 text-xs text-slate-500">
                  {baselineHorizonRun ? runLabel(baselineHorizonRun.config) : ""}
                </p>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={baselineHorizonData}
                      margin={{ top: 8, right: 16, bottom: 8, left: 0 }}
                    >
                      <CartesianGrid stroke="#1e293b" />
                      <XAxis dataKey="hour" stroke="#64748b" fontSize={11} unit="h" />
                      <YAxis stroke="#64748b" fontSize={11} unit="°" />
                      <Tooltip
                        contentStyle={{ background: "#0f172a", border: "1px solid #334155" }}
                      />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="lstm"
                        name="LSTM"
                        stroke={BASELINE_COLORS.lstm}
                        strokeWidth={2}
                        dot={false}
                      />
                      {BASELINES.map((b) => (
                        <Line
                          key={b}
                          type="monotone"
                          dataKey={b}
                          name={BASELINE_LABELS[b]}
                          stroke={BASELINE_COLORS[b]}
                          dot={false}
                        />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}
          </div>

          {horizonData.length > 0 && (
            <SectionHeader title="Грешка спрямо прогнозен хоризонт" />
          )}
          {horizonData.length > 0 && (
            <div className="card">
              <h2 className="mb-3 font-semibold">Тест RMSE спрямо час на хоризонта (финални модели)</h2>
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
                        name={runLabel(r.config)}
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

          {maeHorizonData.length > 0 && (
            <div className="card">
              <h2 className="mb-3 font-semibold">Тест MAE спрямо час на хоризонта (финални модели)</h2>
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={maeHorizonData}
                    margin={{ top: 8, right: 16, bottom: 8, left: 0 }}
                  >
                    <CartesianGrid stroke="#1e293b" />
                    <XAxis dataKey="hour" stroke="#64748b" fontSize={11} unit="h" />
                    <YAxis stroke="#64748b" fontSize={11} unit="°" />
                    <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155" }} />
                    <Legend />
                    {maeRuns.map((r, i) => (
                      <Line
                        key={r.run_id}
                        type="monotone"
                        dataKey={r.run_id}
                        name={runLabel(r.config)}
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

          {(skillData.length > 0 || biasData.length > 0 || r2Data.length > 0 || mapeData.length > 0) && (
            <SectionHeader title="Умение и метрики за качество" />
          )}
          <div className="grid gap-6 lg:grid-cols-2">
            {skillData.length > 0 && (
              <div className="card">
                <h2 className="mb-3 font-semibold">Умение спрямо базови модели (% намаление на RMSE)</h2>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={skillData} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
                      <CartesianGrid stroke="#1e293b" />
                      <XAxis dataKey="name" stroke="#64748b" fontSize={11} />
                      <YAxis stroke="#64748b" fontSize={11} unit="%" />
                      <Tooltip
                        cursor={{ fill: "#1e293b55" }}
                        contentStyle={{ background: "#0f172a", border: "1px solid #334155" }}
                      />
                      <Legend />
                      <Bar
                        dataKey="persistence"
                        name="спрямо персистентност"
                        fill={BASELINE_COLORS.persistence}
                      />
                      <Bar dataKey="diurnal" name="спрямо денонощен" fill={BASELINE_COLORS.diurnal} />
                      <Bar
                        dataKey="climatology"
                        name="спрямо климатология"
                        fill={BASELINE_COLORS.climatology}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {r2Data.length > 0 && (
              <div className="card">
                <h2 className="mb-3 font-semibold">R² (коефициент на детерминация)</h2>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={r2Data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
                      <CartesianGrid stroke="#1e293b" />
                      <XAxis dataKey="name" stroke="#64748b" fontSize={11} />
                      <YAxis stroke="#64748b" fontSize={11} domain={[0, 1]} />
                      <Tooltip
                        cursor={{ fill: "#1e293b55" }}
                        contentStyle={{ background: "#0f172a", border: "1px solid #334155" }}
                      />
                      <Bar dataKey="r2" name="R²">
                        {r2Data.map((_, i) => (
                          <Cell key={i} fill={colorFor(i)} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {biasData.length > 0 && (
              <div className="card">
                <h2 className="mb-3 font-semibold">Отклонение (°C, средна грешка)</h2>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={biasData} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
                      <CartesianGrid stroke="#1e293b" />
                      <XAxis dataKey="name" stroke="#64748b" fontSize={11} />
                      <YAxis stroke="#64748b" fontSize={11} unit="°" />
                      <Tooltip
                        cursor={{ fill: "#1e293b55" }}
                        contentStyle={{ background: "#0f172a", border: "1px solid #334155" }}
                      />
                      <Bar dataKey="bias" name="отклонение °C">
                        {biasData.map((_, i) => (
                          <Cell key={i} fill={colorFor(i)} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {mapeData.length > 0 && (
              <div className="card">
                <h2 className="mb-3 font-semibold">MAPE (% грешка, финални модели)</h2>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={mapeData} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
                      <CartesianGrid stroke="#1e293b" />
                      <XAxis dataKey="name" stroke="#64748b" fontSize={11} />
                      <YAxis stroke="#64748b" fontSize={11} unit="%" />
                      <Tooltip
                        cursor={{ fill: "#1e293b55" }}
                        contentStyle={{ background: "#0f172a", border: "1px solid #334155" }}
                      />
                      <Bar dataKey="mape" name="MAPE %">
                        {mapeData.map((_, i) => (
                          <Cell key={i} fill={colorFor(i)} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}
          </div>

          {(timeScatter.length > 0 || epochScatter.length > 0) && (
            <SectionHeader title="Точност спрямо цена (Парето)" />
          )}
          <div className="grid gap-6 lg:grid-cols-2">
            {timeScatter.length > 0 && (
              <div className="card">
                <h2 className="mb-3 font-semibold">Време за обучение спрямо най-добро вал. RMSE</h2>
                <p className="mb-2 text-xs text-slate-500">
                  Долу-вляво е най-добре: бързо за обучение и точно.
                </p>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 8, right: 16, bottom: 16, left: 0 }}>
                      <CartesianGrid stroke="#1e293b" />
                      <XAxis
                        type="number"
                        dataKey="x"
                        name="време за обучение"
                        stroke="#64748b"
                        fontSize={11}
                        unit="s"
                      />
                      <YAxis
                        type="number"
                        dataKey="y"
                        name="най-добро вал. RMSE"
                        stroke="#64748b"
                        fontSize={11}
                        unit="°"
                      />
                      <ZAxis range={[80, 80]} />
                      <Tooltip cursor={{ strokeDasharray: "3 3" }} content={<ScatterTooltip />} />
                      <Scatter data={timeScatter}>
                        {timeScatter.map((p) => (
                          <Cell key={p.run_id} fill={p.color} />
                        ))}
                      </Scatter>
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {epochScatter.length > 0 && (
              <div className="card">
                <h2 className="mb-3 font-semibold">Епохи до сходимост спрямо най-добро вал. RMSE</h2>
                <p className="mb-2 text-xs text-slate-500">
                  Епоха на ранно спиране (или изпълнени епохи) спрямо точността.
                </p>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 8, right: 16, bottom: 16, left: 0 }}>
                      <CartesianGrid stroke="#1e293b" />
                      <XAxis
                        type="number"
                        dataKey="x"
                        name="епохи"
                        stroke="#64748b"
                        fontSize={11}
                      />
                      <YAxis
                        type="number"
                        dataKey="y"
                        name="най-добро вал. RMSE"
                        stroke="#64748b"
                        fontSize={11}
                        unit="°"
                      />
                      <ZAxis range={[80, 80]} />
                      <Tooltip cursor={{ strokeDasharray: "3 3" }} content={<ScatterTooltip />} />
                      <Scatter data={epochScatter}>
                        {epochScatter.map((p) => (
                          <Cell key={p.run_id} fill={p.color} />
                        ))}
                      </Scatter>
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}
          </div>

          {sensitivity.length > 0 && (
            <SectionHeader title="Чувствителност към хиперпараметри" />
          )}
          {sensitivity.length > 0 && (
            <p className="text-xs text-slate-500">
              Стълбовете са подредени по стойността от серията. Оста Y е мащабирана към обхвата на
              RMSE, така че най-ниският стълб е най-малката (най-добра) грешка.
            </p>
          )}
          {sensitivity.length > 0 && (
            <div className="grid gap-6 lg:grid-cols-2">
              {SENSITIVITY_PARAMS.map(({ key, label }) => {
                const data = [...sensitivity]
                  .sort((a, b) => (a[key] as number) - (b[key] as number))
                  .map((p) => ({
                    run_id: p.run_id,
                    x: String(p[key]),
                    name: p.name,
                    rmse: p.rmse,
                    color: p.color,
                  }));
                return (
                  <div className="card" key={key}>
                    <h2 className="mb-3 font-semibold">Най-добро вал. RMSE спрямо {label}</h2>
                    <div className="h-64">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
                          <CartesianGrid stroke="#1e293b" />
                          <XAxis dataKey="x" stroke="#64748b" fontSize={11} />
                          <YAxis
                            stroke="#64748b"
                            fontSize={11}
                            unit="°"
                            domain={sensYDomain}
                            allowDecimals
                          />
                          <Tooltip
                            cursor={{ fill: "#1e293b55" }}
                            content={<SensitivityTooltip paramLabel={label} />}
                          />
                          <Bar dataKey="rmse" name="най-добро вал. RMSE °C">
                            {data.map((p) => (
                              <Cell key={p.run_id} fill={p.color} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          <SectionHeader title="Пълна таблица с метрики" />
          <div className="card overflow-x-auto">
            <h2 className="mb-3 font-semibold">Таблица за сравнение</h2>
            <table className="w-full text-left text-sm">
              <thead className="text-xs uppercase tracking-wide text-slate-400">
                <tr className="border-b border-slate-800">
                  <th className="py-2 pr-4">Пуск (време)</th>
                  <th className="py-2 pr-4">Скрити</th>
                  <th className="py-2 pr-4">Слоеве</th>
                  <th className="py-2 pr-4">Скорост на обуч.</th>
                  <th className="py-2 pr-4">Прозорец</th>
                  <th className="py-2 pr-4">Хоризонт</th>
                  <th className="py-2 pr-4">Аномалия</th>
                  <th className="py-2 pr-4">Финален</th>
                  <th className="py-2 pr-4">Най-добро вал. RMSE</th>
                  <th className="py-2 pr-4">Вал. MAE</th>
                  <th className="py-2 pr-4">Вал. откл.</th>
                  <th className="py-2 pr-4">Вал. R²</th>
                  <th className="py-2 pr-4">Тест RMSE</th>
                  <th className="py-2 pr-4">Тест MAE</th>
                  <th className="py-2 pr-4">Тест откл.</th>
                  <th className="py-2 pr-4">Тест R²</th>
                  <th className="py-2 pr-4">Умение спрямо денонощен</th>
                </tr>
              </thead>
              <tbody>
                {compared.map((r) => (
                  <tr key={r.run_id} className="border-b border-slate-800/60">
                    <td className="py-2 pr-4 font-mono text-xs text-slate-300">{r.run_id}</td>
                    <td className="py-2 pr-4">{r.config.hidden_size}</td>
                    <td className="py-2 pr-4">{r.config.num_layers}</td>
                    <td className="py-2 pr-4">{r.config.lr}</td>
                    <td className="py-2 pr-4">{r.config.lookback}</td>
                    <td className="py-2 pr-4">{r.config.horizon}</td>
                    <td className="py-2 pr-4">
                      {r.config.use_anomaly ? (
                        <span className="rounded bg-amber-900/40 px-1.5 py-0.5 text-amber-300">
                          аном
                        </span>
                      ) : (
                        <span className="text-slate-600">—</span>
                      )}
                    </td>
                    <td className="py-2 pr-4">{r.is_final ? "да" : "—"}</td>
                    <td className="py-2 pr-4">{fmtNum(r.best_val_rmse_C)}</td>
                    <td className="py-2 pr-4">
                      {r.val_metrics?.lstm ? fmtNum(r.val_metrics.lstm.mae_C) : "—"}
                    </td>
                    <td className="py-2 pr-4">
                      {r.val_metrics?.lstm?.bias_C !== undefined
                        ? fmtNum(r.val_metrics.lstm.bias_C)
                        : "—"}
                    </td>
                    <td className="py-2 pr-4">
                      {r.val_metrics?.lstm?.r2 !== undefined ? fmtNum(r.val_metrics.lstm.r2) : "—"}
                    </td>
                    <td className="py-2 pr-4">
                      {r.test_metrics?.lstm ? fmtNum(r.test_metrics.lstm.rmse_C) : "—"}
                    </td>
                    <td className="py-2 pr-4">
                      {r.test_metrics?.lstm ? fmtNum(r.test_metrics.lstm.mae_C) : "—"}
                    </td>
                    <td className="py-2 pr-4">
                      {r.test_metrics?.lstm?.bias_C !== undefined
                        ? fmtNum(r.test_metrics.lstm.bias_C)
                        : "—"}
                    </td>
                    <td className="py-2 pr-4">
                      {r.test_metrics?.lstm?.r2 !== undefined ? fmtNum(r.test_metrics.lstm.r2) : "—"}
                    </td>
                    <td className="py-2 pr-4">
                      {r.skill_vs?.diurnal !== undefined ? fmtNum(r.skill_vs.diurnal) : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <button className="btn-secondary mt-4" onClick={() => window.print()}>
              Печат / експорт
            </button>
          </div>
        </>
      )}
    </div>
  );
}

type MarginInfo = {
  best: number;
  gap2: number;
  gap5: number;
  hairline: boolean;
  nWithin: number;
} | null;

type HealthInfo = {
  lstm: number | null;
  baselines: { name: string; rmse: number | null; beats: boolean | null; margin: number | null }[];
  beatsAll: boolean;
  bestEpoch: number | null;
  epochsRun: number | null;
  epochsPlanned: number | null;
  luckyEarly: boolean;
  noEarlyStop: boolean;
} | null;

function ModelSelectionPanel({
  leaderboard,
  margin,
  selKind,
  setSelKind,
  candidateId,
  setCandidateId,
  candidateCard,
  candidateErr,
  health,
  onCompareTop,
  seedCount,
  setSeedCount,
  seedBusy,
  seedMsg,
  onQueueSeeds,
  promoteBusy,
  promoteMsg,
  onPromote,
}: {
  leaderboard: RunSummary[];
  margin: MarginInfo;
  selKind: "all" | "anom" | "plain";
  setSelKind: (k: "all" | "anom" | "plain") => void;
  candidateId: string | null;
  setCandidateId: (id: string) => void;
  candidateCard: RunCard | null;
  candidateErr: string | null;
  health: HealthInfo;
  onCompareTop: (n: number) => void;
  seedCount: number;
  setSeedCount: (n: number) => void;
  seedBusy: boolean;
  seedMsg: string | null;
  onQueueSeeds: (n: number) => void;
  promoteBusy: boolean;
  promoteMsg: string | null;
  onPromote: () => void;
}) {
  const FILTERS: { key: "all" | "anom" | "plain"; label: string }[] = [
    { key: "all", label: "Всички пускове" },
    { key: "anom", label: "С аномалия" },
    { key: "plain", label: "Без аномалия" },
  ];

  return (
    <div className="card border-sky-900/60">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="font-semibold">Избор на модел — реален ли е победителят и здрав ли е?</h2>
          <p className="text-xs text-slate-400">
            Подрежда всеки завършен пуск по най-добро валидационно RMSE, отбелязва дали водещият
            резултат е истинска победа или е в рамките на шума от зърното, и проверява здравето на
            кандидата спрямо базовите модели и поведението му при ранно спиране. Тестовият набор
            остава недокоснат, докато не пуснете финалния модел.
          </p>
        </div>
        <div className="flex gap-1 rounded-lg border border-slate-800 bg-slate-900 p-1">
          {FILTERS.map((f) => (
            <button
              key={f.key}
              onClick={() => setSelKind(f.key)}
              className={`rounded-md px-3 py-1 text-xs ${
                selKind === f.key ? "bg-sky-600 text-white" : "text-slate-400 hover:text-slate-200"
              }`}
            >
              {f.label}
            </button>
          ))}
        </div>
      </div>

      {leaderboard.length < 2 ? (
        <p className="mt-4 text-sm text-slate-500">
          Нужни са поне 2 завършени пуска с валидационно RMSE за подреждане.
        </p>
      ) : (
        <div className="mt-4 grid gap-6 lg:grid-cols-2">
          {/* Verdict + leaderboard */}
          <div>
            {margin && (
              <div
                className={`mb-3 rounded-lg border px-4 py-3 text-sm ${
                  margin.hairline
                    ? "border-amber-700/60 bg-amber-950/40 text-amber-200"
                    : "border-emerald-700/60 bg-emerald-950/40 text-emerald-200"
                }`}
              >
                <div className="font-semibold">
                  {margin.hairline ? "На косъм — вероятно шум от зърното" : "Ясна победа"}
                </div>
                <div className="mt-1 text-xs leading-relaxed">
                  Най-добро = <span className="font-mono">{fmtNum(margin.best, 3)} °C</span>. Разлика до #2 ={" "}
                  <span className="font-mono">{fmtNum(margin.gap2, 3)} °C</span>, разлика до #5 ={" "}
                  <span className="font-mono">{fmtNum(margin.gap5, 3)} °C</span>.{" "}
                  <span className="font-mono">{margin.nWithin}</span> пуск(а) са в рамките на{" "}
                  {fmtNum(HAIRLINE_C, 2)} °C от най-добрия.{" "}
                  {margin.hairline
                    ? "Преди да обявите победител, повторете водещите няколко с различни зърна."
                    : "Водещият пуск се откроява ясно от останалите."}
                </div>
              </div>
            )}

            <div className="max-h-80 overflow-y-auto rounded-lg border border-slate-800">
              <table className="w-full text-left text-xs">
                <thead className="sticky top-0 bg-slate-900 text-[10px] uppercase tracking-wide text-slate-400">
                  <tr className="border-b border-slate-800">
                    <th className="px-2 py-2">#</th>
                    <th className="px-2 py-2">Конфигурация</th>
                    <th className="px-2 py-2 text-right">Вал. RMSE</th>
                    <th className="px-2 py-2 text-right">Δ #1</th>
                    <th className="px-2 py-2"></th>
                  </tr>
                </thead>
                <tbody>
                  {leaderboard.slice(0, 15).map((r, i) => {
                    const delta = (r.best_val_rmse_C as number) - (margin?.best ?? 0);
                    const within = delta < HAIRLINE_C;
                    return (
                      <tr
                        key={r.run_id}
                        className={`border-b border-slate-800/60 ${
                          r.run_id === candidateId ? "bg-sky-950/50" : ""
                        }`}
                      >
                        <td className="px-2 py-1.5 font-mono text-slate-500">{i + 1}</td>
                        <td className="px-2 py-1.5">
                          <div className="text-slate-200">{runLabel(r.config)}</div>
                          <div className="font-mono text-[10px] text-slate-500">{r.run_id}</div>
                        </td>
                        <td className="px-2 py-1.5 text-right font-mono text-slate-200">
                          {fmtNum(r.best_val_rmse_C, 3)}
                        </td>
                        <td
                          className={`px-2 py-1.5 text-right font-mono ${
                            i === 0 ? "text-slate-600" : within ? "text-amber-300" : "text-slate-400"
                          }`}
                        >
                          {i === 0 ? "—" : `+${fmtNum(delta, 3)}`}
                        </td>
                        <td className="px-2 py-1.5 text-right">
                          <button
                            onClick={() => setCandidateId(r.run_id)}
                            className="rounded border border-slate-700 px-2 py-0.5 text-[10px] text-slate-300 hover:border-sky-600 hover:text-sky-300"
                          >
                            огледай
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            <div className="mt-3 flex flex-wrap items-center gap-2">
              <button className="btn-secondary text-xs" onClick={() => onCompareTop(5)}>
                Зареди топ 5 в пълното сравнение
              </button>
            </div>

            {!IS_STATIC && (
              <div className="mt-3 rounded-lg border border-slate-800 bg-slate-900/40 p-3">
                <div className="mb-1 text-xs font-semibold text-slate-300">
                  Повторение за стабилност спрямо зърното
                </div>
                <p className="mb-2 text-[11px] leading-relaxed text-slate-500">
                  Поставя отново в опашката <span className="text-slate-300">топ 5</span> конфигурации с
                  нови зърна (1…N; зърно 42 вече е изпълнено). Когато приключат, конфигурацията с
                  най-ниско <em>средно</em> RMSE е истинският ви победител — а не тази с късметлийското зърно.
                </p>
                <div className="flex flex-wrap items-center gap-2">
                  <label className="text-[11px] text-slate-400">
                    зърна на конфигурация
                    <input
                      type="number"
                      min={1}
                      max={10}
                      value={seedCount}
                      onChange={(e) =>
                        setSeedCount(Math.max(1, Math.min(10, Number(e.target.value) || 1)))
                      }
                      className="ml-2 w-16 rounded border border-slate-700 bg-slate-900 px-2 py-1 text-slate-200"
                    />
                  </label>
                  <button
                    className="btn-secondary text-xs"
                    disabled={seedBusy || leaderboard.length < 1}
                    onClick={() => onQueueSeeds(5)}
                  >
                    {seedBusy
                      ? "Добавяне…"
                      : `Добави топ 5 × ${seedCount} зърн${seedCount > 1 ? "а" : "о"} (${
                          Math.min(5, leaderboard.length) * seedCount
                        } пуска)`}
                  </button>
                </div>
                {seedMsg && <p className="mt-2 text-[11px] text-sky-300">{seedMsg}</p>}
              </div>
            )}
          </div>

          {/* Candidate health */}
          <div className="rounded-lg border border-slate-800 bg-slate-900/40 p-4">
            <h3 className="mb-1 font-semibold text-slate-200">Здраве на кандидата</h3>
            {candidateErr && <p className="text-xs text-rose-300">{candidateErr}</p>}
            {!candidateCard && !candidateErr && (
              <p className="text-xs text-slate-500">Зареждане на кандидата…</p>
            )}
            {candidateCard && health && (
              <>
                <p className="mb-3 text-xs text-slate-400">
                  {runLabel(candidateCard.config)}
                  <span className="ml-1 font-mono text-slate-500">({candidateCard.run_id})</span>
                </p>

                {/* Beats baselines */}
                <div className="mb-4">
                  <div className="mb-1 flex items-center gap-2">
                    <span className="text-xs font-semibold text-slate-300">Бие базовите модели (вал.)</span>
                    <HealthPill ok={health.beatsAll} okText="всичко наред" badText="виж по-долу" />
                  </div>
                  <table className="w-full text-left text-xs">
                    <tbody>
                      <tr className="text-slate-400">
                        <td className="py-0.5">LSTM средно RMSE</td>
                        <td className="py-0.5 text-right font-mono text-sky-300">
                          {fmtNum(health.lstm, 3)} °C
                        </td>
                        <td></td>
                      </tr>
                      {health.baselines.map((b) => (
                        <tr key={b.name} className="text-slate-400">
                          <td className="py-0.5">{BASELINE_LABELS[b.name] ?? b.name}</td>
                          <td className="py-0.5 text-right font-mono">{fmtNum(b.rmse, 3)} °C</td>
                          <td className="py-0.5 pl-3 text-right">
                            {b.beats === null ? (
                              <span className="text-slate-600">—</span>
                            ) : b.beats ? (
                              <span className="text-emerald-300">
                                −{fmtNum(b.margin, 2)} ✓
                              </span>
                            ) : (
                              <span className="text-rose-300">+{fmtNum(b.margin, 2)} ✗</span>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {/* Early-stop health */}
                <div>
                  <div className="mb-1 flex items-center gap-2">
                    <span className="text-xs font-semibold text-slate-300">Сходимост</span>
                    <HealthPill
                      ok={!health.luckyEarly && !health.noEarlyStop}
                      okText="изглежда сходящ"
                      badText="изисква проверка"
                    />
                  </div>
                  <p className="text-xs text-slate-400">
                    Най-добра епоха{" "}
                    <span className="font-mono text-slate-200">{health.bestEpoch ?? "—"}</span> от{" "}
                    <span className="font-mono text-slate-200">{health.epochsRun ?? "—"}</span> изпълнени (
                    {health.epochsPlanned ?? "—"} планирани).
                  </p>
                  {health.luckyEarly && (
                    <p className="mt-1 text-xs text-amber-300">
                      ⚠ Най-добра епоха ≤ {LUCKY_EPOCH}: възможно е късметлийска ранна епоха / недообучение.
                      Увеличете търпението или епохите и проверете отново.
                    </p>
                  )}
                  {health.noEarlyStop && (
                    <p className="mt-1 text-xs text-amber-300">
                      ⚠ Достигна тавана на епохите без ранно спиране — може още да се подобрява; опитайте
                      повече епохи.
                    </p>
                  )}
                  {!health.luckyEarly && !health.noEarlyStop && (
                    <p className="mt-1 text-xs text-emerald-300">
                      ✓ Спря рано на плато — здрава сходимост.
                    </p>
                  )}
                </div>

                {/* Promote to final */}
                <div className="mt-4 border-t border-slate-800 pt-3">
                  {IS_STATIC ? (
                    <p className="text-[11px] leading-relaxed text-slate-500">
                      Това е статичен преглед на резултатите — обучение и повишаване до финален пуск
                      не са налични тук. Финалният модел вече е обучен и оценен; вижте тестовите му
                      метрики в таблицата и графиките по-долу.
                    </p>
                  ) : candidateCard.config.is_final ? (
                    <p className="text-xs text-fuchsia-300">
                      Този пуск вече е финален (обучение+валидация → тест). Няма какво да се повишава.
                    </p>
                  ) : (
                    <>
                      <button
                        className="btn-primary w-full text-xs"
                        disabled={promoteBusy || !health.beatsAll}
                        onClick={onPromote}
                        title={
                          !health.beatsAll
                            ? "Кандидатът трябва първо да бие всички базови модели"
                            : "Преобучи върху обучение+валидация и оцени теста веднъж"
                        }
                      >
                        {promoteBusy ? "Добавяне на финален пуск…" : "Повиши до финален пуск (оценка върху тест)"}
                      </button>
                      <p className="mt-1 text-[11px] text-slate-500">
                        Преобучава точно тази конфигурация върху слети обучение+валидация и оценява
                        заделения тестов набор <strong>веднъж</strong>. Направете го само за една
                        конфигурация — тестовото число е вашият докладван резултат.
                      </p>
                    </>
                  )}
                  {promoteMsg && <p className="mt-2 text-[11px] text-sky-300">{promoteMsg}</p>}
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function HealthPill({ ok, okText, badText }: { ok: boolean; okText: string; badText: string }) {
  return (
    <span
      className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${
        ok ? "bg-emerald-900/50 text-emerald-300" : "bg-amber-900/50 text-amber-300"
      }`}
    >
      {ok ? okText : badText}
    </span>
  );
}

function SectionHeader({ title }: { title: string }) {
  return (
    <div className="flex items-center gap-3 pt-2">
      <h2 className="text-lg font-semibold text-slate-200">{title}</h2>
      <div className="h-px flex-1 bg-slate-800" />
    </div>
  );
}

interface BarDatum {
  name: string;
  rmse: number;
  mae: number | null;
}

function RmseMaeTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: { payload: BarDatum }[];
}) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="rounded border border-slate-700 bg-slate-900 px-3 py-2 text-xs">
      <div className="mb-1 font-semibold text-slate-200">{d.name}</div>
      <div className="text-slate-300">най-добро вал. RMSE °C : {fmtNum(d.rmse)}</div>
      <div className="text-slate-300">вал. MAE °C : {fmtNum(d.mae)}</div>
    </div>
  );
}

interface ScatterDatum {
  name: string;
  x: number;
  y: number;
}

function ScatterTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: { payload: ScatterDatum }[];
}) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="rounded border border-slate-700 bg-slate-900 px-3 py-2 text-xs">
      <div className="mb-1 font-semibold text-slate-200">{d.name}</div>
      <div className="text-slate-300">x : {fmtNum(d.x)}</div>
      <div className="text-slate-300">най-добро вал. RMSE °C : {fmtNum(d.y)}</div>
    </div>
  );
}

interface SensitivityDatum {
  name: string;
  rmse: number;
  x: string;
}

function SensitivityTooltip({
  active,
  payload,
  paramLabel,
}: {
  active?: boolean;
  payload?: { payload: SensitivityDatum }[];
  paramLabel: string;
}) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="rounded border border-slate-700 bg-slate-900 px-3 py-2 text-xs">
      <div className="mb-1 font-semibold text-slate-200">{d.name}</div>
      <div className="text-slate-300">
        {paramLabel} : {d.x}
      </div>
      <div className="text-slate-300">най-добро вал. RMSE °C : {fmtNum(d.rmse)}</div>
    </div>
  );
}
