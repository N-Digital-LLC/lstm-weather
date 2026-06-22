"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { api, IS_STATIC, type RunConfig, type RunSummary } from "@/lib/api";
import { FinalBadge, ProgressBar } from "@/components/RunStatus";
import { DataSplitBanner } from "@/components/DataSplit";
import { fmtDateTime, fmtNum } from "@/lib/format";

const DEFAULT_FORM = {
  hidden_size: 128,
  num_layers: 2,
  lookback: 168,
  horizon: 24,
  lr: 0.001,
  stride: 1,
  batch: 256,
  epochs: 30,
  use_anomaly: false,
};

// The compact letters used in the run table's Config column, with their full meaning.
// Single source of truth for the chips, the legend, and the form-label abbreviations.
const CONFIG_PARAMS = [
  { letter: "s", name: "Скрит размер", key: "hidden_size" },
  { letter: "l", name: "Слоеве", key: "num_layers" },
  { letter: "lr", name: "Скорост на обучение", key: "lr" },
  { letter: "Lb", name: "Прозорец", key: "lookback" },
  { letter: "H", name: "Хоризонт", key: "horizon" },
] as const;

type SortValue = number | string | boolean | null | undefined;

// Every sortable column in the Runs table, in display order. The Actions column
// is intentionally excluded since it has no meaningful sort key.
const SORT_COLUMNS: { key: string; label: string; get: (r: RunSummary) => SortValue }[] = [
  { key: "run_id", label: "Пуск", get: (r) => r.run_id },
  { key: "started_at", label: "Стартиран", get: (r) => r.started_at },
  ...CONFIG_PARAMS.map((p) => ({
    key: p.key,
    label: p.name,
    get: (r: RunSummary) => r.config?.[p.key as keyof RunConfig] as SortValue,
  })),
  { key: "use_anomaly", label: "Аномалия", get: (r) => r.config?.use_anomaly },
  { key: "is_final", label: "Финален", get: (r) => r.is_final },
  { key: "status", label: "Статус", get: (r) => r.status },
  { key: "best_val_rmse_C", label: "Най-добро вал. RMSE", get: (r) => r.best_val_rmse_C },
  { key: "val_mae_C", label: "Вал. MAE", get: (r) => r.val_mae_C },
  { key: "val_bias_C", label: "Вал. отклонение", get: (r) => r.val_bias_C },
  { key: "val_r2", label: "Вал. R²", get: (r) => r.val_r2 },
  { key: "test_rmse_C", label: "Тест RMSE", get: (r) => r.test_rmse_C },
  { key: "test_bias_C", label: "Тест отклонение", get: (r) => r.test_bias_C },
  { key: "test_r2", label: "Тест R²", get: (r) => r.test_r2 },
];

// Compares two runs by the active sort column. Null/undefined values always sort
// to the end regardless of direction so empty metrics don't crowd the top.
function compareRuns(a: RunSummary, b: RunSummary, get: (r: RunSummary) => SortValue, dir: "asc" | "desc"): number {
  const va = get(a);
  const vb = get(b);
  const aEmpty = va === null || va === undefined;
  const bEmpty = vb === null || vb === undefined;
  if (aEmpty && bEmpty) return 0;
  if (aEmpty) return 1;
  if (bEmpty) return -1;
  const factor = dir === "asc" ? 1 : -1;
  if (typeof va === "number" && typeof vb === "number") return (va - vb) * factor;
  if (typeof va === "boolean" && typeof vb === "boolean") return (Number(va) - Number(vb)) * factor;
  return String(va).localeCompare(String(vb)) * factor;
}

const DEFAULT_SWEEP = {
  hidden_sizes: "32, 64, 128",
  num_layers_list: "1, 2, 3",
  lr_list: "0.001, 0.0005",
  lookback_list: "72, 168, 336",
};

// Parse a comma/space separated list of numbers, dropping blanks and non-numerics.
function parseList(s: string): number[] {
  return s
    .split(/[,\s]+/)
    .map((x) => x.trim())
    .filter((x) => x.length > 0)
    .map(Number)
    .filter((x) => Number.isFinite(x));
}

export default function TrainingPage() {
  const [form, setForm] = useState(DEFAULT_FORM);
  const [sweepForm, setSweepForm] = useState(DEFAULT_SWEEP);
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [sort, setSort] = useState<{ key: string; dir: "asc" | "desc" } | null>(null);

  // Click a header: sort ascending, then toggle to descending on repeat clicks.
  function toggleSort(key: string) {
    setSort((s) => (s && s.key === key ? { key, dir: s.dir === "asc" ? "desc" : "asc" } : { key, dir: "asc" }));
  }

  // Default order is whatever the API returns (newest first); only re-sort when a column is active.
  const sortedRuns = useMemo(() => {
    if (!sort) return runs;
    const col = SORT_COLUMNS.find((c) => c.key === sort.key);
    if (!col) return runs;
    return [...runs].sort((a, b) => compareRuns(a, b, col.get, sort.dir));
  }, [runs, sort]);

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
    if (IS_STATIC) return; // static package: snapshot is fixed, no polling
    const id = setInterval(refresh, 3000); // live polling
    return () => clearInterval(id);
  }, []);

  function setField<K extends keyof typeof form>(key: K, value: (typeof form)[K]) {
    setForm((f) => ({ ...f, [key]: value }));
  }

  function setSweepField<K extends keyof typeof sweepForm>(key: K, value: string) {
    setSweepForm((f) => ({ ...f, [key]: value }));
  }

  // Lists -> {hidden_sizes, num_layers_list, lr_list, lookback_list}, dropping empties.
  function sweepGrids() {
    const grids: Record<string, number[]> = {
      hidden_sizes: parseList(sweepForm.hidden_sizes),
      num_layers_list: parseList(sweepForm.num_layers_list),
      lr_list: parseList(sweepForm.lr_list),
      lookback_list: parseList(sweepForm.lookback_list),
    };
    return Object.fromEntries(Object.entries(grids).filter(([, v]) => v.length > 0));
  }

  async function launch() {
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const res = await api.createRun({ ...form, is_final: false });
      setNotice(`Настройващ пуск ${res.run_id} е добавен в опашката`);
      refresh();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  async function runSweep(mode: "one_factor" | "grid") {
    const grids = sweepGrids();
    const keys = Object.keys(grids);
    if (keys.length === 0) {
      setError("Въведете поне един списък със стойности за серията.");
      return;
    }

    if (mode === "grid") {
      const total = keys.reduce((n, k) => n * grids[k].length, 1);
      const ok = window.confirm(
        `Матричната серия ще добави ${total} пуска (декартово произведение на ${keys.length} ` +
          `списък${keys.length > 1 ? "а" : ""} с параметри).\n\nТе се обучават един по един на ` +
          "един GPU — това може да отнеме време. Да продължа ли?",
      );
      if (!ok) return;
    }

    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      // hidden_size/num_layers/lookback/lr are swept, so drop the single-value versions; keep the
      // rest of the form (batch/horizon/epochs/anomaly + stride pinned to 1) as the shared config.
      const { hidden_size: _hs, num_layers: _nl, lookback: _lb, lr: _lr, ...shared } = form;
      const res = await api.createSweep({ ...shared, is_final: false, mode, ...grids });
      const label = mode === "grid" ? "Матрична серия" : "Серия по параметър";
      setNotice(`${label}: добавени ${res.run_ids.length} пуска в опашката`);
      refresh();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  async function trainFinal(run: RunSummary) {
    const ok = window.confirm(
      `Да обуча ли ФИНАЛНИЯ модел с конфигурацията на ${run.run_id}?\n\n` +
        "Това преобучава върху слети обучение+валидация и оценява ТЕСТОВИЯ набор — направете го " +
        "точно веднъж, върху избрания победител. Всички настройващи отчети остават без тест по замисъл.",
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
      setNotice(`ФИНАЛЕН пуск ${res.run_id} е добавен в опашката`);
      refresh();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  const _grids = sweepGrids();
  const _lengths = Object.values(_grids).map((v) => v.length);
  const oneFactorCount = _lengths.reduce((n, l) => n + l, 0);
  const matrixCount = _lengths.length === 0 ? 0 : _lengths.reduce((n, l) => n * l, 1);

  async function remove(run: RunSummary) {
    if (!window.confirm(`Да изтрия ли пуск ${run.run_id}?`)) return;
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
        <h1 className="text-2xl font-bold">Обучение</h1>
        <p className="text-sm text-slate-400">
          Стартирайте настройващи пускове (оценявани върху валидацията), правете серии по
          хиперпараметри поотделно или като пълна матрица, после повишете победителя до единствения
          финален модел, който докосва теста.
        </p>
      </div>

      <DataSplitBanner />

      {IS_STATIC && (
        <div className="card border-sky-800 bg-sky-950/40 text-sm text-sky-100">
          <strong>Пакет с резултати само за четене.</strong> Това е статична снимка на завършените
          експерименти — разгледайте всеки пуск, отворете отчета му и сравнявайте модели.
          Стартирането на нови обучения изисква пълния сървър и е изключено тук.
        </div>
      )}

      {!IS_STATIC && (
       <>
      <div className="card">
        <h2 className="mb-4 font-semibold">Стартиране на настройващ пуск</h2>
        <div className="grid gap-4 sm:grid-cols-3 lg:grid-cols-4">
          <Field label="Скрит размер" abbr="s">
            <input
              type="number"
              className="input"
              value={form.hidden_size}
              onChange={(e) => setField("hidden_size", Number(e.target.value))}
            />
          </Field>
          <Field label="Слоеве" abbr="l">
            <input
              type="number"
              className="input"
              value={form.num_layers}
              onChange={(e) => setField("num_layers", Number(e.target.value))}
            />
          </Field>
          <Field label="Прозорец назад" abbr="Lb">
            <input
              type="number"
              className="input"
              value={form.lookback}
              onChange={(e) => setField("lookback", Number(e.target.value))}
            />
          </Field>
          <Field label="Скорост на обучение" abbr="lr">
            <input
              type="number"
              step={0.0005}
              className="input"
              value={form.lr}
              onChange={(e) => setField("lr", Number(e.target.value))}
            />
          </Field>
          <Field label="Хоризонт" abbr="H">
            <input
              type="number"
              className="input"
              value={form.horizon}
              onChange={(e) => setField("horizon", Number(e.target.value))}
            />
          </Field>
          <Field label="Цел: аномалия">
            <label className="mt-2 inline-flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={form.use_anomaly}
                onChange={(e) => setField("use_anomaly", e.target.checked)}
              />
              прогноза на стойност − климатология
            </label>
            <p className="mt-1 text-xs text-slate-500">
              Прогнозира отклонението от сезонната/денонощната норма вместо суровата температура.
            </p>
          </Field>
        </div>

        <div className="mt-4 rounded-lg border border-slate-800 bg-slate-900/40 p-4">
          <h3 className="mb-1 text-sm font-semibold text-slate-300">Настройки на обучението</h3>
          <p className="mb-3 text-xs text-slate-500">
            Влияят на скоростта, паметта и продължителността на обучението — не и на това какво може
            да научи моделът.
          </p>
          <div className="grid gap-4 sm:grid-cols-2">
            <Field label="Партида">
              <input
                type="number"
                className="input"
                value={form.batch}
                onChange={(e) => setField("batch", Number(e.target.value))}
              />
            </Field>
            <Field label="Епохи">
              <input
                type="number"
                className="input"
                value={form.epochs}
                onChange={(e) => setField("epochs", Number(e.target.value))}
              />
            </Field>
          </div>
        </div>
        <div className="mt-4 flex flex-wrap gap-3">
          <button className="btn-primary" onClick={launch} disabled={busy}>
            Стартирай пуск
          </button>
        </div>
        {notice && <p className="mt-3 text-sm text-emerald-300">{notice}</p>}
        {error && <p className="mt-3 text-sm text-rose-300">{error}</p>}
      </div>

      <div className="card">
        <h2 className="mb-1 font-semibold">Серия по хиперпараметри</h2>
        <p className="mb-4 text-sm text-slate-400">
          Въведете стойности, разделени със запетая, за всеки параметър. <strong>По параметър</strong>{" "}
          променя по един наведнъж (останалите се вземат от формата по-горе); <strong>Матрица</strong>{" "}
          обучава всяка комбинация. Оставете списък празен, за да остане параметърът постоянен.
          Партидата остава фиксирана (лост за скорост, не за точност).
        </p>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <Field label="Скрити размери" abbr="s">
            <input
              type="text"
              className="input"
              value={sweepForm.hidden_sizes}
              onChange={(e) => setSweepField("hidden_sizes", e.target.value)}
            />
          </Field>
          <Field label="Слоеве" abbr="l">
            <input
              type="text"
              className="input"
              value={sweepForm.num_layers_list}
              onChange={(e) => setSweepField("num_layers_list", e.target.value)}
            />
          </Field>
          <Field label="Скорости на обучение" abbr="lr">
            <input
              type="text"
              className="input"
              value={sweepForm.lr_list}
              onChange={(e) => setSweepField("lr_list", e.target.value)}
            />
          </Field>
          <Field label="Прозорци назад" abbr="Lb">
            <input
              type="text"
              className="input"
              value={sweepForm.lookback_list}
              onChange={(e) => setSweepField("lookback_list", e.target.value)}
            />
          </Field>
        </div>
        <div className="mt-4 rounded-lg border border-slate-800 bg-slate-900/40 p-3">
          <label className="inline-flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={form.use_anomaly}
              onChange={(e) => setField("use_anomaly", e.target.checked)}
            />
            <span className="font-medium text-slate-200">Цел: аномалия</span>
            <span className="text-slate-400">— прогноза на стойност − климатология за всеки пуск от серията</span>
          </label>
          <p className="mt-1 text-xs text-slate-500">
            Споделя се с формата по-горе; прилага се за всички пускове по параметър и матрични пускове.
          </p>
        </div>
        <div className="mt-4 flex flex-wrap items-center gap-3">
          <button
            className="btn-secondary"
            onClick={() => runSweep("one_factor")}
            disabled={busy}
          >
            Серия по всеки параметър ({oneFactorCount} пуска){form.use_anomaly ? " · аномалия" : ""}
          </button>
          <button className="btn-secondary" onClick={() => runSweep("grid")} disabled={busy}>
            Матрична серия ({matrixCount} пуска){form.use_anomaly ? " · аномалия" : ""}
          </button>
        </div>
      </div>
       </>
      )}

      <div className="card overflow-x-auto">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <h2 className="font-semibold">Пускове</h2>
        </div>
        <table className="w-full text-left text-sm">
          <thead className="text-xs uppercase tracking-wide text-slate-400">
            <tr className="border-b border-slate-800">
              {SORT_COLUMNS.map((c) => {
                const active = sort?.key === c.key;
                return (
                  <th key={c.key} className="py-2 pr-4">
                    <button
                      type="button"
                      onClick={() => toggleSort(c.key)}
                      className="inline-flex items-center gap-1 uppercase tracking-wide hover:text-slate-200"
                    >
                      {c.label}
                      <span className={`text-[10px] ${active ? "text-sky-300" : "text-slate-600"}`}>
                        {active ? (sort?.dir === "asc" ? "▲" : "▼") : "↕"}
                      </span>
                    </button>
                  </th>
                );
              })}
              <th className="py-2 pr-4">Действия</th>
            </tr>
          </thead>
          <tbody>
            {sortedRuns.length === 0 && (
              <tr>
                <td colSpan={SORT_COLUMNS.length + 1} className="py-6 text-center text-slate-500">
                  Все още няма пускове — стартирайте по-горе.
                </td>
              </tr>
            )}
            {sortedRuns.map((r) => (
              <tr key={r.run_id} className="border-b border-slate-800/60 hover:bg-slate-800/30">
                <td className="py-2 pr-4">
                  <Link
                    className="block max-w-[160px] truncate text-sky-300 hover:underline"
                    href={`/runs/${encodeURIComponent(r.run_id)}`}
                    title={r.run_id}
                  >
                    {r.run_id}
                  </Link>
                </td>
                <td className="py-2 pr-4 text-slate-400">{fmtDateTime(r.started_at)}</td>
                {CONFIG_PARAMS.map((p) => (
                  <td key={p.key} className="py-2 pr-4 text-slate-200">
                    {r.config ? (r.config[p.key as keyof RunConfig] as number) : "—"}
                  </td>
                ))}
                <td className="py-2 pr-4">
                  {r.config?.use_anomaly ? (
                    <span className="rounded bg-amber-900/40 px-1.5 py-0.5 text-amber-300">аном</span>
                  ) : (
                    <span className="text-slate-600">—</span>
                  )}
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
                <td className="py-2 pr-4">{fmtNum(r.val_mae_C)}</td>
                <td className="py-2 pr-4">{fmtNum(r.val_bias_C)}</td>
                <td className="py-2 pr-4">{fmtNum(r.val_r2)}</td>
                <td className="py-2 pr-4">
                  {r.is_final ? fmtNum(r.test_rmse_C) : <span className="text-slate-600">—</span>}
                </td>
                <td className="py-2 pr-4">
                  {r.is_final ? fmtNum(r.test_bias_C) : <span className="text-slate-600">—</span>}
                </td>
                <td className="py-2 pr-4">
                  {r.is_final ? fmtNum(r.test_r2) : <span className="text-slate-600">—</span>}
                </td>
                <td className="py-2 pr-4">
                  {IS_STATIC ? (
                    <Link
                      className="btn-secondary px-2 py-1 text-xs"
                      href={`/runs/${encodeURIComponent(r.run_id)}`}
                    >
                      Преглед
                    </Link>
                  ) : (
                    <div className="flex gap-2">
                      {r.status === "done" && !r.is_final && (
                        <button
                          className="btn-secondary px-2 py-1 text-xs"
                          onClick={() => trainFinal(r)}
                          disabled={busy}
                        >
                          Финализирай
                        </button>
                      )}
                      <button className="btn-danger px-2 py-1 text-xs" onClick={() => remove(r)}>
                        Изтрий
                      </button>
                    </div>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Field({
  label,
  abbr,
  children,
}: {
  label: string;
  abbr?: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <label className="label">
        {label}
        {abbr && <span className="ml-1 font-mono text-sky-300">({abbr})</span>}
      </label>
      {children}
    </div>
  );
}

