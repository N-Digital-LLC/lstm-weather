"use client";

import { useEffect, useState } from "react";
import { api, type DataSplit, type SplitSegment } from "@/lib/api";

const SEGMENTS = [
  {
    key: "train" as const,
    label: "Обучение",
    bar: "bg-sky-500",
    text: "text-sky-300",
    dot: "bg-sky-400",
  },
  {
    key: "val" as const,
    label: "Валидация",
    bar: "bg-amber-500",
    text: "text-amber-300",
    dot: "bg-amber-400",
  },
  {
    key: "test" as const,
    label: "Тест (заделен)",
    bar: "bg-emerald-500",
    text: "text-emerald-300",
    dot: "bg-emerald-400",
  },
];

export function DataSplitBanner() {
  const [split, setSplit] = useState<DataSplit | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .dataSplit()
      .then(setSplit)
      .catch((e) => setError(String(e)));
  }, []);

  if (error) {
    return (
      <div className="card text-sm text-rose-300">
        Разделянето на данните не можа да се зареди: {error}
      </div>
    );
  }
  if (!split) {
    return <div className="card text-sm text-slate-400">Зареждане на разделянето на данните…</div>;
  }

  const total = split.total_rows || 1;

  return (
    <div className="card">
      <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="font-semibold">Разделяне на данните — кои години къде отиват</h2>
        <span className="text-xs text-slate-500">
          {split.total_rows.toLocaleString()} почасови реда · хронологично 70 / 15 / 15
        </span>
      </div>

      <div className="flex h-3 w-full overflow-hidden rounded-full bg-slate-800">
        {SEGMENTS.map(({ key, bar }) => {
          const seg = split[key];
          const pct = (seg.rows / total) * 100;
          if (pct <= 0) return null;
          return <div key={key} className={bar} style={{ width: `${pct}%` }} />;
        })}
      </div>

      <div className="mt-4 grid gap-3 sm:grid-cols-3">
        {SEGMENTS.map(({ key, label, text, dot }) => (
          <SegmentCard
            key={key}
            label={label}
            textClass={text}
            dotClass={dot}
            seg={split[key]}
          />
        ))}
      </div>

      <p className="mt-4 text-xs text-slate-500">
        Строго хронологично разделяне (от най-ранните към най-късните данни, без разбъркване).
        Настройващите пускове се обучават върху годините за обучение и се оценяват върху
        валидацията; тестовите години остават недокоснати. Финалният пуск слива обучение и
        валидация и оценява тестовите години точно веднъж.
      </p>
    </div>
  );
}

function SegmentCard({
  label,
  textClass,
  dotClass,
  seg,
}: {
  label: string;
  textClass: string;
  dotClass: string;
  seg: SplitSegment;
}) {
  const empty = seg.rows <= 0;
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900 p-3">
      <div className="flex items-center gap-2">
        <span className={`h-2.5 w-2.5 rounded-full ${dotClass}`} />
        <span className="text-xs font-medium uppercase tracking-wide text-slate-400">
          {label}
        </span>
      </div>
      <div className={`mt-1 text-lg font-semibold ${textClass}`}>
        {empty ? "—" : seg.years}
      </div>
      <div className="text-xs text-slate-500">
        {empty ? "слято с обучението" : `${seg.rows.toLocaleString()} часа`}
      </div>
    </div>
  );
}
