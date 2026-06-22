"use client";

import Link from "next/link";
import { BASE_PATH } from "@/lib/api";

const FINAL_RUN = "2026-06-19_11-00-30__h128_l1_L168_lr0.001";

interface Figure {
  src: string;
  num: number;
  caption: string;
  /** Optional link to the interactive equivalent in the app. */
  live?: { href: string; label: string };
  /** Render narrower (diagrams look better not stretched full width). */
  narrow?: boolean;
}

const RESULTS: Figure[] = [
  {
    src: "/report/training_curve.png",
    num: 1,
    caption:
      "Крива на обучение на финалния модел (обучаваща загуба по епохи). Финалният модел се обучава " +
      "върху train+val без валидационен монитор, затова валидационните криви липсват.",
    narrow: true,
    live: { href: `/runs/${encodeURIComponent(FINAL_RUN)}`, label: "Виж интерактивно в отчета на пуска" },
  },
  {
    src: "/report/rmse_vs_horizon.png",
    num: 2,
    caption:
      "RMSE спрямо хоризонта на прогнозата (тест). LSTM (синьо) превъзхожда всеки базов модел на " +
      "всеки хоризонт. Персистентността достига връх около +12–13 h, а денонощната и климатологията " +
      "са плоски по конструкция.",
    live: { href: `/runs/${encodeURIComponent(FINAL_RUN)}`, label: "Виж интерактивно в отчета на пуска" },
  },
  {
    src: "/report/pred_vs_actual.png",
    num: 3,
    caption:
      "Предсказана (LSTM, +1 h) срещу реална температура върху тестов отрязък (юли 2013 г.). " +
      "Моделът следва плътно денонощната динамика.",
    live: { href: "/", label: "Виж интерактивно в раздел «Прогноза»" },
  },
  {
    src: "/report/fig_bias_vs_horizon.png",
    num: 4,
    caption:
      "Отклонение (bias) на LSTM спрямо хоризонта (тест). Лек систематичен студен наклон около −0.16 °C.",
    narrow: true,
    live: { href: `/runs/${encodeURIComponent(FINAL_RUN)}`, label: "Виж интерактивно в отчета на пуска" },
  },
];

function FigureCard({ fig }: { fig: Figure }) {
  return (
    <figure className="card">
      <div className="flex justify-center rounded-lg bg-white p-3">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={`${BASE_PATH}${fig.src}`}
          alt={`Фиг. ${fig.num}`}
          className={`h-auto w-full ${fig.narrow ? "max-w-2xl" : "max-w-4xl"}`}
        />
      </div>
      <figcaption className="mt-3 text-sm text-slate-400">
        <span className="font-semibold text-slate-300">Фиг. {fig.num}.</span> {fig.caption}
      </figcaption>
      {fig.live && (
        <Link
          href={fig.live.href}
          className="mt-2 inline-block text-xs text-sky-300 hover:underline"
        >
          → {fig.live.label}
        </Link>
      )}
    </figure>
  );
}

export default function ReportPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Графики от доклада</h1>
        <p className="text-sm text-slate-400">
          Графиките от финалния модел в писмения доклад за защитата, показани тук на живо. За всяка
          фигура има и интерактивна версия в приложението (връзките под нея).
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {RESULTS.map((f) => (
          <FigureCard key={f.src} fig={f} />
        ))}
      </div>
    </div>
  );
}
