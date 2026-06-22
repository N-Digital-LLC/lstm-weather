import type { RunConfig } from "./api";

// Human-readable run label spelling out the defining hyperparameters, e.g.
// "Hidden 128 · Layers 1 · Lookback 168 · lr 0.001". Used for chart series/legends.
export function runLabel(config: RunConfig | null | undefined): string {
  if (!config) return "—";
  const parts = [
    `Скрити ${config.hidden_size}`,
    `Слоеве ${config.num_layers}`,
    `Прозорец ${config.lookback}`,
    `Хоризонт ${config.horizon}`,
    `lr ${config.lr}`,
  ];
  if (config.use_anomaly) parts.push("аномалия");
  return parts.join(" · ");
}

export function fmtNum(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return v.toFixed(digits);
}

export function fmtDateTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString("bg-BG", {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function fmtDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return "—";
  const s = Math.round(seconds);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  const rem = s % 60;
  if (m < 60) return `${m}m ${rem}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

// Deterministic palette so the same run keeps its color across charts.
const PALETTE = [
  "#38bdf8",
  "#f97316",
  "#a78bfa",
  "#34d399",
  "#f43f5e",
  "#facc15",
  "#22d3ee",
  "#fb7185",
];

export function colorFor(index: number): string {
  return PALETTE[index % PALETTE.length];
}
