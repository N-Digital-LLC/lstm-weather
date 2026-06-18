import type { Progress } from "@/lib/api";

const STATUS_STYLES: Record<string, string> = {
  queued: "bg-slate-700 text-slate-200",
  running: "bg-amber-500/20 text-amber-300",
  done: "bg-emerald-500/20 text-emerald-300",
  failed: "bg-rose-500/20 text-rose-300",
};

export function StatusBadge({ status }: { status: string }) {
  return (
    <span className={`badge ${STATUS_STYLES[status] ?? "bg-slate-700 text-slate-200"}`}>
      {status}
    </span>
  );
}

export function FinalBadge({ isFinal }: { isFinal: boolean }) {
  if (!isFinal) return <span className="text-slate-600">—</span>;
  return <span className="badge bg-fuchsia-500/20 text-fuchsia-300">final</span>;
}

export function ProgressBar({
  status,
  progress,
}: {
  status: string;
  progress: Progress | null | undefined;
}) {
  if (status === "running" && progress && progress.total_epochs > 0) {
    const pct = Math.min(100, Math.round((progress.current_epoch / progress.total_epochs) * 100));
    return (
      <div className="w-36">
        <div className="h-2 w-full overflow-hidden rounded-full bg-slate-700">
          <div className="h-full bg-amber-400 transition-all" style={{ width: `${pct}%` }} />
        </div>
        <div className="mt-1 text-[11px] text-slate-400">
          epoch {progress.current_epoch}/{progress.total_epochs}
        </div>
      </div>
    );
  }
  return <StatusBadge status={status} />;
}
