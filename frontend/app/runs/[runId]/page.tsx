import fs from "node:fs";
import path from "node:path";
import RunDetailClient from "./RunDetailClient";

// The static export can only serve run pages it pre-rendered, so lock params to
// the snapshot set. (Ignored by `next dev`, which renders run pages on demand.)
export const dynamicParams = false;

export function generateStaticParams() {
  try {
    const p = path.join(process.cwd(), "public", "snapshots", "runs.json");
    const data = JSON.parse(fs.readFileSync(p, "utf-8")) as { runs?: { run_id: string }[] };
    return (data.runs ?? []).map((r) => ({ runId: r.run_id }));
  } catch {
    return [];
  }
}

export default async function RunDetailPage({
  params,
}: {
  params: Promise<{ runId: string }>;
}) {
  const { runId } = await params;
  return <RunDetailClient runId={decodeURIComponent(runId)} />;
}
