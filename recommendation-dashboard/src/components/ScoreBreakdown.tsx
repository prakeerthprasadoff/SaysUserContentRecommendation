import type { PoolStats } from "../types";

const CLAMP = 3; // z-scores beyond +/-3 render at full bar length

const ROWS: { key: keyof PoolStats; zKey: "z_clap" | "z_yamnet" | "z_transcript"; label: string }[] = [
  { key: "clap_sim", zKey: "z_clap", label: "CLAP" },
  { key: "yamnet_overlap", zKey: "z_yamnet", label: "YAMNet" },
  { key: "transcript_sim", zKey: "z_transcript", label: "Transcript" },
];

interface ScoreBreakdownProps {
  clap_sim: number;
  yamnet_overlap: number | null;
  transcript_sim: number | null;
  z_clap: number;
  z_yamnet: number | null;
  z_transcript: number | null;
  score: number;
  poolStats: PoolStats;
}

export function ScoreBreakdown({
  clap_sim,
  yamnet_overlap,
  transcript_sim,
  z_clap,
  z_yamnet,
  z_transcript,
  score,
  poolStats,
}: ScoreBreakdownProps) {
  const raw: Record<string, number | null> = { clap_sim, yamnet_overlap, transcript_sim };
  const z: Record<string, number | null> = { z_clap, z_yamnet, z_transcript };

  return (
    <div className="flex flex-col gap-2">
      {ROWS.map(({ key, zKey, label }) => {
        const v = z[zKey];
        const r = raw[key];
        const mean = poolStats[key]?.mean ?? null;
        const pct = v === null ? 0 : Math.min(Math.abs(v), CLAMP) / CLAMP / 2;
        const isPos = (v ?? 0) >= 0;

        return (
          <div key={key} className="flex flex-col gap-0.5">
            <div className="flex items-center gap-2">
              <span className="w-16 shrink-0 text-[11px] text-[var(--ink-muted)]">{label}</span>
              <div className="relative h-1.5 flex-1 rounded-full bg-[var(--surface-2)]">
                <div className="absolute inset-y-0 left-1/2 w-px bg-[var(--line)]" />
                {v !== null && (
                  <div
                    className="absolute inset-y-0 rounded-full"
                    style={{
                      left: isPos ? "50%" : `${50 - pct * 100}%`,
                      width: `${pct * 100}%`,
                      backgroundColor: isPos ? "var(--pos)" : "var(--neg)",
                    }}
                  />
                )}
              </div>
              <span className="w-11 shrink-0 text-right font-mono text-[11px] tabular-nums text-[var(--ink-muted)]">
                {v === null ? "n/a" : `z ${v.toFixed(1)}`}
              </span>
            </div>
            {r !== null && (
              <span className="pl-[4.5rem] font-mono text-[10px] tabular-nums text-[var(--ink-muted)]">
                cosine {r.toFixed(2)}
                {mean !== null && <> &middot; pool avg {mean.toFixed(2)}</>}
              </span>
            )}
          </div>
        );
      })}
      <div className="mt-0.5 flex items-center justify-between border-t border-[var(--line)] pt-1.5">
        <span className="text-[11px] font-medium text-[var(--ink-soft)]">Total score</span>
        <span className="font-mono text-[12px] font-medium tabular-nums text-[var(--ink)]">{score.toFixed(2)}</span>
      </div>
    </div>
  );
}
