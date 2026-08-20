import { Quotes, WarningCircle } from "@phosphor-icons/react";
import type { Category, PoolStats, YamnetClass } from "../types";
import { accentFor } from "../accent";
import { AudioPlayer } from "./AudioPlayer";
import { YamnetBars } from "./YamnetBars";
import { ScoreBreakdown } from "./ScoreBreakdown";
import { CategoryBadge } from "./CategoryBadge";
import { IdTag } from "./IdTag";

interface ScoreProps {
  clap_sim: number;
  yamnet_overlap: number | null;
  transcript_sim: number | null;
  z_clap: number;
  z_yamnet: number | null;
  z_transcript: number | null;
  score: number;
  poolStats: PoolStats;
}

interface ClipCardProps {
  id: string;
  filename: string;
  isSpeech: boolean;
  yamnet: YamnetClass[];
  transcript: string | null;
  category: Category;
  rank?: number;
  mismatch?: boolean;
  scoreBreakdown?: ScoreProps;
}

export function ClipCard({ id, filename, isSpeech, yamnet, transcript, category, rank, mismatch, scoreBreakdown }: ClipCardProps) {
  const accent = accentFor(category);

  return (
    <div
      className="flex flex-col gap-3 rounded-2xl border border-[var(--line)] bg-[var(--surface-1)] p-4"
      style={{ borderLeft: `3px solid ${accent}` }}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2">
          {rank !== undefined && (
            <span className="font-mono text-[11px] text-[var(--ink-muted)]">{String(rank).padStart(2, "0")}</span>
          )}
          <CategoryBadge category={category} />
        </div>
        <div className="flex flex-col items-end gap-1">
          <IdTag id={id} />
          {mismatch && (
            <span
              className="inline-flex items-center gap-1 text-[10px] text-[var(--ink-muted)]"
              title={`Retrieved via the ${category === "speech" ? "speech" : "non-speech"} profile, but this clip is itself labeled ${isSpeech ? "speech" : "non-speech"}.`}
            >
              <WarningCircle size={12} />
              actually {isSpeech ? "speech" : "non-speech"}
            </span>
          )}
        </div>
      </div>

      <AudioPlayer src={`/audio/${filename}`} accent={accent} />

      <YamnetBars classes={yamnet} accent={accent} />

      {transcript && (
        <div className="flex gap-1.5 rounded-lg bg-[var(--surface-2)] px-2.5 py-2 text-[12px] leading-snug text-[var(--ink-soft)]">
          <Quotes size={13} weight="fill" className="mt-0.5 shrink-0 text-[var(--ink-muted)]" />
          <span>{transcript}</span>
        </div>
      )}

      {scoreBreakdown && (
        <div className="border-t border-[var(--line)] pt-3">
          <ScoreBreakdown {...scoreBreakdown} />
        </div>
      )}
    </div>
  );
}
