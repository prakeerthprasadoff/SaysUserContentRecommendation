import type { YamnetClass } from "../types";

export function YamnetBars({ classes, accent }: { classes: YamnetClass[]; accent: string }) {
  if (classes.length === 0) return null;
  const max = classes[0].score;

  return (
    <div className="flex flex-col gap-1.5">
      {classes.map((c) => (
        <div key={c.class_name} className="flex items-center gap-2">
          <span className="w-28 shrink-0 truncate text-[12px] text-[var(--ink-soft)]" title={c.class_name}>
            {c.class_name}
          </span>
          <div className="h-1.5 flex-1 rounded-full bg-[var(--surface-2)]">
            <div
              className="h-full rounded-full"
              style={{ width: `${(c.score / max) * 100}%`, backgroundColor: accent }}
            />
          </div>
          <span className="w-9 shrink-0 text-right font-mono text-[11px] tabular-nums text-[var(--ink-muted)]">
            {Math.round(c.score * 100)}%
          </span>
        </div>
      ))}
    </div>
  );
}
