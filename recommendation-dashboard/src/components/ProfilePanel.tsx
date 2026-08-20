import type { Category, CategoryProfile } from "../types";
import { accentFor } from "../accent";
import { CategoryBadge } from "./CategoryBadge";

const CATEGORY_ORDER: Category[] = ["non_speech", "speech"];

interface CombinedRow {
  class_name: string;
  score: number;
  category: Category;
}

export function ProfilePanel({ profiles }: { profiles: Partial<Record<Category, CategoryProfile>> }) {
  // Each category's top_classes already sums to 100% *within that category alone* (a speech-only
  // pie and a non-speech-only pie). Merging them unweighted would show two independent 100%s side
  // by side, which looks like it should add to 100 and doesn't. Weighting each by its share of the
  // 10 mock clips turns it into one real combined distribution instead.
  const totalClips = CATEGORY_ORDER.reduce((sum, c) => sum + (profiles[c]?.count ?? 0), 0);

  const rows: CombinedRow[] = CATEGORY_ORDER.flatMap((category) => {
    const profile = profiles[category];
    if (!profile || totalClips === 0) return [];
    const weight = profile.count / totalClips;
    return profile.top_classes.map((c) => ({ ...c, score: c.score * weight, category }));
  })
    .sort((a, b) => b.score - a.score)
    .slice(0, 8);

  const max = rows[0]?.score ?? 1;

  return (
    <div className="flex flex-col gap-3 rounded-2xl border border-[var(--line)] bg-[var(--surface-1)] p-4">
      <div className="flex items-center justify-between">
        <span className="text-[13px] font-medium text-[var(--ink)]">Combined YAMNet signal</span>
        <div className="flex items-center gap-3">
          {CATEGORY_ORDER.filter((c) => profiles[c]).map((category) => (
            <div key={category} className="flex items-center gap-1.5">
              <CategoryBadge category={category} />
              <span className="text-[11px] text-[var(--ink-muted)]">{profiles[category]!.count} clips</span>
            </div>
          ))}
        </div>
      </div>

      {rows.length > 0 ? (
        <div className="flex flex-col gap-1.5">
          {rows.map((row) => {
            const accent = accentFor(row.category);
            return (
              <div key={`${row.category}-${row.class_name}`} className="flex items-center gap-2">
                <span className="size-1.5 shrink-0 rounded-full" style={{ backgroundColor: accent }} />
                <span className="w-36 shrink-0 truncate text-[12px] text-[var(--ink-soft)]" title={row.class_name}>
                  {row.class_name}
                </span>
                <div className="h-1.5 flex-1 rounded-full bg-[var(--surface-2)]">
                  <div
                    className="h-full rounded-full"
                    style={{ width: `${(row.score / max) * 100}%`, backgroundColor: accent }}
                  />
                </div>
                <span className="w-9 shrink-0 text-right font-mono text-[11px] tabular-nums text-[var(--ink-muted)]">
                  {Math.round(row.score * 100)}%
                </span>
              </div>
            );
          })}
        </div>
      ) : (
        <p className="text-[12px] text-[var(--ink-muted)]">No YAMNet signal for this history.</p>
      )}

      <p className="border-t border-[var(--line)] pt-2 text-[11px] leading-relaxed text-[var(--ink-muted)]">
        Weighted by clip count ({CATEGORY_ORDER.filter((c) => profiles[c]).map((c) => `${profiles[c]!.count} ${c === "speech" ? "speech" : "non-speech"}`).join(", ")}) so this is one combined distribution, not two 100% pies shown side by side.
      </p>
    </div>
  );
}
