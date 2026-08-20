import { SpeakerHigh, Waveform } from "@phosphor-icons/react";
import type { Category } from "../types";
import { accentFor } from "../accent";

export function CategoryBadge({ category }: { category: Category }) {
  const accent = accentFor(category);
  const Icon = category === "speech" ? SpeakerHigh : Waveform;
  const label = category === "speech" ? "Speech" : "Non-speech";

  return (
    <span
      className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium"
      style={{
        color: accent,
        backgroundColor: `color-mix(in oklab, ${accent} 14%, var(--surface-1))`,
      }}
    >
      <Icon size={12} weight="bold" />
      {label}
    </span>
  );
}
