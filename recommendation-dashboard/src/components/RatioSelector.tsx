import type { RatioData } from "../types";

export function RatioSelector({
  ratios,
  activeIndex,
  onSelect,
}: {
  ratios: RatioData[];
  activeIndex: number;
  onSelect: (index: number) => void;
}) {
  return (
    <div className="flex flex-wrap gap-2">
      {ratios.map((ratio, i) => {
        const active = i === activeIndex;
        return (
          <button
            key={ratio.label}
            type="button"
            onClick={() => onSelect(i)}
            className="flex flex-col gap-1.5 rounded-xl border px-3 py-2 text-left transition-colors active:scale-[0.98]"
            style={{
              borderColor: active ? "var(--ink)" : "var(--line)",
              backgroundColor: active ? "var(--surface-2)" : "var(--surface-1)",
            }}
          >
            <span className="font-mono text-[12px] font-medium tabular-nums text-[var(--ink)]">{ratio.label}</span>
            <div className="flex h-1 w-14 overflow-hidden rounded-full">
              <div style={{ width: `${(ratio.non_speech_count / 10) * 100}%`, backgroundColor: "var(--non-speech)" }} />
              <div style={{ width: `${(ratio.speech_count / 10) * 100}%`, backgroundColor: "var(--speech)" }} />
            </div>
          </button>
        );
      })}
    </div>
  );
}
