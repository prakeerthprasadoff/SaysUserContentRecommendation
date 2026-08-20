import { CaretDown } from "@phosphor-icons/react";

export function HowItWorks() {
  return (
    <details className="group rounded-2xl border border-[var(--line)] bg-[var(--surface-1)] px-4 py-3 open:pb-4">
      <summary className="flex cursor-pointer list-none items-center justify-between text-[13px] font-medium text-[var(--ink)]">
        How the recommendations are built
        <CaretDown size={14} className="text-[var(--ink-muted)] transition-transform group-open:rotate-180" />
      </summary>
      <div className="mt-3 flex flex-col gap-2.5 text-[13px] leading-relaxed text-[var(--ink-soft)]">
        <p>
          The 10 mock clips are split into two groups by their true speech label, and each group gets its own
          taste profile (a CLAP embedding average, a YAMNet class-distribution average, a transcript-embedding
          average). Averaging everything into one profile would let whichever group clusters more tightly in
          CLAP space (usually speech) drown out the other, so retrieval runs separately per group instead.
        </p>
        <p>
          Each group searches the full catalog by CLAP similarity, then reranks its own candidates by
          z-scoring CLAP, YAMNet, and transcript similarity within that search (so the three differently-scaled
          signals are combined without hand-picked weights) and summing them.
        </p>
        <p>
          The final 10 recommendations are split proportionally to your ratio, at least one slot per group
          present, so a minority interest is never fully crowded out by the other.
        </p>
      </div>
    </details>
  );
}
