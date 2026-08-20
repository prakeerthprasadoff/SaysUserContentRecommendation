import { useMemo, useState } from "react";
import pipeline from "./data/pipeline.json";
import type { Pipeline, Category } from "./types";
import { RatioSelector } from "./components/RatioSelector";
import { ClipCard } from "./components/ClipCard";
import { ProfilePanel } from "./components/ProfilePanel";
import { HowItWorks } from "./components/HowItWorks";
import { ThemeToggle } from "./components/ThemeToggle";

const data = pipeline as Pipeline;
const CATEGORY_ORDER: Category[] = ["non_speech", "speech"];
const CATEGORY_LABEL: Record<Category, string> = { non_speech: "Non-speech", speech: "Speech" };

function SectionHeading({ eyebrow, title }: { eyebrow: string; title: string }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-[11px] font-medium uppercase tracking-wide text-[var(--ink-muted)]">{eyebrow}</span>
      <h2 className="text-lg font-semibold text-[var(--ink)]">{title}</h2>
    </div>
  );
}

export default function App() {
  const [ratioIndex, setRatioIndex] = useState(
    data.ratios.findIndex((r) => r.non_speech_count === 7) ?? 0,
  );
  const ratio = data.ratios[ratioIndex];

  const historyByCategory = useMemo(() => {
    const groups: Record<Category, typeof ratio.mock_history> = { speech: [], non_speech: [] };
    for (const clip of ratio.mock_history) {
      groups[clip.is_speech ? "speech" : "non_speech"].push(clip);
    }
    return groups;
  }, [ratio]);

  const recsByCategory = useMemo(() => {
    const groups: Record<Category, typeof ratio.recommendations> = { speech: [], non_speech: [] };
    for (const rec of ratio.recommendations) {
      groups[rec.recommended_because].push(rec);
    }
    return groups;
  }, [ratio]);

  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-10 px-5 py-10 md:px-8">
      <header className="flex items-start justify-between gap-4">
        <div className="flex flex-col gap-1.5">
          <h1 className="text-2xl font-semibold tracking-tight text-[var(--ink)]">
            Mock listening history &rarr; recommendations
          </h1>
          <p className="max-w-[60ch] text-[13px] leading-relaxed text-[var(--ink-soft)]">
            A speech / non-speech ratio stands in for a real listening history. Pick one to see the resulting
            taste profile and the top {data.top_k} recommendations it produces from a {data.candidate_pool}-clip
            candidate pool per group.
          </p>
        </div>
        <ThemeToggle />
      </header>

      <RatioSelector ratios={data.ratios} activeIndex={ratioIndex} onSelect={setRatioIndex} />

      <HowItWorks />

      <section className="flex flex-col gap-4">
        <SectionHeading eyebrow="Input" title="Your listening history" />
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          {CATEGORY_ORDER.map((category) => (
            <div key={category} className="flex flex-col gap-3">
              {historyByCategory[category].map((clip) => (
                <ClipCard
                  key={clip.id}
                  id={clip.id}
                  filename={clip.filename}
                  isSpeech={clip.is_speech}
                  yamnet={clip.yamnet}
                  transcript={clip.transcript}
                  category={category}
                />
              ))}
            </div>
          ))}
        </div>
      </section>

      <section className="flex flex-col gap-4">
        <SectionHeading eyebrow="Derived" title="Taste profile" />
        <ProfilePanel profiles={ratio.category_profiles} />
      </section>

      <section className="flex flex-col gap-6">
        <div className="flex flex-col gap-2">
          <SectionHeading eyebrow="Output" title="Recommendations" />
          <p className="max-w-[70ch] text-[12px] leading-relaxed text-[var(--ink-muted)]">
            Each bar's <span className="font-mono">z</span> is how many standard deviations that similarity sat
            above or below the average across every candidate the group searched. The line underneath is the
            same thing ungrounded from that curve: the raw cosine similarity, next to the pool's average, so
            you can judge whether a strong z-score reflects a genuinely close match or a pool that was
            uniformly weak.
          </p>
        </div>
        {CATEGORY_ORDER.filter((c) => recsByCategory[c].length > 0).map((category) => (
          <div key={category} className="flex flex-col gap-3">
            <h3 className="text-[13px] font-medium text-[var(--ink-soft)]">
              Because of your {CATEGORY_LABEL[category].toLowerCase()} listens
              <span className="ml-1.5 font-mono text-[11px] text-[var(--ink-muted)]">
                {recsByCategory[category].length} slot{recsByCategory[category].length === 1 ? "" : "s"}
              </span>
            </h3>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {recsByCategory[category].map((rec, i) => (
                <ClipCard
                  key={rec.recordingId}
                  id={rec.recordingId}
                  filename={rec.filename}
                  isSpeech={rec.is_speech}
                  yamnet={rec.yamnet}
                  transcript={rec.transcript}
                  category={category}
                  rank={i + 1}
                  mismatch={rec.is_speech !== (category === "speech")}
                  scoreBreakdown={{
                    clap_sim: rec.clap_sim,
                    yamnet_overlap: rec.yamnet_overlap,
                    transcript_sim: rec.transcript_sim,
                    z_clap: rec.z_clap,
                    z_yamnet: rec.z_yamnet,
                    z_transcript: rec.z_transcript,
                    score: rec.score,
                    poolStats: ratio.pool_stats[category]!,
                  }}
                />
              ))}
            </div>
          </div>
        ))}
      </section>
    </div>
  );
}
