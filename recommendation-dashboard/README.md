# Mock listening history → recommendations

A dashboard for notebook 6's mock-history recommendation pipeline: pick a speech/non-speech ratio, see the
resulting per-category taste profile, and the top-10 recommendations it produces, with a full score breakdown
per result.

## Run it

```bash
npm install
npm run dev
```

## Regenerating the data

`src/data/pipeline.json` and `public/audio/*.mp3` are a **static export**, not a live backend. If notebook 6's
curated pools, the CLAP index, or the z-score reranker change, re-run the export from the repo root:

```bash
.venv/bin/python3 recommendation-dashboard/export_data.py
```

This mirrors notebook 6's logic exactly (same `NON_SPEECH_POOL` / `SPEECH_POOL`, same per-category retrieval,
same z-score reranking) for all 9 ratios (9/1 through 1/9), and copies whichever audio clips are referenced
into `public/audio/`.

## Stack

Vite + React + TypeScript + Tailwind v4, Phosphor icons, self-hosted Geist Sans/Mono. Two accent colors carry
meaning throughout (violet = speech, teal = non-speech); a separate blue/red pair is reserved for the
score-breakdown chart's positive/negative z-scores, so no color means two different things on the page. Both
palettes are validated against colorblind-safety and contrast checks (see `dataviz` skill).
