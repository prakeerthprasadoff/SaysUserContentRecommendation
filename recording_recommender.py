"""
Recording recommender: fuses CLAP audio embeddings with transcript text
embeddings to recommend "Says" recordings similar to a user's history.

Unlike the podcast recommender, this corpus is not "just people talking" --
recordings can be music, sound effects, ambience, or speech. So:

  - CLAP embeddings (clap_embeddings.py) are computed for every recording
    and are the universal similarity signal (audio "vibe": tone, energy,
    background music/noise), regardless of content type.
  - Transcript embeddings (transcript_embeddings.py) only exist for
    speech-labeled recordings with a successful, non-empty transcription,
    and add a topical/semantic signal on top of CLAP for that subset.

A candidate's final score is CLAP-only unless it has a transcript embedding
AND the user's own history produced a transcript query vector -- there is
no zero-padding of a missing component into the blend.

USAGE:
  python recording_recommender.py --account-id <uuid> [OPTIONS]

OPTIONS:
  --top-k <int>            Number of recommendations to return. Default: 10.
  --verbose                Show detailed debug logs.
  --clap-weight <float>    Weight for CLAP similarity when both signals exist.
  --transcript-weight <float>  Weight for transcript similarity when both exist.
  --same-type-only         Restrict candidates to the user's dominant content
                            type (speech vs non_speech), instead of letting
                            CLAP compare across types.
"""

import argparse
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()


# ─── ENV + DATABASE CONFIG ──────────────────────────────────────────────────

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")

DATABASE_URL = (
    f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)


# ─── ARTIFACT PATHS ──────────────────────────────────────────────────────────

SPEECH_LABELS_CSV = os.getenv("SPEECH_LABELS_CSV", "audio_speech_labels.csv")
TRANSCRIPTS_CSV = os.getenv("TRANSCRIPTS_CSV", "audio_transcriptions.csv")
CLAP_EMBEDDINGS_CSV = os.getenv("CLAP_OUTPUT_CSV", "audio_clap_embeddings.csv")
TRANSCRIPT_EMBEDDINGS_CSV = os.getenv(
    "TRANSCRIPT_EMBEDDINGS_OUTPUT_CSV", "audio_transcript_embeddings.csv"
)

ID_COLUMN = os.getenv("ID_COLUMN", "id")


# ─── RECOMMENDER CONFIGURATION ───────────────────────────────────────────────

# Prak - 94ffa848-519e-424c-a343-ba2e021bf75c
# Tavishi - 19749402-c4dc-429a-a664-425240ab2f0b
# Utkarsh - dd2b5ff0-9a5a-44b7-b5cd-e3a082155c48
DEFAULT_ACCOUNT_ID = "dd2b5ff0-9a5a-44b7-b5cd-e3a082155c48"

LIKE_BOOST = 1.0
RECENCY_DECAY_RATE = 0.2
MIN_WEIGHT = 0.1

DEFAULT_CLAP_WEIGHT = 0.5
DEFAULT_TRANSCRIPT_WEIGHT = 0.5

DEFAULT_TOP_K = 10

VERBOSE = False


def log(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)


# ─── ARGUMENT PARSING ────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recommend Says recordings using fused CLAP + transcript embeddings."
    )
    parser.add_argument("--account-id", type=str, default=DEFAULT_ACCOUNT_ID)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--clap-weight", type=float, default=DEFAULT_CLAP_WEIGHT)
    parser.add_argument("--transcript-weight", type=float, default=DEFAULT_TRANSCRIPT_WEIGHT)
    parser.add_argument(
        "--same-type-only",
        action="store_true",
        help="Restrict candidates to the user's dominant content type (speech vs non_speech).",
    )
    return parser.parse_args()


# ─── VECTOR HELPERS ──────────────────────────────────────────────────────────

def decode_vector(text_value: str) -> np.ndarray:
    return np.array(text_value.split(","), dtype=np.float32)


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector if norm == 0 else vector / norm


def normalize_matrix_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


# ─── ARTIFACT LOADING ────────────────────────────────────────────────────────

def load_embedding_artifact(path: str, embedding_column: str) -> tuple[list[str], np.ndarray]:
    """Load a {id, <embedding_column>} CSV into (ids, matrix)."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing embedding artifact: {path}\n"
            f"Run the corresponding embedding pipeline first."
        )

    dataframe = pd.read_csv(path)
    dataframe = dataframe[dataframe[embedding_column].notna()]

    ids = dataframe[ID_COLUMN].astype(str).tolist()
    matrix = np.stack([decode_vector(v) for v in dataframe[embedding_column]]).astype(np.float32)

    return ids, matrix


def load_content_types() -> dict[str, str]:
    """Map recording id -> speech_label ('speech' / 'non_speech')."""
    if not os.path.exists(SPEECH_LABELS_CSV):
        return {}

    dataframe = pd.read_csv(SPEECH_LABELS_CSV)
    dataframe = dataframe[dataframe["speech_label"].notna()]

    return dict(zip(dataframe[ID_COLUMN].astype(str), dataframe["speech_label"]))


def load_transcript_snippets() -> dict[str, str]:
    """Map recording id -> transcript text, for display purposes only."""
    if not os.path.exists(TRANSCRIPTS_CSV):
        return {}

    dataframe = pd.read_csv(TRANSCRIPTS_CSV)
    dataframe["transcript"] = dataframe["transcript"].fillna("").astype(str)

    return dict(zip(dataframe[ID_COLUMN].astype(str), dataframe["transcript"]))


# ─── DATABASE FUNCTIONS ─────────────────────────────────────────────────────

INTERACTIONS_SQL = """
WITH user_plays AS (
    SELECT u.id AS user_id, rp."recordingId" AS recording_id, rp.position, rp."updatedAt"
    FROM "user" u
    JOIN recording_play rp ON u.id = rp."userId"
    WHERE u."accountId" = :account_id
),
recording_posts AS (
    SELECT id AS post_id, "recordingId" AS recording_id
    FROM post
    WHERE "recordingId" IS NOT NULL
),
user_likes AS (
    SELECT rpost.recording_id AS recording_id, pl."accountId"
    FROM post_like pl
    JOIN recording_posts rpost ON pl."postId" = rpost.post_id
    WHERE pl."accountId" = :account_id
)
SELECT
    COALESCE(up.recording_id, ul.recording_id) AS recording_id,
    r.duration                                 AS total_seconds,
    up.position                                AS listened_seconds,
    up."updatedAt"                             AS last_played_at,
    CASE WHEN ul.recording_id IS NOT NULL THEN true ELSE false END AS is_liked
FROM user_plays up
FULL OUTER JOIN user_likes ul ON up.recording_id = ul.recording_id
LEFT JOIN recording r ON r.id = COALESCE(up.recording_id, ul.recording_id)
"""


def fetch_user_interactions(engine, account_id: str) -> pd.DataFrame:
    dataframe = pd.read_sql(text(INTERACTIONS_SQL), engine, params={"account_id": account_id})
    log(f"Fetched {len(dataframe)} interaction rows for account {account_id[:8]}...")
    return dataframe


def fetch_recording_captions(engine, recording_ids: list[str]) -> dict[str, str]:
    """Best-effort captions from the post wrapping each recording, for display only."""
    if not recording_ids:
        return {}

    placeholders = ", ".join(f":id_{i}" for i in range(len(recording_ids)))
    params = {f"id_{i}": rid for i, rid in enumerate(recording_ids)}

    query = text(f"""
        SELECT "recordingId" AS recording_id, description
        FROM post
        WHERE "recordingId" IN ({placeholders})
        AND description IS NOT NULL
    """)

    rows = pd.read_sql(query, engine, params=params)
    return dict(zip(rows["recording_id"].astype(str), rows["description"]))


# ─── INTERACTION CLEANING ────────────────────────────────────────────────────

def convert_time_string_to_seconds(time_value) -> float:
    if pd.isna(time_value):
        return 0.0

    if isinstance(time_value, (int, float)):
        return float(time_value)

    if isinstance(time_value, str):
        try:
            return float(time_value)
        except ValueError:
            pass

        try:
            parts = time_value.split(":")
            if len(parts) == 3:
                h, m, s = map(float, parts)
                return h * 3600 + m * 60 + s
            if len(parts) == 2:
                m, s = map(float, parts)
                return m * 60 + s
        except Exception:
            pass

    return 0.0


def clean_interaction_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe[dataframe["recording_id"].notna()].copy()

    if dataframe.empty:
        return dataframe

    dataframe["recording_id"] = dataframe["recording_id"].astype(str)
    dataframe["total_seconds"] = dataframe["total_seconds"].apply(convert_time_string_to_seconds)
    dataframe["listened_seconds"] = (
        dataframe["listened_seconds"].apply(convert_time_string_to_seconds).fillna(0)
    )
    dataframe["last_played_at"] = pd.to_datetime(dataframe["last_played_at"], errors="coerce")
    dataframe["has_play"] = dataframe["last_played_at"].notna()

    dataframe["recency_rank"] = 0
    played_mask = dataframe["has_play"]
    if played_mask.any():
        dataframe.loc[played_mask, "recency_rank"] = (
            dataframe.loc[played_mask, "last_played_at"]
            .rank(method="min", ascending=False)
            .sub(1)
            .astype(int)
        )

    dataframe["completion_pct"] = dataframe.apply(
        lambda row: min(1.0, max(0.0, row["listened_seconds"] / row["total_seconds"]))
        if row["total_seconds"] > 0
        else 0.0,
        axis=1,
    )

    return dataframe[
        ["recording_id", "completion_pct", "recency_rank", "is_liked", "has_play"]
    ]


def aggregate_interactions_by_recording(dataframe: pd.DataFrame) -> pd.DataFrame:
    aggregated = (
        dataframe.groupby("recording_id")
        .agg(
            completion_pct=("completion_pct", "mean"),
            recency_rank=("recency_rank", "min"),
            is_liked=("is_liked", "any"),
            play_count=("has_play", "sum"),
        )
        .reset_index()
    )
    aggregated["play_count"] = aggregated["play_count"].astype(int)
    return aggregated


def calculate_weights(dataframe: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    recording_ids = []
    weights = []

    for _, row in dataframe.iterrows():
        recency_multiplier = float(np.exp(-RECENCY_DECAY_RATE * int(row["recency_rank"])))
        play_count = int(row["play_count"])
        play_count_boost = 1.0 + float(np.log(play_count)) if play_count > 0 else 1.0
        play_weight = float(row["completion_pct"]) * recency_multiplier * play_count_boost
        explicit_weight = LIKE_BOOST if row["is_liked"] else 0.0
        final_weight = max(MIN_WEIGHT, play_weight + explicit_weight)

        recording_ids.append(str(row["recording_id"]))
        weights.append(final_weight)

    return recording_ids, np.array(weights, dtype=np.float32)


# ─── QUERY EMBEDDING CONSTRUCTION ────────────────────────────────────────────

def build_query_vector(
    ids: list[str],
    matrix: np.ndarray,
    history_ids: list[str],
    history_weights: np.ndarray,
) -> np.ndarray | None:
    """Weighted-average query vector over whichever history ids exist in this
    embedding space. Returns None if none of the history overlaps."""
    id_to_idx = {rid: i for i, rid in enumerate(ids)}

    indices = []
    weights = []
    for rid, weight in zip(history_ids, history_weights):
        if rid in id_to_idx:
            indices.append(id_to_idx[rid])
            weights.append(weight)

    if not indices:
        return None

    selected = normalize_matrix_rows(matrix[np.array(indices)])
    query = np.average(selected, axis=0, weights=np.array(weights, dtype=np.float32))
    return normalize_vector(query)


# ─── SCORING ──────────────────────────────────────────────────────────────

def score_candidates(
    clap_ids: list[str],
    clap_matrix: np.ndarray,
    clap_query: np.ndarray,
    transcript_ids: list[str],
    transcript_matrix: np.ndarray,
    transcript_query: np.ndarray | None,
    clap_weight: float,
    transcript_weight: float,
) -> pd.DataFrame:
    """
    CLAP covers every recording, so it's the master candidate list.
    Transcript similarity is only blended in for candidates that have a
    transcript embedding AND when the user's history produced a transcript
    query vector -- otherwise the candidate's score is CLAP-only.
    """
    clap_scores = normalize_matrix_rows(clap_matrix) @ clap_query

    transcript_score_by_id: dict[str, float] = {}
    if transcript_query is not None:
        transcript_scores = normalize_matrix_rows(transcript_matrix) @ transcript_query
        transcript_score_by_id = dict(zip(transcript_ids, transcript_scores))

    rows = []
    for idx, recording_id in enumerate(clap_ids):
        clap_score = float(clap_scores[idx])
        transcript_score = transcript_score_by_id.get(recording_id)

        if transcript_score is not None:
            final_score = clap_weight * clap_score + transcript_weight * transcript_score
        else:
            final_score = clap_score

        rows.append({
            "recording_id": recording_id,
            "final_score": final_score,
            "clap_score": clap_score,
            "transcript_score": transcript_score,
        })

    return pd.DataFrame(rows)


# ─── PRINTING ────────────────────────────────────────────────────────────────

def _snippet(text_value: str, max_len: int = 60) -> str:
    text_value = (text_value or "").replace("\n", " ").strip()
    if not text_value:
        return "—"
    return text_value if len(text_value) <= max_len else text_value[: max_len - 1] + "…"


def _pct(score: float | None) -> str:
    return "—" if score is None or (isinstance(score, float) and np.isnan(score)) else f"{score * 100:.0f}%"


def print_table(rows: list[dict]) -> None:
    if not rows:
        print("  (no rows)")
        return

    columns = list(rows[0].keys())
    widths = {c: max(len(c), max(len(str(r[c])) for r in rows)) for c in columns}

    def line(values):
        return " │ ".join(str(v).ljust(widths[c]) for v, c in zip(values, columns))

    print(line(columns))
    print("─┼─".join("─" * widths[c] for c in columns))
    for r in rows:
        print(line([r[c] for c in columns]))


# ─── MAIN ───────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    global VERBOSE
    VERBOSE = args.verbose

    engine = create_engine(DATABASE_URL)

    try:
        clap_ids, clap_matrix = load_embedding_artifact(CLAP_EMBEDDINGS_CSV, "clap_embedding")
        transcript_ids, transcript_matrix = load_embedding_artifact(
            TRANSCRIPT_EMBEDDINGS_CSV, "transcript_embedding"
        )
        content_types = load_content_types()
        transcripts = load_transcript_snippets()

        log(f"CLAP embeddings:       {len(clap_ids)}")
        log(f"Transcript embeddings: {len(transcript_ids)}")

        raw_interactions = fetch_user_interactions(engine, args.account_id)
        if raw_interactions.empty:
            print(f"No interactions found for account {args.account_id}")
            return

        cleaned = clean_interaction_data(raw_interactions)
        if cleaned.empty:
            print("No valid interactions after cleaning.")
            return

        aggregated = aggregate_interactions_by_recording(cleaned)
        history_ids, history_weights = calculate_weights(aggregated)

        if args.same_type_only:
            type_weight_totals: dict[str, float] = {}
            for rid, weight in zip(history_ids, history_weights):
                content_type = content_types.get(rid, "unknown")
                type_weight_totals[content_type] = type_weight_totals.get(content_type, 0.0) + weight
            dominant_type = max(type_weight_totals, key=type_weight_totals.get)
            log(f"Dominant content type: {dominant_type} ({type_weight_totals})")

            keep_mask = [content_types.get(rid) == dominant_type for rid in clap_ids]
            clap_ids = [rid for rid, keep in zip(clap_ids, keep_mask) if keep]
            clap_matrix = clap_matrix[np.array(keep_mask)]

        clap_query = build_query_vector(clap_ids, clap_matrix, history_ids, history_weights)
        if clap_query is None:
            print("None of the user's history has a CLAP embedding -- cannot recommend.")
            return

        transcript_query = build_query_vector(
            transcript_ids, transcript_matrix, history_ids, history_weights
        )
        if transcript_query is None:
            log("No transcript overlap with user history -- recommendations will be CLAP-only.")

        scores = score_candidates(
            clap_ids=clap_ids,
            clap_matrix=clap_matrix,
            clap_query=clap_query,
            transcript_ids=transcript_ids,
            transcript_matrix=transcript_matrix,
            transcript_query=transcript_query,
            clap_weight=args.clap_weight,
            transcript_weight=args.transcript_weight,
        )

        history_id_set = set(history_ids)
        scores = scores[~scores["recording_id"].isin(history_id_set)]
        scores = scores.sort_values("final_score", ascending=False).head(args.top_k)

        display_ids = list(history_id_set) + scores["recording_id"].tolist()
        captions = fetch_recording_captions(engine, display_ids)

    finally:
        engine.dispose()

    total_weight = float(history_weights.sum()) or 1.0
    history_rows = sorted(
        zip(history_ids, history_weights), key=lambda pair: -pair[1]
    )

    print(f"\n=== LISTENING HISTORY for account {args.account_id[:8]}... ===\n")
    print_table([
        {
            "#": rank,
            "Type": content_types.get(rid, "?"),
            "Caption/Transcript": _snippet(captions.get(rid) or transcripts.get(rid, "")),
            "Weight": f"{weight:.2f}",
            "Share": f"{weight / total_weight * 100:.0f}%",
        }
        for rank, (rid, weight) in enumerate(history_rows, start=1)
    ])

    print(f"\n=== RECOMMENDATIONS (top {len(scores)}) ===\n")
    print_table([
        {
            "#": rank,
            "Type": content_types.get(row["recording_id"], "?"),
            "Caption/Transcript": _snippet(
                captions.get(row["recording_id"]) or transcripts.get(row["recording_id"], "")
            ),
            "Score": _pct(row["final_score"]),
            "CLAP": _pct(row["clap_score"]),
            "Transcript": _pct(row["transcript_score"]),
        }
        for rank, (_, row) in enumerate(scores.iterrows(), start=1)
    ])


if __name__ == "__main__":
    main()
