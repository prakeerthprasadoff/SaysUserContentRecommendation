"""
Exports the notebook 6 mock-history -> recommendation pipeline (all 9 speech/non-speech
ratios) to a single JSON file the dashboard reads at build/runtime, and copies the
referenced audio files into public/audio/.

Mirrors notebook 6's logic exactly (same pools, same category-profile building, same
z-score reranking) -- this is a read-only export, it does not modify the notebook or
any pipeline output.
"""
import json
import shutil
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
INDEX_DIR = REPO_ROOT / "data_index"
AUDIO_DIR = DATA_DIR / "audio"

OUT_DIR = Path(__file__).resolve().parent
PUBLIC_AUDIO_DIR = OUT_DIR / "public" / "audio"
OUT_JSON = OUT_DIR / "src" / "data" / "pipeline.json"

TOP_K = 10
CANDIDATE_POOL = 200

# Curated for a legible before/after: each pool is now internally coherent (all Music,
# all casual English speech in the same register) rather than deliberately diverse, so a
# recommendation visibly resembles what's in the history instead of just sharing a category.
NON_SPEECH_POOL = [
    "3bc572a3-edcd-462d-a754-a2eb999f9fdd",  # Music (0.840)
    "39af460c-a543-4532-bd7b-d7bc2bc742d7",  # Music (0.865)
    "c7ca8ed4-336f-4ea7-8533-62afd9cc385f",  # Music (0.989)
    "d5780b63-4a09-4a1c-b71f-86b6c6ac4562",  # Music (0.988)
    "9375aa9b-130c-43c6-8af4-221838a2ee9c",  # Music (0.982)
    "2d1c8707-4fff-4681-83c4-0c25567a5de6",  # Music (0.982)
    "7e9a93b0-17e4-48cb-8ec3-1b5174a1c9b5",  # Music (0.979)
    "2451e82b-f687-46aa-bc61-d4ce9af5e490",  # Music (0.978)
    "80d6d8e5-0d1a-4423-9732-ed6d6d1a9b10",  # Music (0.976)
]

SPEECH_POOL = [
    "bc5a14b2-3a32-40b1-9e36-08287de50216",  # en -- "...New episode of Lenny's Podcast..."
    "27b5596c-b371-4804-8491-d207c7543788",  # en -- "...seen this dog's tail..."
    "0448e815-738e-42cf-b9ca-93e467e2aad1",  # en -- "...fiction recs..."
    "33b92342-87fe-4092-a113-b0a460abfbe6",  # en -- "Let's test testing the new release."
    "67536d24-aa3a-465c-acf0-2d66a6dc4a3b",  # en -- "Reconnecting with David..."
    "6adf9446-b53f-4c11-a623-4edd45668641",  # en -- "...deciphered baby talk..."
    "2fd44fbe-3be1-4787-9699-053710da2bb4",  # en -- "Yo, I love this podcast..."
    "384c4540-47ae-47ff-8e16-78a4d60baaaf",  # en -- "Yo, Liam. That was probably the better way..."
    "625322fa-3775-49fb-936c-314e1ceb3262",  # en -- "This may be the most Australian podcast ever."
]

RATIOS = [(n, 10 - n) for n in range(9, 0, -1)]  # (non_speech_count, speech_count): 9/1 .. 1/9


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def decode_vector(text: str) -> np.ndarray:
    return np.array(text.split(","), dtype=np.float32)


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.full_like(x, np.nan, dtype=np.float64)
    valid = np.isfinite(x)
    if not valid.any():
        return out
    values = x[valid]
    sigma = values.std()
    if sigma == 0:
        out[valid] = 0.0
        return out
    out[valid] = (values - values.mean()) / sigma
    return out


print("Loading CLAP index...")
clap_index = faiss.read_index(str(INDEX_DIR / "clap_embeddings.faiss"))
clap_metadata = json.loads((INDEX_DIR / "clap_metadata.json").read_text())
clap_ids = [str(m["id"]) for m in clap_metadata]
id_to_clap_idx = {rid: i for i, rid in enumerate(clap_ids)}
clap_matrix = clap_index.reconstruct_n(0, clap_index.ntotal).astype(np.float32)
is_speech_map = {str(m["id"]): bool(m["is_speech"]) for m in clap_metadata}
filename_map = {str(m["id"]): m["filename"] for m in clap_metadata}

print("Loading YAMNet classifications...")
yamnet_df = pd.read_csv(DATA_DIR / "yamnet_top5_classifications.csv")
yamnet_df["id"] = yamnet_df["id"].astype(str)
yamnet_df = yamnet_df[yamnet_df["yamnet_status"] == "completed"]
long_rows = []
for i in range(1, 6):
    sub = yamnet_df[["id", f"yamnet_rank_{i}_class", f"yamnet_rank_{i}_mean_score"]].dropna()
    sub.columns = ["id", "class_name", "score"]
    long_rows.append(sub)
yamnet_long = pd.concat(long_rows, ignore_index=True)
yamnet_long = yamnet_long.groupby(["id", "class_name"], as_index=False)["score"].sum()
yamnet_vocab = sorted(yamnet_long["class_name"].unique())
class_to_col = {c: i for i, c in enumerate(yamnet_vocab)}
yamnet_ids = sorted(yamnet_long["id"].unique())
id_to_yamnet_idx = {rid: i for i, rid in enumerate(yamnet_ids)}
yamnet_matrix = np.zeros((len(yamnet_ids), len(yamnet_vocab)), dtype=np.float32)
for row in yamnet_long.itertuples(index=False):
    yamnet_matrix[id_to_yamnet_idx[row.id], class_to_col[row.class_name]] = row.score
row_sums = yamnet_matrix.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
yamnet_matrix /= row_sums


def top_yamnet_classes(recording_id: str, n: int = 3):
    if recording_id not in id_to_yamnet_idx:
        return []
    row = yamnet_matrix[id_to_yamnet_idx[recording_id]]
    top = sorted(zip(yamnet_vocab, row), key=lambda x: -x[1])[:n]
    return [{"class_name": c, "score": float(s)} for c, s in top if s > 0]


print("Loading transcripts...")
transcript_map = {}
if (DATA_DIR / "audio_transcriptions.csv").exists():
    t = pd.read_csv(DATA_DIR / "audio_transcriptions.csv")[["id", "transcript"]]
    t["id"] = t["id"].astype(str)
    transcript_map = dict(zip(t["id"], t["transcript"]))

id_to_transcript_idx = {}
transcript_matrix = np.empty((0, 0), dtype=np.float32)
if (DATA_DIR / "audio_transcript_embeddings.csv").exists():
    emb_df = pd.read_csv(DATA_DIR / "audio_transcript_embeddings.csv")
    emb_df["id"] = emb_df["id"].astype(str)
    transcript_ids = emb_df["id"].tolist()
    id_to_transcript_idx = {rid: i for i, rid in enumerate(transcript_ids)}
    transcript_matrix = np.vstack(emb_df["transcript_embedding"].apply(decode_vector).values).astype(np.float32)


def build_profile_from_recording_ids(recording_ids):
    ids = [r for r in dict.fromkeys(recording_ids) if r in id_to_clap_idx]
    clap_vec = l2_normalize(clap_matrix[[id_to_clap_idx[r] for r in ids]].mean(axis=0))

    yp = [r for r in ids if r in id_to_yamnet_idx]
    if yp:
        yv = yamnet_matrix[[id_to_yamnet_idx[r] for r in yp]].mean(axis=0)
        total = yv.sum()
        yv = yv / total if total > 0 else None
    else:
        yv = None

    tp = [r for r in ids if r in id_to_transcript_idx]
    if tp:
        tv = l2_normalize(transcript_matrix[[id_to_transcript_idx[r] for r in tp]].mean(axis=0))
    else:
        tv = None

    return {"recording_ids": ids, "clap_profile": clap_vec, "yamnet_profile": yv, "transcript_profile": tv}


def build_category_profiles(recording_ids):
    ids = [r for r in dict.fromkeys(recording_ids) if r in id_to_clap_idx]
    by_cat = {}
    for r in ids:
        label = "speech" if is_speech_map.get(r, False) else "non_speech"
        by_cat.setdefault(label, []).append(r)
    return {label: build_profile_from_recording_ids(cids) for label, cids in by_cat.items()}


def recommend_from_profile(profile, k, exclude_ids):
    already = set(exclude_ids)
    pool = min(CANDIDATE_POOL, clap_index.ntotal)
    scores, idxs = clap_index.search(profile["clap_profile"].reshape(1, -1).astype("float32"), pool)
    py, pt = profile["yamnet_profile"], profile["transcript_profile"]

    results = []
    for clap_sim, idx in zip(scores[0], idxs[0]):
        if idx < 0:
            continue
        rid = clap_ids[idx]
        if rid in already:
            continue
        yo = cosine_sim(py, yamnet_matrix[id_to_yamnet_idx[rid]]) if (py is not None and rid in id_to_yamnet_idx) else None
        ts = cosine_sim(pt, transcript_matrix[id_to_transcript_idx[rid]]) if (pt is not None and rid in id_to_transcript_idx) else None
        results.append({"recordingId": rid, "is_speech": is_speech_map.get(rid), "clap_sim": float(clap_sim), "yamnet_overlap": yo, "transcript_sim": ts})

    if not results:
        return pd.DataFrame(), {}

    ranked = pd.DataFrame(results)

    # Pool stats (mean/std across every candidate this profile searched, before z-scoring)
    # so the UI can show each recommendation's raw similarity next to the pool it was drawn
    # from ("cosine 0.69, pool average 0.42") instead of the z-score alone.
    pool_stats = {}
    for raw_col in ["clap_sim", "yamnet_overlap", "transcript_sim"]:
        vals = pd.to_numeric(ranked[raw_col], errors="coerce").to_numpy()
        valid = vals[np.isfinite(vals)]
        pool_stats[raw_col] = (
            {"mean": float(valid.mean()), "std": float(valid.std()), "n": int(valid.size)}
            if valid.size > 0
            else None
        )

    ranked["z_clap"] = zscore(pd.to_numeric(ranked["clap_sim"], errors="coerce").to_numpy())
    ranked["z_yamnet"] = zscore(pd.to_numeric(ranked["yamnet_overlap"], errors="coerce").to_numpy())
    ranked["z_transcript"] = zscore(pd.to_numeric(ranked["transcript_sim"], errors="coerce").to_numpy())
    ranked["score"] = ranked["z_clap"].fillna(0) + ranked["z_yamnet"].fillna(0) + ranked["z_transcript"].fillna(0)
    ranked = ranked.sort_values("score", ascending=False).head(k).reset_index(drop=True)
    return ranked, pool_stats


def recommend_from_mixed_history(recording_ids, k=TOP_K):
    profiles = build_category_profiles(recording_ids)
    already = set(recording_ids)
    counts = {label: len(p["recording_ids"]) for label, p in profiles.items()}
    total = sum(counts.values())
    slots = {label: max(1, round(k * c / total)) for label, c in counts.items()}
    biggest = max(slots, key=lambda label: counts[label])
    slots[biggest] += k - sum(slots.values())

    blocks = []
    pool_stats_by_category = {}
    for label, profile in profiles.items():
        block, pool_stats = recommend_from_profile(profile, slots[label], already)
        block.insert(0, "recommended_because", label)
        blocks.append(block)
        pool_stats_by_category[label] = pool_stats
    return pd.concat(blocks, ignore_index=True), pool_stats_by_category


def clip_summary(recording_id: str) -> dict:
    return {
        "id": recording_id,
        "filename": filename_map.get(recording_id),
        "is_speech": is_speech_map.get(recording_id),
        "yamnet": top_yamnet_classes(recording_id, n=3),
        "transcript": transcript_map.get(recording_id) or None,
    }


print("Computing all 9 ratios...")
referenced_audio_ids = set()
ratios_out = []

for n_non_speech, n_speech in RATIOS:
    mock_ids = NON_SPEECH_POOL[:n_non_speech] + SPEECH_POOL[:n_speech]
    referenced_audio_ids.update(mock_ids)

    profiles = build_category_profiles(mock_ids)
    profile_summaries = {}
    for label, profile in profiles.items():
        top = []
        if profile["yamnet_profile"] is not None:
            top = sorted(zip(yamnet_vocab, profile["yamnet_profile"]), key=lambda x: -x[1])[:5]
            top = [{"class_name": c, "score": float(s)} for c, s in top if s > 0]
        profile_summaries[label] = {"count": len(profile["recording_ids"]), "top_classes": top}

    recs_df, pool_stats_by_category = recommend_from_mixed_history(mock_ids, k=TOP_K)
    recommendations = []
    for row in recs_df.itertuples():
        referenced_audio_ids.add(row.recordingId)
        recommendations.append({
            "recordingId": row.recordingId,
            "filename": filename_map.get(row.recordingId),
            "is_speech": is_speech_map.get(row.recordingId),
            "recommended_because": row.recommended_because,
            "clap_sim": float(row.clap_sim),
            "yamnet_overlap": None if pd.isna(row.yamnet_overlap) else float(row.yamnet_overlap),
            "transcript_sim": None if pd.isna(row.transcript_sim) else float(row.transcript_sim),
            "z_clap": None if pd.isna(row.z_clap) else float(row.z_clap),
            "z_yamnet": None if pd.isna(row.z_yamnet) else float(row.z_yamnet),
            "z_transcript": None if pd.isna(row.z_transcript) else float(row.z_transcript),
            "score": float(row.score),
            "yamnet": top_yamnet_classes(row.recordingId, n=3),
            "transcript": transcript_map.get(row.recordingId) or None,
        })

    ratios_out.append({
        "non_speech_count": n_non_speech,
        "speech_count": n_speech,
        "label": f"{n_non_speech}/{n_speech}",
        "mock_history": [clip_summary(rid) for rid in mock_ids],
        "category_profiles": profile_summaries,
        "recommendations": recommendations,
        "pool_stats": pool_stats_by_category,
    })

payload = {
    "top_k": TOP_K,
    "candidate_pool": CANDIDATE_POOL,
    "ratios": ratios_out,
}

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUT_JSON.write_text(json.dumps(payload, indent=2))
print(f"wrote {OUT_JSON} ({len(ratios_out)} ratios, {len(referenced_audio_ids)} distinct audio clips referenced)")

print("Copying referenced audio files...")
PUBLIC_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
copied = 0
for rid in referenced_audio_ids:
    filename = filename_map.get(rid)
    if not filename:
        continue
    src = AUDIO_DIR / filename
    dst = PUBLIC_AUDIO_DIR / filename
    if src.exists() and not dst.exists():
        shutil.copy2(src, dst)
        copied += 1
print(f"copied {copied} new audio files to {PUBLIC_AUDIO_DIR} ({len(list(PUBLIC_AUDIO_DIR.glob('*')))} total)")
