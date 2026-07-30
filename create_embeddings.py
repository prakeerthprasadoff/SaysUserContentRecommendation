"""
Download audio (if needed), extract BEATs embeddings, build FAISS index.

Safe to re-run: skips already-embedded clips and saves progress after every batch.
Run from terminal (not Jupyter) to avoid kernel OOM crashes:

    python create_embeddings.py
"""

from __future__ import annotations

import gc
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import miniaudio
import numpy as np
import pandas as pd
import requests
import torch
from tqdm import tqdm

from beats_setup import load_beats_model

CSV_PATH = Path("audio_speech_labels.csv")
AUDIO_URL_COLUMN = "streamableUrl"
AUDIO_DIR = Path("audio_cache")
EMBEDDINGS_PATH = Path("embeddings.npy")
ID_MAP_PATH = Path("id_map.json")
INDEX_PATH = Path("nn_index.faiss")
PROGRESS_DIR = Path("embed_progress")

BATCH_SIZE = 4  # keep small on CPU to avoid OOM / kernel crashes
TARGET_SR = 16000
MAX_AUDIO_LENGTH_SEC = 30
DOWNLOAD_WORKERS = 8
DOWNLOAD_TIMEOUT = 60
EMBED_DIM = 768


def load_audio(filepath: str | Path, target_sr: int = TARGET_SR) -> torch.Tensor:
    decoded = miniaudio.decode_file(str(filepath), nchannels=1, sample_rate=target_sr)
    if not decoded.samples:
        raise ValueError("Empty waveform")
    return torch.tensor(decoded.samples, dtype=torch.float32)


def pad_or_trim(waveform: torch.Tensor, max_samples: int) -> torch.Tensor:
    if waveform.shape[0] > max_samples:
        return waveform[:max_samples]
    if waveform.shape[0] < max_samples:
        return torch.nn.functional.pad(waveform, (0, max_samples - waveform.shape[0]))
    return waveform


def download_audio(row: dict) -> tuple[str, str | None, str | None]:
    clip_id = row["id"]
    url = str(row[AUDIO_URL_COLUMN]).strip()
    ext = ".mp3" if ".mp3" in url else ".wav" if ".wav" in url else ".mp3"
    filepath = AUDIO_DIR / f"{clip_id}{ext}"

    if filepath.exists() and filepath.stat().st_size > 0:
        return clip_id, str(filepath), None

    try:
        resp = requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT)
        resp.raise_for_status()
        with open(filepath, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return clip_id, str(filepath), None
    except Exception as e:
        return clip_id, None, f"{type(e).__name__}: {e}"


def load_progress() -> tuple[dict[str, np.ndarray], list[str]]:
    """Load previously saved per-clip embeddings from embed_progress/."""
    PROGRESS_DIR.mkdir(exist_ok=True)
    done: dict[str, np.ndarray] = {}
    order: list[str] = []

    meta_path = PROGRESS_DIR / "done_ids.json"
    if meta_path.exists():
        with open(meta_path) as f:
            order = json.load(f)
        for clip_id in order:
            npy = PROGRESS_DIR / f"{clip_id}.npy"
            if npy.exists():
                done[clip_id] = np.load(npy)
            else:
                # incomplete — truncate order at first missing
                order = order[: order.index(clip_id)]
                break
    return done, order


def save_clip_embedding(clip_id: str, vec: np.ndarray, order: list[str]) -> None:
    PROGRESS_DIR.mkdir(exist_ok=True)
    np.save(PROGRESS_DIR / f"{clip_id}.npy", vec.astype(np.float32))
    if clip_id not in order:
        order.append(clip_id)
    with open(PROGRESS_DIR / "done_ids.json", "w") as f:
        json.dump(order, f)


def main() -> None:
    AUDIO_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | batch_size={BATCH_SIZE}")

    df = pd.read_csv(CSV_PATH)
    df = df[
        (df["speech_label"] == "non_speech")
        & (df["processing_status"] == "completed")
        & (df[AUDIO_URL_COLUMN].notna())
        & (df[AUDIO_URL_COLUMN].astype(str).str.strip() != "")
    ].reset_index(drop=True)
    print(f"Clips to process: {len(df)}")

    # Download
    downloaded: dict[str, str] = {}
    rows = df.to_dict("records")
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as ex:
        futures = {ex.submit(download_audio, row): row["id"] for row in rows}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Download"):
            clip_id, filepath, err = fut.result()
            if filepath:
                downloaded[clip_id] = filepath

    print(f"Downloaded/cached: {len(downloaded)}")

    # Collect valid clips (existence + decodable)
    valid: list[tuple[str, str]] = []
    for clip_id, filepath in tqdm(downloaded.items(), desc="Validate"):
        try:
            load_audio(filepath)
            valid.append((clip_id, filepath))
        except Exception:
            continue
    print(f"Valid clips: {len(valid)}")

    # Resume from progress
    done, order = load_progress()
    remaining = [(cid, fp) for cid, fp in valid if cid not in done]
    print(f"Already embedded: {len(done)} | Remaining: {len(remaining)}")

    if remaining:
        print("Loading BEATs...")
        model = load_beats_model(device)
        max_samples = TARGET_SR * MAX_AUDIO_LENGTH_SEC

        with torch.no_grad():
            for i in tqdm(range(0, len(remaining), BATCH_SIZE), desc="Embed"):
                batch = remaining[i : i + BATCH_SIZE]
                waveforms = []
                batch_ids = []
                for clip_id, filepath in batch:
                    try:
                        w = pad_or_trim(load_audio(filepath), max_samples)
                        waveforms.append(w)
                        batch_ids.append(clip_id)
                    except Exception as e:
                        print(f"  skip {clip_id}: {e}")

                if not waveforms:
                    continue

                batch_w = torch.stack(waveforms).to(device)
                padding_mask = torch.zeros_like(batch_w, dtype=torch.bool)
                features = model.extract_features(batch_w, padding_mask=padding_mask)[0]
                pooled = features.mean(dim=1).cpu().numpy().astype(np.float32)

                for clip_id, vec in zip(batch_ids, pooled):
                    save_clip_embedding(clip_id, vec, order)
                    done[clip_id] = vec

                del batch_w, features, pooled, waveforms
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        del model
        gc.collect()

    # Assemble final artifacts in stable order (progress order, then any leftover)
    final_ids = [cid for cid in order if cid in done]
    for cid, _ in valid:
        if cid not in done:
            continue
        if cid not in final_ids:
            final_ids.append(cid)

    all_embeddings = np.vstack([done[cid] for cid in final_ids]).astype(np.float32)
    np.save(EMBEDDINGS_PATH, all_embeddings)
    with open(ID_MAP_PATH, "w") as f:
        json.dump(final_ids, f)
    print(f"Saved {EMBEDDINGS_PATH} shape={all_embeddings.shape}")

    # FAISS index
    try:
        import faiss
    except ImportError:
        print("faiss not installed; skipping index build")
        return

    norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    emb_n = (all_embeddings / norms).astype(np.float32)
    index = faiss.IndexFlatIP(emb_n.shape[1])
    index.add(emb_n)
    faiss.write_index(index, str(INDEX_PATH))
    print(f"Saved {INDEX_PATH} ({index.ntotal} vectors)")
    print("Done. Open recommend.ipynb for recommendations.")


if __name__ == "__main__":
    main()
