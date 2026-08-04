import os
import subprocess
import tempfile
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from transformers import ClapModel, ClapProcessor
from tqdm import tqdm

load_dotenv()


# ============================================================
# INPUT / OUTPUT CONFIGURATION
# ============================================================

# Output CSV produced by audio_download_and_categorize.py. This pipeline
# runs over ALL rows (speech and non_speech alike) -- CLAP is the universal
# acoustic signal that works across music, sound effects, ambience, and
# speech, unlike the transcript embeddings which only apply to speech.
INPUT_CSV = os.getenv("SPEECH_LABELS_CSV", "audio_speech_labels.csv")

OUTPUT_CSV = os.getenv("CLAP_OUTPUT_CSV", "audio_clap_embeddings.csv")

AUDIO_URL_COLUMN = os.getenv("AUDIO_URL_COLUMN", "streamableUrl")
ID_COLUMN = os.getenv("ID_COLUMN", "id")

DOWNLOAD_TIMEOUT_SECONDS = int(os.getenv("DOWNLOAD_TIMEOUT_SECONDS", "180"))

# Write results to disk after every N processed rows.
SAVE_EVERY = int(os.getenv("SAVE_EVERY", "10"))

# Rows processed concurrently. Each one downloads/decodes audio via ffmpeg
# (network + CPU bound); GPU inference itself is serialized (see GPU lock
# below), so this mostly controls download concurrency.
MAX_WORKERS = int(os.getenv("CLAP_MAX_WORKERS", "16"))


# ============================================================
# CLAP CONFIGURATION
# ============================================================

CLAP_MODEL_NAME = os.getenv("CLAP_MODEL_NAME", "laion/clap-htsat-unfused")

# Defaults below are set for a CUDA GPU box (e.g. an H100). Override via
# .env if you ever need to run this on CPU instead.
CLAP_DEVICE = os.getenv("CLAP_DEVICE", "cuda")

# CLAP's HTSAT audio encoder expects 48kHz mono audio in fixed ~10s windows
# (repeat-padded if shorter). These come from the model's own feature
# extractor config, not an arbitrary choice.
CLAP_SAMPLE_RATE = 48000
CLAP_WINDOW_SECONDS = 10
CLAP_WINDOW_SAMPLES = CLAP_SAMPLE_RATE * CLAP_WINDOW_SECONDS

# Drop a trailing window shorter than this many seconds instead of feeding
# CLAP a window that's almost entirely repeat-padding.
MIN_TRAILING_WINDOW_SECONDS = float(os.getenv("CLAP_MIN_TRAILING_WINDOW_SECONDS", "1.0"))


# ============================================================
# SHARED CLAP MODEL
# ============================================================
#
# One model + processor, built once before any worker threads start.
# Unlike faster-whisper/CTranslate2, transformers models don't document an
# explicit "safe for concurrent forward passes" contract, so GPU inference
# itself is serialized behind a lock. The actual parallelism win comes from
# letting many threads download/decode/window audio concurrently while only
# one thread touches the GPU at a time -- download was already the measured
# bottleneck in the VAD/transcription stages of this pipeline.

_model: ClapModel | None = None
_processor: ClapProcessor | None = None
_gpu_lock = threading.Lock()


def build_clap_model() -> tuple[ClapModel, ClapProcessor]:
    processor = ClapProcessor.from_pretrained(CLAP_MODEL_NAME)
    model = ClapModel.from_pretrained(CLAP_MODEL_NAME)
    model.to(CLAP_DEVICE)
    model.eval()
    return model, processor


def get_clap_model() -> tuple[ClapModel, ClapProcessor]:
    assert _model is not None and _processor is not None, (
        "build_clap_model() must run before workers start."
    )
    return _model, _processor


# ============================================================
# AUDIO LOADING (download in Python, decode with ffmpeg locally)
# ============================================================
#
# ffmpeg is asked to decode a local temp file, never the remote URL
# directly. Some HPC/module ffmpeg builds are compiled without HTTPS
# protocol support ("Protocol not found" for https:// inputs) -- fetching
# the bytes in Python sidesteps that dependency entirely and works the
# same regardless of which ffmpeg build happens to be on PATH.

def download_to_temp_file(url: str, timeout_seconds: int) -> str:
    file_descriptor, download_path = tempfile.mkstemp(suffix=".download")
    os.close(file_descriptor)

    try:
        request = urllib.request.Request(url, headers={"User-Agent": "curl/8.0"})
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            with open(download_path, "wb") as f:
                f.write(response.read())
        return download_path
    except Exception:
        if os.path.exists(download_path):
            os.remove(download_path)
        raise


def load_audio_from_url(
    url,
    target_sample_rate=CLAP_SAMPLE_RATE,
    timeout_seconds=DOWNLOAD_TIMEOUT_SECONDS,
):
    download_path = None
    file_descriptor, temporary_path = tempfile.mkstemp(suffix=".f32")
    os.close(file_descriptor)

    try:
        try:
            download_path = download_to_temp_file(url, timeout_seconds)
        except urllib.error.URLError as error:
            raise RuntimeError(f"Failed to download audio: {error}") from error
        except TimeoutError as error:
            raise TimeoutError(
                f"Audio download exceeded {timeout_seconds} seconds."
            ) from error

        command = [
            "ffmpeg",
            "-y",
            "-loglevel", "error",
            "-i", download_path,
            "-vn",
            "-ac", "1",
            "-ar", str(target_sample_rate),
            "-f", "f32le",
            temporary_path,
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
        )

        if result.returncode != 0:
            error_message = result.stderr.decode("utf-8", errors="ignore").strip()
            raise RuntimeError(error_message or "ffmpeg failed to decode the audio.")

        waveform = np.fromfile(temporary_path, dtype=np.float32)

        if waveform.size == 0:
            raise ValueError("The decoded waveform is empty.")

        waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)

        return waveform

    except subprocess.TimeoutExpired as error:
        raise TimeoutError(
            f"Audio decoding exceeded {timeout_seconds} seconds."
        ) from error

    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
        if download_path and os.path.exists(download_path):
            os.remove(download_path)


# ============================================================
# WINDOWING + EMBEDDING
# ============================================================

def split_into_windows(waveform: np.ndarray) -> list[np.ndarray]:
    """
    Split a waveform into consecutive, non-overlapping ~10s windows matching
    CLAP's native window size. A short trailing window is kept as-is (CLAP's
    feature extractor repeat-pads it) unless it's too short to be meaningful.
    """
    min_trailing_samples = int(MIN_TRAILING_WINDOW_SECONDS * CLAP_SAMPLE_RATE)

    windows = []
    for start in range(0, len(waveform), CLAP_WINDOW_SAMPLES):
        chunk = waveform[start:start + CLAP_WINDOW_SAMPLES]

        if len(chunk) < min_trailing_samples and windows:
            # Too short to matter and we already have at least one window.
            continue

        windows.append(chunk)

    return windows


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def embed_audio_url(audio_url: str) -> np.ndarray:
    waveform = load_audio_from_url(audio_url)
    windows = split_into_windows(waveform)

    model, processor = get_clap_model()

    inputs = processor(
        audio=windows,
        sampling_rate=CLAP_SAMPLE_RATE,
        return_tensors="pt",
    ).to(CLAP_DEVICE)

    with _gpu_lock:
        with torch.no_grad():
            # get_audio_features returns a BaseModelOutputWithPooling; the
            # projected, L2-normalized 512-dim embedding is in pooler_output.
            window_embeddings = model.get_audio_features(**inputs).pooler_output

    window_embeddings = window_embeddings.detach().cpu().numpy().astype(np.float32)
    window_embeddings = np.stack([normalize_vector(v) for v in window_embeddings])

    pooled = window_embeddings.mean(axis=0)
    return normalize_vector(pooled)


def encode_vector(vector: np.ndarray) -> str:
    return ",".join(f"{value:.6f}" for value in vector)


def decode_vector(text: str) -> np.ndarray:
    return np.array(text.split(","), dtype=np.float32)


# ============================================================
# ROW WORKER
# ============================================================

def process_row(row: dict) -> dict:
    result = {
        ID_COLUMN: row.get(ID_COLUMN),
        "clap_embedding": None,
        "clap_status": None,
        "clap_error": None,
    }

    audio_url = row.get(AUDIO_URL_COLUMN)

    try:
        if pd.isna(audio_url):
            raise ValueError("Audio URL is missing.")

        audio_url = str(audio_url).strip()

        if not audio_url:
            raise ValueError("Audio URL is empty.")

        embedding = embed_audio_url(audio_url)
        result["clap_embedding"] = encode_vector(embedding)
        result["clap_status"] = "completed"

    except Exception as error:
        result["clap_status"] = "failed"
        result["clap_error"] = str(error)

    return result


# ============================================================
# RESUME SUPPORT
# ============================================================

def load_existing_results() -> pd.DataFrame:
    if not os.path.exists(OUTPUT_CSV):
        return pd.DataFrame()

    try:
        return pd.read_csv(OUTPUT_CSV)
    except Exception as error:
        print(f"Could not read existing CSV: {error}")
        return pd.DataFrame()


def get_processed_row_identifiers(
    existing_results: pd.DataFrame,
    id_column: str,
) -> set:
    if existing_results.empty or id_column not in existing_results.columns:
        return set()

    return set(existing_results[id_column].astype(str).tolist())


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    dataframe = pd.read_csv(INPUT_CSV)

    if "processing_status" in dataframe.columns:
        dataframe = dataframe[dataframe["processing_status"] == "completed"]

    if dataframe.empty:
        print("No rows to process.")
        return

    if AUDIO_URL_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Column '{AUDIO_URL_COLUMN}' was not found.\n"
            f"Available columns: {list(dataframe.columns)}"
        )

    if ID_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Column '{ID_COLUMN}' was not found.\n"
            f"Available columns: {list(dataframe.columns)}"
        )

    existing_results = load_existing_results()
    processed_ids = get_processed_row_identifiers(existing_results, ID_COLUMN)

    rows_to_process = dataframe[
        ~dataframe[ID_COLUMN].astype(str).isin(processed_ids)
    ]

    print(f"Total candidate rows:  {len(dataframe)}")
    print(f"Already embedded:      {len(processed_ids)}")
    print(f"Remaining rows:        {len(rows_to_process)}")
    print(f"Download workers:      {MAX_WORKERS}")
    print(f"Model:                 {CLAP_MODEL_NAME} ({CLAP_DEVICE})")

    global _model, _processor
    _model, _processor = build_clap_model()

    new_results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_row, row.to_dict()): idx
            for idx, (_, row) in enumerate(rows_to_process.iterrows())
        }

        with tqdm(total=len(futures), desc="Embedding audio", unit="clip") as progress:
            for future in as_completed(futures):
                new_results.append(future.result())
                progress.update(1)

                if len(new_results) % SAVE_EVERY == 0:
                    save_results(existing_results, new_results)

    save_results(existing_results, new_results)

    print(f"\nFinished embedding.")
    print(f"Results saved to: {OUTPUT_CSV}")


def save_results(
    existing_results: pd.DataFrame,
    new_results: list[dict],
) -> None:
    new_dataframe = pd.DataFrame(new_results)

    if existing_results.empty:
        combined_dataframe = new_dataframe
    else:
        combined_dataframe = pd.concat(
            [existing_results, new_dataframe],
            ignore_index=True,
        )

    combined_dataframe.to_csv(OUTPUT_CSV, index=False)


if __name__ == "__main__":
    main()
