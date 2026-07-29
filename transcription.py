import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from tqdm import tqdm

load_dotenv()


# ============================================================
# INPUT / OUTPUT CONFIGURATION
# ============================================================

# Output CSV produced by audio_download_and_categorize.py.
INPUT_CSV = os.getenv("SPEECH_LABELS_CSV", "audio_speech_labels.csv")

OUTPUT_CSV = os.getenv("TRANSCRIPTION_OUTPUT_CSV", "audio_transcriptions.csv")

AUDIO_URL_COLUMN = os.getenv("AUDIO_URL_COLUMN", "streamableUrl")
ID_COLUMN = os.getenv("ID_COLUMN", "id")

SAMPLE_RATE = 16000  # faster-whisper expects 16kHz mono audio.

DOWNLOAD_TIMEOUT_SECONDS = int(os.getenv("DOWNLOAD_TIMEOUT_SECONDS", "180"))

# Write results to disk after every N processed rows.
SAVE_EVERY = int(os.getenv("SAVE_EVERY", "10"))


# ============================================================
# WHISPER CONFIGURATION
# ============================================================

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "large-v3")

# Defaults below are set for a CUDA GPU box (e.g. an H100). Override via
# .env if you ever need to run this on CPU instead.
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")

# Leave unset to let Whisper auto-detect the spoken language.
_language_env = os.getenv("WHISPER_LANGUAGE", "").strip()
WHISPER_LANGUAGE = _language_env if _language_env else None

# Threads CTranslate2 uses per inference stream when WHISPER_DEVICE=cpu.
# 0 lets CTranslate2 pick automatically. Ignored on cuda.
WHISPER_CPU_THREADS = int(os.getenv("WHISPER_CPU_THREADS", "0"))

# Parallel inference streams inside the (single, shared) model -- this is
# what faster-whisper/CTranslate2 uses to safely serve transcribe() calls
# from multiple Python threads at once. Raise this to keep the GPU busier;
# it is independent of MAX_WORKERS below. Watch `nvidia-smi` and raise
# further if utilization is low.
WHISPER_NUM_WORKERS = int(os.getenv("WHISPER_NUM_WORKERS", "4"))

# Rows processed concurrently. Each one downloads/decodes audio via
# ffmpeg (network + CPU bound) and then calls the shared model above.
# Since the model itself now handles its own internal parallelism via
# WHISPER_NUM_WORKERS, this mostly controls download concurrency --
# keep it >= WHISPER_NUM_WORKERS so the model always has work queued.
# Set to match the 16 cores available for ffmpeg decoding.
MAX_WORKERS = int(os.getenv("TRANSCRIBE_MAX_WORKERS", "16"))


# ============================================================
# SHARED WHISPER MODEL
# ============================================================
#
# A single WhisperModel instance is created once (before any worker
# threads start) and shared across all of them. faster-whisper is
# explicitly designed to be called concurrently from multiple threads
# this way -- see the `num_workers` parameter -- so we get parallelism
# without paying for N separate model copies in memory.

_model: WhisperModel | None = None


def build_whisper_model() -> WhisperModel:
    return WhisperModel(
        WHISPER_MODEL_SIZE,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE,
        cpu_threads=WHISPER_CPU_THREADS,
        num_workers=WHISPER_NUM_WORKERS,
    )


def get_whisper_model() -> WhisperModel:
    assert _model is not None, "build_whisper_model() must run before workers start."
    return _model


# ============================================================
# AUDIO LOADING (ffmpeg)
# ============================================================

def load_audio_from_url(
    url,
    target_sample_rate=SAMPLE_RATE,
    timeout_seconds=DOWNLOAD_TIMEOUT_SECONDS,
):
    file_descriptor, temporary_path = tempfile.mkstemp(suffix=".f32")
    os.close(file_descriptor)

    command = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-rw_timeout", str(timeout_seconds * 1_000_000),
        "-i", str(url),
        "-vn",
        "-ac", "1",
        "-ar", str(target_sample_rate),
        "-f", "f32le",
        temporary_path,
    ]

    try:
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
            f"Audio loading exceeded {timeout_seconds} seconds."
        ) from error

    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


# ============================================================
# WHISPER TRANSCRIPTION
# ============================================================

def transcribe_audio(audio_url: str) -> dict:
    waveform = load_audio_from_url(audio_url)

    segments, info = get_whisper_model().transcribe(
        waveform,
        language=WHISPER_LANGUAGE,
    )

    segments = list(segments)
    transcript = " ".join(segment.text.strip() for segment in segments).strip()

    return {
        "transcript": transcript,
        "language": info.language,
        "language_probability": info.language_probability,
        "audio_duration_seconds": info.duration,
        "segment_count": len(segments),
    }


# ============================================================
# ROW WORKER
# ============================================================

def process_row(row: dict) -> dict:
    result = dict(row)
    result.update(
        {
            "transcript": None,
            "language": None,
            "language_probability": None,
            "audio_duration_seconds": None,
            "segment_count": None,
            "transcription_status": None,
            "transcription_error": None,
        }
    )

    audio_url = row.get(AUDIO_URL_COLUMN)

    try:
        if pd.isna(audio_url):
            raise ValueError("Audio URL is missing.")

        audio_url = str(audio_url).strip()

        if not audio_url:
            raise ValueError("Audio URL is empty.")

        result.update(transcribe_audio(audio_url))
        result["transcription_status"] = "completed"

    except Exception as error:
        result["transcription_status"] = "failed"
        result["transcription_error"] = str(error)

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

    if "speech_label" not in dataframe.columns:
        raise ValueError(
            f"Column 'speech_label' was not found in {INPUT_CSV}.\n"
            f"Available columns: {list(dataframe.columns)}"
        )

    dataframe = dataframe[dataframe["speech_label"] == "speech"]

    if dataframe.empty:
        print("No rows labelled 'speech' found.")
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

    print(f"Speech-labelled rows: {len(dataframe)}")
    print(f"Already transcribed:  {len(processed_ids)}")
    print(f"Remaining rows:       {len(rows_to_process)}")
    print(f"Download workers:     {MAX_WORKERS}")
    print(f"Inference workers:    {WHISPER_NUM_WORKERS}")
    print(f"Model:                {WHISPER_MODEL_SIZE} ({WHISPER_DEVICE}, {WHISPER_COMPUTE_TYPE})")

    global _model
    _model = build_whisper_model()

    new_results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_row, row.to_dict()): idx
            for idx, (_, row) in enumerate(rows_to_process.iterrows())
        }

        with tqdm(total=len(futures), desc="Transcribing audio", unit="audio") as progress:
            for future in as_completed(futures):
                new_results.append(future.result())
                progress.update(1)

                if len(new_results) % SAVE_EVERY == 0:
                    save_results(existing_results, new_results)

    save_results(existing_results, new_results)

    print(f"\nFinished transcribing.")
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
