import os
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import psycopg
import torch
from dotenv import load_dotenv
from silero_vad import (
    get_speech_timestamps,
    load_silero_vad,
)
from tqdm import tqdm

load_dotenv()


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# Include the schema if needed, for example:
# TABLE_NAME = "public.audio_records"
TABLE_NAME = os.getenv("TABLE_NAME")

# Name of the column containing the streamable MP3 URL.
AUDIO_URL_COLUMN = os.getenv("AUDIO_URL_COLUMN", "streamable_audio_url")

# Optional:
# Set this to a SQL condition such as:
# WHERE_CLAUSE = '"status" = \'Complete\''
#
# Leave it as None (or unset in .env) to process every row.
_where_clause_env = os.getenv("WHERE_CLAUSE", "").strip()
WHERE_CLAUSE = _where_clause_env if _where_clause_env else None

# Start with a small number such as 100.
# Set LIMIT= (empty) in .env after confirming everything works.
_limit_env = os.getenv("LIMIT", "100").strip()
LIMIT = int(_limit_env) if _limit_env else None


# ============================================================
# OUTPUT CONFIGURATION
# ============================================================

OUTPUT_CSV = os.getenv("OUTPUT_CSV", "audio_speech_labels.csv")

# Write results to disk after every N processed rows.
SAVE_EVERY = int(os.getenv("SAVE_EVERY", "10"))


# ============================================================
# SPEECH DETECTION CONFIGURATION
# ============================================================

SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))

# The recording is labelled as speech when at least this proportion
# of the recording contains speech.
SPEECH_RATIO_THRESHOLD = float(os.getenv("SPEECH_RATIO_THRESHOLD", "0.20"))

MIN_SPEECH_DURATION_MS = int(os.getenv("MIN_SPEECH_DURATION_MS", "250"))
MIN_SILENCE_DURATION_MS = int(os.getenv("MIN_SILENCE_DURATION_MS", "300"))
SPEECH_PAD_MS = int(os.getenv("SPEECH_PAD_MS", "100"))

DOWNLOAD_TIMEOUT_SECONDS = int(os.getenv("DOWNLOAD_TIMEOUT_SECONDS", "180"))

# Number of parallel workers. Each gets its own Silero model instance.
# Start with 8 and increase if your network allows it.
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "8"))


# ============================================================
# THREAD-LOCAL SILERO MODEL
# ============================================================

torch.set_num_threads(1)

_thread_local = threading.local()


def get_vad_model():
    if not hasattr(_thread_local, "model"):
        _thread_local.model = load_silero_vad()
    return _thread_local.model


# ============================================================
# DATABASE FUNCTIONS
# ============================================================

def connect_to_database():
    return psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


def load_database_rows() -> pd.DataFrame:
    query = f"SELECT * FROM {TABLE_NAME}"

    if WHERE_CLAUSE:
        query += f" WHERE {WHERE_CLAUSE}"

    if LIMIT is not None:
        query += f" LIMIT {int(LIMIT)}"

    print("Reading rows from PostgreSQL...")

    with connect_to_database() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)

            column_names = [
                description.name
                for description in cursor.description
            ]

            rows = cursor.fetchall()

    dataframe = pd.DataFrame(rows, columns=column_names)

    print(f"Loaded {len(dataframe)} rows.")

    return dataframe


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
# SILERO CLASSIFICATION
# ============================================================

def classify_audio(audio_url: str) -> dict:
    waveform = torch.from_numpy(load_audio_from_url(audio_url))

    total_duration_seconds = waveform.numel() / SAMPLE_RATE

    speech_segments = get_speech_timestamps(
        waveform,
        get_vad_model(),
        sampling_rate=SAMPLE_RATE,
        return_seconds=True,
        min_speech_duration_ms=MIN_SPEECH_DURATION_MS,
        min_silence_duration_ms=MIN_SILENCE_DURATION_MS,
        speech_pad_ms=SPEECH_PAD_MS,
    )

    speech_duration_seconds = sum(
        max(0.0, float(s["end"]) - float(s["start"]))
        for s in speech_segments
    )

    if total_duration_seconds > 0:
        speech_ratio = speech_duration_seconds / total_duration_seconds
    else:
        speech_ratio = 0.0

    speech_ratio = max(0.0, min(1.0, speech_ratio))

    return {
        "speech_label": "speech" if speech_ratio >= SPEECH_RATIO_THRESHOLD else "non_speech",
        "speech_ratio": speech_ratio,
        "speech_duration_seconds": speech_duration_seconds,
        "total_duration_seconds": total_duration_seconds,
        "speech_segment_count": len(speech_segments),
    }


# ============================================================
# ROW WORKER
# ============================================================

def process_row(row: dict) -> dict:
    result = dict(row)
    result.update(
        {
            "speech_label": None,
            "speech_ratio": None,
            "speech_duration_seconds": None,
            "total_duration_seconds": None,
            "speech_segment_count": None,
            "processing_status": None,
            "processing_error": None,
        }
    )

    audio_url = row.get(AUDIO_URL_COLUMN)

    try:
        if pd.isna(audio_url):
            raise ValueError("Audio URL is missing.")

        audio_url = str(audio_url).strip()

        if not audio_url:
            raise ValueError("Audio URL is empty.")

        result.update(classify_audio(audio_url))
        result["processing_status"] = "completed"

    except Exception as error:
        result["processing_status"] = "failed"
        result["processing_error"] = str(error)

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
    dataframe = load_database_rows()

    if dataframe.empty:
        print("No database rows found.")
        return

    if AUDIO_URL_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Column '{AUDIO_URL_COLUMN}' was not found.\n"
            f"Available columns: {list(dataframe.columns)}"
        )

    # Change this if your table does not use "id".
    id_column = "id"

    if id_column not in dataframe.columns:
        raise ValueError(
            f"Column '{id_column}' was not found.\n"
            f"Available columns: {list(dataframe.columns)}"
        )

    existing_results = load_existing_results()
    processed_ids = get_processed_row_identifiers(existing_results, id_column)

    rows_to_process = dataframe[
        ~dataframe[id_column].astype(str).isin(processed_ids)
    ]

    print(f"Already processed: {len(processed_ids)}")
    print(f"Remaining rows:    {len(rows_to_process)}")
    print(f"Workers:           {MAX_WORKERS}")

    new_results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_row, row.to_dict()): idx
            for idx, (_, row) in enumerate(rows_to_process.iterrows())
        }

        with tqdm(total=len(futures), desc="Processing audio", unit="audio") as progress:
            for future in as_completed(futures):
                new_results.append(future.result())
                progress.update(1)

                if len(new_results) % SAVE_EVERY == 0:
                    save_results(existing_results, new_results)

    save_results(existing_results, new_results)

    print(f"\nFinished processing.")
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
