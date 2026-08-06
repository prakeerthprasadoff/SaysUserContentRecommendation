"""
Check streamable audio URLs and download reachable files into audio_cache/.

Safe to re-run: skips files that are already present and non-empty.

    python download_streamable_audio.py
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

CSV_PATH = Path(os.getenv("CSV_PATH", "audio_speech_labels.csv"))
AUDIO_URL_COLUMN = os.getenv("AUDIO_URL_COLUMN", "streamableUrl")
ID_COLUMN = os.getenv("ID_COLUMN", "id")
AUDIO_CACHE_DIR = Path(os.getenv("AUDIO_CACHE_DIR", "audio_cache"))

DOWNLOAD_TIMEOUT_SECONDS = int(os.getenv("DOWNLOAD_TIMEOUT_SECONDS", "60"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "8"))
MIN_FILE_BYTES = int(os.getenv("MIN_FILE_BYTES", "1024"))

# Optional: only download rows with this processing_status (e.g. "completed").
# Leave empty to attempt every row that has a streamable URL.
_status_filter = os.getenv("STATUS_FILTER", "").strip()
STATUS_FILTER = _status_filter if _status_filter else None

_limit_env = os.getenv("LIMIT", "").strip()
LIMIT = int(_limit_env) if _limit_env else None

DOWNLOAD_LOG_CSV = Path(os.getenv("DOWNLOAD_LOG_CSV", "audio_download_log.csv"))


# ============================================================
# HELPERS
# ============================================================

def cache_path_for(row_id: str, url: str) -> Path:
    suffix = Path(urlparse(url).path).suffix.lower() or ".mp3"
    if suffix not in {".mp3", ".m4a", ".wav", ".ogg", ".aac", ".flac"}:
        suffix = ".mp3"
    return AUDIO_CACHE_DIR / f"{row_id}{suffix}"


def is_cached(path: Path) -> bool:
    return path.exists() and path.stat().st_size >= MIN_FILE_BYTES


def check_streamable_url(url: str, timeout: int = DOWNLOAD_TIMEOUT_SECONDS) -> None:
    """Raise if the streamable URL is not reachable / not an audio response."""
    try:
        response = requests.head(
            url,
            timeout=timeout,
            allow_redirects=True,
        )
        # Some S3 buckets reject HEAD; fall through to a ranged GET.
        if response.status_code in {403, 405}:
            response = requests.get(
                url,
                timeout=timeout,
                stream=True,
                headers={"Range": "bytes=0-0"},
            )
    except requests.RequestException as error:
        raise RuntimeError(f"URL check failed: {error}") from error

    if response.status_code >= 400:
        raise RuntimeError(
            f"URL not reachable (HTTP {response.status_code})."
        )

    content_type = (response.headers.get("Content-Type") or "").lower()
    if content_type and not (
        content_type.startswith("audio/")
        or content_type in {"application/octet-stream", "binary/octet-stream"}
        or "mpeg" in content_type
    ):
        # Soft warning only — S3 sometimes omits / mislabels Content-Type.
        pass

    response.close()


def download_audio(url: str, destination: Path) -> int:
    temporary_path = destination.with_suffix(destination.suffix + ".partial")

    try:
        with requests.get(
            url,
            timeout=DOWNLOAD_TIMEOUT_SECONDS,
            stream=True,
        ) as response:
            if response.status_code >= 400:
                raise RuntimeError(
                    f"Download failed (HTTP {response.status_code})."
                )

            with open(temporary_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        handle.write(chunk)

        size = temporary_path.stat().st_size
        if size < MIN_FILE_BYTES:
            raise RuntimeError(
                f"Downloaded file too small ({size} bytes)."
            )

        temporary_path.replace(destination)
        return size

    finally:
        if temporary_path.exists():
            temporary_path.unlink(missing_ok=True)


def process_row(row: dict) -> dict:
    row_id = str(row[ID_COLUMN])
    url_value = row.get(AUDIO_URL_COLUMN)

    result = {
        "id": row_id,
        "streamable_url": None,
        "local_path": None,
        "bytes": None,
        "download_status": None,
        "download_error": None,
    }

    try:
        if pd.isna(url_value):
            raise ValueError("Streamable URL is missing.")

        url = str(url_value).strip()
        if not url:
            raise ValueError("Streamable URL is empty.")

        result["streamable_url"] = url
        destination = cache_path_for(row_id, url)
        result["local_path"] = str(destination)

        if is_cached(destination):
            result["bytes"] = destination.stat().st_size
            result["download_status"] = "skipped_cached"
            return result

        check_streamable_url(url)
        size = download_audio(url, destination)
        result["bytes"] = size
        result["download_status"] = "downloaded"

    except Exception as error:
        result["download_status"] = "failed"
        result["download_error"] = str(error)

    return result


# ============================================================
# MAIN
# ============================================================

def load_rows() -> pd.DataFrame:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    dataframe = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(dataframe)} rows from {CSV_PATH}")

    if ID_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Column '{ID_COLUMN}' was not found.\n"
            f"Available columns: {list(dataframe.columns)}"
        )

    if AUDIO_URL_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Column '{AUDIO_URL_COLUMN}' was not found.\n"
            f"Available columns: {list(dataframe.columns)}"
        )

    if STATUS_FILTER and "processing_status" in dataframe.columns:
        before = len(dataframe)
        dataframe = dataframe[
            dataframe["processing_status"].astype(str) == STATUS_FILTER
        ]
        print(
            f"STATUS_FILTER={STATUS_FILTER!r}: "
            f"{len(dataframe)} / {before} rows kept."
        )

    # Drop rows with no streamable URL up front.
    dataframe = dataframe[
        dataframe[AUDIO_URL_COLUMN].notna()
        & (dataframe[AUDIO_URL_COLUMN].astype(str).str.strip() != "")
        & (dataframe[AUDIO_URL_COLUMN].astype(str).str.strip().str.lower() != "nan")
    ].copy()

    if LIMIT is not None:
        dataframe = dataframe.head(LIMIT)

    return dataframe


def main() -> None:
    AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    dataframe = load_rows()

    if dataframe.empty:
        print("No rows with streamable URLs to download.")
        return

    already_cached = 0
    pending_rows = []

    for _, row in dataframe.iterrows():
        row_dict = row.to_dict()
        url = str(row_dict[AUDIO_URL_COLUMN]).strip()
        destination = cache_path_for(str(row_dict[ID_COLUMN]), url)
        if is_cached(destination):
            already_cached += 1
        else:
            pending_rows.append(row_dict)

    print(f"Already cached: {already_cached}")
    print(f"To download:    {len(pending_rows)}")
    print(f"Workers:        {MAX_WORKERS}")
    print(f"Cache dir:      {AUDIO_CACHE_DIR.resolve()}")

    results: list[dict] = []

    # Record already-cached rows for the log.
    for _, row in dataframe.iterrows():
        row_dict = row.to_dict()
        url = str(row_dict[AUDIO_URL_COLUMN]).strip()
        destination = cache_path_for(str(row_dict[ID_COLUMN]), url)
        if is_cached(destination):
            results.append(
                {
                    "id": str(row_dict[ID_COLUMN]),
                    "streamable_url": url,
                    "local_path": str(destination),
                    "bytes": destination.stat().st_size,
                    "download_status": "skipped_cached",
                    "download_error": None,
                }
            )

    if pending_rows:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(process_row, row): row[ID_COLUMN]
                for row in pending_rows
            }

            with tqdm(
                total=len(futures),
                desc="Downloading audio",
                unit="file",
            ) as progress:
                for future in as_completed(futures):
                    results.append(future.result())
                    progress.update(1)

    results_df = pd.DataFrame(results)
    results_df.to_csv(DOWNLOAD_LOG_CSV, index=False)

    counts = results_df["download_status"].value_counts().to_dict()
    print("\nFinished.")
    print(f"Status counts: {counts}")
    print(f"Files in cache: {sum(1 for p in AUDIO_CACHE_DIR.iterdir() if p.is_file())}")
    print(f"Log saved to:   {DOWNLOAD_LOG_CSV}")


if __name__ == "__main__":
    main()
