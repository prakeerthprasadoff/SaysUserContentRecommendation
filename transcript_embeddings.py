import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()


# ============================================================
# INPUT / OUTPUT CONFIGURATION
# ============================================================

# Output CSV produced by transcription.py. Only speech-labeled recordings
# have transcripts, so this pipeline is inherently a subset of the full
# recording corpus (unlike clap_embeddings.py, which runs over everything).
INPUT_CSV = os.getenv("TRANSCRIPTS_CSV", "audio_transcriptions.csv")

OUTPUT_CSV = os.getenv("TRANSCRIPT_EMBEDDINGS_OUTPUT_CSV", "audio_transcript_embeddings.csv")

ID_COLUMN = os.getenv("ID_COLUMN", "id")


# ============================================================
# TEXT EMBEDDING CONFIGURATION
# ============================================================

# Same model family as the podcast recommender's description/title/genre/
# artist embeddings, for consistency if the two projects ever need to be
# compared or merged.
TEXT_MODEL_NAME = os.getenv("TEXT_EMBEDDING_MODEL", "all-mpnet-base-v2")

# Defaults below are set for a CUDA GPU box (e.g. an H100). Override via
# .env if you ever need to run this on CPU instead.
TEXT_EMBEDDING_DEVICE = os.getenv("TEXT_EMBEDDING_DEVICE", "cuda")

TEXT_EMBEDDING_BATCH_SIZE = int(os.getenv("TEXT_EMBEDDING_BATCH_SIZE", "64"))


# ============================================================
# VECTOR ENCODING (matches clap_embeddings.py's CSV format)
# ============================================================

def encode_vector(vector: np.ndarray) -> str:
    return ",".join(f"{value:.6f}" for value in vector)


def decode_vector(text: str) -> np.ndarray:
    return np.array(text.split(","), dtype=np.float32)


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    dataframe = pd.read_csv(INPUT_CSV)

    if "transcription_status" in dataframe.columns:
        dataframe = dataframe[dataframe["transcription_status"] == "completed"]

    if ID_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Column '{ID_COLUMN}' was not found.\n"
            f"Available columns: {list(dataframe.columns)}"
        )

    if "transcript" not in dataframe.columns:
        raise ValueError(
            f"Column 'transcript' was not found.\n"
            f"Available columns: {list(dataframe.columns)}"
        )

    dataframe["transcript"] = dataframe["transcript"].fillna("").astype(str).str.strip()
    dataframe = dataframe[dataframe["transcript"] != ""]

    if dataframe.empty:
        print("No non-empty transcripts found.")
        return

    print(f"Transcripts to embed: {len(dataframe)}")
    print(f"Model:                {TEXT_MODEL_NAME} ({TEXT_EMBEDDING_DEVICE})")

    model = SentenceTransformer(TEXT_MODEL_NAME, device=TEXT_EMBEDDING_DEVICE)

    embeddings = model.encode(
        dataframe["transcript"].tolist(),
        batch_size=TEXT_EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    result = pd.DataFrame({
        ID_COLUMN: dataframe[ID_COLUMN].values,
        "transcript_embedding": [encode_vector(vec) for vec in embeddings],
    })

    result.to_csv(OUTPUT_CSV, index=False)

    print(f"\nFinished embedding.")
    print(f"Results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
