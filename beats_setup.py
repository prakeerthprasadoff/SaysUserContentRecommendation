"""Download Microsoft BEATs source/checkpoint and load the model."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import torch

BEATS_BASE_URL = "https://raw.githubusercontent.com/microsoft/unilm/master/beats"
CHECKPOINT_URL = (
    "https://huggingface.co/lpepino/beats_ckpts/resolve/main/BEATs_iter3_plus_AS2M.pt"
)
CHECKPOINT_MIN_BYTES = 300_000_000  # ~361 MB expected

BEATS_REPO_DIR = Path(__file__).resolve().parent / "unilm" / "beats"
BEATS_CHECKPOINT = Path(__file__).resolve().parent / "BEATs_iter3_plus_AS2M.pt"

BEATS_SOURCE_FILES = ("BEATs.py", "backbone.py", "modules.py")


def _download_file(url: str, dest: Path, min_bytes: int = 1) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size >= min_bytes:
        return
    if dest.exists():
        dest.unlink()
    print(f"Downloading {dest.name}...")
    urllib.request.urlretrieve(url, dest)


def ensure_beats_ready(
    repo_dir: Path = BEATS_REPO_DIR,
    checkpoint_path: Path = BEATS_CHECKPOINT,
) -> tuple[Path, Path]:
    """Download BEATs source files and checkpoint if missing."""
    for filename in BEATS_SOURCE_FILES:
        _download_file(f"{BEATS_BASE_URL}/{filename}", repo_dir / filename)

    if not checkpoint_path.exists() or checkpoint_path.stat().st_size == 0:
        print("Downloading BEATs checkpoint (~360 MB)...")
        _download_file(CHECKPOINT_URL, checkpoint_path, min_bytes=CHECKPOINT_MIN_BYTES)

    return repo_dir, checkpoint_path


def load_beats_model(
    device: torch.device | str = "cpu",
    repo_dir: Path = BEATS_REPO_DIR,
    checkpoint_path: Path = BEATS_CHECKPOINT,
):
    """Ensure assets exist, then return a BEATs model in eval mode."""
    repo_dir, checkpoint_path = ensure_beats_ready(repo_dir, checkpoint_path)

    repo_str = str(repo_dir.resolve())
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    from BEATs import BEATs, BEATsConfig

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = BEATsConfig(checkpoint["cfg"])
    model = BEATs(cfg)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()
    return model
