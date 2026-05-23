"""Model downloader for sherpa-onnx speaker diarization models.

Downloads and caches the required ONNX models to ~/.cache/meetandread/diarization-models/:
  - pyannote-segmentation-3.0 (speaker segmentation)
  - 3dspeaker CAM++ / eres2net (speaker embedding extraction)

Model files are verified by existence; re-downloads are skipped if present.
"""

import logging
import shutil
import tarfile
import urllib.request
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Base URL for sherpa-onnx model releases
SHERPA_BASE_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download"
)

# Segmentation model (pyannote-segmentation-3.0)
SEGMENTATION_MODEL_NAME = "sherpa-onnx-pyannote-segmentation-3-0"
SEGMENTATION_TARBALL = f"{SEGMENTATION_MODEL_NAME}.tar.bz2"
SEGMENTATION_URL = (
    f"{SHERPA_BASE_URL}/speaker-segmentation-models/{SEGMENTATION_TARBALL}"
)

# Speaker embedding model (3D-Speaker eres2net, 16kHz)
EMBEDDING_MODEL_NAME = "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
EMBEDDING_URL = (
    f"{SHERPA_BASE_URL}/speaker-recongition-models/{EMBEDDING_MODEL_NAME}"
)

# Default cache directory
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "meetandread" / "diarization-models"


def get_cache_dir(cache_dir: Optional[Path] = None) -> Path:
    """Return the model cache directory, creating it if needed.

    Args:
        cache_dir: Override the default cache location. If None, uses
            ~/.cache/meetandread/diarization-models/.

    Returns:
        Path to the cache directory.
    """
    directory = cache_dir or DEFAULT_CACHE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _download_file(url: str, dest: Path, label: str = "") -> None:
    """Download a file from *url* to *dest* with progress logging.

    Args:
        url: Remote URL to download.
        dest: Local destination path.
        label: Human-readable label for log messages.
    """
    if dest.exists():
        logger.debug("Already cached: %s (%s)", label or dest.name, dest)
        return

    logger.info("Downloading %s from %s", label or dest.name, url)
    tmp_dest = dest.with_suffix(dest.suffix + ".tmp")
    try:
        urllib.request.urlretrieve(url, tmp_dest)  # nosec B310
        shutil.move(str(tmp_dest), str(dest))
        logger.info("Downloaded %s (%.1f MB)", label or dest.name, dest.stat().st_size / 1e6)
    except Exception:
        # Clean up partial download on failure
        if tmp_dest.exists():
            tmp_dest.unlink()
        raise


def ensure_segmentation_model(cache_dir: Optional[Path] = None) -> Path:
    """Download and extract the pyannote segmentation model if not cached.

    Returns the directory containing model.onnx and model.int8.onnx.
    """
    cache = get_cache_dir(cache_dir)
    model_dir = cache / SEGMENTATION_MODEL_NAME

    if model_dir.is_dir() and (model_dir / "model.onnx").exists():
        logger.info(
            "Segmentation model already cached at %s", model_dir
        )
        return model_dir

    tarball_path = cache / SEGMENTATION_TARBALL
    _download_file(SEGMENTATION_URL, tarball_path, label="segmentation model tarball")

    logger.info("Extracting segmentation model to %s", cache)
    with tarfile.open(str(tarball_path), "r:bz2") as tar:
        # Validate members before extraction to prevent path traversal (B202)
        for member in tar.getmembers():
            if member.name.startswith("/") or ".." in member.name:
                raise ValueError(f"Unsafe tar member: {member.name}")
        tar.extractall(path=str(cache))  # nosec B202
    tarball_path.unlink()

    model_onnx = model_dir / "model.onnx"
    if not model_onnx.exists():
        raise FileNotFoundError(
            f"Segmentation model not found after extraction: {model_onnx}"
        )
    logger.info(
        "Segmentation model ready (%.1f MB)",
        model_onnx.stat().st_size / 1e6,
    )
    return model_dir


def ensure_embedding_model(cache_dir: Optional[Path] = None) -> Path:
    """Download the 3D-Speaker embedding model if not cached.

    Returns the path to the .onnx file.
    """
    cache = get_cache_dir(cache_dir)
    model_path = cache / EMBEDDING_MODEL_NAME

    _download_file(EMBEDDING_URL, model_path, label="embedding model")
    return model_path


def ensure_all_models(cache_dir: Optional[Path] = None) -> dict:
    """Download all required diarization models.

    Returns a dict with keys:
        segmentation_dir: Path to the segmentation model directory
        embedding_model:  Path to the embedding .onnx file
    """
    logger.info("Ensuring all speaker diarization models are available…")
    seg_dir = ensure_segmentation_model(cache_dir)
    emb_path = ensure_embedding_model(cache_dir)
    logger.info("All speaker diarization models ready.")
    return {
        "segmentation_dir": seg_dir,
        "embedding_model": emb_path,
    }
