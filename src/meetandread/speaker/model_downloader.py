"""Model downloader for sherpa-onnx speaker diarization models.

Downloads and caches the required ONNX models to ~/.cache/meetandread/diarization-models/:
  - pyannote-segmentation-3.0 (speaker segmentation)
  - 3dspeaker CAM++ / eres2net (speaker embedding extraction)

Model files are verified by SHA256 checksum; corrupted or truncated artifacts are
deleted and re-downloaded automatically.
"""

import hashlib
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

# SHA256 checksums for integrity verification (lowercase hex digests)
EMBEDDING_SHA256 = (
    "1a331345f04805badbb495c775a6ddffcdd1a732567d5ec8b3d5749e3c7a5e4b"
)
SEGMENTATION_MODEL_ONNX_SHA256 = (
    "220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079"
)

# Chunk size for streaming SHA256 computation (64 KiB)
_CHECKSUM_CHUNK_SIZE = 65536

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


def sha256_checksum(path: Path) -> str:
    """Compute the SHA256 hex digest of *path* using streaming reads.

    Args:
        path: File to hash.

    Returns:
        Lowercase hex digest string.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_CHECKSUM_CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _verify_checksum(path: Path, expected: str, label: str = "") -> bool:
    """Verify the SHA256 checksum of *path* against *expected*.

    Args:
        path: File to verify.
        expected: Expected lowercase hex digest.
        label: Human-readable label for log messages.

    Returns:
        True if checksum matches, False otherwise.
    """
    actual = sha256_checksum(path)
    if actual == expected:
        logger.debug("Checksum verified for %s: %s", label or path.name, actual[:16])
        return True
    logger.warning(
        "Checksum mismatch for %s: expected %s, got %s",
        label or path.name,
        expected[:16],
        actual[:16],
    )
    return False


def _download_file(
    url: str,
    dest: Path,
    expected_sha256: Optional[str] = None,
    label: str = "",
) -> None:
    """Download a file from *url* to *dest* with checksum verification.

    If *dest* already exists and *expected_sha256* is provided, the cached
    file is verified.  A checksum mismatch triggers deletion and re-download.

    Downloads land in a temporary file first.  If verification is requested
    and fails after download, the temporary file is removed and an exception
    is raised.

    Args:
        url: Remote URL to download.
        dest: Local destination path.
        expected_sha256: Expected SHA256 hex digest, or None to skip
            checksum verification.
        label: Human-readable label for log messages.

    Raises:
        RuntimeError: If checksum verification fails after download.
    """
    # Check existing cache
    if dest.exists():
        if expected_sha256:
            if _verify_checksum(dest, expected_sha256, label or dest.name):
                logger.info("Cached %s verified, skipping download", label or dest.name)
                return
            # Corrupt cache — delete and re-download
            logger.warning(
                "Deleting corrupt cached file %s (checksum mismatch)",
                dest,
            )
            dest.unlink()
        else:
            logger.debug("Already cached: %s (%s)", label or dest.name, dest)
            return

    logger.info("Downloading %s from %s", label or dest.name, url)
    tmp_dest = dest.with_suffix(dest.suffix + ".tmp")
    try:
        urllib.request.urlretrieve(url, tmp_dest)  # nosec B310

        # Verify checksum of downloaded temp file before moving
        if expected_sha256:
            if not _verify_checksum(
                tmp_dest, expected_sha256, label or dest.name
            ):
                tmp_dest.unlink()
                raise RuntimeError(
                    f"Checksum verification failed for {label or dest.name} "
                    f"after download from {url}"
                )

        shutil.move(str(tmp_dest), str(dest))
        logger.info(
            "Downloaded %s (%.1f MB)",
            label or dest.name,
            dest.stat().st_size / 1e6,
        )
    except Exception:
        # Clean up partial download on failure
        if tmp_dest.exists():
            tmp_dest.unlink()
        raise


def ensure_segmentation_model(cache_dir: Optional[Path] = None) -> Path:
    """Download and extract the pyannote segmentation model if not cached.

    Verifies the SHA256 checksum of the extracted ``model.onnx``.  If the
    cached model fails verification it is deleted and re-downloaded.

    Returns the directory containing model.onnx and model.int8.onnx.
    """
    cache = get_cache_dir(cache_dir)
    model_dir = cache / SEGMENTATION_MODEL_NAME
    model_onnx = model_dir / "model.onnx"

    # Fast path: existing verified cache
    if model_onnx.exists():
        if _verify_checksum(
            model_onnx, SEGMENTATION_MODEL_ONNX_SHA256, "segmentation model.onnx"
        ):
            logger.info("Segmentation model already cached at %s", model_dir)
            return model_dir
        # Corrupt extracted model — remove directory and re-download
        logger.warning(
            "Deleting corrupt segmentation model directory %s", model_dir
        )
        shutil.rmtree(str(model_dir), ignore_errors=True)

    tarball_path = cache / SEGMENTATION_TARBALL
    _download_file(
        SEGMENTATION_URL,
        tarball_path,
        label="segmentation model tarball",
        # We verify the extracted model.onnx rather than the tarball,
        # because tarball checksums are not stored (tarball is deleted
        # after extraction).
    )

    logger.info("Extracting segmentation model to %s", cache)
    with tarfile.open(str(tarball_path), "r:bz2") as tar:
        # Validate members before extraction to prevent path traversal (B202)
        for member in tar.getmembers():
            if member.name.startswith("/") or ".." in member.name:
                raise ValueError(f"Unsafe tar member: {member.name}")
        tar.extractall(path=str(cache))  # nosec B202
    tarball_path.unlink()

    if not model_onnx.exists():
        raise FileNotFoundError(
            f"Segmentation model not found after extraction: {model_onnx}"
        )

    # Verify the extracted model.onnx
    if not _verify_checksum(
        model_onnx, SEGMENTATION_MODEL_ONNX_SHA256, "segmentation model.onnx"
    ):
        logger.error(
            "Extracted segmentation model.onnx has wrong checksum — "
            "removing and raising"
        )
        shutil.rmtree(str(model_dir), ignore_errors=True)
        raise RuntimeError(
            "Extracted segmentation model.onnx failed checksum verification"
        )

    logger.info(
        "Segmentation model ready (%.1f MB)",
        model_onnx.stat().st_size / 1e6,
    )
    return model_dir


def ensure_embedding_model(cache_dir: Optional[Path] = None) -> Path:
    """Download the 3D-Speaker embedding model if not cached.

    Verifies the SHA256 checksum of the cached file.  If verification
    fails the corrupt artifact is deleted and re-downloaded.

    Returns the path to the .onnx file.
    """
    cache = get_cache_dir(cache_dir)
    model_path = cache / EMBEDDING_MODEL_NAME

    _download_file(
        EMBEDDING_URL,
        model_path,
        expected_sha256=EMBEDDING_SHA256,
        label="embedding model",
    )
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
