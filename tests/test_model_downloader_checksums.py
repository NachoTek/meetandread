"""Tests for model downloader SHA256 checksum verification.

Covers:
  - Valid download with correct checksum
  - Corrupt download (wrong checksum after download)
  - Corrupt existing cache (wrong checksum, triggers re-download)
  - Temp file cleanup on failure
  - Segmentation extraction validation
  - Segmentation extracted model.onnx checksum mismatch
  - Unsafe tar member rejection
  - Missing extracted model.onnx after extraction
  - Checksum helper correctness

All tests mock urlretrieve and use inline fixture bytes — no network required.
"""

import hashlib
import tarfile
import io
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from meetandread.speaker.model_downloader import (
    EMBEDDING_SHA256,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_URL,
    SEGMENTATION_MODEL_NAME,
    SEGMENTATION_MODEL_ONNX_SHA256,
    SEGMENTATION_TARBALL,
    SEGMENTATION_URL,
    sha256_checksum,
    _verify_checksum,
    _download_file,
    ensure_embedding_model,
    ensure_segmentation_model,
    ensure_all_models,
    get_cache_dir,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cache_dir(tmp_path):
    """Provide a clean temporary cache directory."""
    d = tmp_path / "models"
    d.mkdir()
    return d


def _make_onnx_bytes(size: int = 1024, seed: int = 42) -> bytes:
    """Create deterministic fake ONNX bytes of given size."""
    import random
    rng = random.Random(seed)
    return bytes(rng.getrandbits(8) for _ in range(size))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# sha256_checksum / _verify_checksum tests
# ---------------------------------------------------------------------------

class TestChecksumHelpers:
    """Verify streaming SHA256 helper and _verify_checksum."""

    def test_sha256_checksum_matches_hashlib(self, tmp_path):
        data = _make_onnx_bytes(4096)
        p = tmp_path / "test.bin"
        p.write_bytes(data)
        assert sha256_checksum(p) == _sha256(data)

    def test_sha256_checksum_empty_file(self, tmp_path):
        p = tmp_path / "empty.bin"
        p.write_bytes(b"")
        assert sha256_checksum(p) == hashlib.sha256(b"").hexdigest()

    def test_verify_checksum_match(self, tmp_path):
        data = _make_onnx_bytes()
        p = tmp_path / "model.onnx"
        p.write_bytes(data)
        assert _verify_checksum(p, _sha256(data), "test") is True

    def test_verify_checksum_mismatch(self, tmp_path):
        data = _make_onnx_bytes()
        p = tmp_path / "model.onnx"
        p.write_bytes(data)
        assert _verify_checksum(p, "0" * 64, "test") is False


# ---------------------------------------------------------------------------
# _download_file tests
# ---------------------------------------------------------------------------

class TestDownloadFile:
    """Test _download_file with mocked urlretrieve."""

    def test_valid_download_moves_tmp_to_dest(self, cache_dir):
        """Successful download moves temp file to destination."""
        data = _make_onnx_bytes()
        dest = cache_dir / "model.onnx"
        checksum = _sha256(data)

        def fake_urlretrieve(url, path):
            Path(path).write_bytes(data)

        with patch("meetandread.speaker.model_downloader.urllib.request.urlretrieve", side_effect=fake_urlretrieve):
            _download_file("http://example.com/model.onnx", dest, expected_sha256=checksum, label="test")

        assert dest.exists()
        assert dest.read_bytes() == data
        # No temp file left behind
        assert not dest.with_suffix(dest.suffix + ".tmp").exists()

    def test_corrupt_download_deletes_tmp_and_raises(self, cache_dir):
        """Wrong checksum after download: tmp deleted, RuntimeError raised."""
        data = _make_onnx_bytes()
        dest = cache_dir / "model.onnx"

        def fake_urlretrieve(url, path):
            Path(path).write_bytes(data)

        with patch("meetandread.speaker.model_downloader.urllib.request.urlretrieve", side_effect=fake_urlretrieve):
            with pytest.raises(RuntimeError, match="Checksum verification failed"):
                _download_file("http://example.com/model.onnx", dest, expected_sha256="0" * 64, label="test")

        assert not dest.exists()
        tmp_path = dest.with_suffix(dest.suffix + ".tmp")
        assert not tmp_path.exists()

    def test_urlretrieve_exception_cleans_tmp(self, cache_dir):
        """urlretrieve exception: tmp deleted, exception propagated."""
        dest = cache_dir / "model.onnx"

        def fake_urlretrieve(url, path):
            Path(path).write_bytes(b"partial")
            raise ConnectionError("network down")

        with patch("meetandread.speaker.model_downloader.urllib.request.urlretrieve", side_effect=fake_urlretrieve):
            with pytest.raises(ConnectionError):
                _download_file("http://example.com/model.onnx", dest, expected_sha256=None, label="test")

        assert not dest.exists()
        tmp_path = dest.with_suffix(dest.suffix + ".tmp")
        assert not tmp_path.exists()

    def test_existing_valid_cache_skips_download(self, cache_dir):
        """Existing file with matching checksum skips download entirely."""
        data = _make_onnx_bytes()
        checksum = _sha256(data)
        dest = cache_dir / "model.onnx"
        dest.write_bytes(data)

        # urlretrieve should NOT be called
        with patch("meetandread.speaker.model_downloader.urllib.request.urlretrieve") as mock_ur:
            _download_file("http://example.com/model.onnx", dest, expected_sha256=checksum, label="test")
            mock_ur.assert_not_called()

    def test_corrupt_cache_deleted_and_redownloaded(self, cache_dir):
        """Existing file with wrong checksum is deleted and re-downloaded."""
        corrupt_data = b"corrupt"
        good_data = _make_onnx_bytes(seed=99)
        checksum = _sha256(good_data)
        dest = cache_dir / "model.onnx"
        dest.write_bytes(corrupt_data)

        def fake_urlretrieve(url, path):
            Path(path).write_bytes(good_data)

        with patch("meetandread.speaker.model_downloader.urllib.request.urlretrieve", side_effect=fake_urlretrieve):
            _download_file("http://example.com/model.onnx", dest, expected_sha256=checksum, label="test")

        assert dest.exists()
        assert dest.read_bytes() == good_data

    def test_no_checksum_skips_verification(self, cache_dir):
        """Without expected_sha256, existing file is used without checksum."""
        dest = cache_dir / "model.onnx"
        dest.write_bytes(b"whatever")

        with patch("meetandread.speaker.model_downloader.urllib.request.urlretrieve") as mock_ur:
            _download_file("http://example.com/model.onnx", dest, expected_sha256=None, label="test")
            mock_ur.assert_not_called()

    def test_download_without_checksum_no_verification(self, cache_dir):
        """Download without checksum does not verify after download."""
        data = _make_onnx_bytes()
        dest = cache_dir / "model.onnx"

        def fake_urlretrieve(url, path):
            Path(path).write_bytes(data)

        with patch("meetandread.speaker.model_downloader.urllib.request.urlretrieve", side_effect=fake_urlretrieve):
            # No checksum — should succeed even though data is arbitrary
            _download_file("http://example.com/model.onnx", dest, expected_sha256=None, label="test")

        assert dest.read_bytes() == data


# ---------------------------------------------------------------------------
# ensure_embedding_model tests
# ---------------------------------------------------------------------------

class TestEnsureEmbeddingModel:
    """Test ensure_embedding_model with mocked urlretrieve."""

    def test_valid_embedding_download(self, cache_dir):
        """Valid embedding model download succeeds."""
        data = _make_onnx_bytes(2048, seed=10)
        checksum = _sha256(data)

        def fake_urlretrieve(url, path):
            Path(path).write_bytes(data)

        with patch("meetandread.speaker.model_downloader.urllib.request.urlretrieve", side_effect=fake_urlretrieve), \
             patch("meetandread.speaker.model_downloader.EMBEDDING_SHA256", checksum):
            path = ensure_embedding_model(cache_dir)

        assert path.name == EMBEDDING_MODEL_NAME
        assert path.exists()
        assert path.read_bytes() == data

    def test_cached_valid_embedding_skips_download(self, cache_dir):
        """Valid cached embedding model skips download."""
        data = _make_onnx_bytes(2048, seed=10)
        checksum = _sha256(data)
        model_path = cache_dir / EMBEDDING_MODEL_NAME
        model_path.write_bytes(data)

        with patch("meetandread.speaker.model_downloader.EMBEDDING_SHA256", checksum), \
             patch("meetandread.speaker.model_downloader.urllib.request.urlretrieve") as mock_ur:
            path = ensure_embedding_model(cache_dir)
            mock_ur.assert_not_called()

        assert path == model_path

    def test_corrupt_cached_embedding_redownloads(self, cache_dir):
        """Corrupt cached embedding model is deleted and re-downloaded."""
        corrupt_data = b"not-an-onnx"
        good_data = _make_onnx_bytes(2048, seed=20)
        checksum = _sha256(good_data)
        model_path = cache_dir / EMBEDDING_MODEL_NAME
        model_path.write_bytes(corrupt_data)

        def fake_urlretrieve(url, path):
            Path(path).write_bytes(good_data)

        with patch("meetandread.speaker.model_downloader.EMBEDDING_SHA256", checksum), \
             patch("meetandread.speaker.model_downloader.urllib.request.urlretrieve", side_effect=fake_urlretrieve):
            path = ensure_embedding_model(cache_dir)

        assert path.read_bytes() == good_data


# ---------------------------------------------------------------------------
# ensure_segmentation_model tests
# ---------------------------------------------------------------------------

def _make_tarball_bytes(model_onnx_data: bytes) -> bytes:
    """Create an in-memory tar.bz2 containing the segmentation model dir."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:bz2") as tar:
        # Add model.onnx
        info = tarfile.TarInfo(name=f"{SEGMENTATION_MODEL_NAME}/model.onnx")
        info.size = len(model_onnx_data)
        tar.addfile(info, io.BytesIO(model_onnx_data))

        # Add a dummy file so the dir isn't empty
        readme = b"test"
        info2 = tarfile.TarInfo(name=f"{SEGMENTATION_MODEL_NAME}/README.md")
        info2.size = len(readme)
        tar.addfile(info2, io.BytesIO(readme))
    return buf.getvalue()


class TestEnsureSegmentationModel:
    """Test ensure_segmentation_model with mocked downloads."""

    def test_valid_segmentation_download_and_extract(self, cache_dir):
        """Valid tarball downloads, extracts, and verifies model.onnx."""
        onnx_data = _make_onnx_bytes(4096, seed=30)
        checksum = _sha256(onnx_data)
        tarball_data = _make_tarball_bytes(onnx_data)

        def fake_urlretrieve(url, path):
            Path(path).write_bytes(tarball_data)

        with patch("meetandread.speaker.model_downloader.urllib.request.urlretrieve", side_effect=fake_urlretrieve), \
             patch("meetandread.speaker.model_downloader.SEGMENTATION_MODEL_ONNX_SHA256", checksum):
            model_dir = ensure_segmentation_model(cache_dir)

        model_onnx = model_dir / "model.onnx"
        assert model_onnx.exists()
        assert model_onnx.read_bytes() == onnx_data
        # Tarball should be removed after extraction
        assert not (cache_dir / SEGMENTATION_TARBALL).exists()

    def test_cached_valid_segmentation_skips_download(self, cache_dir):
        """Valid cached segmentation model skips download."""
        onnx_data = _make_onnx_bytes(4096, seed=30)
        checksum = _sha256(onnx_data)

        # Create the cached model directory
        model_dir = cache_dir / SEGMENTATION_MODEL_NAME
        model_dir.mkdir()
        (model_dir / "model.onnx").write_bytes(onnx_data)

        with patch("meetandread.speaker.model_downloader.SEGMENTATION_MODEL_ONNX_SHA256", checksum), \
             patch("meetandread.speaker.model_downloader.urllib.request.urlretrieve") as mock_ur:
            result = ensure_segmentation_model(cache_dir)
            mock_ur.assert_not_called()

        assert result == model_dir

    def test_corrupt_cached_segmentation_redownloads(self, cache_dir):
        """Corrupt cached model.onnx is deleted and re-downloaded."""
        corrupt_data = b"corrupt-onnx"
        good_data = _make_onnx_bytes(4096, seed=40)
        checksum = _sha256(good_data)

        # Create corrupt cache
        model_dir = cache_dir / SEGMENTATION_MODEL_NAME
        model_dir.mkdir()
        (model_dir / "model.onnx").write_bytes(corrupt_data)

        tarball_data = _make_tarball_bytes(good_data)

        def fake_urlretrieve(url, path):
            Path(path).write_bytes(tarball_data)

        with patch("meetandread.speaker.model_downloader.SEGMENTATION_MODEL_ONNX_SHA256", checksum), \
             patch("meetandread.speaker.model_downloader.urllib.request.urlretrieve", side_effect=fake_urlretrieve):
            result = ensure_segmentation_model(cache_dir)

        assert (result / "model.onnx").read_bytes() == good_data

    def test_extracted_model_wrong_checksum_raises(self, cache_dir):
        """If extracted model.onnx has wrong checksum, RuntimeError is raised."""
        onnx_data = _make_onnx_bytes(4096, seed=50)
        tarball_data = _make_tarball_bytes(onnx_data)

        def fake_urlretrieve(url, path):
            Path(path).write_bytes(tarball_data)

        with patch("meetandread.speaker.model_downloader.urllib.request.urlretrieve", side_effect=fake_urlretrieve), \
             patch("meetandread.speaker.model_downloader.SEGMENTATION_MODEL_ONNX_SHA256", "0" * 64):
            with pytest.raises(RuntimeError, match="failed checksum verification"):
                ensure_segmentation_model(cache_dir)

        # Model directory should have been cleaned up
        model_dir = cache_dir / SEGMENTATION_MODEL_NAME
        assert not model_dir.exists()

    def test_missing_model_onnx_after_extraction_raises(self, cache_dir):
        """Tarball without model.onnx raises FileNotFoundError."""
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:bz2") as tar:
            info = tarfile.TarInfo(name=f"{SEGMENTATION_MODEL_NAME}/other.txt")
            info.size = 4
            tar.addfile(info, io.BytesIO(b"test"))
        tarball_data = buf.getvalue()

        def fake_urlretrieve(url, path):
            Path(path).write_bytes(tarball_data)

        with patch("meetandread.speaker.model_downloader.urllib.request.urlretrieve", side_effect=fake_urlretrieve):
            with pytest.raises(FileNotFoundError, match="not found after extraction"):
                ensure_segmentation_model(cache_dir)

    def test_unsafe_tar_member_raises(self, cache_dir):
        """Tarball with path traversal member raises ValueError."""
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:bz2") as tar:
            info = tarfile.TarInfo(name=f"{SEGMENTATION_MODEL_NAME}/../escape.txt")
            info.size = 4
            tar.addfile(info, io.BytesIO(b"test"))
        tarball_data = buf.getvalue()

        def fake_urlretrieve(url, path):
            Path(path).write_bytes(tarball_data)

        with patch("meetandread.speaker.model_downloader.urllib.request.urlretrieve", side_effect=fake_urlretrieve):
            with pytest.raises(ValueError, match="Unsafe tar member"):
                ensure_segmentation_model(cache_dir)

    def test_absolute_path_tar_member_raises(self, cache_dir):
        """Tarball with absolute path member raises ValueError."""
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:bz2") as tar:
            info = tarfile.TarInfo(name="/etc/passwd")
            info.size = 4
            tar.addfile(info, io.BytesIO(b"test"))
        tarball_data = buf.getvalue()

        def fake_urlretrieve(url, path):
            Path(path).write_bytes(tarball_data)

        with patch("meetandread.speaker.model_downloader.urllib.request.urlretrieve", side_effect=fake_urlretrieve):
            with pytest.raises(ValueError, match="Unsafe tar member"):
                ensure_segmentation_model(cache_dir)


# ---------------------------------------------------------------------------
# ensure_all_models tests
# ---------------------------------------------------------------------------

class TestEnsureAllModels:
    """Test ensure_all_models delegates correctly."""

    def test_returns_both_model_paths(self, cache_dir):
        """ensure_all_models returns dict with segmentation_dir and embedding_model."""
        emb_data = _make_onnx_bytes(2048, seed=10)
        emb_checksum = _sha256(emb_data)
        seg_data = _make_onnx_bytes(4096, seed=30)
        seg_checksum = _sha256(seg_data)
        tarball_data = _make_tarball_bytes(seg_data)

        def fake_urlretrieve(url, path):
            if "segmentation" in url:
                Path(path).write_bytes(tarball_data)
            else:
                Path(path).write_bytes(emb_data)

        with patch("meetandread.speaker.model_downloader.urllib.request.urlretrieve", side_effect=fake_urlretrieve), \
             patch("meetandread.speaker.model_downloader.EMBEDDING_SHA256", emb_checksum), \
             patch("meetandread.speaker.model_downloader.SEGMENTATION_MODEL_ONNX_SHA256", seg_checksum):
            result = ensure_all_models(cache_dir)

        assert "segmentation_dir" in result
        assert "embedding_model" in result
        assert result["segmentation_dir"].name == SEGMENTATION_MODEL_NAME
        assert result["embedding_model"].name == EMBEDDING_MODEL_NAME


# ---------------------------------------------------------------------------
# Edge cases and negative tests (Q7)
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Negative tests covering failure modes and edge cases."""

    def test_interrupted_download_no_dest_file(self, cache_dir):
        """Download that creates tmp but fails before move leaves no dest."""
        dest = cache_dir / "model.onnx"

        def fake_urlretrieve(url, path):
            Path(path).write_bytes(b"partial")
            raise OSError("connection reset")

        with patch("meetandread.speaker.model_downloader.urllib.request.urlretrieve", side_effect=fake_urlretrieve):
            with pytest.raises(OSError, match="connection reset"):
                _download_file("http://example.com/model.onnx", dest, expected_sha256=None, label="test")

        assert not dest.exists()
        assert not dest.with_suffix(dest.suffix + ".tmp").exists()

    def test_sha256_helper_large_file(self, tmp_path):
        """SHA256 helper works on files larger than one chunk."""
        # Create a file > _CHECKSUM_CHUNK_SIZE
        data = b"A" * (65536 * 3 + 100)
        p = tmp_path / "big.bin"
        p.write_bytes(data)
        assert sha256_checksum(p) == hashlib.sha256(data).hexdigest()

    def test_download_file_network_error_no_tmp_leak(self, cache_dir):
        """Network error before tmp creation: no tmp file left."""
        dest = cache_dir / "model.onnx"

        def fake_urlretrieve(url, path):
            raise OSError("DNS failure")

        with patch("meetandread.speaker.model_downloader.urllib.request.urlretrieve", side_effect=fake_urlretrieve):
            with pytest.raises(OSError):
                _download_file("http://x.com/m.onnx", dest, expected_sha256=None, label="test")

        assert not dest.with_suffix(dest.suffix + ".tmp").exists()
