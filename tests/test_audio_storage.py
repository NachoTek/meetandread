"""Tests for crash-safe audio storage primitives.

Validates:
- Path utilities (directory resolution, filename generation)
- PCM streaming writer (append-only, metadata sidecar)
- WAV finalization (stdlib wave module produces valid WAVs)
- Recovery (partial files can be converted to playable WAVs)
"""

import math
import struct
import wave
from datetime import datetime
from pathlib import Path

import pytest

from metamemory.audio.storage import (
    get_recordings_dir,
    new_recording_stem,
    PcmPartWriter,
    PcmMetadata,
    load_metadata,
    finalize_part_to_wav,
    find_part_files,
    recover_part_file,
    recover_part_files,
)


class TestPathUtilities:
    """Test path resolution and filename generation."""

    def test_get_recordings_dir_creates_directory(self, tmp_path: Path):
        """Recording directory is created if it doesn't exist."""
        base_dir = tmp_path / "test_docs"
        recordings_dir = get_recordings_dir(base_dir=base_dir)

        assert recordings_dir.exists()
        assert recordings_dir.name == "recordings"
        assert recordings_dir.parent.name == "metamemory"

    def test_get_recordings_dir_returns_existing(self, tmp_path: Path):
        """Existing directory is returned without error."""
        base_dir = tmp_path / "test_docs"
        expected = base_dir / "metamemory" / "recordings"
        expected.mkdir(parents=True)

        recordings_dir = get_recordings_dir(base_dir=base_dir)
        assert recordings_dir == expected

    def test_new_recording_stem_format(self):
        """Stem follows recording-YYYY-MM-DD-HHMMSS format."""
        now = datetime(2026, 2, 1, 14, 30, 45)
        stem = new_recording_stem(now)

        assert stem == "recording-2026-02-01-143045"
        assert stem.startswith("recording-")
        assert len(stem) == 27  # "recording-" (10) + "YYYY-MM-DD-HHMMSS" (17)

    def test_new_recording_stem_uses_utc(self):
        """Default uses current UTC time."""
        stem = new_recording_stem()

        assert stem.startswith("recording-")
        # Format: recording-YYYY-MM-DD-HHMMSS (e.g., recording-2026-02-01-143045)
        remainder = stem.replace("recording-", "")
        parts = remainder.split("-")
        # Should have date parts (YYYY, MM, DD) and time (HHMMSS)
        assert len(parts) == 4  # YYYY, MM, DD, HHMMSS
        assert len(parts[0]) == 4  # Year
        assert len(parts[1]) == 2  # Month
        assert len(parts[2]) == 2  # Day
        assert len(parts[3]) == 6  # HHMMSS


class TestPcmPartWriter:
    """Test streaming PCM writer."""

    def test_create_writer(self, tmp_path: Path):
        """Writer creates part file and metadata sidecar."""
        stem = "test-recording"
        writer = PcmPartWriter.create(
            stem=stem,
            sample_rate=16000,
            channels=1,
            sample_width_bytes=2,
            recordings_dir=tmp_path,
        )

        assert writer.part_path.exists()
        assert writer.metadata_path.exists()
        assert not writer.is_closed

        writer.close()

    def test_write_frames_i16(self, tmp_path: Path):
        """Frames are written to file in binary format."""
        writer = PcmPartWriter.create(
            stem="test",
            recordings_dir=tmp_path,
        )

        # Create simple sine wave frames (int16)
        frames = struct.pack("<h", 1000) * 100  # 100 samples of value 1000
        writer.write_frames_i16(frames)
        writer.close()

        # Verify file size
        assert writer.part_path.stat().st_size == len(frames)
        assert writer.frames_written == 100

    def test_metadata_sidecar_content(self, tmp_path: Path):
        """Metadata sidecar contains correct values."""
        writer = PcmPartWriter.create(
            stem="test",
            sample_rate=16000,
            channels=1,
            sample_width_bytes=2,
            recordings_dir=tmp_path,
        )
        writer.close()

        # Load and verify metadata
        metadata = load_metadata(writer.metadata_path)
        assert metadata.sample_rate == 16000
        assert metadata.channels == 1
        assert metadata.sample_width_bytes == 2

    def test_flush_writes_to_disk(self, tmp_path: Path):
        """Flush ensures data is on disk."""
        writer = PcmPartWriter.create(
            stem="test",
            recordings_dir=tmp_path,
        )

        frames = struct.pack("<h", 500) * 50
        writer.write_frames_i16(frames)
        writer.flush()

        # File should be readable even before close
        with open(writer.part_path, "rb") as f:
            data = f.read()
            assert len(data) == len(frames)

        writer.close()

    def test_context_manager(self, tmp_path: Path):
        """Writer works as context manager."""
        with PcmPartWriter.create(
            stem="test",
            recordings_dir=tmp_path,
        ) as writer:
            frames = struct.pack("<h", 100) * 10
            writer.write_frames_i16(frames)
            assert not writer.is_closed

        # Should be closed after exiting context
        assert writer.is_closed

    def test_reject_odd_length_frames(self, tmp_path: Path):
        """Odd-length frame data is rejected (int16 = 2 bytes)."""
        writer = PcmPartWriter.create(
            stem="test",
            recordings_dir=tmp_path,
        )

        with pytest.raises(ValueError, match="even"):
            writer.write_frames_i16(b"\x01\x02\x03")  # 3 bytes

        writer.close()

    def test_reject_write_after_close(self, tmp_path: Path):
        """Writing to closed writer raises error."""
        writer = PcmPartWriter.create(
            stem="test",
            recordings_dir=tmp_path,
        )
        writer.close()

        with pytest.raises(ValueError, match="closed"):
            writer.write_frames_i16(b"\x01\x02")

    def test_reject_duplicate_stem(self, tmp_path: Path):
        """Creating writer with existing stem raises error."""
        PcmPartWriter.create(stem="test", recordings_dir=tmp_path).close()

        with pytest.raises(FileExistsError):
            PcmPartWriter.create(stem="test", recordings_dir=tmp_path)


class TestWavFinalize:
    """Test WAV finalization from PCM part files."""

    def _create_synthetic_audio(self, duration_sec: float = 1.0, freq: float = 440.0) -> bytes:
        """Create synthetic sine wave audio as int16 bytes.

        Args:
            duration_sec: Duration in seconds.
            freq: Frequency of sine wave in Hz.

        Returns:
            int16 little-endian PCM data.
        """
        sample_rate = 16000
        num_samples = int(sample_rate * duration_sec)

        frames = []
        for i in range(num_samples):
            # Generate sine wave and scale to int16 range
            sample = int(32767 * 0.5 * math.sin(2 * math.pi * freq * i / sample_rate))
            frames.append(struct.pack("<h", sample))

        return b"".join(frames)

    def test_finalize_creates_valid_wav(self, tmp_path: Path):
        """Finalization produces valid, readable WAV file."""
        # Create PCM part file with synthetic audio
        writer = PcmPartWriter.create(
            stem="test",
            sample_rate=16000,
            channels=1,
            sample_width_bytes=2,
            recordings_dir=tmp_path,
        )

        audio = self._create_synthetic_audio(duration_sec=0.5)
        writer.write_frames_i16(audio)
        writer.close()

        # Finalize to WAV
        wav_path = tmp_path / "test.wav"
        result = finalize_part_to_wav(writer.part_path, wav_path)

        assert result == wav_path
        assert wav_path.exists()

        # Verify WAV is valid using stdlib wave
        with wave.open(str(wav_path), "rb") as wav_file:
            assert wav_file.getnchannels() == 1
            assert wav_file.getframerate() == 16000
            assert wav_file.getsampwidth() == 2
            assert wav_file.getnframes() > 0

    def test_finalize_with_preloaded_metadata(self, tmp_path: Path):
        """Can finalize with pre-loaded metadata."""
        writer = PcmPartWriter.create(
            stem="test",
            sample_rate=22050,
            channels=1,
            sample_width_bytes=2,
            recordings_dir=tmp_path,
        )

        audio = self._create_synthetic_audio(duration_sec=0.1)
        writer.write_frames_i16(audio)
        writer.close()

        # Load metadata separately
        metadata = load_metadata(writer.metadata_path)

        # Finalize with pre-loaded metadata
        wav_path = tmp_path / "test.wav"
        finalize_part_to_wav(writer.part_path, wav_path, metadata=metadata)

        with wave.open(str(wav_path), "rb") as wav_file:
            assert wav_file.getframerate() == 22050

    def test_finalize_raises_on_missing_part(self, tmp_path: Path):
        """Error raised if part file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            finalize_part_to_wav(tmp_path / "nonexistent.pcm.part", tmp_path / "out.wav")

    def test_finalize_raises_on_missing_metadata(self, tmp_path: Path):
        """Error raised if metadata file doesn't exist."""
        # Create empty part file without metadata
        part_path = tmp_path / "orphan.pcm.part"
        part_path.write_bytes(b"\x00\x00")

        with pytest.raises(FileNotFoundError):
            finalize_part_to_wav(part_path, tmp_path / "out.wav")


class TestRecovery:
    """Test recovery of partial recordings."""

    def _create_partial_recording(self, tmp_path: Path, stem: str) -> Path:
        """Create a partial recording that simulates a crash."""
        writer = PcmPartWriter.create(
            stem=stem,
            sample_rate=16000,
            channels=1,
            sample_width_bytes=2,
            recordings_dir=tmp_path,
        )

        # Write some audio but don't close (simulates crash)
        audio = struct.pack("<h", 1000) * 1000
        writer.write_frames_i16(audio)
        writer.flush()
        # Note: not calling writer.close() - simulates crash

        return writer.part_path

    def test_find_part_files(self, tmp_path: Path):
        """find_part_files locates all .pcm.part files."""
        # Create some partial files
        self._create_partial_recording(tmp_path, "crash1")
        self._create_partial_recording(tmp_path, "crash2")

        # Create a regular file (should not be found)
        (tmp_path / "other.txt").write_text("hello")

        found = find_part_files(tmp_path)
        assert len(found) == 2
        assert all(p.suffixes == [".pcm", ".part"] for p in found)

    def test_find_part_files_empty(self, tmp_path: Path):
        """Empty list returned when no partial files exist."""
        found = find_part_files(tmp_path)
        assert found == []

    def test_recover_part_file(self, tmp_path: Path):
        """Single partial file can be recovered to WAV."""
        part_path = self._create_partial_recording(tmp_path, "crash")

        recovered = recover_part_file(part_path)

        assert recovered is not None
        assert recovered.exists()
        assert recovered.suffix == ".wav"
        assert "recovered" in recovered.name

        # Verify it's a valid WAV
        with wave.open(str(recovered), "rb") as wav_file:
            assert wav_file.getnframes() > 0

        # Originals should be backed up
        backup_part = tmp_path / "crash.pcm.part.recovered.bak"
        backup_meta = tmp_path / "crash.pcm.part.json.recovered.bak"
        assert backup_part.exists()
        assert backup_meta.exists()

    def test_recover_part_file_delete_original(self, tmp_path: Path):
        """Can delete originals instead of backing up."""
        part_path = self._create_partial_recording(tmp_path, "crash")
        metadata_path = part_path.with_suffix(".part.json")

        recovered = recover_part_file(part_path, delete_original=True)

        assert recovered is not None
        # Originals should be gone
        assert not part_path.exists()
        assert not metadata_path.exists()

    def test_recover_part_files_batch(self, tmp_path: Path):
        """Multiple partial files can be recovered in batch."""
        self._create_partial_recording(tmp_path, "crash1")
        self._create_partial_recording(tmp_path, "crash2")
        self._create_partial_recording(tmp_path, "crash3")

        recovered = recover_part_files(tmp_path)

        assert len(recovered) == 3
        for wav_path in recovered:
            assert wav_path.exists()
            assert wav_path.suffix == ".wav"

    def test_recover_part_files_with_progress(self, tmp_path: Path):
        """Progress callback is invoked during batch recovery."""
        self._create_partial_recording(tmp_path, "crash1")
        self._create_partial_recording(tmp_path, "crash2")

        progress_calls = []

        def on_progress(filename, current, total):
            progress_calls.append((filename, current, total))

        recover_part_files(tmp_path, progress_callback=on_progress)

        assert len(progress_calls) == 2
        assert progress_calls[0] == ("crash1.pcm.part", 1, 2)
        assert progress_calls[1] == ("crash2.pcm.part", 2, 2)

    def test_recover_part_files_continues_on_error(self, tmp_path: Path):
        """Batch recovery continues even if one file fails."""
        # Create one valid partial
        self._create_partial_recording(tmp_path, "good")

        # Create an orphaned part file (no metadata)
        orphan = tmp_path / "orphan.pcm.part"
        orphan.write_bytes(b"\x00\x00")

        recovered = recover_part_files(tmp_path)

        # Should still recover the valid one
        assert len(recovered) == 1
        assert "good" in recovered[0].name


class TestIntegration:
    """Integration tests for full workflow."""

    def test_full_record_finalize_workflow(self, tmp_path: Path):
        """Complete workflow: record -> finalize -> verify WAV."""
        # 1. Create recording
        stem = "my-recording"
        writer = PcmPartWriter.create(
            stem=stem,
            sample_rate=16000,
            channels=1,
            sample_width_bytes=2,
            recordings_dir=tmp_path,
        )

        # 2. Write synthetic audio (1 second of 440Hz sine)
        sample_rate = 16000
        num_samples = sample_rate
        frames = []
        for i in range(num_samples):
            sample = int(32767 * 0.5 * math.sin(2 * math.pi * 440 * i / sample_rate))
            frames.append(struct.pack("<h", sample))

        audio_data = b"".join(frames)
        writer.write_frames_i16(audio_data)
        writer.close()

        # 3. Finalize to WAV
        wav_path = tmp_path / f"{stem}.wav"
        finalize_part_to_wav(writer.part_path, wav_path)

        # 4. Verify the WAV
        with wave.open(str(wav_path), "rb") as wav_file:
            assert wav_file.getnchannels() == 1
            assert wav_file.getframerate() == 16000
            assert wav_file.getsampwidth() == 2
            assert wav_file.getnframes() == sample_rate  # 1 second at 16kHz

    def test_crash_recovery_workflow(self, tmp_path: Path):
        """Simulate crash and verify recovery produces valid audio."""
        # 1. Simulate recording that crashes partway
        stem = "crashed-recording"
        writer = PcmPartWriter.create(
            stem=stem,
            sample_rate=16000,
            channels=1,
            sample_width_bytes=2,
            recordings_dir=tmp_path,
        )

        # Write some audio
        audio = struct.pack("<h", 1000) * 5000
        writer.write_frames_i16(audio)
        writer.flush()

        # Simulate crash: close file handle but leave .pcm.part (not finalized to WAV)
        # In a real crash, the OS closes file handles, leaving the partial file
        writer.close()

        # 2. Discover partial recording
        partials = find_part_files(tmp_path)
        assert len(partials) == 1

        # 3. Recover the partial
        recovered = recover_part_files(tmp_path)
        assert len(recovered) == 1

        # 4. Verify recovered audio is valid
        wav_path = recovered[0]
        with wave.open(str(wav_path), "rb") as wav_file:
            assert wav_file.getnchannels() == 1
            assert wav_file.getframerate() == 16000
            assert wav_file.getsampwidth() == 2
            assert wav_file.getnframes() == 5000

        # 5. Verify original partial is backed up
        backup = tmp_path / f"{stem}.pcm.part.recovered.bak"
        assert backup.exists()
