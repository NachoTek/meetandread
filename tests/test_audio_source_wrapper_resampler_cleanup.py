"""Test for nanobind memory leak fix in AudioSourceWrapper.

This test verifies that AudioSourceWrapper properly cleans up the soxr.ResampleStream
to prevent nanobind memory leaks on application exit.

Issue: #18 - nanobind leak of soxr.soxr_ext.CSoxr instances
"""
import gc
import weakref
from pathlib import Path
import pytest

from meetandread.audio.capture.fake_module import FakeAudioModule
from meetandread.audio.session import (
    AudioSourceWrapper,
    SourceConfig,
)


FIXTURES = Path(__file__).resolve().parent / "fixtures"
SILENCE_WAV_16K = str(FIXTURES / "SAMPLE-Audio1.wav")  # 16kHz file
SILENCE_WAV_48K = str(FIXTURES / "SAMPLE-Audio1.wav")  # Will need 48kHz source


class TestAudioSourceWrapperResamplerCleanup:
    """Verify resampler cleanup to prevent nanobind leaks."""

    def test_resampler_created_when_rate_mismatch(self):
        """When source_rate != target_rate, a resampler should be created."""
        source = FakeAudioModule(
            wav_path=SILENCE_WAV_16K,
            blocksize=1024,
            queue_size=10,
            loop=True
        )
        config = SourceConfig(type="fake", fake_path=SILENCE_WAV_16K, loop=True)
        
        # Create wrapper with target_rate=48000 (different from source 16000)
        wrapper = AudioSourceWrapper(
            source,
            config,
            target_rate=48000,  # Different from source rate to trigger resampling
            target_channels=1
        )
        
        assert wrapper._resampler is not None, "Resampler should be created when rates differ"
        assert hasattr(wrapper._resampler, 'resample_chunk'), "Should be a soxr.ResampleStream"

    def test_resampler_none_when_rate_match(self):
        """When source_rate == target_rate, no resampler should be created."""
        source = FakeAudioModule(
            wav_path=SILENCE_WAV_16K,
            blocksize=1024,
            queue_size=10,
            loop=True
        )
        config = SourceConfig(type="fake", fake_path=SILENCE_WAV_16K, loop=True)
        
        # Create wrapper with target_rate=16000 (same as source)
        wrapper = AudioSourceWrapper(
            source,
            config,
            target_rate=16000,  # Same as source rate
            target_channels=1
        )
        
        assert wrapper._resampler is None, "Resampler should be None when rates match"

    def test_stop_clears_resampler(self):
        """Calling stop() should explicitly clean up the resampler."""
        source = FakeAudioModule(
            wav_path=SILENCE_WAV_16K,
            blocksize=1024,
            queue_size=10,
            loop=True
        )
        config = SourceConfig(type="fake", fake_path=SILENCE_WAV_16K, loop=True)
        
        # Create wrapper with resampler
        wrapper = AudioSourceWrapper(
            source,
            config,
            target_rate=48000,  # Different rate to trigger resampling
            target_channels=1
        )
        
        assert wrapper._resampler is not None, "Resampler should be created"
        
        # Create weak reference to verify cleanup
        resampler_ref = weakref.ref(wrapper._resampler)
        
        # Stop the wrapper
        wrapper.stop()
        
        # After stop, resampler should be cleared
        assert wrapper._resampler is None, "Resampler should be None after stop()"
        
        # Force garbage collection
        gc.collect()
        
        # Verify the resampler was actually cleaned up
        assert resampler_ref() is None, "Resampler should be garbage collected after cleanup"

    def test_stop_with_none_resampler_is_safe(self):
        """Calling stop() when resampler is None should be safe (no-op)."""
        source = FakeAudioModule(
            wav_path=SILENCE_WAV_16K,
            blocksize=1024,
            queue_size=10,
            loop=True
        )
        config = SourceConfig(type="fake", fake_path=SILENCE_WAV_16K, loop=True)
        
        # Create wrapper without resampler (matching rates)
        wrapper = AudioSourceWrapper(
            source,
            config,
            target_rate=16000,  # Same as source rate
            target_channels=1
        )
        
        assert wrapper._resampler is None
        
        # Stop should not raise an exception
        wrapper.stop()
        
        # Should still be None
        assert wrapper._resampler is None

    def test_multiple_start_stop_cycles_cleanup(self):
        """Multiple start/stop cycles should properly cleanup resampler each time."""
        source = FakeAudioModule(
            wav_path=SILENCE_WAV_16K,
            blocksize=1024,
            queue_size=10,
            loop=True
        )
        config = SourceConfig(type="fake", fake_path=SILENCE_WAV_16K, loop=True)
        
        for _ in range(3):
            # Create fresh wrapper each cycle
            wrapper = AudioSourceWrapper(
                source,
                config,
                target_rate=48000,  # Different rate to trigger resampling
                target_channels=1
            )
            
            assert wrapper._resampler is not None, "Resampler should be created"
            
            # Create weak reference
            resampler_ref = weakref.ref(wrapper._resampler)
            
            wrapper.stop()
            
            assert wrapper._resampler is None, "Resampler should be None after stop()"
            
            # Force garbage collection
            gc.collect()
            
            # Verify cleanup
            assert resampler_ref() is None, f"Resampler should be garbage collected (cycle {_ + 1})"