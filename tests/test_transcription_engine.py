"""Integration tests for the transcription engine pipeline.

Tests the full pipeline: AudioRingBuffer -> VADChunkingProcessor -> 
WhisperTranscriptionEngine -> LocalAgreementBuffer.

Uses FakeAudioModule to provide deterministic audio for testing.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
from pathlib import Path

from metamemory.audio.capture import FakeAudioModule
from metamemory.transcription import (
    AudioRingBuffer,
    VADChunkingProcessor,
    LocalAgreementBuffer,
    WhisperTranscriptionEngine,
)


class TestAudioRingBuffer:
    """Test AudioRingBuffer functionality."""
    
    def test_basic_append_and_duration(self):
        """Test basic append and duration calculation."""
        buffer = AudioRingBuffer(max_seconds=5)
        
        # Append 1 second of audio
        buffer.append(np.zeros(16000, dtype=np.float32))
        assert buffer.get_total_duration() == 1.0
        
        # Append another second
        buffer.append(np.zeros(16000, dtype=np.float32))
        assert buffer.get_total_duration() == 2.0
    
    def test_get_recent(self):
        """Test retrieving recent audio."""
        buffer = AudioRingBuffer(max_seconds=5)
        
        # Append 2 seconds
        buffer.append(np.ones(16000, dtype=np.float32))
        buffer.append(np.zeros(16000, dtype=np.float32))
        
        # Get last 0.5 seconds
        recent = buffer.get_recent(0.5)
        assert len(recent) == 8000
        assert np.all(recent == 0)  # Should be the zeros we appended last
    
    def test_trim_committed(self):
        """Test trimming committed audio."""
        buffer = AudioRingBuffer(max_seconds=5)
        
        # Append 2 seconds
        buffer.append(np.zeros(16000, dtype=np.float32))
        buffer.append(np.zeros(16000, dtype=np.float32))
        
        # Trim 0.5 seconds (8000 samples)
        buffer.trim_committed(8000)
        assert buffer.get_total_duration() == 1.5
    
    def test_auto_trimming(self):
        """Test automatic trimming when buffer exceeds max."""
        buffer = AudioRingBuffer(max_seconds=2)  # 2 second max
        
        # Append 3 seconds (should be trimmed to 2)
        for _ in range(3):
            buffer.append(np.zeros(16000, dtype=np.float32))
        
        assert buffer.get_total_duration() <= 2.0
    
    def test_thread_safety(self):
        """Test that buffer operations are thread-safe."""
        import threading
        
        buffer = AudioRingBuffer(max_seconds=10)
        errors = []
        
        def append_audio():
            try:
                for _ in range(100):
                    buffer.append(np.zeros(1600, dtype=np.float32))  # 0.1s chunks
            except Exception as e:
                errors.append(e)
        
        # Run from multiple threads
        threads = [threading.Thread(target=append_audio) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Thread errors: {errors}"
        # Should have 4 threads * 100 chunks * 0.1s = 40s, but capped at 10s
        assert buffer.get_total_duration() <= 10.0


class TestVADChunkingProcessor:
    """Test VADChunkingProcessor functionality."""
    
    def test_min_chunk_size_triggering(self):
        """Test that processor triggers when min chunk size reached."""
        processor = VADChunkingProcessor(min_chunk_size_sec=1.0)
        
        # Feed 0.5s with speech
        processor.feed_audio(np.zeros(8000, dtype=np.float32), True)
        assert not processor.should_process()
        
        # Feed another 0.5s with speech
        processor.feed_audio(np.zeros(8000, dtype=np.float32), True)
        assert processor.should_process()
        
        # Get chunk
        chunk = processor.get_chunk()
        assert len(chunk) == 16000  # 1 second
    
    def test_speech_end_triggering(self):
        """Test that processor triggers on speech end."""
        processor = VADChunkingProcessor(min_chunk_size_sec=1.0)
        
        # Feed 0.5s with speech
        processor.feed_audio(np.zeros(8000, dtype=np.float32), True)
        assert not processor.should_process()
        
        # Feed 0.3s without speech (speech end)
        processor.feed_audio(np.zeros(4800, dtype=np.float32), False)
        # Should trigger even though below min chunk size
        assert processor.should_process()
        
        # Get chunk
        chunk = processor.get_chunk()
        assert len(chunk) == 8000 + 4800  # All audio fed so far
    
    def test_remainder_keeping(self):
        """Test that remainder is kept for next chunk."""
        processor = VADChunkingProcessor(min_chunk_size_sec=1.0)
        
        # Feed 1.5 seconds
        processor.feed_audio(np.zeros(24000, dtype=np.float32), True)
        assert processor.should_process()
        
        # Get chunk - should return 1.0s and keep 0.5s
        chunk = processor.get_chunk()
        assert len(chunk) == 16000
        
        # Buffer should have 0.5s remaining
        assert processor.get_buffer_duration() == 0.5


class TestLocalAgreementBuffer:
    """Test LocalAgreementBuffer functionality."""
    
    def test_basic_agreement(self):
        """Test basic agreement and commitment."""
        buffer = LocalAgreementBuffer(agreement_threshold=2)
        
        # First iteration - no commit
        result = buffer.process_iteration('Hello world')
        assert result == ''
        
        # Second iteration - building agreement
        result = buffer.process_iteration('Hello world')
        assert result == ''
        
        # Third iteration - threshold reached, commit
        result = buffer.process_iteration('Hello world')
        assert result == 'Hello world'
        assert buffer.get_committed() == 'Hello world'
    
    def test_extension_needs_fresh_agreement(self):
        """Test that extensions need fresh agreement cycles."""
        buffer = LocalAgreementBuffer(agreement_threshold=2)
        
        # Commit base text
        buffer.process_iteration('Hello world')
        buffer.process_iteration('Hello world')
        buffer.process_iteration('Hello world')
        assert buffer.get_committed() == 'Hello world'
        
        # Extension needs 2 more agreements
        buffer.process_iteration('Hello world today')
        assert buffer.get_pending() == ' today'
        
        buffer.process_iteration('Hello world today')
        assert buffer.get_committed() == 'Hello world'  # Still not committed
        
        result = buffer.process_iteration('Hello world today')
        assert result == ' today'
        assert buffer.get_committed() == 'Hello world today'
    
    def test_divergence_resets_agreement(self):
        """Test that divergence resets agreement and truncates to common prefix."""
        buffer = LocalAgreementBuffer(agreement_threshold=2)
        
        # Build agreement and commit
        buffer.process_iteration('Hello world')
        buffer.process_iteration('Hello world')
        result = buffer.process_iteration('Hello world')
        assert result == 'Hello world'
        assert buffer.get_committed() == 'Hello world'
        
        # Divergence - different text
        # Common prefix is 'Hello ', so buffer gets truncated
        # The committed text stays, but tracking is reset to common prefix
        result = buffer.process_iteration('Hello there')
        assert result == ''  # Not committed yet
        # Buffer is now 'Hello ' (common prefix)
        assert buffer.get_buffer() == 'Hello '
        
        # The new text 'Hello there' needs fresh agreement
        # Step 1 after divergence
        result = buffer.process_iteration('Hello there')
        assert result == ''
        
        # Step 2 after divergence  
        result = buffer.process_iteration('Hello there')
        assert result == ''
        
        # Step 3 after divergence - commits the corrected text
        result = buffer.process_iteration('Hello there')
        assert result == 'there'  # New extension is committed
        # Note: 'Hello ' was already committed, 'there' is the new part
        assert buffer.get_committed() == 'Hello there'
    
    def test_reset(self):
        """Test buffer reset."""
        buffer = LocalAgreementBuffer(agreement_threshold=2)
        
        buffer.process_iteration('Hello world')
        buffer.process_iteration('Hello world')
        buffer.process_iteration('Hello world')
        
        buffer.reset()
        assert buffer.get_committed() == ''
        assert buffer.get_pending() == ''


class TestWhisperTranscriptionEngine:
    """Test WhisperTranscriptionEngine functionality."""
    
    def test_initialization(self):
        """Test engine initialization."""
        engine = WhisperTranscriptionEngine(model_size='tiny')
        assert engine.model_size == 'tiny'
        assert not engine.is_model_loaded()
    
    def test_confidence_normalization(self):
        """Test confidence score normalization."""
        engine = WhisperTranscriptionEngine()
        
        # High confidence
        high = engine._normalize_confidence(-0.5)
        assert high == 95
        
        # Low confidence
        low = engine._normalize_confidence(-4.0)
        assert low == 30
        
        # Medium confidence
        mid = engine._normalize_confidence(-2.0)
        assert 30 < mid < 95
    
    def test_model_info(self):
        """Test getting model info."""
        engine = WhisperTranscriptionEngine(model_size='base', device='cpu')
        info = engine.get_model_info()
        
        assert info['model_size'] == 'base'
        assert info['device'] == 'cpu'
        assert not info['loaded']
    
    @pytest.mark.slow
    def test_model_loading_and_transcription(self):
        """Test actual model loading and transcription.
        
        This test is marked as slow because it downloads and loads the model.
        Run with: pytest -m slow
        """
        engine = WhisperTranscriptionEngine(model_size='tiny')
        engine.load_model()
        
        assert engine.is_model_loaded()
        
        # Create 2 seconds of silence (not ideal for transcription,
        # but faster-whisper with VAD should handle it gracefully)
        audio = np.zeros(16000 * 2, dtype=np.float32)
        
        # Should not raise an error
        segments = engine.transcribe_chunk(audio)
        
        # Should return a list (might be empty for silence)
        assert isinstance(segments, list)


class TestTranscriptionPipeline:
    """Test the full transcription pipeline integration."""
    
    def test_buffer_to_vad_pipeline(self):
        """Test AudioRingBuffer feeding into VADChunkingProcessor."""
        ring_buffer = AudioRingBuffer(max_seconds=10)
        vad_processor = VADChunkingProcessor(min_chunk_size_sec=1.0)
        
        # Simulate: audio capture -> ring buffer -> VAD processor
        for _ in range(10):  # 1 second of audio in 0.1s chunks
            chunk = np.zeros(1600, dtype=np.float32)
            ring_buffer.append(chunk)
            
            # Feed to VAD (simulate speech detected)
            vad_processor.feed_audio(chunk, vad_is_speech=True)
        
        # Should have enough for a chunk
        assert vad_processor.should_process()
        
        chunk = vad_processor.get_chunk()
        assert len(chunk) == 16000  # 1 second
    
    def test_vad_to_agreement_pipeline(self):
        """Test VADChunkingProcessor feeding into LocalAgreementBuffer."""
        # This simulates the flow without actual Whisper inference
        agreement_buffer = LocalAgreementBuffer(agreement_threshold=2)
        
        # Simulate transcription results coming from engine
        # In reality, these would come from WhisperTranscriptionEngine
        simulated_transcriptions = [
            'Hello world',
            'Hello world',
            'Hello world',  # Should commit here
            'Hello world today',
            'Hello world today',
            'Hello world today',  # Should commit extension here
        ]
        
        committed_texts = []
        for text in simulated_transcriptions:
            result = agreement_buffer.process_iteration(text)
            if result:
                committed_texts.append(result)
        
        assert 'Hello world' in committed_texts
        assert ' today' in committed_texts
        assert agreement_buffer.get_committed() == 'Hello world today'
    
    def test_fake_audio_integration(self):
        """Test integration with FakeAudioModule.
        
        This demonstrates how the transcription pipeline would work
        with the FakeAudioModule from Phase 1.
        """
        # Note: This test doesn't actually load the Whisper model
        # It just demonstrates the pipeline structure
        
        # Create a simple test "audio" - just zeros (would be real WAV in practice)
        # For actual testing with model, would use:
        # fake_module = FakeAudioModule('path/to/test.wav')
        
        ring_buffer = AudioRingBuffer(max_seconds=30)
        vad_processor = VADChunkingProcessor(min_chunk_size_sec=1.0)
        
        # Simulate 3 seconds of audio
        for i in range(30):  # 30 chunks of 0.1s
            chunk = np.zeros(1600, dtype=np.float32)
            ring_buffer.append(chunk)
            
            # Simulate VAD (speech for first 2 seconds, then silence)
            is_speech = i < 20
            vad_processor.feed_audio(chunk, vad_is_speech=is_speech)
            
            # In real pipeline, we would check should_process() and feed to Whisper
            if vad_processor.should_process():
                # Check speech end BEFORE getting chunk (flag resets in get_chunk)
                is_end = vad_processor.is_speech_end()
                audio_chunk = vad_processor.get_chunk()
                # Would feed to WhisperTranscriptionEngine here
                # Note: chunks from min_chunk_size are >= 16000 (1s)
                # Chunks from speech end can be smaller (remaining audio)
                if is_end:
                    # Speech end chunks can be any size (remaining audio)
                    assert len(audio_chunk) > 0
                else:
                    # Min chunk size chunks should be full size
                    assert len(audio_chunk) >= 16000  # At least 1 second


if __name__ == '__main__':
    # Run basic tests
    print("Running AudioRingBuffer tests...")
    test_buffer = TestAudioRingBuffer()
    test_buffer.test_basic_append_and_duration()
    test_buffer.test_get_recent()
    test_buffer.test_trim_committed()
    test_buffer.test_auto_trimming()
    test_buffer.test_thread_safety()
    print("OK AudioRingBuffer tests passed")
    
    print("\nRunning VADChunkingProcessor tests...")
    test_vad = TestVADChunkingProcessor()
    test_vad.test_min_chunk_size_triggering()
    test_vad.test_speech_end_triggering()
    test_vad.test_remainder_keeping()
    print("OK VADChunkingProcessor tests passed")
    
    print("\nRunning LocalAgreementBuffer tests...")
    test_agreement = TestLocalAgreementBuffer()
    test_agreement.test_basic_agreement()
    test_agreement.test_extension_needs_fresh_agreement()
    test_agreement.test_divergence_resets_agreement()
    test_agreement.test_reset()
    print("OK LocalAgreementBuffer tests passed")
    
    print("\nRunning WhisperTranscriptionEngine tests...")
    test_engine = TestWhisperTranscriptionEngine()
    test_engine.test_initialization()
    test_engine.test_confidence_normalization()
    test_engine.test_model_info()
    print("OK WhisperTranscriptionEngine tests passed")
    
    print("\nRunning pipeline integration tests...")
    test_pipeline = TestTranscriptionPipeline()
    test_pipeline.test_buffer_to_vad_pipeline()
    test_pipeline.test_vad_to_agreement_pipeline()
    test_pipeline.test_fake_audio_integration()
    print("OK Pipeline integration tests passed")
    
    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
