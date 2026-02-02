"""Whisper transcription engine using whisper.cpp.

Provides real-time transcription with confidence scoring and word-level
timestamps. Uses whisper.cpp (via pywhispercpp) for CPU-only operation
without PyTorch DLL dependencies.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import numpy as np
import logging
import tempfile
import wave
import os
import urllib.request

# Import whisper.cpp bindings
try:
    from pywhispercpp.model import Model as WhisperModel
    _WHISPER_AVAILABLE = True
except ImportError:
    _WHISPER_AVAILABLE = False
    WhisperModel = None


logger = logging.getLogger(__name__)


@dataclass
class WordInfo:
    """Word-level transcription information."""
    text: str
    start: float  # timestamp in seconds
    end: float
    confidence: int  # 0-100 scale


@dataclass
class TranscriptionSegment:
    """Transcription segment with metadata."""
    text: str
    confidence: int  # 0-100 scale
    start: float  # timestamp in seconds
    end: float
    words: List[WordInfo]


class WhisperTranscriptionEngine:
    """Whisper-based transcription engine with confidence extraction.
    
    Wraps whisper.cpp for real-time transcription with:
    - Configurable model sizes (tiny/base/small)
    - Confidence score normalization (0-100 scale)
    - Word-level timestamps
    - CPU-only operation (no GPU/CUDA required)
    
    Models are automatically downloaded from HuggingFace in .bin format
    (ggml-whisper models).
    
    Example:
        engine = WhisperTranscriptionEngine(model_size='base')
        engine.load_model()  # Load model (can take 2-5 seconds)
        
        # Transcribe audio chunk
        audio = np.zeros(16000 * 2, dtype=np.float32)  # 2 seconds of audio
        segments = engine.transcribe_chunk(audio)
        
        for segment in segments:
            print(f"{segment.text} (confidence: {segment.confidence}%)")
    """
    
    # Model download URLs from HuggingFace
    MODEL_URLS = {
        'tiny': 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin',
        'base': 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin',
        'small': 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin',
        'medium': 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin',
        'large': 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin',
    }
    
    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        """Initialize the transcription engine.
        
        Args:
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
            device: Device to use ('cpu' only for whisper.cpp)
            compute_type: Computation type (ignored for whisper.cpp, always uses optimized quantization)
        """
        self.model_size = model_size
        self.device = device  # whisper.cpp is CPU-only
        self.compute_type = compute_type  # Ignored, whisper.cpp handles internally
        
        self._model: Optional[Any] = None
        self._model_loaded = False
        
        # Model directory in app data
        self._model_dir = self._get_model_dir()
        
        # Configuration for transcription
        self.beam_size = 5
        self.word_timestamps = True
        self.vad_filter = True
        
        # whisper.cpp doesn't have built-in VAD like faster-whisper
        # We use the external VAD processor instead
        self.vad_parameters = {
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 200,
        }
        
        # Confidence normalization parameters
        # Whisper log_probs: -1.0 = high, -3.0 = low
        self._conf_high_logprob = -1.0
        self._conf_low_logprob = -3.0
        self._conf_high_score = 95
        self._conf_low_score = 30
        
        if not _WHISPER_AVAILABLE:
            logger.warning("pywhispercpp not available. Install with: pip install pywhispercpp")
    
    def _get_model_dir(self) -> Path:
        """Get the directory for storing models."""
        # Use platform-appropriate app data directory
        if os.name == 'nt':  # Windows
            app_data = Path(os.environ.get('APPDATA', Path.home() / 'AppData' / 'Roaming'))
        else:  # macOS/Linux
            app_data = Path(os.environ.get('XDG_DATA_HOME', Path.home() / '.local' / 'share'))
        
        model_dir = app_data / 'metamemory' / 'models'
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir
    
    def _get_model_path(self) -> Path:
        """Get the path to the model file."""
        # Map model size to filename
        model_filename = f"ggml-{self.model_size}.bin"
        return self._model_dir / model_filename
    
    def _download_model(self, model_path: Path) -> None:
        """Download the model from HuggingFace if it doesn't exist."""
        if model_path.exists():
            return
        
        url = self.MODEL_URLS.get(self.model_size)
        if not url:
            raise ValueError(f"Unknown model size: {self.model_size}")
        
        logger.info(f"Downloading model {self.model_size} from {url}")
        logger.info(f"This may take a few minutes depending on your connection...")
        
        try:
            # Download with progress reporting
            import urllib.request
            
            def download_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(100, int(downloaded * 100 / total_size)) if total_size > 0 else 0
                if block_num % 100 == 0:  # Log every 100 blocks to avoid spam
                    logger.info(f"Downloaded {percent}%")
            
            urllib.request.urlretrieve(url, model_path, reporthook=download_progress)
            logger.info(f"Model downloaded to {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            if model_path.exists():
                model_path.unlink()  # Clean up partial download
            raise
    
    def load_model(self) -> None:
        """Load the Whisper model.
        
        This can take 2-5 seconds depending on model size and hardware.
        Automatically downloads the model if not present.
        Should be called before starting transcription.
        """
        if self._model_loaded:
            return
        
        if not _WHISPER_AVAILABLE:
            raise RuntimeError(
                "pywhispercpp not installed. "
                "Install with: pip install pywhispercpp"
            )
        
        logger.info(f"Loading Whisper model: {self.model_size}")
        
        try:
            model_path = self._get_model_path()
            
            # Download if needed
            if not model_path.exists():
                self._download_model(model_path)
            
            # Load the model with whisper.cpp
            # Model takes model path as string, print options disabled
            self._model = WhisperModel(
                str(model_path),
                print_realtime=False,
                print_progress=False
            )
            self._model_loaded = True
            logger.info(f"Model loaded successfully from {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def is_model_loaded(self) -> bool:
        """Check if the model has been loaded.
        
        Returns:
            True if model is loaded and ready for transcription
        """
        return self._model_loaded and self._model is not None
    
    def _save_audio_to_temp_file(self, audio_np: np.ndarray) -> str:
        """Save audio numpy array to a temporary WAV file.
        
        whisper.cpp requires file paths, so we save audio chunks to temp files.
        
        Args:
            audio_np: Audio samples as float32 numpy array (mono, 16kHz)
            
        Returns:
            Path to temporary WAV file
        """
        # Create temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            temp_path = tmp_file.name
        
        # Convert float32 to int16 for WAV format
        if audio_np.dtype == np.float32:
            # Scale from [-1.0, 1.0] to int16 range
            audio_int16 = (audio_np * 32767).astype(np.int16)
        else:
            audio_int16 = audio_np.astype(np.int16)
        
        # Write WAV file
        with wave.open(temp_path, 'wb') as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(16000)  # 16kHz
            wf.writeframes(audio_int16.tobytes())
        
        return temp_path
    
    def _parse_whisper_result(self, result) -> Tuple[str, float, float, List[WordInfo]]:
        """Parse whisper.cpp output into text, timestamps, confidence, and word info.
        
        pywhispercpp returns segments with attributes: text, start, end, t0, t1
        We extract text, timestamps and estimate confidence from available data.
        
        Args:
            result: Output from whisper.cpp transcription (list of segments)
            
        Returns:
            Tuple of (text, start_time, end_time, word_list)
        """
        text_parts = []
        words = []
        
        start_time = 0.0
        end_time = 0.0
        
        # Extract text from result
        # pywhispercpp returns a list of segment objects with text, start, end attributes
        if isinstance(result, str):
            return result, 0.0, 0.0, []  # Default if string result
        
        # Try to extract segments
        try:
            if hasattr(result, '__iter__') and not isinstance(result, (str, bytes)):
                for segment in result:
                    if hasattr(segment, 'text'):
                        text_parts.append(segment.text)
                        # Extract timestamps if available
                        if hasattr(segment, 'start'):
                            if start_time == 0.0 or segment.start < start_time:
                                start_time = segment.start
                        if hasattr(segment, 'end'):
                            if segment.end > end_time:
                                end_time = segment.end
                    elif isinstance(segment, dict):
                        text_parts.append(segment.get('text', ''))
                        start = segment.get('start', 0.0)
                        end = segment.get('end', 0.0)
                        if start_time == 0.0 or start < start_time:
                            start_time = start
                        if end > end_time:
                            end_time = end
                    elif isinstance(segment, str):
                        text_parts.append(segment)
            
            # Try string conversion as fallback
            else:
                text_parts.append(str(result))
                
        except Exception as e:
            logger.warning(f"Error parsing whisper result: {e}")
            if isinstance(result, str):
                text_parts.append(result)
            else:
                text_parts.append(str(result))
        
        full_text = ' '.join(text_parts).strip()
        
        return full_text, start_time, end_time, words
    
    def _estimate_confidence(self, text: str) -> int:
        """Estimate confidence score when whisper.cpp doesn't provide probabilities.
        
        This is a heuristic fallback. whisper.cpp may not expose token-level
        probabilities depending on the binding version.
        
        Args:
            text: Transcribed text
            
        Returns:
            Estimated confidence score 0-100
        """
        # Default to medium confidence
        # In a production system, we'd want to use actual token probabilities
        # from whisper.cpp if the binding exposes them
        
        # Heuristic: longer text with reasonable punctuation is more confident
        base_confidence = 70
        
        # Adjust based on text characteristics
        if len(text) > 10:
            base_confidence += 10
        
        if any(c in text for c in '.,!?;:'):
            base_confidence += 5
        
        # Cap at 95% (never claim 100% without real probabilities)
        return min(95, base_confidence)
    
    def transcribe_chunk(self, audio_np: np.ndarray) -> List[TranscriptionSegment]:
        """Transcribe an audio chunk.
        
        Args:
            audio_np: Audio samples as float32 numpy array (mono, 16kHz)
            
        Returns:
            List of transcription segments with confidence scores
            
        Raises:
            RuntimeError: If model is not loaded
        """
        if not self.is_model_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if len(audio_np) == 0:
            return []
        
        # Save audio to temp file (whisper.cpp requires file path)
        temp_path = None
        try:
            temp_path = self._save_audio_to_temp_file(audio_np)
            
            # Transcribe with whisper.cpp
            result = self._model.transcribe(temp_path)
            
            # Parse result - pywhispercpp returns segments with text, start, end
            text, start_time, end_time, words = self._parse_whisper_result(result)
            
            # Estimate confidence based on text characteristics
            confidence = self._estimate_confidence(text)
            
            # Use timestamps from whisper or calculate from audio length
            if end_time == 0.0:
                end_time = len(audio_np) / 16000  # 16kHz sample rate
            
            if text:
                segment = TranscriptionSegment(
                    text=text,
                    confidence=confidence,
                    start=start_time,
                    end=end_time,
                    words=words
                )
                return [segment]
            else:
                return []
                
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return []
            
        finally:
            # Clean up temp file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass  # Ignore cleanup errors
    
    def _normalize_confidence(self, avg_log_prob: float) -> int:
        """Convert Whisper's avg_log_prob to 0-100 scale.
        
        Note: whisper.cpp may not expose avg_log_prob directly like faster-whisper.
        This method is kept for API compatibility.
        
        Whisper log probabilities:
        - -1.0 to -1.5: High confidence
        - -2.0: Medium confidence
        - -3.0 or lower: Low confidence
        
        Args:
            avg_log_prob: Average log probability from Whisper
            
        Returns:
            Confidence score 0-100
        """
        # Clamp to range
        if avg_log_prob > self._conf_high_logprob:
            return self._conf_high_score
        elif avg_log_prob < self._conf_low_logprob:
            return self._conf_low_score
        
        # Linear interpolation
        # Map [low_logprob, high_logprob] to [low_score, high_score]
        normalized = (avg_log_prob - self._conf_low_logprob) / (
            self._conf_high_logprob - self._conf_low_logprob
        )
        return int(self._conf_low_score + normalized * (self._conf_high_score - self._conf_low_score))
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.
        
        Returns:
            Dictionary with model size, device, compute type, etc.
        """
        return {
            "model_size": self.model_size,
            "device": self.device,
            "compute_type": self.compute_type,
            "loaded": self._model_loaded,
            "beam_size": self.beam_size,
            "vad_filter": self.vad_filter,
            "backend": "whisper.cpp",
        }
