"""
Enhancement module for dual-mode transcription processing.

This module implements the enhancement architecture for processing low-confidence
segments using background workers without blocking real-time transcription.
"""

import asyncio
import logging
from queue import Queue
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EnhancementConfig:
    """Configuration for enhancement processing."""
    confidence_threshold: float = 0.7  # Default: 70%
    num_workers: int = 4  # Default: 4 workers
    max_queue_size: int = 100  # Default: 100 segments
    enhancement_model: str = "medium"  # Large model for enhancement
    
    def update_settings(self, **kwargs):
        """Update enhancement settings."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


class EnhancementQueue:
    """Bounded queue for low-confidence segments waiting for enhancement."""
    
    def __init__(self, max_size: int = 100, confidence_threshold: float = 0.7):
        """
        Initialize the enhancement queue with bounded capacity.
        
        Args:
            max_size: Maximum number of segments to hold in queue (default: 100)
            confidence_threshold: Confidence threshold for enhancement (default: 0.7)
        """
        self.queue = Queue(maxsize=max_size)
        self.total_enqueued = 0
        self.total_processed = 0
        self.dropped_segments = 0
        self.confidence_threshold = confidence_threshold
        
    def should_enhance(self, segment: Dict[str, Any]) -> bool:
        """
        Determine if segment should be enhanced based on confidence threshold.
        
        Args:
            segment: Segment dictionary containing at least 'confidence' key
            
        Returns:
            bool: True if segment confidence is below threshold and should be enhanced
        """
        confidence = segment.get('confidence')
        threshold_score = self.confidence_threshold * 100
        
        # Handle edge cases
        if confidence is None:
            logger.debug(f"Segment {segment.get('id')} has no confidence, skipping enhancement")
            return False
        
        # Check if confidence is below threshold
        if confidence < threshold_score:
            logger.debug(f"Segment {segment.get('id')} confidence {confidence}% < threshold {threshold_score}%, eligible for enhancement")
            return True
        else:
            logger.debug(f"Segment {segment.get('id')} confidence {confidence}% >= threshold {threshold_score}%, not eligible for enhancement")
            return False
    
    def enqueue(self, segment: Dict[str, Any]) -> bool:
        """
        Add segment to queue if space available and meets enhancement criteria.
        
        Args:
            segment: Dictionary containing segment data with at least 'id', 'text', and 'confidence'
            
        Returns:
            bool: True if segment was enqueued, False if queue was full or not eligible
        """
        # Check if segment should be enhanced
        if not self.should_enhance(segment):
            return False
            
        if self.queue.full():
            self.dropped_segments += 1
            logger.warning(f"Enhancement queue full, dropped segment {segment['id']}")
            return False
            
        self.queue.put(segment)
        self.total_enqueued += 1
        logger.debug(f"Enqueued segment {segment['id']} (queue size: {self.queue.qsize()}")
        return True
    
    def dequeue(self) -> Optional[Dict[str, Any]]:
        """
        Get next segment from queue.
        
        Returns:
            Optional[Dict[str, Any]]: Segment dictionary or None if queue is empty
        """
        try:
            segment = self.queue.get_nowait()
            self.total_processed += 1
            logger.debug(f"Dequeued segment {segment['id']} (queue size: {self.queue.qsize()}")
            return segment
        except:
            return None
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current queue status.
        
        Returns:
            Dict[str, Any]: Dictionary with queue statistics
        """
        return {
            'size': self.queue.qsize(),
            'max_size': self.queue.maxsize,
            'total_enqueued': self.total_enqueued,
            'total_processed': self.total_processed,
            'dropped_segments': self.dropped_segments,
            'is_full': self.queue.full(),
            'is_empty': self.queue.empty(),
            'confidence_threshold': self.confidence_threshold
        }
    
    def set_confidence_threshold(self, threshold: float):
        """Update the confidence threshold for enhancement eligibility.
        
        Args:
            threshold: New confidence threshold (0.0-1.0)
        """
        if 0.0 <= threshold <= 1.0:
            self.confidence_threshold = threshold
            logger.info(f"Updated enhancement confidence threshold to {threshold}")
        else:
            logger.warning(f"Invalid confidence threshold {threshold}, must be between 0.0 and 1.0")


class EnhancementWorkerPool:
    """Async worker pool for background enhancement processing."""
    
    def __init__(self, num_workers: int = 4):
        """
        Initialize the worker pool with specified number of workers.
        
        Args:
            num_workers: Number of parallel workers to use (default: 4)
        """
        self.num_workers = num_workers
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.tasks = []
        self.is_running = False
        self.pending_tasks = 0
        
    async def process_segment(self, segment: Dict[str, Any], 
                            processor: 'EnhancementProcessor') -> Dict[str, Any]:
        """
        Process a segment using the enhancement processor.
        
        Args:
            segment: Segment dictionary to process
            processor: EnhancementProcessor instance to use for processing
            
        Returns:
            Dict[str, Any]: Enhanced segment with results
        """
        loop = asyncio.get_event_loop()
        
        try:
            enhanced = await loop.run_in_executor(
                self.executor,
                processor.enhance,
                segment
            )
            return enhanced
        except Exception as e:
            logger.error(f"Error processing segment {segment['id']}: {e}")
            return {
                'id': segment['id'],
                'error': str(e),
                'original_text': segment['text'],
                'confidence': segment['confidence']
            }
    
    def start(self):
        """Start the worker pool."""
        self.is_running = True
        logger.info(f"Started EnhancementWorkerPool with {self.num_workers} workers")
    
    def stop(self):
        """Stop the worker pool and clean up resources."""
        self.is_running = False
        self.executor.shutdown(wait=True)
        logger.info("Stopped EnhancementWorkerPool")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current worker pool status.
        
        Returns:
            Dict[str, Any]: Dictionary with worker pool statistics
        """
        return {
            'num_workers': self.num_workers,
            'is_running': self.is_running,
            'pending_tasks': self.pending_tasks,
            'active_threads': len(self.executor._threads) if hasattr(self.executor, '_threads') else 0
        }


class EnhancementProcessor:
    """Large model inference engine for segment enhancement using whisper.cpp."""
    
    def __init__(self, config: EnhancementConfig):
        """
        Initialize the enhancement processor with configuration.
        
        Args:
            config: EnhancementConfig instance with model size and settings
        """
        self.config = config
        self.model_name = config.enhancement_model
        self.model = None
        self._model_loaded = False
        
        # Import WhisperTranscriptionEngine for model management
        from .engine import WhisperTranscriptionEngine
        
        # Create engine with enhancement model size
        self.engine = WhisperTranscriptionEngine(model_size=self.model_name)
        
        self.load_model()
    
    def load_model(self):
        """Load the Whisper model for enhancement using whisper.cpp."""
        try:
            logger.info(f"Loading Whisper {self.model_name} model for enhancement...")
            self.engine.load_model()
            self.model = self.engine
            self._model_loaded = True
            logger.info(f"Successfully loaded Whisper {self.model_name} model")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            self._model_loaded = False
    
    def is_model_loaded(self) -> bool:
        """Check if the enhancement model has been loaded.
        
        Returns:
            True if model is loaded and ready for enhancement
        """
        return self._model_loaded and self.model is not None
    
    def transcribe_segment(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Transcribe audio segment using large Whisper model for enhancement.
        
        Args:
            audio: Audio samples as float32 numpy array (mono, 16kHz)
            
        Returns:
            Dict[str, Any]: Enhanced transcription segment
        """
        if not self.is_model_loaded():
            logger.warning("Enhancement model not loaded")
            return {
                'text': '',
                'confidence': 0,
                'enhanced': False,
                'error': 'Enhancement model not available'
            }
        
        try:
            # Use the transcription engine to enhance the audio
            segments = self.engine.transcribe_chunk(audio)
            
            if segments and len(segments) > 0:
                segment = segments[0]
                return {
                    'text': segment.text,
                    'confidence': segment.confidence,
                    'start': segment.start,
                    'end': segment.end,
                    'words': [{'text': w.text, 'start': w.start, 'end': w.end, 'confidence': w.confidence} 
                             for w in segment.words] if segment.words else [],
                    'enhanced': True,
                    'model': self.model_name
                }
            else:
                return {
                    'text': '',
                    'confidence': 0,
                    'enhanced': False,
                    'error': 'No transcription produced'
                }
                
        except Exception as e:
            logger.error(f"Error transcribing segment for enhancement: {e}")
            return {
                'text': '',
                'confidence': 0,
                'enhanced': False,
                'error': str(e)
            }
    
    def enhance(self, segment: Dict[str, Any], audio: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Enhance the segment using large Whisper model.
        
        Args:
            segment: Segment dictionary containing text, confidence, and optionally audio
            audio: Optional audio array for enhancement (transcribe_segment if provided)
            
        Returns:
            Dict[str, Any]: Enhanced segment with improved transcription
        """
        if not self.is_model_loaded():
            # If model not available, return original segment
            return {
                'id': segment.get('id'),
                'original_text': segment.get('text', ''),
                'enhanced_text': segment.get('text', ''),
                'original_confidence': segment.get('confidence'),
                'confidence': segment.get('confidence'),
                'enhanced': False,
                'message': 'Enhancement model not available'
            }
        
        try:
            # If audio is provided, use transcribe_segment
            if audio is not None:
                result = self.transcribe_segment(audio)
                
                enhanced_segment = {
                    'id': segment.get('id'),
                    'original_text': segment.get('text', ''),
                    'enhanced_text': result.get('text', segment.get('text', '')),
                    'original_confidence': segment.get('confidence'),
                    'confidence': result.get('confidence', segment.get('confidence')),
                    'enhanced': result.get('enhanced', False),
                    'model': self.model_name,
                    'start': result.get('start', 0.0),
                    'end': result.get('end', 0.0)
                }
                
                if result.get('error'):
                    enhanced_segment['error'] = result['error']
                
                logger.debug(f"Enhanced segment {segment.get('id')} with {self.model_name} model")
                return enhanced_segment
            else:
                # If no audio, just return original segment
                return {
                    'id': segment.get('id'),
                    'original_text': segment.get('text', ''),
                    'enhanced_text': segment.get('text', ''),
                    'original_confidence': segment.get('confidence'),
                    'confidence': segment.get('confidence'),
                    'enhanced': False,
                    'message': 'No audio provided for enhancement'
                }
                
        except Exception as e:
            logger.error(f"Error enhancing segment {segment.get('id')}: {e}")
            return {
                'id': segment.get('id'),
                'original_text': segment.get('text', ''),
                'enhanced_text': segment.get('text', ''),
                'original_confidence': segment.get('confidence'),
                'confidence': segment.get('confidence'),
                'enhanced': False,
                'error': str(e)
            }


class TranscriptUpdater:
    """Real-time transcript update mechanism for enhanced segments."""
    
    def __init__(self):
        """Initialize the transcript updater."""
        self.updates = []
        self.lock = asyncio.Lock()
        
    async def add_update(self, update: Dict[str, Any]):
        """
        Add transcript update.
        
        Args:
            update: Dictionary containing update information
        """
        async with self.lock:
            self.updates.append(update)
    
    async def get_updates(self) -> List[Dict[str, Any]]:
        """
        Get all pending updates.
        
        Returns:
            List[Dict[str, Any]]: List of pending updates
        """
        async with self.lock:
            updates = self.updates.copy()
            self.updates.clear()
            return updates
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current update status.
        
        Returns:
            Dict[str, Any]: Dictionary with update statistics
        """
        return {
            'pending_updates': len(self.updates)
        }