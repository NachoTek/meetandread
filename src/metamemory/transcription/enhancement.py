"""
Enhancement module for dual-mode transcription processing.

This module implements the enhancement architecture for processing low-confidence
segments using background workers without blocking real-time transcription.
"""

import asyncio
import logging
from queue import Queue
from typing import Dict, Any, Optional, List, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import numpy as np
import time
import psutil

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
    """Async worker pool for background enhancement processing.

    Features:
    - Parallel processing using asyncio + ThreadPoolExecutor
    - Dynamic worker scaling based on system load
    - Completion callbacks for real-time transcript updates
    - Error handling with retry logic
    - Graceful degradation under resource constraints
    - Performance metrics and completion timing
    - Context tracking for recording vs post-stop scenarios
    """

    def __init__(self, num_workers: int = 4,
                 min_workers: int = 2,
                 max_workers: int = 8,
                 dynamic_scaling: bool = True,
                 cpu_usage_threshold: float = 0.8,
                 worker_scaling_algorithm: str = "adaptive"):
        """
        Initialize the worker pool with specified number of workers.

        Args:
            num_workers: Initial number of parallel workers (default: 4)
            min_workers: Minimum workers when scaling down (default: 2)
            max_workers: Maximum workers when scaling up (default: 8)
            dynamic_scaling: Enable auto-scaling based on system load (default: True)
            cpu_usage_threshold: CPU usage threshold for scaling (default: 0.8)
            worker_scaling_algorithm: Scaling algorithm - "adaptive", "linear", or "none" (default: "adaptive")
        """
        self.num_workers = num_workers
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.dynamic_scaling = dynamic_scaling
        self.cpu_usage_threshold = cpu_usage_threshold
        self.worker_scaling_algorithm = worker_scaling_algorithm

        # Initialize executor with current worker count
        self.executor = ThreadPoolExecutor(max_workers=num_workers)

        # Task tracking
        self.pending_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.retry_count = 0
        self.max_retries = 2

        # Async task tracking
        self.async_tasks: List[asyncio.Task] = []
        self.active_tasks: Dict[str, asyncio.Task] = {}

        # Worker pool state
        self.is_running = False
        self._scaling_lock = asyncio.Lock()

        # Performance metrics
        self.task_start_times: Dict[str, float] = {}
        self.avg_processing_time = 0.0
        self.last_scale_time = 0.0

        # Completion timing metrics
        self.completion_times: List[float] = []
        self.max_completion_time = 0.0
        self.min_completion_time = float('inf')

        # Context tracking (recording vs post-stop)
        self.recording_active = True
        self.tasks_during_recording = 0
        self.tasks_after_recording = 0

        # Completion callbacks
        self.completion_callbacks: List[Callable[[Dict[str, Any]], None]] = []

        logger.info(f"Initialized EnhancementWorkerPool: {num_workers} workers, scaling={dynamic_scaling}")

    async def process_segment(self, segment: Dict[str, Any],
                            processor: 'EnhancementProcessor',
                            retry_count: int = 0) -> Dict[str, Any]:
        """
        Process a segment using the enhancement processor.

        Args:
            segment: Segment dictionary to process
            processor: EnhancementProcessor instance to use for processing
            retry_count: Current retry attempt (internal use)

        Returns:
            Dict[str, Any]: Enhanced segment with results
        """
        segment_id = segment.get('id', 'unknown')
        start_time = time.time()
        self.task_start_times[segment_id] = start_time

        # Track context (during recording vs after stop)
        is_during_recording = self.recording_active
        if is_during_recording:
            self.tasks_during_recording += 1
        else:
            self.tasks_after_recording += 1

        # Check if pool is running
        if not self.is_running:
            logger.warning(f"Worker pool not running, skipping segment {segment_id}")
            return {
                'id': segment_id,
                'error': 'Worker pool not running',
                'original_text': segment.get('text', ''),
                'confidence': segment.get('confidence'),
                'enhanced': False
            }

        # Check if dynamic scaling is needed
        if self.dynamic_scaling:
            await self._maybe_scale_workers()

        self.pending_tasks += 1

        loop = asyncio.get_event_loop()

        try:
            context_str = "during recording" if is_during_recording else "after recording stopped"
            logger.debug(f"Processing segment {segment_id} {context_str} (retry {retry_count}/{self.max_retries})")

            enhanced = await loop.run_in_executor(
                self.executor,
                processor.enhance,
                segment
            )

            # Update metrics
            processing_time = time.time() - start_time
            self._update_completion_metrics(processing_time)
            self._update_avg_processing_time(processing_time)
            self.completed_tasks += 1
            self.pending_tasks -= 1

            # Remove from active tasks tracking
            if segment_id in self.active_tasks:
                del self.active_tasks[segment_id]

            # Mark as enhanced with bold formatting
            if enhanced:
                enhanced['enhanced'] = enhanced.get('enhanced', True)
                enhanced['is_enhanced'] = True  # Additional flag for UI
                enhanced['context'] = 'during_recording' if is_during_recording else 'post_recording'
                enhanced['processing_time'] = processing_time

            logger.info(
                f"Completed segment {segment_id} {context_str} in {processing_time:.2f}s "
                f"(avg: {self.avg_processing_time:.2f}s, enhanced: {enhanced.get('enhanced', False)})"
            )

            # Trigger completion callbacks
            await self._notify_completion_callbacks(enhanced)

            return enhanced

        except Exception as e:
            self.pending_tasks -= 1
            error_msg = str(e)
            logger.error(f"Error processing segment {segment_id} {context_str}: {error_msg}")

            # Retry logic with graceful degradation
            if retry_count < self.max_retries:
                self.retry_count += 1
                logger.info(f"Retrying segment {segment_id} (attempt {retry_count + 1}/{self.max_retries})")
                await asyncio.sleep(0.5 * (retry_count + 1))  # Exponential backoff
                return await self.process_segment(segment, processor, retry_count + 1)
            else:
                self.failed_tasks += 1
                logger.error(f"Max retries exceeded for segment {segment_id}, returning original")

                # Remove from active tasks tracking
                if segment_id in self.active_tasks:
                    del self.active_tasks[segment_id]

                # Graceful degradation: return original segment with error flag
                return {
                    'id': segment_id,
                    'error': error_msg,
                    'original_text': segment.get('text', ''),
                    'enhanced_text': segment.get('text', ''),  # Fallback to original
                    'confidence': segment.get('confidence'),
                    'enhanced': False,
                    'is_enhanced': False,
                    'graceful_degradation': True,
                    'context': 'during_recording' if is_during_recording else 'post_recording'
                }
        finally:
            if segment_id in self.task_start_times:
                del self.task_start_times[segment_id]
    
    async def process_segment_async(self, segment: Dict[str, Any],
                                   processor: 'EnhancementProcessor') -> asyncio.Task:
        """
        Process a segment asynchronously and return the task.
        
        Args:
            segment: Segment dictionary to process
            processor: EnhancementProcessor instance to use for processing
            
        Returns:
            asyncio.Task: Task that will complete with the enhanced segment
        """
        segment_id = segment.get('id', 'unknown')
        
        # Create async task
        task = asyncio.create_task(self.process_segment(segment, processor))
        
        # Track active task
        self.active_tasks[segment_id] = task
        self.async_tasks.append(task)
        
        # Clean up completed tasks periodically
        self._cleanup_completed_tasks()
        
        return task
    
    async def _notify_completion_callbacks(self, result: Dict[str, Any]) -> None:
        """
        Notify all registered completion callbacks.
        
        Args:
            result: Enhancement result dictionary
        """
        for callback in self.completion_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(result)
                else:
                    callback(result)
            except Exception as e:
                logger.error(f"Error in completion callback: {e}")
    
    def add_completion_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register a callback to be called when enhancement completes.
        
        Args:
            callback: Function to call with enhancement result
        """
        self.completion_callbacks.append(callback)
        logger.debug(f"Added completion callback (total: {len(self.completion_callbacks)})")
    
    def remove_completion_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Remove a completion callback.
        
        Args:
            callback: Function to remove
        """
        if callback in self.completion_callbacks:
            self.completion_callbacks.remove(callback)
            logger.debug(f"Removed completion callback (remaining: {len(self.completion_callbacks)})")
    
    async def _maybe_scale_workers(self) -> None:
        """
        Dynamically scale workers based on system load.
        Only scales if enough time has passed since last scale operation.
        """
        async with self._scaling_lock:
            now = time.time()
            time_since_last_scale = now - self.last_scale_time
            
            # Only check scaling every 30 seconds
            if time_since_last_scale < 30:
                return
            
            cpu_usage = psutil.cpu_percent(interval=0.1)
            
            if self.worker_scaling_algorithm == "none":
                return
            
            if cpu_usage > self.cpu_usage_threshold and self.num_workers > self.min_workers:
                # Scale down
                new_worker_count = max(self.min_workers, self.num_workers - 1)
                await self._scale_workers(new_worker_count)
                logger.info(f"Scaled down workers: {self.num_workers} -> {new_worker_count} (CPU: {cpu_usage}%)")
            elif cpu_usage < (self.cpu_usage_threshold * 0.7) and self.num_workers < self.max_workers:
                # Scale up
                new_worker_count = min(self.max_workers, self.num_workers + 1)
                await self._scale_workers(new_worker_count)
                logger.info(f"Scaled up workers: {self.num_workers} -> {new_worker_count} (CPU: {cpu_usage}%)")
    
    async def _scale_workers(self, new_worker_count: int) -> None:
        """
        Scale the worker pool to a new worker count.
        
        Args:
            new_worker_count: New number of workers
        """
        if new_worker_count == self.num_workers:
            return
        
        self.last_scale_time = time.time()
        
        # Shutdown old executor
        old_executor = self.executor
        old_executor.shutdown(wait=False)
        
        # Create new executor with new worker count
        self.num_workers = new_worker_count
        self.executor = ThreadPoolExecutor(max_workers=new_worker_count)
        
        # Give old executor time to finish
        await asyncio.sleep(0.1)
    
    def _update_avg_processing_time(self, processing_time: float) -> None:
        """
        Update average processing time using exponential moving average.

        Args:
            processing_time: Processing time for the last task
        """
        if self.avg_processing_time == 0:
            self.avg_processing_time = processing_time
        else:
            # EMA with alpha = 0.1
            self.avg_processing_time = 0.9 * self.avg_processing_time + 0.1 * processing_time

    def _update_completion_metrics(self, processing_time: float) -> None:
        """
        Update completion time metrics.

        Args:
            processing_time: Processing time for the last task
        """
        self.completion_times.append(processing_time)

        # Keep only last 100 completion times to avoid memory growth
        if len(self.completion_times) > 100:
            self.completion_times = self.completion_times[-100:]

        # Update min/max
        self.max_completion_time = max(self.max_completion_time, processing_time)
        self.min_completion_time = min(self.min_completion_time, processing_time)

    def set_recording_state(self, is_recording: bool) -> None:
        """
        Set the recording state for context tracking.

        Args:
            is_recording: True if recording is active, False if stopped
        """
        old_state = "active" if self.recording_active else "stopped"
        self.recording_active = is_recording
        new_state = "active" if self.recording_active else "stopped"
        logger.debug(f"Recording state changed: {old_state} -> {new_state}")

    def get_completion_metrics(self) -> Dict[str, Any]:
        """
        Get completion timing metrics.

        Returns:
            Dict[str, Any]: Dictionary with completion timing statistics
        """
        if not self.completion_times:
            return {
                'avg_completion_time': 0.0,
                'max_completion_time': 0.0,
                'min_completion_time': 0.0,
                'total_completions': 0
            }

        return {
            'avg_completion_time': sum(self.completion_times) / len(self.completion_times),
            'max_completion_time': self.max_completion_time,
            'min_completion_time': self.min_completion_time if self.min_completion_time != float('inf') else 0.0,
            'total_completions': len(self.completion_times),
            'tasks_during_recording': self.tasks_during_recording,
            'tasks_after_recording': self.tasks_after_recording
        }
    
    def _cleanup_completed_tasks(self) -> None:
        """Remove completed tasks from the tracking list."""
        self.async_tasks = [task for task in self.async_tasks if not task.done()]
    
    async def wait_for_completion(self, timeout: float = 30.0) -> None:
        """
        Wait for all pending tasks to complete.
        
        Args:
            timeout: Maximum time to wait in seconds (default: 30)
        """
        if not self.active_tasks:
            return
        
        logger.info(f"Waiting for {len(self.active_tasks)} pending tasks to complete...")
        
        try:
            await asyncio.wait_for(
                asyncio.gather(*self.active_tasks.values(), return_exceptions=True),
                timeout=timeout
            )
            logger.info("All tasks completed successfully")
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for tasks after {timeout}s")
            # Cancel remaining tasks
            for task in self.active_tasks.values():
                if not task.done():
                    task.cancel()
    
    def start(self) -> None:
        """Start the worker pool."""
        if self.is_running:
            logger.warning("Worker pool already running")
            return
            
        self.is_running = True
        self.pending_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.retry_count = 0
        logger.info(f"Started EnhancementWorkerPool with {self.num_workers} workers")
    
    async def stop(self, timeout: float = 5.0) -> None:
        """
        Stop the worker pool and clean up resources.
        
        Args:
            timeout: Maximum time to wait for pending tasks (default: 5)
        """
        if not self.is_running:
            return
            
        self.is_running = False
        logger.info("Stopping EnhancementWorkerPool...")
        
        # Wait for pending tasks to complete
        if self.active_tasks:
            await self.wait_for_completion(timeout=timeout)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        self.active_tasks.clear()
        self.async_tasks.clear()
        
        logger.info(f"Stopped EnhancementWorkerPool (completed: {self.completed_tasks}, failed: {self.failed_tasks}, retries: {self.retry_count})")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current worker pool status.

        Returns:
            Dict[str, Any]: Dictionary with worker pool statistics
        """
        cpu_usage = psutil.cpu_percent(interval=0.1) if self.dynamic_scaling else 0.0
        completion_metrics = self.get_completion_metrics()

        return {
            'num_workers': self.num_workers,
            'min_workers': self.min_workers,
            'max_workers': self.max_workers,
            'is_running': self.is_running,
            'recording_active': self.recording_active,
            'pending_tasks': self.pending_tasks,
            'completed_tasks': self.completed_tasks,
            'failed_tasks': self.failed_tasks,
            'retry_count': self.retry_count,
            'active_tasks': len(self.active_tasks),
            'dynamic_scaling': self.dynamic_scaling,
            'cpu_usage': cpu_usage,
            'cpu_usage_threshold': self.cpu_usage_threshold,
            'avg_processing_time': self.avg_processing_time,
            'active_threads': len(self.executor._threads) if hasattr(self.executor, '_threads') else 0,
            'completion_metrics': completion_metrics
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