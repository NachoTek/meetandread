"""
Enhancement module for dual-mode transcription processing.

This module implements the enhancement architecture for processing low-confidence
segments using background workers without blocking real-time transcription.

Includes accuracy measurement and benchmarking utilities for dual-mode validation.
"""

import asyncio
import logging
from queue import Queue
from typing import Dict, Any, Optional, List, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import numpy as np
import time
import psutil
import json
import os
from datetime import datetime
from statistics import mean, stdev
from difflib import SequenceMatcher

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
        logger.info(f"[ENHANCEMENT ENQUEUE] Enqueued segment {segment['id']} (queue size: {self.queue.qsize()})")
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
            logger.info(f"[ENHANCEMENT DEQUEUE] Dequeued segment {segment['id']} (queue size: {self.queue.qsize()})")
            return segment
        except:
            logger.debug("[ENHANCEMENT DEQUEUE] Queue is empty")
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
    - Dynamic worker scaling based on system load (CPU and RAM)
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
                 ram_usage_threshold: float = 0.85,
                 worker_scaling_algorithm: str = "adaptive",
                 enable_graceful_degradation: bool = True,
                 degradation_cpu_threshold: float = 0.9,
                 degradation_ram_threshold: float = 0.9,
                 degradation_strategy: str = "reduce_workers",
                 fallback_on_failure: bool = True,
                 max_retries_before_fallback: int = 2,
                 degradation_logging: bool = True,
                 queue_overflow_strategy: str = "drop_oldest"):
        """
        Initialize the worker pool with specified number of workers.

        Args:
            num_workers: Initial number of parallel workers (default: 4)
            min_workers: Minimum workers when scaling down (default: 2)
            max_workers: Maximum workers when scaling up (default: 8)
            dynamic_scaling: Enable auto-scaling based on system load (default: True)
            cpu_usage_threshold: CPU usage threshold for scaling (default: 0.8)
            ram_usage_threshold: RAM usage threshold for scaling (default: 0.85)
            worker_scaling_algorithm: Scaling algorithm - "adaptive", "linear", or "none" (default: "adaptive")
            enable_graceful_degradation: Enable graceful degradation under constraints (default: True)
            degradation_cpu_threshold: CPU threshold for degradation mode (default: 0.9)
            degradation_ram_threshold: RAM threshold for degradation mode (default: 0.9)
            degradation_strategy: Strategy for handling degradation (default: "reduce_workers")
            fallback_on_failure: Fall back to original on failure (default: True)
            max_retries_before_fallback: Max retries before fallback (default: 2)
            degradation_logging: Enable detailed degradation logging (default: True)
            queue_overflow_strategy: Strategy for queue overflow (default: "drop_oldest")
        """
        self.num_workers = num_workers
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.dynamic_scaling = dynamic_scaling
        self.cpu_usage_threshold = cpu_usage_threshold
        self.ram_usage_threshold = ram_usage_threshold
        self.worker_scaling_algorithm = worker_scaling_algorithm
        
        # Graceful degradation configuration
        self.enable_graceful_degradation = enable_graceful_degradation
        self.degradation_cpu_threshold = degradation_cpu_threshold
        self.degradation_ram_threshold = degradation_ram_threshold
        self.degradation_strategy = degradation_strategy
        self.fallback_on_failure = fallback_on_failure
        self.max_retries = max_retries_before_fallback
        self.degradation_logging = degradation_logging
        self.queue_overflow_strategy = queue_overflow_strategy

        # Initialize executor with current worker count
        self.executor = ThreadPoolExecutor(max_workers=num_workers)

        # Task tracking
        self.pending_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.retry_count = 0

        # Async task tracking
        self.async_tasks: List[asyncio.Task] = []
        self.active_tasks: Dict[str, asyncio.Task] = {}

        # Worker pool state
        self.is_running = False
        self._scaling_lock = asyncio.Lock()
        
        # Degradation state tracking
        self._degradation_mode = False
        self._degradation_start_time: Optional[float] = None
        self._degradation_events: List[Dict[str, Any]] = []
        self._max_degradation_events = 100
        self._segments_skipped_during_degradation = 0
        self._segments_fallback_during_degradation = 0
        self._last_degradation_check = 0.0
        self._degradation_check_interval = 5.0  # Check every 5 seconds

        # Performance metrics
        self.task_start_times: Dict[str, float] = {}
        self.avg_processing_time = 0.0
        self.last_scale_time = 0.0
        self.scale_check_interval = 30.0  # Seconds between scaling checks

        # System load monitoring (CPU and RAM)
        self._cpu_history: List[float] = []
        self._ram_history: List[float] = []
        self._history_max_size = 10  # Keep last 10 readings
        self._current_cpu_usage = 0.0
        self._current_ram_usage = 0.0

        # Scaling decisions log
        self._scaling_decisions: List[Dict[str, Any]] = []
        self._max_scaling_log = 50

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

        logger.info(f"Initialized EnhancementWorkerPool: {num_workers} workers, scaling={dynamic_scaling}, "
                   f"cpu_threshold={cpu_usage_threshold}, ram_threshold={ram_usage_threshold}, "
                   f"degradation={'enabled' if enable_graceful_degradation else 'disabled'}")

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
        context_str = "during recording" if is_during_recording else "after recording stopped"

        if is_during_recording:
            self.tasks_during_recording += 1
        else:
            self.tasks_after_recording += 1

        logger.info(f"[ENHANCEMENT PROCESS] Starting segment {segment_id} ({context_str}, retry: {retry_count})")

        # Check if pool is running
        if not self.is_running:
            logger.warning(f"[ENHANCEMENT PROCESS] Worker pool not running, skipping segment {segment_id}")
            return self.get_fallback_result(segment, error='Worker pool not running')

        # Check degradation state
        await self.check_degradation_state()
        
        # Apply degradation strategy if in degradation mode
        if self._degradation_mode:
            degraded_segment = await self.apply_degradation_strategy(segment)
            if degraded_segment is None:
                # Segment was skipped or queued only
                return self.get_fallback_result(segment, error='Skipped due to degradation mode')

        # Check if dynamic scaling is needed
        if self.dynamic_scaling:
            await self._maybe_scale_workers()

        self.pending_tasks += 1

        loop = asyncio.get_event_loop()

        try:
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

                # Graceful degradation: return fallback result
                return self.get_fallback_result(segment, error=error_msg)
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
        Dynamically scale workers based on system load (CPU and RAM).
        Only scales if enough time has passed since last scale operation.
        Uses smoothed metrics from history to avoid reactive scaling.
        """
        async with self._scaling_lock:
            now = time.time()
            time_since_last_scale = now - self.last_scale_time
            
            # Only check scaling at configured interval
            if time_since_last_scale < self.scale_check_interval:
                return
            
            if self.worker_scaling_algorithm == "none":
                return
            
            # Collect current system metrics
            cpu_usage = psutil.cpu_percent(interval=0.1)
            ram_usage = psutil.virtual_memory().percent / 100.0
            
            # Update history for smoothing
            self._update_resource_history(cpu_usage, ram_usage)
            
            # Get smoothed averages
            avg_cpu = self._get_avg_cpu_usage()
            avg_ram = self._get_avg_ram_usage()
            
            # Store current readings
            self._current_cpu_usage = cpu_usage
            self._current_ram_usage = ram_usage
            
            # Determine scaling action based on algorithm
            scaling_action = self._determine_scaling_action(avg_cpu, avg_ram)
            
            if scaling_action['action'] == 'scale_down':
                new_worker_count = max(self.min_workers, self.num_workers - scaling_action['delta'])
                await self._scale_workers(new_worker_count)
                self._log_scaling_decision('scale_down', new_worker_count, cpu_usage, ram_usage, scaling_action['reason'])
                
            elif scaling_action['action'] == 'scale_up':
                new_worker_count = min(self.max_workers, self.num_workers + scaling_action['delta'])
                await self._scale_workers(new_worker_count)
                self._log_scaling_decision('scale_up', new_worker_count, cpu_usage, ram_usage, scaling_action['reason'])
    
    def _update_resource_history(self, cpu_usage: float, ram_usage: float) -> None:
        """
        Update resource usage history for trend analysis.
        
        Args:
            cpu_usage: Current CPU usage (0.0-1.0)
            ram_usage: Current RAM usage (0.0-1.0)
        """
        self._cpu_history.append(cpu_usage)
        self._ram_history.append(ram_usage)
        
        # Trim to max size
        if len(self._cpu_history) > self._history_max_size:
            self._cpu_history = self._cpu_history[-self._history_max_size:]
        if len(self._ram_history) > self._history_max_size:
            self._ram_history = self._ram_history[-self._history_max_size:]
    
    def _get_avg_cpu_usage(self) -> float:
        """Get smoothed average CPU usage from history."""
        if not self._cpu_history:
            return 0.0
        return sum(self._cpu_history) / len(self._cpu_history)
    
    def _get_avg_ram_usage(self) -> float:
        """Get smoothed average RAM usage from history."""
        if not self._ram_history:
            return 0.0
        return sum(self._ram_history) / len(self._ram_history)
    
    def _determine_scaling_action(self, avg_cpu: float, avg_ram: float) -> Dict[str, Any]:
        """
        Determine scaling action based on resource metrics and algorithm.
        
        Args:
            avg_cpu: Smoothed CPU usage (0-100)
            avg_ram: Smoothed RAM usage (0.0-1.0)
            
        Returns:
            Dict with 'action', 'delta', and 'reason' keys
        """
        # Default: no action
        result = {'action': 'none', 'delta': 0, 'reason': 'Within thresholds'}
        
        # Check for resource pressure (scale down conditions)
        cpu_pressure = avg_cpu > (self.cpu_usage_threshold * 100)
        ram_pressure = avg_ram > self.ram_usage_threshold
        
        if cpu_pressure and ram_pressure:
            # Both resources under pressure - aggressive scale down
            delta = 2 if self.worker_scaling_algorithm == "adaptive" else 1
            result = {
                'action': 'scale_down',
                'delta': delta,
                'reason': f'High resource pressure (CPU: {avg_cpu:.1f}%, RAM: {avg_ram*100:.1f}%)'
            }
        elif cpu_pressure:
            # CPU pressure - moderate scale down
            result = {
                'action': 'scale_down',
                'delta': 1,
                'reason': f'High CPU usage ({avg_cpu:.1f}% > {self.cpu_usage_threshold * 100:.0f}%)'
            }
        elif ram_pressure:
            # RAM pressure - moderate scale down
            result = {
                'action': 'scale_down',
                'delta': 1,
                'reason': f'High RAM usage ({avg_ram*100:.1f}% > {self.ram_usage_threshold * 100:.0f}%)'
            }
        elif avg_cpu < (self.cpu_usage_threshold * 100 * 0.7) and avg_ram < (self.ram_usage_threshold * 0.7):
            # Low resource usage - scale up opportunity
            if self.pending_tasks > self.num_workers:
                # There's queue pressure, scale up
                delta = 2 if self.worker_scaling_algorithm == "adaptive" else 1
                result = {
                    'action': 'scale_up',
                    'delta': delta,
                    'reason': f'Low resource usage with pending tasks (CPU: {avg_cpu:.1f}%, RAM: {avg_ram*100:.1f}%)'
                }
        
        return result
    
    def _log_scaling_decision(self, action: str, new_count: int, cpu: float, ram: float, reason: str) -> None:
        """
        Log a scaling decision for monitoring and debugging.
        
        Args:
            action: 'scale_up' or 'scale_down'
            new_count: New worker count
            cpu: CPU usage at decision time
            ram: RAM usage at decision time
            reason: Human-readable reason for the decision
        """
        decision = {
            'timestamp': time.time(),
            'action': action,
            'from_workers': self.num_workers,
            'to_workers': new_count,
            'cpu_usage': cpu,
            'ram_usage': ram,
            'reason': reason
        }
        
        self._scaling_decisions.append(decision)
        
        # Trim log
        if len(self._scaling_decisions) > self._max_scaling_log:
            self._scaling_decisions = self._scaling_decisions[-self._max_scaling_log:]
        
        # Log to logger
        direction = "up" if action == "scale_up" else "down"
        logger.info(f"[SCALING] Scaled {direction}: {self.num_workers} -> {new_count} workers | "
                   f"CPU: {cpu:.1f}% | RAM: {ram*100:.1f}% | Reason: {reason}")
    
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
        ram_info = psutil.virtual_memory() if self.dynamic_scaling else None
        completion_metrics = self.get_completion_metrics()
        resource_metrics = self.get_system_metrics()
        degradation_status = self.get_degradation_status()

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
            'ram_usage': ram_info.percent / 100.0 if ram_info else 0.0,
            'ram_usage_threshold': self.ram_usage_threshold,
            'avg_processing_time': self.avg_processing_time,
            'active_threads': len(self.executor._threads) if hasattr(self.executor, '_threads') else 0,
            'completion_metrics': completion_metrics,
            'resource_metrics': resource_metrics,
            'degradation_status': degradation_status
        }
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """
        Get current system resource metrics for performance monitoring.
        
        Returns:
            Dict[str, Any]: Dictionary with CPU, RAM, and system metrics
        """
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            return {
                'cpu': {
                    'usage_percent': cpu_percent,
                    'count': cpu_count,
                    'frequency_mhz': cpu_freq.current if cpu_freq else 0,
                    'avg_usage': self._get_avg_cpu_usage()
                },
                'ram': {
                    'total_gb': memory.total / (1024**3),
                    'available_gb': memory.available / (1024**3),
                    'used_gb': memory.used / (1024**3),
                    'usage_percent': memory.percent,
                    'avg_usage': self._get_avg_ram_usage() * 100
                },
                'swap': {
                    'total_gb': swap.total / (1024**3),
                    'used_gb': swap.used / (1024**3),
                    'usage_percent': swap.percent
                },
                'thresholds': {
                    'cpu_threshold': self.cpu_usage_threshold,
                    'ram_threshold': self.ram_usage_threshold,
                    'cpu_pressure': cpu_percent > (self.cpu_usage_threshold * 100),
                    'ram_pressure': memory.percent / 100 > self.ram_usage_threshold
                }
            }
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {
                'cpu': {'usage_percent': 0, 'count': 0, 'frequency_mhz': 0, 'avg_usage': 0},
                'ram': {'total_gb': 0, 'available_gb': 0, 'used_gb': 0, 'usage_percent': 0, 'avg_usage': 0},
                'swap': {'total_gb': 0, 'used_gb': 0, 'usage_percent': 0},
                'thresholds': {
                    'cpu_threshold': self.cpu_usage_threshold,
                    'ram_threshold': self.ram_usage_threshold,
                    'cpu_pressure': False,
                    'ram_pressure': False
                },
                'error': str(e)
            }
    
    def get_worker_performance_metrics(self) -> Dict[str, Any]:
        """
        Get worker performance metrics for monitoring.
        
        Returns:
            Dict[str, Any]: Dictionary with worker performance statistics
        """
        return {
            'current_workers': self.num_workers,
            'worker_bounds': {
                'min': self.min_workers,
                'max': self.max_workers
            },
            'task_statistics': {
                'pending': self.pending_tasks,
                'completed': self.completed_tasks,
                'failed': self.failed_tasks,
                'total_retries': self.retry_count
            },
            'performance': {
                'avg_processing_time_sec': self.avg_processing_time,
                'success_rate': self.completed_tasks / max(1, self.completed_tasks + self.failed_tasks),
                'throughput_per_min': self._calculate_throughput()
            },
            'scaling': {
                'algorithm': self.worker_scaling_algorithm,
                'dynamic_scaling_enabled': self.dynamic_scaling,
                'recent_decisions': self._scaling_decisions[-5:] if self._scaling_decisions else []
            }
        }
    
    def _calculate_throughput(self) -> float:
        """Calculate tasks completed per minute."""
        if not self.completion_times:
            return 0.0
        
        # Use recent completion times to estimate throughput
        recent_count = min(len(self.completion_times), 20)
        if recent_count < 2:
            return 0.0
        
        # Estimate based on average processing time
        avg_time = sum(self.completion_times[-recent_count:]) / recent_count
        if avg_time > 0:
            return 60.0 / avg_time  # Tasks per minute per worker
        return 0.0
    
    def get_response_time_metrics(self) -> Dict[str, Any]:
        """
        Get response time tracking metrics.
        
        Returns:
            Dict[str, Any]: Dictionary with response time statistics
        """
        if not self.completion_times:
            return {
                'avg_response_time_ms': 0,
                'min_response_time_ms': 0,
                'max_response_time_ms': 0,
                'p50_response_time_ms': 0,
                'p95_response_time_ms': 0,
                'p99_response_time_ms': 0,
                'sample_count': 0
            }
        
        sorted_times = sorted(self.completion_times)
        count = len(sorted_times)
        
        def percentile(data: List[float], p: float) -> float:
            """Calculate the p-th percentile of data."""
            if not data:
                return 0.0
            k = (len(data) - 1) * p / 100
            f = int(k)
            c = f + 1 if f + 1 < len(data) else f
            return data[f] + (k - f) * (data[c] - data[f]) if c != f else data[f]
        
        return {
            'avg_response_time_ms': sum(sorted_times) / count * 1000,
            'min_response_time_ms': sorted_times[0] * 1000,
            'max_response_time_ms': sorted_times[-1] * 1000,
            'p50_response_time_ms': percentile(sorted_times, 50) * 1000,
            'p95_response_time_ms': percentile(sorted_times, 95) * 1000,
            'p99_response_time_ms': percentile(sorted_times, 99) * 1000,
            'sample_count': count
        }
    
    def check_performance_thresholds(self) -> Dict[str, Any]:
        """
        Check if performance metrics are within acceptable thresholds.
        
        Returns:
            Dict[str, Any]: Dictionary with threshold check results
        """
        warnings = []
        critical = []
        
        # Check CPU usage
        avg_cpu = self._get_avg_cpu_usage()
        if avg_cpu > self.cpu_usage_threshold * 100:
            critical.append(f"CPU usage critical: {avg_cpu:.1f}% (threshold: {self.cpu_usage_threshold * 100:.0f}%)")
        elif avg_cpu > self.cpu_usage_threshold * 100 * 0.9:
            warnings.append(f"CPU usage high: {avg_cpu:.1f}%")
        
        # Check RAM usage
        avg_ram = self._get_avg_ram_usage()
        if avg_ram > self.ram_usage_threshold:
            critical.append(f"RAM usage critical: {avg_ram * 100:.1f}% (threshold: {self.ram_usage_threshold * 100:.0f}%)")
        elif avg_ram > self.ram_usage_threshold * 0.9:
            warnings.append(f"RAM usage high: {avg_ram * 100:.1f}%")
        
        # Check failure rate
        total_tasks = self.completed_tasks + self.failed_tasks
        if total_tasks > 0:
            failure_rate = self.failed_tasks / total_tasks
            if failure_rate > 0.1:
                critical.append(f"High failure rate: {failure_rate * 100:.1f}%")
            elif failure_rate > 0.05:
                warnings.append(f"Elevated failure rate: {failure_rate * 100:.1f}%")
        
        return {
            'healthy': len(critical) == 0,
            'warnings': warnings,
            'critical': critical,
            'cpu_status': 'critical' if avg_cpu > self.cpu_usage_threshold * 100 else ('warning' if avg_cpu > self.cpu_usage_threshold * 100 * 0.9 else 'ok'),
            'ram_status': 'critical' if avg_ram > self.ram_usage_threshold else ('warning' if avg_ram > self.ram_usage_threshold * 0.9 else 'ok')
        }
    
    # ==========================================
    # Graceful Degradation Handling
    # ==========================================
    
    async def check_degradation_state(self) -> bool:
        """
        Check if system is in degradation mode based on resource usage.
        
        Returns:
            bool: True if in degradation mode
        """
        if not self.enable_graceful_degradation:
            return False
        
        now = time.time()
        
        # Throttle degradation checks
        if now - self._last_degradation_check < self._degradation_check_interval:
            return self._degradation_mode
        
        self._last_degradation_check = now
        
        # Get current resource usage
        cpu_usage = psutil.cpu_percent(interval=0.1)
        ram_usage = psutil.virtual_memory().percent / 100.0
        
        # Check degradation thresholds
        cpu_degraded = cpu_usage > (self.degradation_cpu_threshold * 100)
        ram_degraded = ram_usage > self.degradation_ram_threshold
        
        was_degraded = self._degradation_mode
        self._degradation_mode = cpu_degraded or ram_degraded
        
        # Log state transitions
        if self._degradation_mode and not was_degraded:
            self._degradation_start_time = now
            self._log_degradation_event('degradation_started', {
                'cpu_usage': cpu_usage,
                'ram_usage': ram_usage,
                'trigger': 'cpu' if cpu_degraded else 'ram'
            })
            logger.warning(f"[DEGRADATION] Entering degradation mode - CPU: {cpu_usage:.1f}%, RAM: {ram_usage*100:.1f}%")
            
        elif not self._degradation_mode and was_degraded:
            duration = now - (self._degradation_start_time or now)
            self._log_degradation_event('degradation_ended', {
                'duration_sec': duration,
                'segments_skipped': self._segments_skipped_during_degradation,
                'segments_fallback': self._segments_fallback_during_degradation
            })
            logger.info(f"[DEGRADATION] Exiting degradation mode after {duration:.1f}s")
            # Reset counters
            self._degradation_start_time = None
            self._segments_skipped_during_degradation = 0
            self._segments_fallback_during_degradation = 0
        
        return self._degradation_mode
    
    def _log_degradation_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """
        Log a degradation event for monitoring.
        
        Args:
            event_type: Type of event (degradation_started, degradation_ended, etc.)
            details: Event details dictionary
        """
        if not self.degradation_logging:
            return
        
        event = {
            'timestamp': time.time(),
            'event_type': event_type,
            'details': details
        }
        
        self._degradation_events.append(event)
        
        # Trim events log
        if len(self._degradation_events) > self._max_degradation_events:
            self._degradation_events = self._degradation_events[-self._max_degradation_events:]
    
    async def apply_degradation_strategy(self, segment: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Apply degradation strategy to a segment.
        
        Args:
            segment: Segment dictionary to process
            
        Returns:
            Optional[Dict[str, Any]]: Processed segment or None if skipped
        """
        if not self._degradation_mode:
            return segment  # Not in degradation mode, process normally
        
        self._segments_skipped_during_degradation += 1
        
        if self.degradation_strategy == "skip_low_confidence":
            # Skip segments with very low confidence during degradation
            confidence = segment.get('confidence', 100)
            if confidence < 50:  # Skip very low confidence
                self._log_degradation_event('segment_skipped', {
                    'segment_id': segment.get('id'),
                    'confidence': confidence,
                    'reason': 'low_confidence_during_degradation'
                })
                logger.debug(f"[DEGRADATION] Skipping low-confidence segment {segment.get('id')}")
                return None
            return segment
            
        elif self.degradation_strategy == "queue_only":
            # Don't process, just queue for later
            self._log_degradation_event('segment_queued', {
                'segment_id': segment.get('id'),
                'reason': 'queue_only_mode'
            })
            logger.debug(f"[DEGRADATION] Queueing segment {segment.get('id')} for later processing")
            return None  # Signal to queue but not process
            
        else:  # "reduce_workers" - default
            # Continue processing but with reduced capacity (handled by scaling)
            return segment
    
    def get_fallback_result(self, segment: Dict[str, Any], error: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a fallback result when enhancement fails.
        
        Args:
            segment: Original segment dictionary
            error: Optional error message
            
        Returns:
            Dict[str, Any]: Fallback result dictionary
        """
        self._segments_fallback_during_degradation += 1
        
        result = {
            'id': segment.get('id'),
            'original_text': segment.get('text', ''),
            'enhanced_text': segment.get('text', ''),  # Fallback to original
            'original_confidence': segment.get('confidence'),
            'confidence': segment.get('confidence'),
            'enhanced': False,
            'is_enhanced': False,
            'graceful_degradation': True,
            'fallback': True,
            'context': 'during_recording' if self.recording_active else 'post_recording'
        }
        
        if error:
            result['error'] = error
        
        if self._degradation_mode:
            result['degradation_mode'] = True
            self._log_degradation_event('fallback_applied', {
                'segment_id': segment.get('id'),
                'error': error
            })
        
        return result
    
    def get_degradation_status(self) -> Dict[str, Any]:
        """
        Get current degradation status.
        
        Returns:
            Dict[str, Any]: Dictionary with degradation status information
        """
        return {
            'degradation_enabled': self.enable_graceful_degradation,
            'in_degradation_mode': self._degradation_mode,
            'degradation_start_time': self._degradation_start_time,
            'degradation_duration_sec': time.time() - self._degradation_start_time if self._degradation_start_time else 0,
            'current_strategy': self.degradation_strategy,
            'segments_skipped': self._segments_skipped_during_degradation,
            'segments_fallback': self._segments_fallback_during_degradation,
            'recent_events': self._degradation_events[-10:] if self._degradation_events else [],
            'thresholds': {
                'cpu_threshold': self.degradation_cpu_threshold,
                'ram_threshold': self.degradation_ram_threshold
            }
        }
    
    async def handle_queue_overflow(self, queue: 'EnhancementQueue', segment: Dict[str, Any]) -> bool:
        """
        Handle queue overflow based on configured strategy.
        
        Args:
            queue: EnhancementQueue instance
            segment: Segment to enqueue
            
        Returns:
            bool: True if segment was handled, False if dropped
        """
        if self.queue_overflow_strategy == "drop_oldest":
            # Remove oldest segment to make room
            try:
                old_segment = queue.queue.get_nowait()
                self._log_degradation_event('queue_overflow_drop', {
                    'dropped_segment_id': old_segment.get('id'),
                    'new_segment_id': segment.get('id'),
                    'strategy': 'drop_oldest'
                })
                logger.warning(f"[DEGRADATION] Dropped oldest segment {old_segment.get('id')} to make room")
            except:
                pass  # Queue might be empty now
            return queue.enqueue(segment)
            
        elif self.queue_overflow_strategy == "drop_newest":
            # Drop the new segment
            self._log_degradation_event('queue_overflow_drop', {
                'dropped_segment_id': segment.get('id'),
                'strategy': 'drop_newest'
            })
            logger.warning(f"[DEGRADATION] Dropped new segment {segment.get('id')} due to queue overflow")
            queue.dropped_segments += 1
            return False
            
        else:  # "pause_enqueue"
            # Pause enqueueing (handled by caller)
            self._log_degradation_event('queue_overflow_pause', {
                'segment_id': segment.get('id'),
                'queue_size': queue.queue.qsize()
            })
            logger.warning(f"[DEGRADATION] Pausing enqueue - queue full with {queue.queue.qsize()} segments")
            return False


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


# =============================================================================
# WER (Word Error Rate) Calculation Functions
# =============================================================================

def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Calculate Word Error Rate (WER) between reference and hypothesis.
    
    WER = (S + D + I) / N where:
    - S = number of substitutions
    - D = number of deletions
    - I = number of insertions
    - N = number of words in reference
    
    Args:
        reference: Ground truth text
        hypothesis: Transcribed text to evaluate
        
    Returns:
        float: WER value (0.0 = perfect match, 1.0 = completely wrong)
    """
    ref_words = reference.lower().strip().split()
    hyp_words = hypothesis.lower().strip().split()
    
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    
    # Use dynamic programming for edit distance
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=int)
    
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)
    
    # Extract operations
    i, j = len(ref_words), len(hyp_words)
    substitutions = 0
    insertions = 0
    deletions = 0
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i - 1] == hyp_words[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
        elif j > 0 and d[i][j] == d[i][j - 1] + 1:
            insertions += 1
            j -= 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            deletions += 1
            i -= 1
    
    wer = (substitutions + deletions + insertions) / len(ref_words)
    return min(1.0, wer)  # Cap at 1.0


def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    Calculate Character Error Rate (CER) between reference and hypothesis.
    
    Args:
        reference: Ground truth text
        hypothesis: Transcribed text to evaluate
        
    Returns:
        float: CER value (0.0 = perfect match)
    """
    ref_chars = list(reference.lower().strip().replace(" ", ""))
    hyp_chars = list(hypothesis.lower().strip().replace(" ", ""))
    
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    
    # Use difflib for simpler calculation
    matcher = SequenceMatcher(None, ref_chars, hyp_chars)
    matches = sum(triple.size for triple in matcher.get_matching_blocks())
    
    errors = len(ref_chars) - matches + len(hyp_chars) - matches
    cer = errors / len(ref_chars)
    
    return min(1.0, cer)


def calculate_accuracy(reference: str, hypothesis: str) -> float:
    """
    Calculate accuracy as 1.0 - WER.
    
    Args:
        reference: Ground truth text
        hypothesis: Transcribed text to evaluate
        
    Returns:
        float: Accuracy value (0.0-1.0, higher is better)
    """
    wer = calculate_wer(reference, hypothesis)
    return max(0.0, 1.0 - wer)


# =============================================================================
# Benchmarking Data Structures
# =============================================================================

@dataclass
class AccuracyMetrics:
    """Accuracy metrics for a transcription mode."""
    wer: float = 0.0
    cer: float = 0.0
    accuracy: float = 0.0
    word_count: int = 0
    segment_count: int = 0
    avg_confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'wer': self.wer,
            'cer': self.cer,
            'accuracy': self.accuracy,
            'word_count': self.word_count,
            'segment_count': self.segment_count,
            'avg_confidence': self.avg_confidence,
        }


@dataclass
class PerformanceMetrics:
    """Performance metrics for benchmarking."""
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    total_time_s: float = 0.0
    avg_cpu_percent: float = 0.0
    max_cpu_percent: float = 0.0
    avg_ram_mb: float = 0.0
    max_ram_mb: float = 0.0
    segments_processed: int = 0
    throughput_segments_per_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'avg_latency_ms': self.avg_latency_ms,
            'max_latency_ms': self.max_latency_ms,
            'min_latency_ms': self.min_latency_ms if self.min_latency_ms != float('inf') else 0.0,
            'total_time_s': self.total_time_s,
            'avg_cpu_percent': self.avg_cpu_percent,
            'max_cpu_percent': self.max_cpu_percent,
            'avg_ram_mb': self.avg_ram_mb,
            'max_ram_mb': self.max_ram_mb,
            'segments_processed': self.segments_processed,
            'throughput_segments_per_sec': self.throughput_segments_per_sec,
        }


@dataclass
class BenchmarkResult:
    """Complete benchmark result for a single test scenario."""
    name: str
    mode: str  # "single" or "dual"
    accuracy: AccuracyMetrics = field(default_factory=AccuracyMetrics)
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'mode': self.mode,
            'accuracy': self.accuracy.to_dict(),
            'performance': self.performance.to_dict(),
            'timestamp': self.timestamp,
            'metadata': self.metadata,
        }


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking runs."""
    name: str = "default"
    warmup_segments: int = 5
    test_segments: int = 50
    confidence_threshold: float = 0.7
    collect_system_metrics: bool = True
    metrics_interval_ms: int = 100
    output_dir: str = "benchmarks"
    save_results: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'warmup_segments': self.warmup_segments,
            'test_segments': self.test_segments,
            'confidence_threshold': self.confidence_threshold,
            'collect_system_metrics': self.collect_system_metrics,
            'metrics_interval_ms': self.metrics_interval_ms,
            'output_dir': self.output_dir,
            'save_results': self.save_results,
        }


class AccuracyMeasurer:
    """
    Accuracy measurement utility for comparing transcriptions.
    
    Supports:
    - WER (Word Error Rate) calculation
    - CER (Character Error Rate) calculation
    - Accuracy comparison between modes
    - Aggregation across multiple segments
    """
    
    def __init__(self):
        """Initialize the accuracy measurer."""
        self.reset()
    
    def reset(self) -> None:
        """Reset all accumulated measurements."""
        self._wer_values: List[float] = []
        self._cer_values: List[float] = []
        self._accuracy_values: List[float] = []
        self._confidences: List[float] = []
        self._word_counts: List[int] = []
        self._segment_count: int = 0
    
    def measure(
        self,
        reference: str,
        hypothesis: str,
        confidence: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Measure accuracy for a single segment.
        
        Args:
            reference: Ground truth text
            hypothesis: Transcribed text
            confidence: Optional confidence score for the hypothesis
            
        Returns:
            Dict[str, float]: Metrics for this measurement
        """
        wer = calculate_wer(reference, hypothesis)
        cer = calculate_cer(reference, hypothesis)
        accuracy = calculate_accuracy(reference, hypothesis)
        
        self._wer_values.append(wer)
        self._cer_values.append(cer)
        self._accuracy_values.append(accuracy)
        self._word_counts.append(len(reference.split()))
        self._segment_count += 1
        
        if confidence is not None:
            self._confidences.append(confidence)
        
        return {
            'wer': wer,
            'cer': cer,
            'accuracy': accuracy,
        }
    
    def get_aggregated_metrics(self) -> AccuracyMetrics:
        """
        Get aggregated accuracy metrics across all measured segments.
        
        Returns:
            AccuracyMetrics: Aggregated metrics
        """
        if not self._wer_values:
            return AccuracyMetrics()
        
        return AccuracyMetrics(
            wer=mean(self._wer_values),
            cer=mean(self._cer_values),
            accuracy=mean(self._accuracy_values),
            word_count=sum(self._word_counts),
            segment_count=self._segment_count,
            avg_confidence=mean(self._confidences) if self._confidences else 0.0,
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get detailed statistics including standard deviation.
        
        Returns:
            Dict[str, Any]: Detailed statistics
        """
        metrics = self.get_aggregated_metrics()
        
        stats = {
            'mean': metrics.to_dict(),
            'count': self._segment_count,
        }
        
        # Add standard deviation if we have enough samples
        if len(self._wer_values) >= 2:
            stats['std'] = {
                'wer': stdev(self._wer_values),
                'cer': stdev(self._cer_values),
                'accuracy': stdev(self._accuracy_values),
            }
        
        return stats


class PerformanceMonitor:
    """
    Performance monitoring utility for benchmarking.
    
    Tracks:
    - Latency (processing time per segment)
    - CPU usage
    - RAM usage
    - Throughput
    """
    
    def __init__(self, collect_system_metrics: bool = True):
        """
        Initialize the performance monitor.
        
        Args:
            collect_system_metrics: Whether to collect CPU/RAM metrics
        """
        self._collect_system = collect_system_metrics
        self.reset()
    
    def reset(self) -> None:
        """Reset all accumulated measurements."""
        self._latencies: List[float] = []
        self._cpu_samples: List[float] = []
        self._ram_samples: List[float] = []
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._segments_processed: int = 0
    
    def start(self) -> None:
        """Start monitoring session."""
        self.reset()
        self._start_time = time.time()
    
    def stop(self) -> None:
        """Stop monitoring session."""
        self._end_time = time.time()
    
    def record_segment(
        self,
        latency_ms: float,
        cpu_percent: Optional[float] = None,
        ram_mb: Optional[float] = None
    ) -> None:
        """
        Record metrics for a processed segment.
        
        Args:
            latency_ms: Processing latency in milliseconds
            cpu_percent: CPU usage percentage (optional)
            ram_mb: RAM usage in MB (optional)
        """
        self._latencies.append(latency_ms)
        self._segments_processed += 1
        
        if self._collect_system:
            if cpu_percent is not None:
                self._cpu_samples.append(cpu_percent)
            if ram_mb is not None:
                self._ram_samples.append(ram_mb)
    
    def get_current_system_metrics(self) -> Tuple[float, float]:
        """
        Get current system metrics.
        
        Returns:
            Tuple[float, float]: (CPU percent, RAM MB)
        """
        if not self._collect_system:
            return 0.0, 0.0
        
        cpu_percent = psutil.cpu_percent(interval=0.01)
        process = psutil.Process()
        ram_mb = process.memory_info().rss / (1024 * 1024)
        
        return cpu_percent, ram_mb
    
    def get_metrics(self) -> PerformanceMetrics:
        """
        Get aggregated performance metrics.
        
        Returns:
            PerformanceMetrics: Aggregated metrics
        """
        if not self._latencies:
            return PerformanceMetrics()
        
        total_time = (self._end_time or time.time()) - (self._start_time or 0)
        
        return PerformanceMetrics(
            avg_latency_ms=mean(self._latencies),
            max_latency_ms=max(self._latencies),
            min_latency_ms=min(self._latencies),
            total_time_s=total_time,
            avg_cpu_percent=mean(self._cpu_samples) if self._cpu_samples else 0.0,
            max_cpu_percent=max(self._cpu_samples) if self._cpu_samples else 0.0,
            avg_ram_mb=mean(self._ram_samples) if self._ram_samples else 0.0,
            max_ram_mb=max(self._ram_samples) if self._ram_samples else 0.0,
            segments_processed=self._segments_processed,
            throughput_segments_per_sec=self._segments_processed / total_time if total_time > 0 else 0.0,
        )


class BenchmarkRunner:
    """
    Automated benchmark runner for dual-mode validation.
    
    Supports:
    - Running benchmarks with configurable scenarios
    - Comparing single-mode vs dual-mode performance
    - Aggregating and reporting results
    - Saving benchmark results to files
    """
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        """
        Initialize the benchmark runner.
        
        Args:
            config: Benchmark configuration
        """
        self.config = config or BenchmarkConfig()
        self.results: List[BenchmarkResult] = []
        self._accuracy_measurer = AccuracyMeasurer()
        self._performance_monitor = PerformanceMonitor(
            collect_system_metrics=self.config.collect_system_metrics
        )
    
    def run_benchmark(
        self,
        segments: List[Dict[str, Any]],
        ground_truths: List[str],
        mode: str = "single"
    ) -> BenchmarkResult:
        """
        Run a benchmark for the given segments.
        
        Args:
            segments: List of segment dictionaries with 'text' and 'confidence'
            ground_truths: List of ground truth strings for each segment
            mode: "single" or "dual" mode identifier
            
        Returns:
            BenchmarkResult: Complete benchmark result
        """
        # Reset measurers
        self._accuracy_measurer.reset()
        self._performance_monitor.start()
        
        logger.info(f"Starting benchmark: {self.config.name} (mode: {mode}, segments: {len(segments)})")
        
        # Process segments (warmup + test)
        warmup_count = min(self.config.warmup_segments, len(segments))
        test_count = min(self.config.test_segments, len(segments) - warmup_count)
        
        for i, (segment, ground_truth) in enumerate(zip(segments, ground_truths)):
            is_warmup = i < warmup_count
            
            start_time = time.time()
            
            # Measure accuracy
            text = segment.get('enhanced_text', segment.get('text', ''))
            confidence = segment.get('confidence')
            metrics = self._accuracy_measurer.measure(ground_truth, text, confidence)
            
            # Record performance
            latency_ms = (time.time() - start_time) * 1000
            cpu_percent, ram_mb = self._performance_monitor.get_current_system_metrics()
            self._performance_monitor.record_segment(latency_ms, cpu_percent, ram_mb)
            
            if not is_warmup:
                logger.debug(f"Segment {i}: WER={metrics['wer']:.3f}, latency={latency_ms:.1f}ms")
        
        self._performance_monitor.stop()
        
        # Create result
        result = BenchmarkResult(
            name=self.config.name,
            mode=mode,
            accuracy=self._accuracy_measurer.get_aggregated_metrics(),
            performance=self._performance_monitor.get_metrics(),
            metadata={
                'config': self.config.to_dict(),
                'warmup_segments': warmup_count,
                'test_segments': test_count,
            }
        )
        
        self.results.append(result)
        
        logger.info(
            f"Benchmark complete: WER={result.accuracy.wer:.3f}, "
            f"accuracy={result.accuracy.accuracy:.3f}, "
            f"avg_latency={result.performance.avg_latency_ms:.1f}ms"
        )
        
        # Save results if configured
        if self.config.save_results:
            self._save_result(result)
        
        return result
    
    def _save_result(self, result: BenchmarkResult) -> None:
        """Save benchmark result to file."""
        try:
            output_dir = self.config.output_dir
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_{result.name}_{result.mode}_{timestamp}.json"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
            
            logger.info(f"Saved benchmark result to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save benchmark result: {e}")
    
    def compare_results(
        self,
        single_mode_result: BenchmarkResult,
        dual_mode_result: BenchmarkResult
    ) -> Dict[str, Any]:
        """
        Compare single-mode and dual-mode results.
        
        Args:
            single_mode_result: Benchmark result for single mode
            dual_mode_result: Benchmark result for dual mode
            
        Returns:
            Dict[str, Any]: Comparison summary
        """
        accuracy_improvement = dual_mode_result.accuracy.accuracy - single_mode_result.accuracy.accuracy
        wer_improvement = single_mode_result.accuracy.wer - dual_mode_result.accuracy.wer
        
        latency_overhead = dual_mode_result.performance.avg_latency_ms - single_mode_result.performance.avg_latency_ms
        latency_overhead_percent = (latency_overhead / single_mode_result.performance.avg_latency_ms * 100 
                                   if single_mode_result.performance.avg_latency_ms > 0 else 0)
        
        return {
            'accuracy_improvement': accuracy_improvement,
            'wer_improvement': wer_improvement,
            'latency_overhead_ms': latency_overhead,
            'latency_overhead_percent': latency_overhead_percent,
            'single_mode': single_mode_result.to_dict(),
            'dual_mode': dual_mode_result.to_dict(),
            'improvement_ratio': accuracy_improvement / single_mode_result.accuracy.accuracy 
                               if single_mode_result.accuracy.accuracy > 0 else 0,
        }
    
    def generate_report(self) -> str:
        """
        Generate a text report of all benchmark results.
        
        Returns:
            str: Formatted report
        """
        lines = [
            "=" * 60,
            f"Benchmark Report: {self.config.name}",
            "=" * 60,
            "",
        ]
        
        for result in self.results:
            lines.extend([
                f"Mode: {result.mode}",
                "-" * 40,
                "Accuracy Metrics:",
                f"  WER: {result.accuracy.wer:.4f}",
                f"  CER: {result.accuracy.cer:.4f}",
                f"  Accuracy: {result.accuracy.accuracy:.4f}",
                f"  Segments: {result.accuracy.segment_count}",
                "",
                "Performance Metrics:",
                f"  Avg Latency: {result.performance.avg_latency_ms:.2f}ms",
                f"  Max Latency: {result.performance.max_latency_ms:.2f}ms",
                f"  Throughput: {result.performance.throughput_segments_per_sec:.2f} seg/s",
                f"  Avg CPU: {result.performance.avg_cpu_percent:.1f}%",
                f"  Avg RAM: {result.performance.avg_ram_mb:.1f}MB",
                "",
            ])
        
        return "\n".join(lines)


# =============================================================================
# Dual-Mode vs Single-Mode Comparison Utilities
# =============================================================================

@dataclass
class DualModeComparisonResult:
    """Result of comparing dual-mode vs single-mode transcription."""
    # Accuracy comparison
    single_mode_accuracy: float = 0.0
    dual_mode_accuracy: float = 0.0
    accuracy_improvement: float = 0.0
    accuracy_improvement_percent: float = 0.0
    
    # WER comparison
    single_mode_wer: float = 0.0
    dual_mode_wer: float = 0.0
    wer_improvement: float = 0.0
    wer_improvement_percent: float = 0.0
    
    # Confidence comparison
    single_mode_avg_confidence: float = 0.0
    dual_mode_avg_confidence: float = 0.0
    confidence_improvement: float = 0.0
    
    # Performance comparison
    single_mode_latency_ms: float = 0.0
    dual_mode_latency_ms: float = 0.0
    latency_overhead_ms: float = 0.0
    latency_overhead_percent: float = 0.0
    
    # Segment counts
    segments_compared: int = 0
    segments_improved: int = 0
    segments_degraded: int = 0
    segments_unchanged: int = 0
    
    # Overall assessment
    is_improvement: bool = False
    improvement_ratio: float = 0.0
    
    # Detailed segment comparison
    segment_details: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'accuracy': {
                'single_mode': self.single_mode_accuracy,
                'dual_mode': self.dual_mode_accuracy,
                'improvement': self.accuracy_improvement,
                'improvement_percent': self.accuracy_improvement_percent,
            },
            'wer': {
                'single_mode': self.single_mode_wer,
                'dual_mode': self.dual_mode_wer,
                'improvement': self.wer_improvement,
                'improvement_percent': self.wer_improvement_percent,
            },
            'confidence': {
                'single_mode': self.single_mode_avg_confidence,
                'dual_mode': self.dual_mode_avg_confidence,
                'improvement': self.confidence_improvement,
            },
            'performance': {
                'single_mode_latency_ms': self.single_mode_latency_ms,
                'dual_mode_latency_ms': self.dual_mode_latency_ms,
                'overhead_ms': self.latency_overhead_ms,
                'overhead_percent': self.latency_overhead_percent,
            },
            'segments': {
                'compared': self.segments_compared,
                'improved': self.segments_improved,
                'degraded': self.segments_degraded,
                'unchanged': self.segments_unchanged,
            },
            'overall': {
                'is_improvement': self.is_improvement,
                'improvement_ratio': self.improvement_ratio,
            },
            'segment_details': self.segment_details[:10],  # First 10 for summary
        }


class DualModeComparator:
    """
    Utility for comparing dual-mode vs single-mode transcription accuracy.
    
    Provides detailed comparison analysis including:
    - Accuracy improvement measurement
    - WER reduction analysis
    - Confidence score comparison
    - Per-segment improvement tracking
    - Statistical significance testing
    """
    
    def __init__(self, confidence_threshold: float = 0.7):
        """
        Initialize the dual-mode comparator.
        
        Args:
            confidence_threshold: Threshold below which segments should be enhanced
        """
        self.confidence_threshold = confidence_threshold
        self._comparison_results: List[DualModeComparisonResult] = []
    
    def compare(
        self,
        single_mode_segments: List[Dict[str, Any]],
        dual_mode_segments: List[Dict[str, Any]],
        ground_truths: List[str]
    ) -> DualModeComparisonResult:
        """
        Compare single-mode and dual-mode transcription results.
        
        Args:
            single_mode_segments: Segments from single-mode transcription
            dual_mode_segments: Segments from dual-mode (enhanced) transcription
            ground_truths: Ground truth strings for each segment
            
        Returns:
            DualModeComparisonResult: Detailed comparison result
        """
        if len(single_mode_segments) != len(dual_mode_segments):
            logger.warning(
                f"Segment count mismatch: single={len(single_mode_segments)}, "
                f"dual={len(dual_mode_segments)}"
            )
        
        # Initialize result
        result = DualModeComparisonResult()
        
        # Track metrics
        single_accuracies: List[float] = []
        dual_accuracies: List[float] = []
        single_wers: List[float] = []
        dual_wers: List[float] = []
        single_confidences: List[float] = []
        dual_confidences: List[float] = []
        single_latencies: List[float] = []
        dual_latencies: List[float] = []
        
        # Compare each segment
        for i, (single_seg, dual_seg, truth) in enumerate(
            zip(single_mode_segments, dual_mode_segments, ground_truths)
        ):
            # Get text from segments
            single_text = single_seg.get('text', '')
            dual_text = dual_seg.get('enhanced_text', dual_seg.get('text', ''))
            
            # Calculate accuracy metrics
            single_wer = calculate_wer(truth, single_text)
            dual_wer = calculate_wer(truth, dual_text)
            single_acc = calculate_accuracy(truth, single_text)
            dual_acc = calculate_accuracy(truth, dual_text)
            
            # Get confidence
            single_conf = single_seg.get('confidence', 0.0)
            dual_conf = dual_seg.get('confidence', single_conf)
            
            # Get latency (if available)
            single_lat = single_seg.get('processing_time', 0.0) * 1000
            dual_lat = dual_seg.get('processing_time', 0.0) * 1000
            
            # Track metrics
            single_accuracies.append(single_acc)
            dual_accuracies.append(dual_acc)
            single_wers.append(single_wer)
            dual_wers.append(dual_wer)
            single_confidences.append(single_conf)
            dual_confidences.append(dual_conf)
            single_latencies.append(single_lat)
            dual_latencies.append(dual_lat)
            
            # Track improvement
            if dual_acc > single_acc:
                result.segments_improved += 1
            elif dual_acc < single_acc:
                result.segments_degraded += 1
            else:
                result.segments_unchanged += 1
            
            # Store segment detail
            result.segment_details.append({
                'index': i,
                'ground_truth': truth[:50] + '...' if len(truth) > 50 else truth,
                'single_mode': {
                    'text': single_text[:50] + '...' if len(single_text) > 50 else single_text,
                    'wer': single_wer,
                    'accuracy': single_acc,
                    'confidence': single_conf,
                },
                'dual_mode': {
                    'text': dual_text[:50] + '...' if len(dual_text) > 50 else dual_text,
                    'wer': dual_wer,
                    'accuracy': dual_acc,
                    'confidence': dual_conf,
                },
                'improved': dual_acc > single_acc,
            })
            
            result.segments_compared += 1
        
        # Calculate aggregated metrics
        if single_accuracies:
            result.single_mode_accuracy = mean(single_accuracies)
            result.dual_mode_accuracy = mean(dual_accuracies)
            result.accuracy_improvement = result.dual_mode_accuracy - result.single_mode_accuracy
            result.accuracy_improvement_percent = (
                (result.accuracy_improvement / result.single_mode_accuracy * 100)
                if result.single_mode_accuracy > 0 else 0
            )
            
            result.single_mode_wer = mean(single_wers)
            result.dual_mode_wer = mean(dual_wers)
            result.wer_improvement = result.single_mode_wer - result.dual_mode_wer
            result.wer_improvement_percent = (
                (result.wer_improvement / result.single_mode_wer * 100)
                if result.single_mode_wer > 0 else 0
            )
            
            result.single_mode_avg_confidence = mean(single_confidences)
            result.dual_mode_avg_confidence = mean(dual_confidences)
            result.confidence_improvement = result.dual_mode_avg_confidence - result.single_mode_avg_confidence
            
            if single_latencies and any(l > 0 for l in single_latencies):
                result.single_mode_latency_ms = mean([l for l in single_latencies if l > 0])
            if dual_latencies and any(l > 0 for l in dual_latencies):
                result.dual_mode_latency_ms = mean([l for l in dual_latencies if l > 0])
            
            result.latency_overhead_ms = result.dual_mode_latency_ms - result.single_mode_latency_ms
            result.latency_overhead_percent = (
                (result.latency_overhead_ms / result.single_mode_latency_ms * 100)
                if result.single_mode_latency_ms > 0 else 0
            )
            
            # Overall assessment
            result.is_improvement = result.accuracy_improvement > 0
            result.improvement_ratio = (
                result.accuracy_improvement / result.single_mode_accuracy
                if result.single_mode_accuracy > 0 else 0
            )
        
        # Store result
        self._comparison_results.append(result)
        
        logger.info(
            f"Dual-mode comparison: accuracy={result.accuracy_improvement*100:.2f}% improvement, "
            f"WER={result.wer_improvement*100:.2f}% reduction, "
            f"segments improved={result.segments_improved}/{result.segments_compared}"
        )
        
        return result
    
    def get_comparison_history(self) -> List[DualModeComparisonResult]:
        """
        Get all comparison results.
        
        Returns:
            List[DualModeComparisonResult]: All comparison results
        """
        return self._comparison_results.copy()
    
    def generate_comparison_report(self, result: Optional[DualModeComparisonResult] = None) -> str:
        """
        Generate a detailed comparison report.
        
        Args:
            result: Comparison result to report (uses latest if None)
            
        Returns:
            str: Formatted comparison report
        """
        if result is None:
            if not self._comparison_results:
                return "No comparison results available."
            result = self._comparison_results[-1]
        
        lines = [
            "=" * 70,
            "DUAL-MODE vs SINGLE-MODE COMPARISON REPORT",
            "=" * 70,
            "",
            "ACCURACY COMPARISON",
            "-" * 40,
            f"  Single-Mode Accuracy: {result.single_mode_accuracy:.4f} ({result.single_mode_accuracy*100:.2f}%)",
            f"  Dual-Mode Accuracy:   {result.dual_mode_accuracy:.4f} ({result.dual_mode_accuracy*100:.2f}%)",
            f"  Improvement:          {result.accuracy_improvement:+.4f} ({result.accuracy_improvement_percent:+.2f}%)",
            "",
            "WER (WORD ERROR RATE) COMPARISON",
            "-" * 40,
            f"  Single-Mode WER:      {result.single_mode_wer:.4f}",
            f"  Dual-Mode WER:        {result.dual_mode_wer:.4f}",
            f"  Improvement:          {result.wer_improvement:+.4f} ({result.wer_improvement_percent:+.2f}%)",
            "",
            "CONFIDENCE COMPARISON",
            "-" * 40,
            f"  Single-Mode Avg:      {result.single_mode_avg_confidence:.2f}%",
            f"  Dual-Mode Avg:        {result.dual_mode_avg_confidence:.2f}%",
            f"  Improvement:          {result.confidence_improvement:+.2f}%",
            "",
            "PERFORMANCE COMPARISON",
            "-" * 40,
            f"  Single-Mode Latency:  {result.single_mode_latency_ms:.2f}ms",
            f"  Dual-Mode Latency:    {result.dual_mode_latency_ms:.2f}ms",
            f"  Overhead:             {result.latency_overhead_ms:+.2f}ms ({result.latency_overhead_percent:+.2f}%)",
            "",
            "SEGMENT ANALYSIS",
            "-" * 40,
            f"  Total Segments:       {result.segments_compared}",
            f"  Improved:             {result.segments_improved} ({result.segments_improved/max(1,result.segments_compared)*100:.1f}%)",
            f"  Degraded:             {result.segments_degraded} ({result.segments_degraded/max(1,result.segments_compared)*100:.1f}%)",
            f"  Unchanged:            {result.segments_unchanged} ({result.segments_unchanged/max(1,result.segments_compared)*100:.1f}%)",
            "",
            "OVERALL ASSESSMENT",
            "-" * 40,
            f"  Is Improvement:       {'YES' if result.is_improvement else 'NO'}",
            f"  Improvement Ratio:    {result.improvement_ratio:.4f}",
            "",
            "=" * 70,
        ]
        
        return "\n".join(lines)
    
    def check_significance(self, result: DualModeComparisonResult) -> Dict[str, Any]:
        """
        Check if the improvement is statistically significant.
        
        Args:
            result: Comparison result to analyze
            
        Returns:
            Dict[str, Any]: Significance analysis results
        """
        # Simple heuristic: consider significant if improvement > 5% and improved segments > 50%
        accuracy_significant = result.accuracy_improvement_percent > 5.0
        wer_significant = result.wer_improvement_percent > 5.0
        segment_significant = (
            result.segments_improved > result.segments_degraded and
            result.segments_improved / max(1, result.segments_compared) > 0.5
        )
        
        overall_significant = (accuracy_significant or wer_significant) and segment_significant
        
        return {
            'accuracy_significant': accuracy_significant,
            'wer_significant': wer_significant,
            'segment_significant': segment_significant,
            'overall_significant': overall_significant,
            'confidence_level': 'high' if overall_significant else 'medium' if segment_significant else 'low',
            'recommendation': (
                'Dual-mode provides meaningful accuracy improvement'
                if overall_significant
                else 'Consider tuning confidence threshold for better results'
                if segment_significant
                else 'Dual-mode may not be beneficial for this test case'
            ),
        }


# =============================================================================
# Test Automation and Validation
# =============================================================================

@dataclass
class TestScenario:
    """Definition of a test scenario for dual-mode validation."""
    name: str
    description: str
    audio_config: Dict[str, Any] = field(default_factory=dict)
    expected_confidence_range: Tuple[float, float] = (0.3, 0.9)
    expected_wer_range: Tuple[float, float] = (0.0, 0.5)
    min_segments: int = 10
    max_segments: int = 100
    test_duration_seconds: float = 10.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'audio_config': self.audio_config,
            'expected_confidence_range': self.expected_confidence_range,
            'expected_wer_range': self.expected_wer_range,
            'min_segments': self.min_segments,
            'max_segments': self.max_segments,
            'test_duration_seconds': self.test_duration_seconds,
            'metadata': self.metadata,
        }


@dataclass
class TestResult:
    """Result of a single test run."""
    scenario_name: str
    passed: bool
    accuracy_score: float = 0.0
    wer_score: float = 0.0
    confidence_avg: float = 0.0
    segments_tested: int = 0
    error_message: Optional[str] = None
    duration_seconds: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'scenario_name': self.scenario_name,
            'passed': self.passed,
            'accuracy_score': self.accuracy_score,
            'wer_score': self.wer_score,
            'confidence_avg': self.confidence_avg,
            'segments_tested': self.segments_tested,
            'error_message': self.error_message,
            'duration_seconds': self.duration_seconds,
            'details': self.details,
            'timestamp': self.timestamp,
        }


@dataclass
class TestSuiteResult:
    """Result of a complete test suite run."""
    suite_name: str
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0
    total_duration_seconds: float = 0.0
    test_results: List[TestResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def pass_rate(self) -> float:
        return self.passed_tests / max(1, self.total_tests)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'suite_name': self.suite_name,
            'total_tests': self.total_tests,
            'passed_tests': self.passed_tests,
            'failed_tests': self.failed_tests,
            'skipped_tests': self.skipped_tests,
            'pass_rate': self.pass_rate,
            'total_duration_seconds': self.total_duration_seconds,
            'test_results': [r.to_dict() for r in self.test_results],
            'summary': self.summary,
            'timestamp': self.timestamp,
        }


class TestRunner:
    """
    Automated test runner for dual-mode validation.
    
    Features:
    - Configurable test scenarios
    - Batch test execution
    - Pass/fail criteria validation
    - Test result reporting and logging
    - Support for integration with CI/CD
    """
    
    def __init__(
        self,
        output_dir: str = "test_results",
        save_results: bool = True,
        verbose: bool = True
    ):
        """
        Initialize the test runner.
        
        Args:
            output_dir: Directory to save test results
            save_results: Whether to save results to files
            verbose: Whether to log detailed test progress
        """
        self.output_dir = output_dir
        self.save_results = save_results
        self.verbose = verbose
        self._scenarios: List[TestScenario] = []
        self._last_suite_result: Optional[TestSuiteResult] = None
        
        # Create output directory if needed
        if self.save_results:
            os.makedirs(output_dir, exist_ok=True)
    
    def add_scenario(self, scenario: TestScenario) -> None:
        """
        Add a test scenario to the runner.
        
        Args:
            scenario: TestScenario to add
        """
        self._scenarios.append(scenario)
        logger.info(f"Added test scenario: {scenario.name}")
    
    def add_default_scenarios(self) -> None:
        """Add default test scenarios for dual-mode validation."""
        default_scenarios = [
            TestScenario(
                name="low_confidence_enhancement",
                description="Test enhancement of low-confidence segments (< 70%)",
                audio_config={'confidence_pattern': 'step', 'confidence_min': 0.3, 'confidence_max': 0.65},
                expected_confidence_range=(0.3, 0.65),
                expected_wer_range=(0.0, 0.4),
                min_segments=10,
                max_segments=50
            ),
            TestScenario(
                name="mixed_confidence_processing",
                description="Test processing with mixed confidence levels",
                audio_config={'confidence_pattern': 'sine', 'confidence_min': 0.4, 'confidence_max': 0.9},
                expected_confidence_range=(0.4, 0.9),
                expected_wer_range=(0.0, 0.35),
                min_segments=15,
                max_segments=100
            ),
            TestScenario(
                name="high_noise_accuracy",
                description="Test accuracy improvement with high noise levels",
                audio_config={'noise_level': 0.7, 'confidence_pattern': 'random'},
                expected_confidence_range=(0.2, 0.6),
                expected_wer_range=(0.0, 0.5),
                min_segments=10,
                max_segments=50
            ),
        ]
        
        for scenario in default_scenarios:
            self.add_scenario(scenario)
    
    def run_test(
        self,
        scenario: TestScenario,
        segments: List[Dict[str, Any]],
        ground_truths: List[str]
    ) -> TestResult:
        """
        Run a single test scenario.
        
        Args:
            scenario: TestScenario to run
            segments: Segments to test
            ground_truths: Ground truth strings for validation
            
        Returns:
            TestResult: Result of the test
        """
        start_time = time.time()
        
        if self.verbose:
            logger.info(f"Running test scenario: {scenario.name}")
        
        # Initialize result
        result = TestResult(
            scenario_name=scenario.name,
            passed=False
        )
        
        try:
            # Validate segment count
            if len(segments) < scenario.min_segments:
                result.error_message = f"Insufficient segments: {len(segments)} < {scenario.min_segments}"
                result.passed = False
                return result
            
            if len(segments) > scenario.max_segments:
                segments = segments[:scenario.max_segments]
                ground_truths = ground_truths[:scenario.max_segments]
            
            # Calculate metrics
            accuracies: List[float] = []
            wers: List[float] = []
            confidences: List[float] = []
            
            for segment, truth in zip(segments, ground_truths):
                text = segment.get('enhanced_text', segment.get('text', ''))
                wer = calculate_wer(truth, text)
                acc = calculate_accuracy(truth, text)
                conf = segment.get('confidence', 0.0)
                
                wers.append(wer)
                accuracies.append(acc)
                confidences.append(conf)
            
            # Aggregate results
            result.wer_score = mean(wers) if wers else 1.0
            result.accuracy_score = mean(accuracies) if accuracies else 0.0
            result.confidence_avg = mean(confidences) if confidences else 0.0
            result.segments_tested = len(segments)
            
            # Validate against expected ranges
            wer_in_range = scenario.expected_wer_range[0] <= result.wer_score <= scenario.expected_wer_range[1]
            conf_in_range = scenario.expected_confidence_range[0] <= result.confidence_avg <= scenario.expected_confidence_range[1]
            
            # Determine pass/fail
            result.passed = wer_in_range and conf_in_range
            
            # Add details
            result.details = {
                'wer_in_range': wer_in_range,
                'conf_in_range': conf_in_range,
                'actual_wer': result.wer_score,
                'actual_confidence': result.confidence_avg,
                'expected_wer_range': scenario.expected_wer_range,
                'expected_confidence_range': scenario.expected_confidence_range,
            }
            
            if self.verbose:
                status = "PASSED" if result.passed else "FAILED"
                logger.info(f"Test {scenario.name}: {status} (WER={result.wer_score:.3f}, accuracy={result.accuracy_score:.3f})")
        
        except Exception as e:
            result.error_message = str(e)
            result.passed = False
            logger.error(f"Test {scenario.name} failed with error: {e}")
        
        finally:
            result.duration_seconds = time.time() - start_time
        
        return result
    
    def run_batch(
        self,
        segments: List[Dict[str, Any]],
        ground_truths: List[str],
        suite_name: str = "default_suite"
    ) -> TestSuiteResult:
        """
        Run all test scenarios as a batch.
        
        Args:
            segments: Segments to test
            ground_truths: Ground truth strings for validation
            suite_name: Name for this test suite
            
        Returns:
            TestSuiteResult: Complete suite result
        """
        suite_start = time.time()
        
        suite_result = TestSuiteResult(suite_name=suite_name)
        suite_result.total_tests = len(self._scenarios)
        
        if self.verbose:
            logger.info(f"Starting test suite: {suite_name} ({suite_result.total_tests} scenarios)")
        
        for scenario in self._scenarios:
            result = self.run_test(scenario, segments, ground_truths)
            suite_result.test_results.append(result)
            
            if result.passed:
                suite_result.passed_tests += 1
            elif result.error_message:
                suite_result.failed_tests += 1
            else:
                suite_result.failed_tests += 1
        
        suite_result.total_duration_seconds = time.time() - suite_start
        
        # Generate summary
        suite_result.summary = {
            'pass_rate_percent': suite_result.pass_rate * 100,
            'avg_wer': mean([r.wer_score for r in suite_result.test_results]) if suite_result.test_results else 0,
            'avg_accuracy': mean([r.accuracy_score for r in suite_result.test_results]) if suite_result.test_results else 0,
            'total_segments_tested': sum(r.segments_tested for r in suite_result.test_results),
        }
        
        self._last_suite_result = suite_result
        
        if self.verbose:
            logger.info(
                f"Test suite complete: {suite_result.passed_tests}/{suite_result.total_tests} passed "
                f"({suite_result.pass_rate*100:.1f}%) in {suite_result.total_duration_seconds:.2f}s"
            )
        
        # Save results if configured
        if self.save_results:
            self._save_suite_result(suite_result)
        
        return suite_result
    
    def _save_suite_result(self, result: TestSuiteResult) -> None:
        """Save test suite result to file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_suite_{result.suite_name}_{timestamp}.json"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
            
            logger.info(f"Saved test suite result to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save test suite result: {e}")
    
    def get_last_result(self) -> Optional[TestSuiteResult]:
        """
        Get the last test suite result.
        
        Returns:
            Optional[TestSuiteResult]: Last suite result or None
        """
        return self._last_suite_result
    
    def generate_report(self, result: Optional[TestSuiteResult] = None) -> str:
        """
        Generate a detailed test report.
        
        Args:
            result: TestSuiteResult to report (uses last if None)
            
        Returns:
            str: Formatted test report
        """
        if result is None:
            result = self._last_suite_result
        
        if result is None:
            return "No test results available."
        
        lines = [
            "=" * 70,
            f"TEST SUITE REPORT: {result.suite_name}",
            "=" * 70,
            "",
            "SUMMARY",
            "-" * 40,
            f"  Total Tests:     {result.total_tests}",
            f"  Passed:          {result.passed_tests}",
            f"  Failed:          {result.failed_tests}",
            f"  Pass Rate:       {result.pass_rate*100:.1f}%",
            f"  Duration:        {result.total_duration_seconds:.2f}s",
            "",
            "AGGREGATED METRICS",
            "-" * 40,
            f"  Average WER:     {result.summary.get('avg_wer', 0):.4f}",
            f"  Average Accuracy: {result.summary.get('avg_accuracy', 0):.4f}",
            f"  Segments Tested: {result.summary.get('total_segments_tested', 0)}",
            "",
            "INDIVIDUAL TEST RESULTS",
            "-" * 40,
        ]
        
        for test_result in result.test_results:
            status = "✓ PASSED" if test_result.passed else "✗ FAILED"
            lines.extend([
                f"",
                f"  [{status}] {test_result.scenario_name}",
                f"    WER: {test_result.wer_score:.4f} | Accuracy: {test_result.accuracy_score:.4f}",
                f"    Confidence: {test_result.confidence_avg:.2f}% | Segments: {test_result.segments_tested}",
            ])
            if test_result.error_message:
                lines.append(f"    Error: {test_result.error_message}")
        
        lines.extend([
            "",
            "=" * 70,
            f"Report generated: {result.timestamp}",
            "=" * 70,
        ])
        
        return "\n".join(lines)
    
    def validate_results(
        self,
        result: TestSuiteResult,
        min_pass_rate: float = 0.8,
        max_avg_wer: float = 0.3
    ) -> Dict[str, Any]:
        """
        Validate test results against thresholds.
        
        Args:
            result: TestSuiteResult to validate
            min_pass_rate: Minimum required pass rate (default: 0.8)
            max_avg_wer: Maximum allowed average WER (default: 0.3)
            
        Returns:
            Dict[str, Any]: Validation result with pass/fail status
        """
        pass_rate_ok = result.pass_rate >= min_pass_rate
        wer_ok = result.summary.get('avg_wer', 1.0) <= max_avg_wer
        
        overall_pass = pass_rate_ok and wer_ok
        
        return {
            'overall_pass': overall_pass,
            'checks': {
                'pass_rate': {
                    'actual': result.pass_rate,
                    'threshold': min_pass_rate,
                    'passed': pass_rate_ok,
                },
                'avg_wer': {
                    'actual': result.summary.get('avg_wer', 1.0),
                    'threshold': max_avg_wer,
                    'passed': wer_ok,
                },
            },
            'recommendation': (
                'All validation checks passed'
                if overall_pass
                else 'Validation failed - review test results and adjust parameters'
            ),
        }