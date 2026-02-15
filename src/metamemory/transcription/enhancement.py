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
        status = {
            'size': self.queue.qsize(),
            'max_size': self.queue.maxsize,
            'total_enqueued': self.total_enqueued,
            'total_processed': self.total_processed,
            'dropped_segments': self.dropped_segments,
            'is_full': self.queue.full(),
            'is_empty': self.queue.empty(),
            'confidence_threshold': self.confidence_threshold
        }
        print(f"[QUEUE STATUS] Returning: size={status['size']}, total_enqueued={status['total_enqueued']}, total_processed={status['total_processed']}")
        return status
    
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

        result = {
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
        print(f"[WORKER STATUS] active_tasks={result['active_tasks']}, completed_tasks={result['completed_tasks']}, is_running={result['is_running']}, pending_tasks={result['pending_tasks']}")
        return result
    
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


# =============================================================================
# Go/No-Go Validation Framework
# =============================================================================

@dataclass
class ValidationCriteria:
    """
    Criteria for Go/No-Go validation decision.
    
    Defines the thresholds and targets that determine whether dual-mode
    enhancement provides meaningful benefit and acceptable performance.
    """
    # Accuracy improvement thresholds
    min_accuracy_improvement: float = 0.05  # 5% minimum improvement
    target_accuracy_improvement: float = 0.10  # 10% target improvement
    min_wer_reduction: float = 0.05  # 5% minimum WER reduction
    
    # Performance targets (in seconds)
    min_enhancement_completion_time: float = 15.0  # Minimum acceptable time
    max_enhancement_completion_time: float = 30.0  # Maximum acceptable time
    target_enhancement_completion_time: float = 20.0  # Target time
    
    # Resource usage thresholds
    max_cpu_percent: float = 80.0  # Maximum CPU usage %
    max_ram_gb: float = 4.0  # Maximum RAM usage in GB
    max_latency_overhead_percent: float = 50.0  # Maximum latency overhead
    
    # Segment improvement thresholds
    min_improved_segments_percent: float = 50.0  # Minimum % of segments improved
    max_degraded_segments_percent: float = 20.0  # Maximum % of segments degraded
    
    # Quality thresholds
    min_avg_confidence_improvement: float = 5.0  # Minimum confidence improvement %
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'accuracy': {
                'min_improvement': self.min_accuracy_improvement,
                'target_improvement': self.target_accuracy_improvement,
                'min_wer_reduction': self.min_wer_reduction,
            },
            'performance': {
                'min_completion_time': self.min_enhancement_completion_time,
                'max_completion_time': self.max_enhancement_completion_time,
                'target_completion_time': self.target_enhancement_completion_time,
            },
            'resources': {
                'max_cpu_percent': self.max_cpu_percent,
                'max_ram_gb': self.max_ram_gb,
                'max_latency_overhead_percent': self.max_latency_overhead_percent,
            },
            'segments': {
                'min_improved_percent': self.min_improved_segments_percent,
                'max_degraded_percent': self.max_degraded_segments_percent,
            },
            'quality': {
                'min_confidence_improvement': self.min_avg_confidence_improvement,
            },
        }


@dataclass
class ValidationResult:
    """
    Result of Go/No-Go validation run.
    
    Contains all validation metrics, pass/fail status for each criterion,
    and the overall Go/No-Go decision.
    """
    # Overall decision
    decision: str = "pending"  # "go", "no_go", "conditional_go", "pending"
    decision_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Accuracy validation
    accuracy_improvement: float = 0.0
    accuracy_pass: bool = False
    wer_reduction: float = 0.0
    wer_pass: bool = False
    
    # Performance validation
    enhancement_completion_time: float = 0.0
    performance_pass: bool = False
    latency_overhead_percent: float = 0.0
    latency_pass: bool = False
    
    # Resource validation
    cpu_usage_percent: float = 0.0
    cpu_pass: bool = False
    ram_usage_gb: float = 0.0
    ram_pass: bool = False
    
    # Segment validation
    improved_segments_percent: float = 0.0
    improved_pass: bool = False
    degraded_segments_percent: float = 0.0
    degraded_pass: bool = False
    
    # Quality validation
    confidence_improvement: float = 0.0
    confidence_pass: bool = False
    
    # Detailed metrics
    segments_validated: int = 0
    validation_duration_seconds: float = 0.0
    
    # Failure reasons
    failure_reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'decision': self.decision,
            'decision_timestamp': self.decision_timestamp,
            'validation_summary': {
                'accuracy': {
                    'improvement': self.accuracy_improvement,
                    'pass': self.accuracy_pass,
                    'wer_reduction': self.wer_reduction,
                    'wer_pass': self.wer_pass,
                },
                'performance': {
                    'completion_time': self.enhancement_completion_time,
                    'pass': self.performance_pass,
                    'latency_overhead_percent': self.latency_overhead_percent,
                    'latency_pass': self.latency_pass,
                },
                'resources': {
                    'cpu_percent': self.cpu_usage_percent,
                    'pass': self.cpu_pass,
                    'ram_gb': self.ram_usage_gb,
                    'pass': self.ram_pass,
                },
                'segments': {
                    'improved_percent': self.improved_segments_percent,
                    'pass': self.improved_pass,
                    'degraded_percent': self.degraded_segments_percent,
                    'pass': self.degraded_pass,
                },
                'quality': {
                    'confidence_improvement': self.confidence_improvement,
                    'pass': self.confidence_pass,
                },
            },
            'metrics': {
                'segments_validated': self.segments_validated,
                'duration_seconds': self.validation_duration_seconds,
            },
            'issues': {
                'failures': self.failure_reasons,
                'warnings': self.warnings,
            },
            'recommendations': self.recommendations,
        }


@dataclass
class FallbackGuidance:
    """
    Guidance for fallback scenarios when dual-mode doesn't meet criteria.
    
    Provides recommendations for single-mode fallback, resource optimization,
    and next steps for improvement.
    """
    # Fallback type
    fallback_type: str = "single_mode"  # "single_mode", "optimized_dual", "conditional_dual"
    
    # Primary recommendation
    primary_recommendation: str = ""
    
    # Resource optimization suggestions
    resource_suggestions: List[str] = field(default_factory=list)
    
    # Configuration adjustments
    config_adjustments: Dict[str, Any] = field(default_factory=dict)
    
    # Next steps
    next_steps: List[str] = field(default_factory=list)
    
    # Reason for fallback
    fallback_reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'fallback_type': self.fallback_type,
            'primary_recommendation': self.primary_recommendation,
            'resource_suggestions': self.resource_suggestions,
            'config_adjustments': self.config_adjustments,
            'next_steps': self.next_steps,
            'fallback_reason': self.fallback_reason,
        }


class GoNoGoValidator:
    """
    Automated Go/No-Go validation framework for dual-mode enhancement.
    
    Validates that dual-mode enhancement provides:
    - Meaningful accuracy improvement over single-mode
    - Acceptable performance (15-30s enhancement completion)
    - Acceptable resource usage (CPU < 80%, RAM < 4GB)
    - Net positive impact (more segments improved than degraded)
    
    Provides automated Go/No-Go decision with detailed reporting and
    fallback guidance when criteria are not met.
    """
    
    def __init__(
        self,
        criteria: Optional[ValidationCriteria] = None,
        output_dir: str = "validation_results",
        save_results: bool = True,
        verbose: bool = True
    ):
        """
        Initialize the Go/No-Go validator.
        
        Args:
            criteria: Validation criteria (uses defaults if None)
            output_dir: Directory to save validation results
            save_results: Whether to save results to files
            verbose: Whether to log detailed validation progress
        """
        self.criteria = criteria or ValidationCriteria()
        self.output_dir = output_dir
        self.save_results = save_results
        self.verbose = verbose
        
        # Validation history
        self._validation_history: List[ValidationResult] = []
        
        # Create output directory if needed
        if self.save_results:
            os.makedirs(output_dir, exist_ok=True)
        
        logger.info(
            f"Initialized GoNoGoValidator with criteria: "
            f"accuracy_improvement>={self.criteria.min_accuracy_improvement*100:.0f}%, "
            f"completion_time<={self.criteria.max_enhancement_completion_time:.0f}s"
        )
    
    def validate(
        self,
        comparison_result: DualModeComparisonResult,
        performance_metrics: Optional[PerformanceMetrics] = None,
        system_metrics: Optional[Dict[str, Any]] = None,
        enhancement_completion_time: Optional[float] = None
    ) -> ValidationResult:
        """
        Validate dual-mode enhancement against Go/No-Go criteria.
        
        Args:
            comparison_result: DualModeComparisonResult from DualModeComparator
            performance_metrics: Optional performance metrics from benchmarking
            system_metrics: Optional system resource metrics
            enhancement_completion_time: Time taken for enhancement completion
            
        Returns:
            ValidationResult: Complete validation result with Go/No-Go decision
        """
        start_time = time.time()
        
        result = ValidationResult()
        result.segments_validated = comparison_result.segments_compared
        
        if self.verbose:
            logger.info(f"Starting Go/No-Go validation for {result.segments_validated} segments")
        
        # === ACCURACY VALIDATION ===
        result.accuracy_improvement = comparison_result.accuracy_improvement
        result.accuracy_pass = result.accuracy_improvement >= self.criteria.min_accuracy_improvement
        result.wer_reduction = comparison_result.wer_improvement
        result.wer_pass = result.wer_reduction >= self.criteria.min_wer_reduction
        
        if not result.accuracy_pass:
            result.failure_reasons.append(
                f"Accuracy improvement ({result.accuracy_improvement*100:.1f}%) "
                f"below minimum ({self.criteria.min_accuracy_improvement*100:.0f}%)"
            )
        
        if not result.wer_pass:
            result.warnings.append(
                f"WER reduction ({result.wer_reduction*100:.1f}%) "
                f"below target ({self.criteria.min_wer_reduction*100:.0f}%)"
            )
        
        # === PERFORMANCE VALIDATION ===
        if enhancement_completion_time is not None:
            result.enhancement_completion_time = enhancement_completion_time
            result.performance_pass = (
                self.criteria.min_enhancement_completion_time <= 
                enhancement_completion_time <= 
                self.criteria.max_enhancement_completion_time
            )
            
            if enhancement_completion_time < self.criteria.min_enhancement_completion_time:
                result.warnings.append(
                    f"Enhancement completed very quickly ({enhancement_completion_time:.1f}s) - "
                    f"may indicate insufficient processing"
                )
            elif enhancement_completion_time > self.criteria.max_enhancement_completion_time:
                result.failure_reasons.append(
                    f"Enhancement completion time ({enhancement_completion_time:.1f}s) "
                    f"exceeds maximum ({self.criteria.max_enhancement_completion_time:.0f}s)"
                )
        
        # Latency overhead validation
        result.latency_overhead_percent = comparison_result.latency_overhead_percent
        result.latency_pass = result.latency_overhead_percent <= self.criteria.max_latency_overhead_percent
        
        if not result.latency_pass:
            result.warnings.append(
                f"Latency overhead ({result.latency_overhead_percent:.1f}%) "
                f"exceeds target ({self.criteria.max_latency_overhead_percent:.0f}%)"
            )
        
        # === RESOURCE VALIDATION ===
        if system_metrics:
            # CPU validation
            cpu_metrics = system_metrics.get('cpu', {})
            result.cpu_usage_percent = cpu_metrics.get('avg_usage', 0)
            result.cpu_pass = result.cpu_usage_percent <= self.criteria.max_cpu_percent
            
            if not result.cpu_pass:
                result.failure_reasons.append(
                    f"CPU usage ({result.cpu_usage_percent:.1f}%) "
                    f"exceeds maximum ({self.criteria.max_cpu_percent:.0f}%)"
                )
            
            # RAM validation
            ram_metrics = system_metrics.get('ram', {})
            result.ram_usage_gb = ram_metrics.get('used_gb', 0)
            result.ram_pass = result.ram_usage_gb <= self.criteria.max_ram_gb
            
            if not result.ram_pass:
                result.failure_reasons.append(
                    f"RAM usage ({result.ram_usage_gb:.1f}GB) "
                    f"exceeds maximum ({self.criteria.max_ram_gb:.0f}GB)"
                )
        else:
            # No system metrics available - mark as pass with warning
            result.cpu_pass = True
            result.ram_pass = True
            result.warnings.append("System metrics not provided - resource validation skipped")
        
        # === SEGMENT VALIDATION ===
        if comparison_result.segments_compared > 0:
            result.improved_segments_percent = (
                comparison_result.segments_improved / comparison_result.segments_compared * 100
            )
            result.degraded_segments_percent = (
                comparison_result.segments_degraded / comparison_result.segments_compared * 100
            )
        else:
            result.improved_segments_percent = 0
            result.degraded_segments_percent = 0
        
        result.improved_pass = result.improved_segments_percent >= self.criteria.min_improved_segments_percent
        result.degraded_pass = result.degraded_segments_percent <= self.criteria.max_degraded_segments_percent
        
        if not result.improved_pass:
            result.failure_reasons.append(
                f"Improved segments ({result.improved_segments_percent:.1f}%) "
                f"below minimum ({self.criteria.min_improved_segments_percent:.0f}%)"
            )
        
        if not result.degraded_pass:
            result.failure_reasons.append(
                f"Degraded segments ({result.degraded_segments_percent:.1f}%) "
                f"exceeds maximum ({self.criteria.max_degraded_segments_percent:.0f}%)"
            )
        
        # === QUALITY VALIDATION ===
        result.confidence_improvement = comparison_result.confidence_improvement
        result.confidence_pass = result.confidence_improvement >= self.criteria.min_avg_confidence_improvement
        
        if not result.confidence_pass:
            result.warnings.append(
                f"Confidence improvement ({result.confidence_improvement:.1f}%) "
                f"below target ({self.criteria.min_avg_confidence_improvement:.0f}%)"
            )
        
        # === OVERALL DECISION ===
        result.validation_duration_seconds = time.time() - start_time
        
        # Determine Go/No-Go decision
        critical_passes = [
            result.accuracy_pass,
            result.improved_pass,
            result.degraded_pass,
        ]
        
        performance_passes = [
            result.performance_pass,
            result.cpu_pass,
            result.ram_pass,
        ]
        
        if all(critical_passes) and all(performance_passes):
            result.decision = "go"
            result.recommendations.append("Dual-mode enhancement is ready for production use")
            
            if result.accuracy_improvement >= self.criteria.target_accuracy_improvement:
                result.recommendations.append(
                    f"Excellent accuracy improvement ({result.accuracy_improvement*100:.1f}%) "
                    f"exceeds target ({self.criteria.target_accuracy_improvement*100:.0f}%)"
                )
        elif all(critical_passes) and not all(performance_passes):
            result.decision = "conditional_go"
            result.recommendations.append(
                "Dual-mode provides accuracy benefits but has performance constraints"
            )
            result.recommendations.append(
                "Consider using dual-mode for high-priority recordings only"
            )
        else:
            result.decision = "no_go"
            result.recommendations.append(
                "Dual-mode enhancement does not meet validation criteria"
            )
        
        # Store result
        self._validation_history.append(result)
        
        if self.verbose:
            logger.info(
                f"Validation complete: {result.decision.upper()} "
                f"(accuracy: {'PASS' if result.accuracy_pass else 'FAIL'}, "
                f"segments: {'PASS' if result.improved_pass else 'FAIL'}, "
                f"duration: {result.validation_duration_seconds:.2f}s)"
            )
        
        # Save results if configured
        if self.save_results:
            self._save_validation_result(result)
        
        return result
    
    def validate_benchmark_results(
        self,
        single_mode_result: BenchmarkResult,
        dual_mode_result: BenchmarkResult,
        comparison_result: Optional[DualModeComparisonResult] = None
    ) -> ValidationResult:
        """
        Validate using benchmark results directly.
        
        Args:
            single_mode_result: Benchmark result for single mode
            dual_mode_result: Benchmark result for dual mode
            comparison_result: Optional pre-computed comparison result
            
        Returns:
            ValidationResult: Complete validation result
        """
        # Create comparison if not provided
        if comparison_result is None:
            comparator = DualModeComparator()
            # Create synthetic segments for comparison
            # This is a simplified validation path
            comparison_result = DualModeComparisonResult(
                single_mode_accuracy=1.0 - single_mode_result.accuracy.wer,
                dual_mode_accuracy=1.0 - dual_mode_result.accuracy.wer,
                accuracy_improvement=(1.0 - dual_mode_result.accuracy.wer) - (1.0 - single_mode_result.accuracy.wer),
                single_mode_wer=single_mode_result.accuracy.wer,
                dual_mode_wer=dual_mode_result.accuracy.wer,
                wer_improvement=single_mode_result.accuracy.wer - dual_mode_result.accuracy.wer,
                single_mode_avg_confidence=single_mode_result.accuracy.avg_confidence,
                dual_mode_avg_confidence=dual_mode_result.accuracy.avg_confidence,
                segments_compared=max(
                    single_mode_result.accuracy.segment_count,
                    dual_mode_result.accuracy.segment_count
                ),
                segments_improved=max(
                    1,
                    int(dual_mode_result.accuracy.segment_count * 
                        (1.0 if dual_mode_result.accuracy.wer < single_mode_result.accuracy.wer else 0.5))
                ),
                segments_degraded=max(
                    0,
                    int(dual_mode_result.accuracy.segment_count * 
                        (0.0 if dual_mode_result.accuracy.wer < single_mode_result.accuracy.wer else 0.3))
                ),
                is_improvement=dual_mode_result.accuracy.wer < single_mode_result.accuracy.wer,
                latency_overhead_percent=(
                    (dual_mode_result.performance.avg_latency_ms - single_mode_result.performance.avg_latency_ms) /
                    single_mode_result.performance.avg_latency_ms * 100
                    if single_mode_result.performance.avg_latency_ms > 0 else 0
                ),
            )
        
        # Extract system metrics
        system_metrics = {
            'cpu': {
                'avg_usage': dual_mode_result.performance.avg_cpu_percent,
            },
            'ram': {
                'used_gb': dual_mode_result.performance.avg_ram_mb / 1024,
            },
        }
        
        return self.validate(
            comparison_result=comparison_result,
            performance_metrics=dual_mode_result.performance,
            system_metrics=system_metrics,
            enhancement_completion_time=dual_mode_result.performance.total_time_s
        )
    
    def get_fallback_guidance(self, result: ValidationResult) -> FallbackGuidance:
        """
        Generate fallback guidance for No-Go or conditional results.
        
        Args:
            result: ValidationResult to generate guidance for
            
        Returns:
            FallbackGuidance: Detailed fallback recommendations
        """
        guidance = FallbackGuidance()
        
        if result.decision == "go":
            guidance.fallback_type = "none"
            guidance.primary_recommendation = "No fallback needed - dual-mode validation passed"
            return guidance
        
        # Determine fallback type based on failure reasons
        if not result.accuracy_pass or not result.improved_pass:
            guidance.fallback_type = "single_mode"
            guidance.fallback_reason = "Dual-mode does not provide meaningful accuracy improvement"
            guidance.primary_recommendation = (
                "Use single-mode transcription as the default mode. "
                "Consider dual-mode only for specific use cases where accuracy is critical."
            )
        elif not result.performance_pass or not result.cpu_pass or not result.ram_pass:
            guidance.fallback_type = "optimized_dual"
            guidance.fallback_reason = "Dual-mode has performance constraints"
            guidance.primary_recommendation = (
                "Optimize dual-mode configuration for better performance, "
                "or use dual-mode selectively for important recordings."
            )
        else:
            guidance.fallback_type = "conditional_dual"
            guidance.fallback_reason = "Dual-mode has some limitations"
            guidance.primary_recommendation = (
                "Use dual-mode with caution and monitor performance closely."
            )
        
        # Add resource optimization suggestions
        if not result.cpu_pass:
            guidance.resource_suggestions.extend([
                "Reduce number of enhancement workers",
                "Use smaller enhancement model (e.g., base instead of medium)",
                "Implement more aggressive confidence threshold",
                "Process enhancement only during idle periods",
            ])
        
        if not result.ram_pass:
            guidance.resource_suggestions.extend([
                "Limit concurrent enhancement tasks",
                "Use memory-efficient model variants",
                "Implement streaming enhancement instead of batch",
                "Clear model cache between sessions",
            ])
        
        if not result.performance_pass:
            guidance.resource_suggestions.extend([
                "Optimize audio chunk size for enhancement",
                "Use GPU acceleration if available",
                "Implement lazy loading for enhancement model",
                "Consider asynchronous enhancement pipeline",
            ])
        
        # Add configuration adjustments
        guidance.config_adjustments = {
            'confidence_threshold': 0.75,  # More selective
            'num_workers': 2,  # Reduce workers
            'enhancement_model': 'base',  # Smaller model
            'max_queue_size': 50,  # Smaller queue
        }
        
        if result.accuracy_improvement < 0.03:
            guidance.config_adjustments['confidence_threshold'] = 0.6  # Less selective
        
        # Add next steps
        if result.decision == "no_go":
            guidance.next_steps.extend([
                "Review validation results and identify specific issues",
                "Tune confidence threshold and worker configuration",
                "Re-run validation with adjusted parameters",
                "Consider alternative enhancement strategies",
                "Document findings and update configuration",
            ])
        else:  # conditional_go
            guidance.next_steps.extend([
                "Monitor dual-mode performance in production",
                "Set up alerts for resource usage thresholds",
                "Implement graceful degradation handling",
                "Periodically re-run validation to track improvements",
            ])
        
        return guidance
    
    def _save_validation_result(self, result: ValidationResult) -> None:
        """Save validation result to file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"validation_result_{result.decision}_{timestamp}.json"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
            
            # Also save a markdown report
            report = self.generate_report(result)
            report_filepath = os.path.join(self.output_dir, f"validation_report_{timestamp}.md")
            with open(report_filepath, 'w') as f:
                f.write(report)
            
            logger.info(f"Saved validation result to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save validation result: {e}")
    
    def generate_report(self, result: Optional[ValidationResult] = None) -> str:
        """
        Generate a detailed validation report.
        
        Args:
            result: ValidationResult to report (uses latest if None)
            
        Returns:
            str: Formatted validation report in markdown
        """
        if result is None:
            if not self._validation_history:
                return "No validation results available."
            result = self._validation_history[-1]
        
        # Determine emoji for decision
        decision_emoji = {
            "go": "✅",
            "no_go": "❌",
            "conditional_go": "⚠️",
            "pending": "⏳",
        }.get(result.decision, "❓")
        
        lines = [
            "# Go/No-Go Validation Report",
            "",
            f"**Decision:** {decision_emoji} **{result.decision.upper()}**",
            f"**Timestamp:** {result.decision_timestamp}",
            f"**Segments Validated:** {result.segments_validated}",
            f"**Validation Duration:** {result.validation_duration_seconds:.2f}s",
            "",
            "---",
            "",
            "## Validation Summary",
            "",
            "| Category | Metric | Value | Threshold | Status |",
            "|----------|--------|-------|-----------|--------|",
            f"| Accuracy | Improvement | {result.accuracy_improvement*100:.1f}% | ≥{self.criteria.min_accuracy_improvement*100:.0f}% | {'✅ PASS' if result.accuracy_pass else '❌ FAIL'} |",
            f"| Accuracy | WER Reduction | {result.wer_reduction*100:.1f}% | ≥{self.criteria.min_wer_reduction*100:.0f}% | {'✅ PASS' if result.wer_pass else '⚠️ WARN'} |",
            f"| Performance | Completion Time | {result.enhancement_completion_time:.1f}s | {self.criteria.min_enhancement_completion_time:.0f}-{self.criteria.max_enhancement_completion_time:.0f}s | {'✅ PASS' if result.performance_pass else '❌ FAIL'} |",
            f"| Performance | Latency Overhead | {result.latency_overhead_percent:.1f}% | ≤{self.criteria.max_latency_overhead_percent:.0f}% | {'✅ PASS' if result.latency_pass else '⚠️ WARN'} |",
            f"| Resources | CPU Usage | {result.cpu_usage_percent:.1f}% | ≤{self.criteria.max_cpu_percent:.0f}% | {'✅ PASS' if result.cpu_pass else '❌ FAIL'} |",
            f"| Resources | RAM Usage | {result.ram_usage_gb:.1f}GB | ≤{self.criteria.max_ram_gb:.0f}GB | {'✅ PASS' if result.ram_pass else '❌ FAIL'} |",
            f"| Segments | Improved | {result.improved_segments_percent:.1f}% | ≥{self.criteria.min_improved_segments_percent:.0f}% | {'✅ PASS' if result.improved_pass else '❌ FAIL'} |",
            f"| Segments | Degraded | {result.degraded_segments_percent:.1f}% | ≤{self.criteria.max_degraded_segments_percent:.0f}% | {'✅ PASS' if result.degraded_pass else '❌ FAIL'} |",
            f"| Quality | Confidence Improvement | {result.confidence_improvement:.1f}% | ≥{self.criteria.min_avg_confidence_improvement:.0f}% | {'✅ PASS' if result.confidence_pass else '⚠️ WARN'} |",
            "",
        ]
        
        # Add failure reasons if any
        if result.failure_reasons:
            lines.extend([
                "## ❌ Failure Reasons",
                "",
            ])
            for reason in result.failure_reasons:
                lines.append(f"- {reason}")
            lines.append("")
        
        # Add warnings if any
        if result.warnings:
            lines.extend([
                "## ⚠️ Warnings",
                "",
            ])
            for warning in result.warnings:
                lines.append(f"- {warning}")
            lines.append("")
        
        # Add recommendations
        if result.recommendations:
            lines.extend([
                "## 📋 Recommendations",
                "",
            ])
            for rec in result.recommendations:
                lines.append(f"- {rec}")
            lines.append("")
        
        # Add fallback guidance if not a clean Go
        if result.decision != "go":
            guidance = self.get_fallback_guidance(result)
            lines.extend([
                "## 🔄 Fallback Guidance",
                "",
                f"**Type:** {guidance.fallback_type}",
                "",
                f"**Reason:** {guidance.fallback_reason}",
                "",
                f"**Primary Recommendation:** {guidance.primary_recommendation}",
                "",
            ])
            
            if guidance.resource_suggestions:
                lines.append("### Resource Optimization Suggestions")
                lines.append("")
                for suggestion in guidance.resource_suggestions:
                    lines.append(f"- {suggestion}")
                lines.append("")
            
            if guidance.config_adjustments:
                lines.append("### Suggested Configuration Adjustments")
                lines.append("")
                lines.append("```python")
                for key, value in guidance.config_adjustments.items():
                    if isinstance(value, str):
                        lines.append(f'{key} = "{value}"')
                    else:
                        lines.append(f'{key} = {value}')
                lines.append("```")
                lines.append("")
            
            if guidance.next_steps:
                lines.append("### Next Steps")
                lines.append("")
                for i, step in enumerate(guidance.next_steps, 1):
                    lines.append(f"{i}. {step}")
                lines.append("")
        
        lines.extend([
            "---",
            f"*Report generated by GoNoGoValidator*",
        ])
        
        return "\n".join(lines)
    
    def export_results(
        self,
        result: Optional[ValidationResult] = None,
        format: str = "json",
        filepath: Optional[str] = None
    ) -> str:
        """
        Export validation results in specified format.
        
        Args:
            result: ValidationResult to export (uses latest if None)
            format: Export format ("json", "markdown", "summary")
            filepath: Optional filepath to save (auto-generated if None)
            
        Returns:
            str: Path to exported file
        """
        if result is None:
            if not self._validation_history:
                raise ValueError("No validation results available to export")
            result = self._validation_history[-1]
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = "md" if format in ["markdown", "md"] else "json"
            filepath = os.path.join(self.output_dir, f"validation_export_{timestamp}.{ext}")
        
        if format in ["json"]:
            content = json.dumps(result.to_dict(), indent=2)
        elif format in ["markdown", "md"]:
            content = self.generate_report(result)
        elif format == "summary":
            content = self._generate_summary(result)
        else:
            raise ValueError(f"Unknown export format: {format}")
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        logger.info(f"Exported validation results to {filepath}")
        return filepath
    
    def _generate_summary(self, result: ValidationResult) -> str:
        """Generate a one-line summary of validation result."""
        return (
            f"Go/No-Go: {result.decision.upper()} | "
            f"Accuracy: {result.accuracy_improvement*100:.1f}% | "
            f"Segments: {result.improved_segments_percent:.0f}% improved | "
            f"Time: {result.enhancement_completion_time:.1f}s | "
            f"CPU: {result.cpu_usage_percent:.0f}% | "
            f"RAM: {result.ram_usage_gb:.1f}GB"
        )
    
    def generate_summary_report(self, result: Optional[ValidationResult] = None) -> str:
        """
        Generate a concise summary report.
        
        Args:
            result: ValidationResult to report (uses latest if None)
            
        Returns:
            str: Concise summary report
        """
        if result is None:
            if not self._validation_history:
                return "No validation results available."
            result = self._validation_history[-1]
        
        decision_icon = {"go": "✅", "no_go": "❌", "conditional_go": "⚠️"}.get(result.decision, "❓")
        
        lines = [
            f"{decision_icon} **Validation Result: {result.decision.upper()}**",
            "",
            f"| Metric | Value | Status |",
            f"|--------|-------|--------|",
            f"| Accuracy Improvement | {result.accuracy_improvement*100:.1f}% | {'✅' if result.accuracy_pass else '❌'} |",
            f"| WER Reduction | {result.wer_reduction*100:.1f}% | {'✅' if result.wer_pass else '⚠️'} |",
            f"| Completion Time | {result.enhancement_completion_time:.1f}s | {'✅' if result.performance_pass else '❌'} |",
            f"| Improved Segments | {result.improved_segments_percent:.0f}% | {'✅' if result.improved_pass else '❌'} |",
            f"| Degraded Segments | {result.degraded_segments_percent:.0f}% | {'✅' if result.degraded_pass else '❌'} |",
            f"| CPU Usage | {result.cpu_usage_percent:.0f}% | {'✅' if result.cpu_pass else '❌'} |",
            f"| RAM Usage | {result.ram_usage_gb:.1f}GB | {'✅' if result.ram_pass else '❌'} |",
        ]
        
        return "\n".join(lines)
    
    def generate_detailed_report(self, result: Optional[ValidationResult] = None) -> str:
        """
        Generate a detailed validation report with analysis.
        
        Args:
            result: ValidationResult to report (uses latest if None)
            
        Returns:
            str: Detailed validation report in markdown
        """
        if result is None:
            if not self._validation_history:
                return "No validation results available."
            result = self._validation_history[-1]
        
        # Get interpretation
        interpretation = self.interpret_validation_result(result)
        
        # Build detailed report
        lines = [
            "# Go/No-Go Validation Detailed Report",
            "",
            f"**Generated:** {datetime.now().isoformat()}",
            f"**Decision:** {result.decision.upper()}",
            f"**Segments Validated:** {result.segments_validated}",
            "",
            "---",
            "",
            "## Executive Summary",
            "",
            interpretation['overall_assessment'],
            "",
        ]
        
        # Strengths
        if interpretation['strengths']:
            lines.extend([
                "### Strengths",
                "",
            ])
            for strength in interpretation['strengths']:
                lines.append(f"- ✅ {strength}")
            lines.append("")
        
        # Weaknesses
        if interpretation['weaknesses']:
            lines.extend([
                "### Weaknesses",
                "",
            ])
            for weakness in interpretation['weaknesses']:
                lines.append(f"- ⚠️ {weakness}")
            lines.append("")
        
        # Critical issues
        if interpretation['critical_issues']:
            lines.extend([
                "### Critical Issues",
                "",
            ])
            for issue in interpretation['critical_issues']:
                lines.append(f"- ❌ {issue}")
            lines.append("")
        
        # Detailed metrics section
        lines.extend([
            "---",
            "",
            "## Detailed Metrics",
            "",
            "### Accuracy Metrics",
            "",
            f"| Metric | Value | Threshold | Status |",
            f"|--------|-------|-----------|--------|",
            f"| Accuracy Improvement | {result.accuracy_improvement*100:.2f}% | ≥{self.criteria.min_accuracy_improvement*100:.0f}% | {'PASS' if result.accuracy_pass else 'FAIL'} |",
            f"| WER Reduction | {result.wer_reduction*100:.2f}% | ≥{self.criteria.min_wer_reduction*100:.0f}% | {'PASS' if result.wer_pass else 'WARN'} |",
            f"| Confidence Improvement | {result.confidence_improvement:.1f}% | ≥{self.criteria.min_avg_confidence_improvement:.0f}% | {'PASS' if result.confidence_pass else 'WARN'} |",
            "",
            "### Performance Metrics",
            "",
            f"| Metric | Value | Threshold | Status |",
            f"|--------|-------|-----------|--------|",
            f"| Enhancement Time | {result.enhancement_completion_time:.2f}s | {self.criteria.min_enhancement_completion_time:.0f}-{self.criteria.max_enhancement_completion_time:.0f}s | {'PASS' if result.performance_pass else 'FAIL'} |",
            f"| Latency Overhead | {result.latency_overhead_percent:.1f}% | ≤{self.criteria.max_latency_overhead_percent:.0f}% | {'PASS' if result.latency_pass else 'WARN'} |",
            f"| Validation Duration | {result.validation_duration_seconds:.2f}s | - | - |",
            "",
            "### Resource Metrics",
            "",
            f"| Metric | Value | Threshold | Status |",
            f"|--------|-------|-----------|--------|",
            f"| CPU Usage | {result.cpu_usage_percent:.1f}% | ≤{self.criteria.max_cpu_percent:.0f}% | {'PASS' if result.cpu_pass else 'FAIL'} |",
            f"| RAM Usage | {result.ram_usage_gb:.2f}GB | ≤{self.criteria.max_ram_gb:.0f}GB | {'PASS' if result.ram_pass else 'FAIL'} |",
            "",
            "### Segment Analysis",
            "",
            f"| Metric | Value | Threshold | Status |",
            f"|--------|-------|-----------|--------|",
            f"| Segments Validated | {result.segments_validated} | - | - |",
            f"| Improved | {result.improved_segments_percent:.1f}% | ≥{self.criteria.min_improved_segments_percent:.0f}% | {'PASS' if result.improved_pass else 'FAIL'} |",
            f"| Degraded | {result.degraded_segments_percent:.1f}% | ≤{self.criteria.max_degraded_segments_percent:.0f}% | {'PASS' if result.degraded_pass else 'FAIL'} |",
            "",
        ])
        
        # Improvement potential
        if interpretation['improvement_potential']:
            lines.extend([
                "---",
                "",
                "## Improvement Opportunities",
                "",
            ])
            for opportunity in interpretation['improvement_potential']:
                lines.append(f"- 💡 {opportunity}")
            lines.append("")
        
        # Recommendations
        if result.recommendations:
            lines.extend([
                "## Recommendations",
                "",
            ])
            for rec in result.recommendations:
                lines.append(f"- {rec}")
            lines.append("")
        
        # Fallback guidance if applicable
        if result.decision != "go":
            guidance = self.get_fallback_guidance(result)
            lines.extend([
                "## Fallback Guidance",
                "",
                f"**Type:** {guidance.fallback_type}",
                "",
                f"**Reason:** {guidance.fallback_reason}",
                "",
                f"**Recommendation:** {guidance.primary_recommendation}",
                "",
            ])
            
            if guidance.next_steps:
                lines.append("### Next Steps")
                lines.append("")
                for i, step in enumerate(guidance.next_steps, 1):
                    lines.append(f"{i}. {step}")
                lines.append("")
        
        return "\n".join(lines)
    
    def export_results_csv(
        self,
        results: Optional[List[ValidationResult]] = None,
        filepath: Optional[str] = None
    ) -> str:
        """
        Export validation results to CSV format.
        
        Args:
            results: Results to export (uses history if None)
            filepath: Optional filepath (auto-generated if None)
            
        Returns:
            str: Path to exported file
        """
        if results is None:
            results = self._validation_history
        
        if not results:
            raise ValueError("No validation results available to export")
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.output_dir, f"validation_results_{timestamp}.csv")
        
        import csv
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'timestamp', 'decision', 'segments_validated',
                'accuracy_improvement', 'accuracy_pass',
                'wer_reduction', 'wer_pass',
                'completion_time', 'performance_pass',
                'latency_overhead_percent', 'latency_pass',
                'cpu_percent', 'cpu_pass',
                'ram_gb', 'ram_pass',
                'improved_percent', 'improved_pass',
                'degraded_percent', 'degraded_pass',
                'confidence_improvement', 'confidence_pass',
                'validation_duration', 'failure_count', 'warning_count'
            ])
            
            # Data rows
            for result in results:
                writer.writerow([
                    result.decision_timestamp,
                    result.decision,
                    result.segments_validated,
                    f"{result.accuracy_improvement:.4f}",
                    result.accuracy_pass,
                    f"{result.wer_reduction:.4f}",
                    result.wer_pass,
                    f"{result.enhancement_completion_time:.2f}",
                    result.performance_pass,
                    f"{result.latency_overhead_percent:.1f}",
                    result.latency_pass,
                    f"{result.cpu_usage_percent:.1f}",
                    result.cpu_pass,
                    f"{result.ram_usage_gb:.2f}",
                    result.ram_pass,
                    f"{result.improved_segments_percent:.1f}",
                    result.improved_pass,
                    f"{result.degraded_segments_percent:.1f}",
                    result.degraded_pass,
                    f"{result.confidence_improvement:.1f}",
                    result.confidence_pass,
                    f"{result.validation_duration_seconds:.2f}",
                    len(result.failure_reasons),
                    len(result.warnings),
                ])
        
        logger.info(f"Exported {len(results)} validation results to {filepath}")
        return filepath
    
    def export_results_html(
        self,
        result: Optional[ValidationResult] = None,
        filepath: Optional[str] = None
    ) -> str:
        """
        Export validation result as HTML report.
        
        Args:
            result: Result to export (uses latest if None)
            filepath: Optional filepath (auto-generated if None)
            
        Returns:
            str: Path to exported file
        """
        if result is None:
            if not self._validation_history:
                raise ValueError("No validation results available to export")
            result = self._validation_history[-1]
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.output_dir, f"validation_report_{timestamp}.html")
        
        # Generate markdown and convert to HTML
        markdown_report = self.generate_detailed_report(result)
        
        # Simple markdown to HTML conversion
        html_lines = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<title>Go/No-Go Validation Report</title>",
            "<style>",
            "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; }",
            "h1 { color: #333; border-bottom: 2px solid #333; padding-bottom: 10px; }",
            "h2 { color: #555; margin-top: 30px; }",
            "h3 { color: #666; }",
            "table { border-collapse: collapse; width: 100%; margin: 20px 0; }",
            "th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }",
            "th { background-color: #4CAF50; color: white; }",
            "tr:nth-child(even) { background-color: #f2f2f2; }",
            ".pass { color: green; font-weight: bold; }",
            ".fail { color: red; font-weight: bold; }",
            ".warn { color: orange; font-weight: bold; }",
            "code { background-color: #f4f4f4; padding: 2px 6px; border-radius: 3px; }",
            "pre { background-color: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }",
            "</style>",
            "</head>",
            "<body>",
        ]
        
        # Convert markdown to HTML
        in_code_block = False
        in_table = False
        
        for line in markdown_report.split('\n'):
            # Code blocks
            if line.strip().startswith('```'):
                if in_code_block:
                    html_lines.append('</code></pre>')
                    in_code_block = False
                else:
                    html_lines.append('<pre><code>')
                    in_code_block = True
                continue
            
            if in_code_block:
                html_lines.append(line)
                continue
            
            # Tables
            if line.strip().startswith('|'):
                if not in_table:
                    html_lines.append('<table>')
                    in_table = True
                
                # Skip separator rows
                if '---' in line:
                    continue
                
                cells = [c.strip() for c in line.split('|')[1:-1]]
                if cells:
                    # First row is header
                    if html_lines[-1] == '<table>':
                        html_lines.append('<tr>' + ''.join(f'<th>{c}</th>' for c in cells) + '</tr>')
                    else:
                        html_lines.append('<tr>' + ''.join(f'<td>{c}</td>' for c in cells) + '</tr>')
                continue
            elif in_table:
                html_lines.append('</table>')
                in_table = False
            
            # Headers
            if line.startswith('# '):
                html_lines.append(f'<h1>{line[2:]}</h1>')
            elif line.startswith('## '):
                html_lines.append(f'<h2>{line[3:]}</h2>')
            elif line.startswith('### '):
                html_lines.append(f'<h3>{line[4:]}</h3>')
            # Horizontal rule
            elif line.strip() == '---':
                html_lines.append('<hr>')
            # List items
            elif line.strip().startswith('- '):
                content = line.strip()[2:]
                # Add styling for status indicators
                content = content.replace('✅', '<span class="pass">✅</span>')
                content = content.replace('❌', '<span class="fail">❌</span>')
                content = content.replace('⚠️', '<span class="warn">⚠️</span>')
                html_lines.append(f'<li>{content}</li>')
            elif line.strip() and line.strip()[0].isdigit() and '. ' in line:
                # Numbered list
                content = line.strip().split('. ', 1)[1] if '. ' in line else line.strip()
                html_lines.append(f'<li>{content}</li>')
            # Bold text
            elif line.strip().startswith('**') and '**' in line[2:]:
                parts = line.split('**')
                for i in range(1, len(parts) - 1, 2):
                    parts[i] = f'<strong>{parts[i]}</strong>'
                html_lines.append('<p>' + ''.join(parts) + '</p>')
            # Regular paragraph
            elif line.strip():
                html_lines.append(f'<p>{line}</p>')
            else:
                html_lines.append('')
        
        # Close any open tags
        if in_table:
            html_lines.append('</table>')
        
        html_lines.extend([
            "</body>",
            "</html>",
        ])
        
        with open(filepath, 'w') as f:
            f.write('\n'.join(html_lines))
        
        logger.info(f"Exported HTML report to {filepath}")
        return filepath
    
    def persist_results(
        self,
        result: Optional[ValidationResult] = None,
        formats: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Persist validation results in multiple formats.
        
        Args:
            result: Result to persist (uses latest if None)
            formats: List of formats to export (default: all)
            
        Returns:
            Dict[str, str]: Mapping of format to filepath
        """
        if result is None:
            if not self._validation_history:
                raise ValueError("No validation results available to persist")
            result = self._validation_history[-1]
        
        if formats is None:
            formats = ['json', 'markdown', 'html']
        
        exported = {}
        
        if 'json' in formats:
            exported['json'] = self.export_results(result, format='json')
        
        if 'markdown' in formats or 'md' in formats:
            exported['markdown'] = self.export_results(result, format='markdown')
        
        if 'html' in formats:
            exported['html'] = self.export_results_html(result)
        
        if 'csv' in formats:
            exported['csv'] = self.export_results_csv([result])
        
        if 'summary' in formats:
            exported['summary'] = self.export_results(result, format='summary')
        
        logger.info(f"Persisted validation results in {len(exported)} formats")
        return exported
    
    def generate_trend_report(self) -> str:
        """
        Generate a trend report across all validation history.
        
        Returns:
            str: Trend analysis report in markdown
        """
        if len(self._validation_history) < 2:
            return "Insufficient validation history for trend analysis (need at least 2 results)."
        
        lines = [
            "# Validation Trend Report",
            "",
            f"**Analysis Period:** {self._validation_history[0].decision_timestamp} to {self._validation_history[-1].decision_timestamp}",
            f"**Total Validations:** {len(self._validation_history)}",
            "",
        ]
        
        # Count decisions
        decisions = [r.decision for r in self._validation_history]
        go_count = decisions.count('go')
        no_go_count = decisions.count('no_go')
        conditional_count = decisions.count('conditional_go')
        
        lines.extend([
            "## Decision Summary",
            "",
            f"| Decision | Count | Percentage |",
            f"|----------|-------|------------|",
            f"| ✅ Go | {go_count} | {go_count/len(decisions)*100:.1f}% |",
            f"| ⚠️ Conditional Go | {conditional_count} | {conditional_count/len(decisions)*100:.1f}% |",
            f"| ❌ No-Go | {no_go_count} | {no_go_count/len(decisions)*100:.1f}% |",
            "",
        ])
        
        # Calculate averages
        avg_accuracy = sum(r.accuracy_improvement for r in self._validation_history) / len(self._validation_history)
        avg_improved = sum(r.improved_segments_percent for r in self._validation_history) / len(self._validation_history)
        avg_time = sum(r.enhancement_completion_time for r in self._validation_history) / len(self._validation_history)
        avg_cpu = sum(r.cpu_usage_percent for r in self._validation_history) / len(self._validation_history)
        
        lines.extend([
            "## Average Metrics",
            "",
            f"| Metric | Average |",
            f"|--------|---------|",
            f"| Accuracy Improvement | {avg_accuracy*100:.1f}% |",
            f"| Improved Segments | {avg_improved:.1f}% |",
            f"| Completion Time | {avg_time:.1f}s |",
            f"| CPU Usage | {avg_cpu:.0f}% |",
            "",
        ])
        
        # Trend direction
        first_half = self._validation_history[:len(self._validation_history)//2]
        second_half = self._validation_history[len(self._validation_history)//2:]
        
        if first_half and second_half:
            first_acc = sum(r.accuracy_improvement for r in first_half) / len(first_half)
            second_acc = sum(r.accuracy_improvement for r in second_half) / len(second_half)
            
            if second_acc > first_acc + 0.01:
                trend = "📈 Improving"
            elif second_acc < first_acc - 0.01:
                trend = "📉 Declining"
            else:
                trend = "➡️ Stable"
            
            lines.extend([
                "## Trend Analysis",
                "",
                f"**Accuracy Trend:** {trend}",
                f"- First Half Average: {first_acc*100:.1f}%",
                f"- Second Half Average: {second_acc*100:.1f}%",
                "",
            ])
        
        return "\n".join(lines)
    
    # =========================================================================
    # Fallback Guidance and Recommendations
    # =========================================================================
    
    def get_single_mode_fallback_guidance(self, result: ValidationResult) -> FallbackGuidance:
        """
        Get guidance for falling back to single-mode transcription.
        
        Args:
            result: ValidationResult that triggered fallback
            
        Returns:
            FallbackGuidance: Detailed single-mode fallback guidance
        """
        guidance = FallbackGuidance()
        guidance.fallback_type = "single_mode"
        guidance.fallback_reason = (
            "Dual-mode enhancement does not provide sufficient benefit to justify "
            "the additional complexity and resource usage."
        )
        guidance.primary_recommendation = (
            "Use single-mode (real-time only) transcription as the default. "
            "This provides the best balance of simplicity, resource usage, and latency."
        )
        
        # Add specific guidance based on what failed
        if not result.accuracy_pass:
            guidance.resource_suggestions.append(
                "Dual-mode accuracy improvement is insufficient. "
                "Focus on improving real-time model quality instead."
            )
        
        if not result.improved_pass:
            guidance.resource_suggestions.append(
                f"Only {result.improved_segments_percent:.0f}% of segments improved. "
                "The confidence threshold may need adjustment."
            )
        
        if not result.degraded_pass:
            guidance.resource_suggestions.append(
                f"{result.degraded_segments_percent:.0f}% of segments degraded. "
                "Dual-mode may be causing quality regression."
            )
        
        # Configuration for single-mode
        guidance.config_adjustments = {
            'mode': 'single',
            'enhancement_enabled': False,
            'model': 'base',  # Use base model for real-time
            'workers': 0,  # No enhancement workers
        }
        
        guidance.next_steps = [
            "Disable dual-mode enhancement in configuration",
            "Monitor single-mode accuracy and performance",
            "Consider upgrading real-time model size if accuracy is insufficient",
            "Review audio quality and noise levels",
            "Re-evaluate dual-mode after real-time improvements",
        ]
        
        return guidance
    
    def get_optimized_dual_guidance(self, result: ValidationResult) -> FallbackGuidance:
        """
        Get guidance for optimized dual-mode with performance constraints.
        
        Args:
            result: ValidationResult with performance issues
            
        Returns:
            FallbackGuidance: Guidance for optimized dual-mode
        """
        guidance = FallbackGuidance()
        guidance.fallback_type = "optimized_dual"
        guidance.fallback_reason = (
            "Dual-mode provides accuracy benefits but has performance or resource constraints."
        )
        guidance.primary_recommendation = (
            "Optimize dual-mode configuration to balance accuracy benefits with resource usage. "
            "Use dual-mode selectively for important recordings."
        )
        
        # Resource optimization suggestions
        if not result.cpu_pass:
            guidance.resource_suggestions.extend([
                "Reduce enhancement workers from current level",
                "Use base model instead of medium for enhancement",
                "Process enhancement only during idle periods",
                "Implement batch processing instead of streaming enhancement",
                "Add CPU throttling for enhancement tasks",
            ])
        
        if not result.ram_pass:
            guidance.resource_suggestions.extend([
                "Limit concurrent enhancement tasks",
                "Use memory-mapped model loading",
                "Clear model cache between sessions",
                "Reduce max queue size to limit memory footprint",
                "Consider streaming model loading",
            ])
        
        if not result.performance_pass:
            guidance.resource_suggestions.extend([
                "Optimize audio chunk size (try 2-4 second chunks)",
                "Use GPU acceleration if available",
                "Implement lazy model loading",
                "Reduce enhancement model complexity",
                "Consider async enhancement pipeline",
            ])
        
        # Optimized configuration
        guidance.config_adjustments = {
            'mode': 'dual_optimized',
            'enhancement_enabled': True,
            'confidence_threshold': 0.65,  # More selective
            'num_workers': max(1, 2),  # Reduced workers
            'enhancement_model': 'base',  # Smaller model
            'max_queue_size': 30,  # Smaller queue
            'dynamic_scaling': True,  # Enable scaling
            'cpu_threshold': 0.7,  # More conservative
            'ram_threshold': 0.75,
        }
        
        # Adjust based on specific issues
        if result.cpu_usage_percent > 70:
            guidance.config_adjustments['num_workers'] = 1
        
        if result.enhancement_completion_time > 30:
            guidance.config_adjustments['confidence_threshold'] = 0.6  # More aggressive
        
        guidance.next_steps = [
            "Apply optimized configuration",
            "Monitor resource usage closely",
            "Set up alerts for CPU/RAM thresholds",
            "Test with representative recordings",
            "Implement graceful degradation handlers",
            "Re-run validation after optimization",
        ]
        
        return guidance
    
    def get_conditional_dual_guidance(self, result: ValidationResult) -> FallbackGuidance:
        """
        Get guidance for conditional dual-mode use.
        
        Args:
            result: ValidationResult with partial issues
            
        Returns:
            FallbackGuidance: Guidance for conditional dual-mode
        """
        guidance = FallbackGuidance()
        guidance.fallback_type = "conditional_dual"
        guidance.fallback_reason = (
            "Dual-mode shows promise but has some limitations. "
            "Use with monitoring and potential constraints."
        )
        guidance.primary_recommendation = (
            "Enable dual-mode with monitoring and user controls. "
            "Allow users to disable enhancement for low-resource situations."
        )
        
        # Monitoring suggestions
        guidance.resource_suggestions.extend([
            "Set up real-time CPU/RAM monitoring",
            "Add UI indicator for enhancement status",
            "Implement user toggle for dual-mode",
            "Monitor enhancement queue depth",
            "Track enhancement success rate",
        ])
        
        # Conditional configuration
        guidance.config_adjustments = {
            'mode': 'dual_conditional',
            'enhancement_enabled': True,
            'confidence_threshold': 0.7,
            'num_workers': 2,
            'enhancement_model': 'base',
            'max_queue_size': 50,
            'user_control_enabled': True,  # Allow user toggle
            'auto_disable_threshold': {
                'cpu_percent': 85,
                'ram_percent': 90,
                'queue_depth': 80,
            },
        }
        
        guidance.next_steps = [
            "Implement monitoring dashboard",
            "Add user controls for enhancement",
            "Set up auto-disable triggers",
            "Create fallback to single-mode on resource pressure",
            "Log enhancement metrics for analysis",
            "Periodically re-validate dual-mode performance",
        ]
        
        return guidance
    
    def get_resource_optimization_suggestions(
        self,
        cpu_percent: float,
        ram_gb: float,
        completion_time: float
    ) -> List[str]:
        """
        Get specific resource optimization suggestions.
        
        Args:
            cpu_percent: Current CPU usage percentage
            ram_gb: Current RAM usage in GB
            completion_time: Current completion time in seconds
            
        Returns:
            List[str]: List of optimization suggestions
        """
        suggestions = []
        
        # CPU optimization
        if cpu_percent > 90:
            suggestions.extend([
                "🔴 CRITICAL: CPU usage is critically high",
                "Immediately reduce worker count to 1",
                "Consider disabling enhancement entirely",
                "Check for other CPU-intensive processes",
            ])
        elif cpu_percent > 80:
            suggestions.extend([
                "🟠 WARNING: CPU usage is high",
                "Reduce worker count by 50%",
                "Use smaller enhancement model",
                "Implement CPU throttling",
            ])
        elif cpu_percent > 70:
            suggestions.extend([
                "🟡 CPU usage is elevated",
                "Monitor for spikes",
                "Consider dynamic worker scaling",
            ])
        
        # RAM optimization
        if ram_gb > 6:
            suggestions.extend([
                "🔴 CRITICAL: RAM usage is critically high",
                "Clear model cache immediately",
                "Reduce queue size",
                "Consider streaming model loading",
            ])
        elif ram_gb > 4:
            suggestions.extend([
                "🟠 WARNING: RAM usage is high",
                "Limit concurrent tasks",
                "Use memory-efficient model variants",
                "Clear caches between sessions",
            ])
        elif ram_gb > 3:
            suggestions.extend([
                "🟡 RAM usage is elevated",
                "Monitor for memory leaks",
                "Consider model unloading when idle",
            ])
        
        # Completion time optimization
        if completion_time > 60:
            suggestions.extend([
                "🔴 CRITICAL: Enhancement taking too long",
                "Significantly reduce enhancement scope",
                "Consider batch processing",
                "Review model size and complexity",
            ])
        elif completion_time > 30:
            suggestions.extend([
                "🟠 WARNING: Enhancement slower than target",
                "Reduce enhancement scope",
                "Optimize chunk sizes",
                "Consider GPU acceleration",
            ])
        elif completion_time < 10:
            suggestions.extend([
                "⚠️ Enhancement completed very quickly",
                "Verify all segments are being processed",
                "Check if enhancement is actually occurring",
            ])
        
        return suggestions
    
    def get_next_steps_recommendations(
        self,
        result: ValidationResult,
        guidance: FallbackGuidance
    ) -> List[str]:
        """
        Get prioritized next steps based on validation result.
        
        Args:
            result: ValidationResult to analyze
            guidance: Associated FallbackGuidance
            
        Returns:
            List[str]: Prioritized list of next steps
        """
        next_steps = []
        
        # Critical issues first
        if result.decision == "no_go":
            next_steps.extend([
                "1. ⛔ CRITICAL: Do not deploy dual-mode in production",
                "2. 🔍 Review failure reasons and understand root causes",
                "3. 📊 Analyze segment-level data to identify patterns",
                "4. ⚙️ Adjust configuration based on findings",
                "5. 🔄 Re-run validation after changes",
            ])
            
            if not result.accuracy_pass:
                next_steps.append("6. 📉 Investigate why accuracy improvement is insufficient")
            
            if not result.improved_pass:
                next_steps.append("6. 🎯 Review confidence threshold settings")
        
        elif result.decision == "conditional_go":
            next_steps.extend([
                "1. ✅ Dual-mode is approved with conditions",
                "2. 📊 Implement monitoring and alerting",
                "3. 🎛️ Add user controls for enhancement",
                "4. 🔄 Set up automatic fallback mechanisms",
                "5. 📈 Track metrics over time",
                "6. 🔄 Re-validate periodically (weekly recommended)",
            ])
        
        else:  # go
            next_steps.extend([
                "1. ✅ Dual-mode is approved for production",
                "2. 📊 Set up baseline monitoring",
                "3. 📈 Track accuracy metrics over time",
                "4. 🔄 Schedule periodic validation reviews",
                "5. 📝 Document configuration and decisions",
            ])
        
        # Add specific recommendations based on warnings
        if result.warnings:
            next_steps.append("")
            next_steps.append("⚠️ Address Warnings:")
            for warning in result.warnings[:3]:  # Top 3 warnings
                next_steps.append(f"   - {warning}")
        
        # Add suggestions based on improvement potential
        interpretation = self.interpret_validation_result(result)
        if interpretation['improvement_potential']:
            next_steps.append("")
            next_steps.append("💡 Improvement Opportunities:")
            for opportunity in interpretation['improvement_potential'][:3]:
                next_steps.append(f"   - {opportunity}")
        
        return next_steps
    
    def generate_fallback_report(
        self,
        result: ValidationResult
    ) -> str:
        """
        Generate a comprehensive fallback guidance report.
        
        Args:
            result: ValidationResult to generate report for
            
        Returns:
            str: Detailed fallback report in markdown
        """
        guidance = self.get_fallback_guidance(result)
        
        lines = [
            "# Fallback Guidance Report",
            "",
            f"**Validation Decision:** {result.decision.upper()}",
            f"**Fallback Type:** {guidance.fallback_type}",
            "",
            "---",
            "",
            "## Summary",
            "",
            f"**Reason:** {guidance.fallback_reason}",
            "",
            f"**Primary Recommendation:** {guidance.primary_recommendation}",
            "",
        ]
        
        # Resource suggestions
        if guidance.resource_suggestions:
            lines.extend([
                "## Resource Optimization Suggestions",
                "",
            ])
            for suggestion in guidance.resource_suggestions:
                lines.append(f"- {suggestion}")
            lines.append("")
        
        # Configuration adjustments
        if guidance.config_adjustments:
            lines.extend([
                "## Recommended Configuration",
                "",
                "```python",
            ])
            for key, value in guidance.config_adjustments.items():
                if isinstance(value, str):
                    lines.append(f'{key} = "{value}"')
                elif isinstance(value, dict):
                    lines.append(f'{key} = {json.dumps(value)}')
                else:
                    lines.append(f'{key} = {value}')
            lines.extend([
                "```",
                "",
            ])
        
        # Next steps
        if guidance.next_steps:
            next_steps = self.get_next_steps_recommendations(result, guidance)
            lines.extend([
                "## Next Steps",
                "",
            ])
            for step in next_steps:
                lines.append(step)
            lines.append("")
        
        # Specific suggestions
        suggestions = self.get_resource_optimization_suggestions(
            result.cpu_usage_percent,
            result.ram_usage_gb,
            result.enhancement_completion_time
        )
        if suggestions:
            lines.extend([
                "## Resource-Specific Suggestions",
                "",
            ])
            for suggestion in suggestions:
                lines.append(f"- {suggestion}")
            lines.append("")
        
        # Alternative approaches
        if result.decision == "no_go":
            lines.extend([
                "## Alternative Approaches",
                "",
                "### Option 1: Single-Mode Only",
                "",
                "- Use real-time transcription without enhancement",
                "- Simpler deployment and maintenance",
                "- Lower resource requirements",
                "",
                "### Option 2: Post-Processing Enhancement",
                "",
                "- Enhance recordings after capture is complete",
                "- No impact on real-time performance",
                "- May miss real-time accuracy benefits",
                "",
                "### Option 3: Selective Enhancement",
                "",
                "- Enhance only user-selected segments",
                "- Manual control over resource usage",
                "- Best of both worlds for important content",
                "",
            ])
        
        return "\n".join(lines)
    
    def get_validation_history(self) -> List[ValidationResult]:
        """
        Get all validation results.
        
        Returns:
            List[ValidationResult]: All validation results
        """
        return self._validation_history.copy()
    
    def get_latest_result(self) -> Optional[ValidationResult]:
        """
        Get the most recent validation result.
        
        Returns:
            Optional[ValidationResult]: Latest result or None
        """
        return self._validation_history[-1] if self._validation_history else None
    
    # =========================================================================
    # Threshold Validation and Edge Case Handling
    # =========================================================================
    
    def validate_thresholds(self) -> Dict[str, Any]:
        """
        Validate that configured thresholds are sensible and consistent.
        
        Returns:
            Dict[str, Any]: Threshold validation result with any issues found
        """
        issues = []
        warnings = []
        
        # Check accuracy thresholds
        if self.criteria.min_accuracy_improvement > self.criteria.target_accuracy_improvement:
            issues.append("min_accuracy_improvement > target_accuracy_improvement")
        
        if self.criteria.min_accuracy_improvement < 0 or self.criteria.min_accuracy_improvement > 1:
            issues.append("min_accuracy_improvement must be between 0 and 1")
        
        if self.criteria.min_wer_reduction < 0 or self.criteria.min_wer_reduction > 1:
            issues.append("min_wer_reduction must be between 0 and 1")
        
        # Check performance thresholds
        if self.criteria.min_enhancement_completion_time > self.criteria.max_enhancement_completion_time:
            issues.append("min_enhancement_completion_time > max_enhancement_completion_time")
        
        if self.criteria.target_enhancement_completion_time < self.criteria.min_enhancement_completion_time or \
           self.criteria.target_enhancement_completion_time > self.criteria.max_enhancement_completion_time:
            warnings.append("target_enhancement_completion_time not between min and max")
        
        if self.criteria.max_enhancement_completion_time < 1:
            warnings.append("max_enhancement_completion_time < 1s may be too aggressive")
        
        if self.criteria.max_enhancement_completion_time > 300:
            warnings.append("max_enhancement_completion_time > 300s may be too lenient")
        
        # Check resource thresholds
        if self.criteria.max_cpu_percent < 10 or self.criteria.max_cpu_percent > 100:
            issues.append("max_cpu_percent must be between 10 and 100")
        
        if self.criteria.max_ram_gb < 0.5:
            warnings.append("max_ram_gb < 0.5GB may be too restrictive")
        
        if self.criteria.max_ram_gb > 64:
            warnings.append("max_ram_gb > 64GB may be too lenient")
        
        # Check segment thresholds
        if self.criteria.min_improved_segments_percent + self.criteria.max_degraded_segments_percent > 100:
            warnings.append("min_improved + max_degraded > 100% may be inconsistent")
        
        if self.criteria.min_improved_segments_percent < 0 or self.criteria.min_improved_segments_percent > 100:
            issues.append("min_improved_segments_percent must be between 0 and 100")
        
        if self.criteria.max_degraded_segments_percent < 0 or self.criteria.max_degraded_segments_percent > 100:
            issues.append("max_degraded_segments_percent must be between 0 and 100")
        
        # Check latency threshold
        if self.criteria.max_latency_overhead_percent < 0:
            issues.append("max_latency_overhead_percent cannot be negative")
        
        if self.criteria.max_latency_overhead_percent > 200:
            warnings.append("max_latency_overhead_percent > 200% may be too lenient")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'criteria_summary': self.criteria.to_dict(),
        }
    
    def check_accuracy_threshold(
        self,
        accuracy_improvement: float,
        wer_reduction: float
    ) -> Dict[str, Any]:
        """
        Check accuracy improvement against thresholds.
        
        Args:
            accuracy_improvement: Measured accuracy improvement (0.0-1.0)
            wer_reduction: Measured WER reduction (0.0-1.0)
            
        Returns:
            Dict[str, Any]: Accuracy threshold check result
        """
        accuracy_pass = accuracy_improvement >= self.criteria.min_accuracy_improvement
        wer_pass = wer_reduction >= self.criteria.min_wer_reduction
        meets_target = accuracy_improvement >= self.criteria.target_accuracy_improvement
        
        # Determine status
        if meets_target:
            status = "excellent"
        elif accuracy_pass and wer_pass:
            status = "pass"
        elif accuracy_pass:
            status = "partial_pass"
        else:
            status = "fail"
        
        return {
            'status': status,
            'accuracy_improvement': accuracy_improvement,
            'wer_reduction': wer_reduction,
            'accuracy_pass': accuracy_pass,
            'wer_pass': wer_pass,
            'meets_target': meets_target,
            'thresholds': {
                'min_accuracy': self.criteria.min_accuracy_improvement,
                'target_accuracy': self.criteria.target_accuracy_improvement,
                'min_wer_reduction': self.criteria.min_wer_reduction,
            },
            'margin': {
                'accuracy_margin': accuracy_improvement - self.criteria.min_accuracy_improvement,
                'wer_margin': wer_reduction - self.criteria.min_wer_reduction,
            },
        }
    
    def check_performance_threshold(
        self,
        completion_time: float,
        latency_overhead_percent: float
    ) -> Dict[str, Any]:
        """
        Check performance targets against thresholds.
        
        Args:
            completion_time: Enhancement completion time in seconds
            latency_overhead_percent: Latency overhead percentage
            
        Returns:
            Dict[str, Any]: Performance threshold check result
        """
        # Handle edge cases
        if completion_time <= 0:
            return {
                'status': 'invalid',
                'completion_time': completion_time,
                'error': 'Completion time must be positive',
                'performance_pass': False,
                'latency_pass': latency_overhead_percent <= self.criteria.max_latency_overhead_percent,
            }
        
        time_pass = (
            self.criteria.min_enhancement_completion_time <= 
            completion_time <= 
            self.criteria.max_enhancement_completion_time
        )
        latency_pass = latency_overhead_percent <= self.criteria.max_latency_overhead_percent
        
        # Determine status
        if time_pass and latency_pass:
            if completion_time <= self.criteria.target_enhancement_completion_time:
                status = "excellent"
            else:
                status = "pass"
        elif time_pass:
            status = "partial_pass"
        else:
            status = "fail"
        
        # Calculate margin
        if completion_time < self.criteria.min_enhancement_completion_time:
            time_margin = completion_time - self.criteria.min_enhancement_completion_time
        elif completion_time > self.criteria.max_enhancement_completion_time:
            time_margin = self.criteria.max_enhancement_completion_time - completion_time
        else:
            time_margin = min(
                completion_time - self.criteria.min_enhancement_completion_time,
                self.criteria.max_enhancement_completion_time - completion_time
            )
        
        return {
            'status': status,
            'completion_time': completion_time,
            'latency_overhead_percent': latency_overhead_percent,
            'performance_pass': time_pass,
            'latency_pass': latency_pass,
            'thresholds': {
                'min_time': self.criteria.min_enhancement_completion_time,
                'max_time': self.criteria.max_enhancement_completion_time,
                'target_time': self.criteria.target_enhancement_completion_time,
                'max_latency_overhead': self.criteria.max_latency_overhead_percent,
            },
            'margin': {
                'time_margin': time_margin,
                'latency_margin': self.criteria.max_latency_overhead_percent - latency_overhead_percent,
            },
        }
    
    def check_resource_threshold(
        self,
        cpu_percent: float,
        ram_gb: float
    ) -> Dict[str, Any]:
        """
        Check resource usage against thresholds.
        
        Args:
            cpu_percent: CPU usage percentage
            ram_gb: RAM usage in GB
            
        Returns:
            Dict[str, Any]: Resource threshold check result
        """
        # Handle edge cases
        if cpu_percent < 0:
            cpu_percent = 0
        if cpu_percent > 100:
            cpu_percent = 100
        if ram_gb < 0:
            ram_gb = 0
        
        cpu_pass = cpu_percent <= self.criteria.max_cpu_percent
        ram_pass = ram_gb <= self.criteria.max_ram_gb
        
        # Determine status
        if cpu_pass and ram_pass:
            # Check if we're close to limits (within 10%)
            cpu_close = cpu_percent > self.criteria.max_cpu_percent * 0.9
            ram_close = ram_gb > self.criteria.max_ram_gb * 0.9
            
            if cpu_close or ram_close:
                status = "pass_warning"
            else:
                status = "pass"
        elif cpu_pass or ram_pass:
            status = "partial_fail"
        else:
            status = "fail"
        
        return {
            'status': status,
            'cpu_percent': cpu_percent,
            'ram_gb': ram_gb,
            'cpu_pass': cpu_pass,
            'ram_pass': ram_pass,
            'thresholds': {
                'max_cpu': self.criteria.max_cpu_percent,
                'max_ram': self.criteria.max_ram_gb,
            },
            'margin': {
                'cpu_margin': self.criteria.max_cpu_percent - cpu_percent,
                'ram_margin': self.criteria.max_ram_gb - ram_gb,
            },
            'utilization': {
                'cpu_utilization': cpu_percent / self.criteria.max_cpu_percent * 100,
                'ram_utilization': ram_gb / self.criteria.max_ram_gb * 100,
            },
        }
    
    def check_segment_threshold(
        self,
        improved_percent: float,
        degraded_percent: float,
        total_segments: int
    ) -> Dict[str, Any]:
        """
        Check segment improvement against thresholds.
        
        Args:
            improved_percent: Percentage of segments improved
            degraded_percent: Percentage of segments degraded
            total_segments: Total number of segments evaluated
            
        Returns:
            Dict[str, Any]: Segment threshold check result
        """
        # Handle edge case of no segments
        if total_segments == 0:
            return {
                'status': 'no_data',
                'improved_percent': 0,
                'degraded_percent': 0,
                'total_segments': 0,
                'improved_pass': False,
                'degraded_pass': False,
                'error': 'No segments to evaluate',
            }
        
        # Handle edge cases for percentages
        if improved_percent < 0:
            improved_percent = 0
        if improved_percent > 100:
            improved_percent = 100
        if degraded_percent < 0:
            degraded_percent = 0
        if degraded_percent > 100:
            degraded_percent = 100
        
        improved_pass = improved_percent >= self.criteria.min_improved_segments_percent
        degraded_pass = degraded_percent <= self.criteria.max_degraded_segments_percent
        
        # Calculate net improvement
        net_improvement = improved_percent - degraded_percent
        
        # Determine status
        if improved_pass and degraded_pass:
            if net_improvement > 50:
                status = "excellent"
            elif net_improvement > 20:
                status = "good"
            else:
                status = "pass"
        elif improved_pass:
            status = "partial_pass"
        else:
            status = "fail"
        
        return {
            'status': status,
            'improved_percent': improved_percent,
            'degraded_percent': degraded_percent,
            'total_segments': total_segments,
            'improved_pass': improved_pass,
            'degraded_pass': degraded_pass,
            'net_improvement': net_improvement,
            'thresholds': {
                'min_improved': self.criteria.min_improved_segments_percent,
                'max_degraded': self.criteria.max_degraded_segments_percent,
            },
            'margin': {
                'improved_margin': improved_percent - self.criteria.min_improved_segments_percent,
                'degraded_margin': self.criteria.max_degraded_segments_percent - degraded_percent,
            },
        }
    
    def interpret_validation_result(self, result: ValidationResult) -> Dict[str, Any]:
        """
        Interpret validation result and provide detailed analysis.
        
        Args:
            result: ValidationResult to interpret
            
        Returns:
            Dict[str, Any]: Detailed interpretation of the validation result
        """
        interpretation = {
            'decision': result.decision,
            'overall_assessment': '',
            'strengths': [],
            'weaknesses': [],
            'critical_issues': [],
            'improvement_potential': [],
        }
        
        # Overall assessment
        if result.decision == "go":
            interpretation['overall_assessment'] = (
                "Dual-mode enhancement validation passed all critical criteria. "
                "The system is ready for production deployment."
            )
        elif result.decision == "conditional_go":
            interpretation['overall_assessment'] = (
                "Dual-mode enhancement shows promise but has some limitations. "
                "Consider using with monitoring and potential constraints."
            )
        else:
            interpretation['overall_assessment'] = (
                "Dual-mode enhancement does not meet validation criteria. "
                "Review issues and consider fallback options."
            )
        
        # Identify strengths
        if result.accuracy_improvement >= self.criteria.target_accuracy_improvement:
            interpretation['strengths'].append(
                f"Excellent accuracy improvement ({result.accuracy_improvement*100:.1f}%)"
            )
        elif result.accuracy_pass:
            interpretation['strengths'].append(
                f"Acceptable accuracy improvement ({result.accuracy_improvement*100:.1f}%)"
            )
        
        if result.improved_segments_percent >= 70:
            interpretation['strengths'].append(
                f"Strong segment improvement rate ({result.improved_segments_percent:.0f}%)"
            )
        
        if result.cpu_usage_percent < self.criteria.max_cpu_percent * 0.7:
            interpretation['strengths'].append(
                f"Efficient CPU usage ({result.cpu_usage_percent:.0f}%)"
            )
        
        if result.ram_usage_gb < self.criteria.max_ram_gb * 0.7:
            interpretation['strengths'].append(
                f"Efficient RAM usage ({result.ram_usage_gb:.1f}GB)"
            )
        
        # Identify weaknesses
        if not result.accuracy_pass:
            interpretation['weaknesses'].append(
                f"Insufficient accuracy improvement ({result.accuracy_improvement*100:.1f}%)"
            )
        
        if not result.improved_pass:
            interpretation['weaknesses'].append(
                f"Low segment improvement rate ({result.improved_segments_percent:.0f}%)"
            )
        
        if not result.degraded_pass:
            interpretation['weaknesses'].append(
                f"High segment degradation rate ({result.degraded_segments_percent:.0f}%)"
            )
        
        if not result.performance_pass:
            interpretation['weaknesses'].append(
                f"Performance outside target range ({result.enhancement_completion_time:.1f}s)"
            )
        
        if not result.cpu_pass:
            interpretation['weaknesses'].append(
                f"High CPU usage ({result.cpu_usage_percent:.0f}%)"
            )
        
        if not result.ram_pass:
            interpretation['weaknesses'].append(
                f"High RAM usage ({result.ram_usage_gb:.1f}GB)"
            )
        
        # Critical issues
        interpretation['critical_issues'] = result.failure_reasons.copy()
        
        # Improvement potential
        if result.accuracy_improvement > 0 and result.accuracy_improvement < self.criteria.target_accuracy_improvement:
            gap = self.criteria.target_accuracy_improvement - result.accuracy_improvement
            interpretation['improvement_potential'].append(
                f"Could improve accuracy by {gap*100:.1f}% to reach target"
            )
        
        if result.improved_segments_percent < 70:
            interpretation['improvement_potential'].append(
                "Consider adjusting confidence threshold to capture more segments"
            )
        
        if result.degraded_segments_percent > 10:
            interpretation['improvement_potential'].append(
                "Review enhancement model selection to reduce degradation"
            )
        
        return interpretation
    
    def run_automated_validation(
        self,
        test_segments: List[Dict[str, Any]],
        ground_truths: List[str],
        single_mode_processor: Callable,
        dual_mode_processor: Callable,
        confidence_threshold: float = 0.7
    ) -> ValidationResult:
        """
        Run fully automated validation with provided processors.
        
        This is a convenience method that runs the full validation pipeline:
        1. Process segments with single-mode
        2. Process segments with dual-mode
        3. Compare results
        4. Validate against criteria
        5. Return Go/No-Go decision
        
        Args:
            test_segments: Segments to process
            ground_truths: Ground truth strings
            single_mode_processor: Function that processes segments in single-mode
            dual_mode_processor: Function that processes segments in dual-mode
            confidence_threshold: Confidence threshold for enhancement
            
        Returns:
            ValidationResult: Complete validation result
        """
        start_time = time.time()
        
        if self.verbose:
            logger.info(f"Starting automated validation with {len(test_segments)} segments")
        
        # Process with single-mode
        single_mode_segments = []
        for segment in test_segments:
            processed = single_mode_processor(segment)
            single_mode_segments.append(processed)
        
        # Process with dual-mode
        dual_mode_segments = []
        enhancement_start = time.time()
        for segment in test_segments:
            processed = dual_mode_processor(segment)
            dual_mode_segments.append(processed)
        enhancement_time = time.time() - enhancement_start
        
        # Compare results
        comparator = DualModeComparator(confidence_threshold=confidence_threshold)
        comparison_result = comparator.compare(
            single_mode_segments,
            dual_mode_segments,
            ground_truths
        )
        
        # Get system metrics
        system_metrics = {
            'cpu': {
                'avg_usage': psutil.cpu_percent(interval=0.1),
            },
            'ram': {
                'used_gb': psutil.virtual_memory().used / (1024**3),
            },
        }
        
        # Validate
        result = self.validate(
            comparison_result=comparison_result,
            system_metrics=system_metrics,
            enhancement_completion_time=enhancement_time
        )
        
        if self.verbose:
            logger.info(
                f"Automated validation complete: {result.decision.upper()} "
                f"(total time: {time.time() - start_time:.2f}s)"
            )
        
        return result