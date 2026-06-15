"""Recording controller - UI-friendly wrapper around AudioSession.

Provides non-blocking recording control with proper error handling
and state management for UI integration. Includes hybrid transcription:
- Real-time: tiny model for immediate display using accumulating processor
- Post-process: stronger model after recording stops
"""

import logging
import threading
import time as _time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional, Set, Callable, List, Dict, Any

import numpy as np  # noqa: E402
from meetandread.speaker.models import DiarizationResult  # noqa: E402


from meetandread.audio import (  # noqa: E402
    AudioSession,
    SessionConfig,
    SourceConfig,
    SessionState,
    SessionError,
    NoSourcesError,
)
from meetandread.audio.capture import AudioSourceError  # noqa: E402
from meetandread.audio.hotplug import DeviceEvent, DeviceEventType, WindowsDeviceMonitor  # noqa: E402
from meetandread.transcription.accumulating_processor import AccumulatingTranscriptionProcessor, SegmentResult  # noqa: E402
from meetandread.transcription.transcript_store import TranscriptStore, Word  # noqa: E402
from meetandread.transcription.post_processor import PostProcessingQueue  # noqa: E402
from meetandread.config.manager import ConfigManager  # noqa: E402

logger = logging.getLogger(__name__)

# Named constants for runtime thresholds and limits
_LIVE_SPEAKER_MATCH_THRESHOLD = 0.75  # Conservative: no match below this
_SANITIZED_ERROR_MAX_LENGTH = 200  # Truncation for sanitized error messages
_SANITIZED_STATUS_MAX_LENGTH = 120  # Truncation for status/diagnostic messages
_HOTPLUG_RECOVERY_WINDOW_SECONDS = 5.0


class RecoveryOutcome(Enum):
    """Sanitized controller outcomes for device hot-plug recovery."""
    IGNORED = "ignored"
    LOST = "lost"
    DEGRADED = "degraded"
    TOTAL_LOSS = "total_loss"
    AUTO_RECOVERED = "auto_recovered"
    MANUAL_RETRY_REQUIRED = "manual_retry_required"
    MANUAL_RECOVERED = "manual_recovered"


@dataclass(frozen=True)
class ActiveSourceIdentity:
    """Sanitized identity for a capture source tracked during recording."""
    type: str
    device_id: Optional[str]
    friendly_name: Optional[str]
    flow: Optional[str]
    is_active: bool = True
    lost_at: Optional[float] = None

    def as_diagnostics(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "device_id": self.device_id,
            "friendly_name": self.friendly_name,
            "flow": self.flow,
            "is_active": self.is_active,
            "lost_at": self.lost_at,
        }


@dataclass(frozen=True)
class RecoveryResult:
    """Sanitized hot-plug recovery decision emitted by RecordingController."""
    outcome: RecoveryOutcome
    source_type: Optional[str] = None
    device_id: Optional[str] = None
    message: str = ""
    recoverable: bool = True

    def as_diagnostics(self) -> Dict[str, Any]:
        return {
            "outcome": self.outcome.value,
            "source_type": self.source_type,
            "device_id": self.device_id,
            "message": self.message[:_SANITIZED_STATUS_MAX_LENGTH],
            "recoverable": self.recoverable,
        }


class ControllerState(Enum):
    """Controller states for UI state management."""
    IDLE = auto()
    STARTING = auto()
    RECORDING = auto()
    STOPPING = auto()
    ERROR = auto()


@dataclass
class ControllerError:
    """Error information for UI display."""
    message: str
    is_recoverable: bool = True


class RecordingController:
    """UI-friendly controller for recording operations.

    Wraps AudioSession with:
    - Non-blocking stop/finalize (runs on worker thread)
    - Clear error state for UI display
    - Simple source selection API
    - Callback support for state changes
    - Real-time transcription using accumulating processor

    HYBRID TRANSCRIPTION ARCHITECTURE:
    - Real-time: AccumulatingTranscriptionProcessor with tiny model
      * 60s window for context
      * Updates every 2 seconds
      * 3s silence detection for phrase breaks
    - Post-process: Stronger model (base/small) after recording stops

    Example:
        controller = RecordingController()
        controller.on_state_change = lambda state: print(f"State: {state}")
        controller.on_error = lambda err: print(f"Error: {err.message}")
        controller.on_phrase_result = lambda result: print(f"Phrase: {result.text}")

        # Start recording
        error = controller.start({'mic', 'system'})
        if error:
            logger.error("Failed to start: %s", error.message)

        # Stop recording (non-blocking)
        controller.stop()
    """

    def __init__(self, enable_transcription: bool = True):
        """Initialize the recording controller.

        Args:
            enable_transcription: Whether to enable real-time transcription
        """
        self._session = AudioSession()
        self._state = ControllerState.IDLE
        self._error: Optional[ControllerError] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._last_wav_path: Optional[Path] = None
        self._last_transcript_path: Optional[Path] = None

        # HYBRID TRANSCRIPTION
        self.enable_transcription = enable_transcription
        self._transcription_processor: Optional[AccumulatingTranscriptionProcessor] = None
        self._transcript_store: Optional[TranscriptStore] = None
        self._post_processor: Optional[PostProcessingQueue] = None
        self._post_process_job_id: Optional[str] = None
        self._finalizer_thread: Optional[threading.Thread] = None
        self._config_manager = ConfigManager()

        # Callbacks
        self.on_state_change: Optional[Callable[[ControllerState], None]] = None
        self.on_error: Optional[Callable[[ControllerError], None]] = None
        self.on_recording_complete: Optional[Callable[[Path, Optional[Path]], None]] = None
        self.on_phrase_result: Optional[Callable[[SegmentResult], None]] = None  # For accumulating processor results
        self.on_post_process_complete: Optional[Callable[[str, Path], None]] = None  # job_id, transcript_path
        self.on_frames_dropped: Optional[Callable[[int], None]] = None  # Aggregate drop count (UI thread via bridge)
        self.on_device_change: Optional[Callable[[DeviceEvent], None]] = None
        self.on_recovery_attempt: Optional[Callable[[RecoveryResult], None]] = None

        # Hot-plug recovery tracking (sanitized endpoint identities only)
        self._hotplug_monitor: Optional[WindowsDeviceMonitor] = None
        self._hotplug_monitor_active = False
        self._hotplug_lock = threading.Lock()
        self._active_source_identities: Dict[str, ActiveSourceIdentity] = {}
        self._lost_source_identities: Dict[str, ActiveSourceIdentity] = {}
        self._last_recovery_result: Optional[RecoveryResult] = None

        # Audio feed tracking
        self._audio_chunks_fed = 0

        # Speaker diarization result (kept for pin-to-name UX)
        self._last_diarization_result: Optional[object] = None  # DiarizationResult

        # Auto-WER from last post-processing (None until computed)
        self._last_wer: Optional[float] = None

        # --- Live speaker matching state ---
        self._live_audio_buffer = bytearray()  # raw int16 PCM bytes
        self._live_max_buffer_bytes = 12 * 16000 * 2  # 12s at 16kHz int16 = 384000
        self._live_extractor = None  # lazily-created SpeakerEmbeddingExtractor
        self._live_extractor_available: Optional[bool] = None  # None=unchecked, True/False
        self._live_store_available: Optional[bool] = None
        self._live_match_attempts = 0
        self._live_match_hits = 0
        self._live_match_fallbacks = 0
        self._live_last_status: str = "disabled"  # sanitized status string
        self._live_last_error_class: Optional[str] = None
        self._live_last_error_message: Optional[str] = None
        self._live_last_attempt_ts: Optional[float] = None
        self._live_min_audio_bytes = 8 * 16000 * 2  # 8s at 16kHz int16 = 256000
        self._live_attempt_interval = 2.0  # seconds between match attempts
        self._live_extractor_lock = threading.Lock()  # serialize extractor use

        # Thread-safety locks (S02/T01)
        self._state_lock = threading.Lock()   # protects _state, _error, _last_*_path, _last_diarization_result, _last_wer, _post_process_job_id
        self._buffer_lock = threading.Lock()  # protects _live_audio_buffer and live matching counters

    def _on_session_frames_dropped(self, count: int) -> None:
        """Internal handler for session-level frame-drop events.

        Called from the audio callback thread via SessionConfig.on_frames_dropped.
        Sanitizes the count and forwards to the user-provided
        ``on_frames_dropped`` callback. Exceptions in the user callback are
        logged and swallowed so recording continues uninterrupted.
        """
        try:
            # Defensive: coerce non-int / negative to safe int
            safe_count = int(max(0, count))
        except (TypeError, ValueError):
            logger.warning(
                "Non-numeric frame-drop count received: %r - treating as 0",
                count,
            )
            safe_count = 0

        if safe_count == 0:
            return  # no-op for zero

        logger.info(
            "Controller frame-drop forwarded: aggregate=%d", safe_count
        )
        if self.on_frames_dropped:
            try:
                self.on_frames_dropped(safe_count)
            except Exception:
                logger.exception("on_frames_dropped callback error (recording continues)")

    def _on_session_error(self, exc: Exception) -> None:
        """Internal handler for session consumer thread crashes.

        Called from the audio consumer thread via SessionConfig.on_error
        when the consumer loop crashes. Logs the error and transitions
        controller state to ERROR. Never raises — exceptions are logged
        and swallowed so the consumer thread cleanup is not compromised.
        """
        try:
            error_class = type(exc).__name__
            logger.error(
                "Audio session consumer crash: error_class=%s, state=%s",
                error_class,
                self._state.name,
            )
            self._set_error(
                f"Audio consumer error: {error_class}",
                is_recoverable=True,
            )
        except Exception:
            logger.exception("Error in _on_session_error handler")

    def _set_state(self, state: ControllerState) -> None:
        """Update state and notify listeners."""
        callback = None
        with self._state_lock:
            self._state = state
            callback = self.on_state_change
        # Invoke callback outside lock to prevent deadlock
        if callback:
            try:
                callback(state)
            except Exception as e:
                logger.error("State change callback failed: %s", e)

    def _set_error(self, message: str, is_recoverable: bool = True) -> ControllerError:
        """Set error state and notify listeners."""
        error = ControllerError(message, is_recoverable)
        state_cb = error_cb = None
        with self._state_lock:
            self._error = error
            self._state = ControllerState.ERROR
            state_cb = self.on_state_change
            error_cb = self.on_error
        # Invoke callbacks outside lock to prevent deadlock
        if state_cb:
            try:
                state_cb(ControllerState.ERROR)
            except Exception as e:
                logger.error("State change callback failed: %s", e)
        if error_cb:
            try:
                error_cb(error)
            except Exception as e:
                logger.error("Error callback failed: %s", e)
        return error

    def clear_error(self) -> None:
        """Clear error state and return to idle."""
        with self._state_lock:
            self._error = None
            is_error = self._state == ControllerState.ERROR
        if is_error:
            self._set_state(ControllerState.IDLE)

    def get_state(self) -> ControllerState:
        """Get current controller state (thread-safe snapshot)."""
        with self._state_lock:
            current = self._state
        # Sync with underlying session state if needed
        session_state = self._session.get_state()
        if session_state == SessionState.RECORDING and current != ControllerState.RECORDING:
            self._set_state(ControllerState.RECORDING)
            with self._state_lock:
                return self._state
        return current

    def get_error(self) -> Optional[ControllerError]:
        """Get current error if any (thread-safe snapshot)."""
        with self._state_lock:
            return self._error

    def is_recording(self) -> bool:
        """Check if currently recording (thread-safe)."""
        with self._state_lock:
            return self._state == ControllerState.RECORDING

    def is_busy(self) -> bool:
        """Check if controller is busy (starting, stopping, etc.)."""
        with self._state_lock:
            return self._state in (ControllerState.STARTING, ControllerState.STOPPING)


    def _source_identity_from_config(self, config: SourceConfig) -> ActiveSourceIdentity:
        device_id = getattr(config, "device_id", None)
        friendly_name = getattr(config, "friendly_name", None)
        if device_id is not None:
            device_id = str(device_id).strip() or None
        if friendly_name is not None:
            friendly_name = str(friendly_name).strip() or None
        flow = "render" if config.type == "system" else "capture" if config.type == "mic" else None
        return ActiveSourceIdentity(
            type=str(config.type),
            device_id=device_id,
            friendly_name=friendly_name,
            flow=flow,
        )

    def _snapshot_active_sources(self, source_configs: List[SourceConfig]) -> None:
        identities = {
            sc.type: self._source_identity_from_config(sc)
            for sc in source_configs
            if sc.type in {"mic", "system"}
        }
        with self._hotplug_lock:
            self._active_source_identities = identities
            self._lost_source_identities = {}
            self._last_recovery_result = None
        logger.info(
            "Recording hotplug active source snapshot: %s",
            [identity.as_diagnostics() for identity in identities.values()],
        )

    def _start_hotplug_monitor(self) -> None:
        try:
            self._hotplug_monitor = WindowsDeviceMonitor()
            self._hotplug_monitor.start()
            self._hotplug_monitor_active = True
            logger.info("Recording hotplug monitor started")
        except Exception as exc:
            self._hotplug_monitor = None
            self._hotplug_monitor_active = False
            logger.warning("Recording hotplug monitor unavailable: %s", exc)

    def _stop_hotplug_monitor(self) -> None:
        monitor = self._hotplug_monitor
        self._hotplug_monitor = None
        self._hotplug_monitor_active = False
        if monitor is None:
            return
        try:
            monitor.stop()
            logger.info("Recording hotplug monitor stopped")
        except Exception as exc:
            logger.warning("Recording hotplug monitor stop failed: %s", exc)

    def drain_hotplug_events(self, max_events: int = 100) -> List[RecoveryResult]:
        """Drain queued monitor events through controller recovery decisions."""
        monitor = self._hotplug_monitor
        if monitor is None:
            return []
        try:
            events = monitor.drain_events(max_events=max_events)
        except Exception as exc:
            logger.warning("Recording hotplug event drain failed: %s", exc)
            return []
        results: List[RecoveryResult] = []
        for event in events:
            result = self.handle_device_event(event)
            if result is not None:
                results.append(result)
        return results

    def _emit_device_change(self, event: DeviceEvent) -> None:
        callback = self.on_device_change
        if callback:
            try:
                callback(event)
            except Exception as exc:
                logger.error("Device change callback failed: %s", exc)

    def _emit_recovery_result(self, result: RecoveryResult) -> None:
        with self._hotplug_lock:
            self._last_recovery_result = result
        callback = self.on_recovery_attempt
        if callback:
            try:
                callback(result)
            except Exception as exc:
                logger.error("Recovery callback failed: %s", exc)

    def _source_type_for_event(self, event: DeviceEvent) -> Optional[str]:
        event_device_id = (event.device_id or "").strip() or None
        event_friendly_name = (event.friendly_name or "").strip() or None
        with self._hotplug_lock:
            identities = list(self._active_source_identities.values()) + list(self._lost_source_identities.values())
        for identity in identities:
            if identity.device_id and event_device_id and identity.device_id == event_device_id:
                return identity.type
            if identity.friendly_name and event_friendly_name and identity.friendly_name == event_friendly_name:
                return identity.type
        flow = (event.flow or "").lower()
        if flow == "render":
            return "system"
        if flow == "capture":
            return "mic"
        return None

    def _is_loss_event(self, event: DeviceEvent) -> bool:
        state = (event.state or "").lower()
        return event.event_type is DeviceEventType.REMOVED or state in {"inactive", "disabled", "unplugged", "notpresent"}

    def _is_reconnect_event(self, event: DeviceEvent) -> bool:
        state = (event.state or "").lower()
        return event.event_type is DeviceEventType.ADDED or state == "active"

    def handle_device_event(self, event: DeviceEvent, *, now: Optional[float] = None) -> Optional[RecoveryResult]:
        """Route a sanitized hot-plug event through recording recovery decisions."""
        with self._state_lock:
            current_state = self._state
        if current_state not in (ControllerState.RECORDING, ControllerState.ERROR):
            return None
        if current_state is ControllerState.ERROR and not self._is_reconnect_event(event):
            return None

        now = _time.time() if now is None else now
        self._emit_device_change(event)
        source_type = self._source_type_for_event(event)
        if source_type is None:
            result = RecoveryResult(RecoveryOutcome.IGNORED, message="Device event did not match an active recording source")
            self._emit_recovery_result(result)
            return result

        event_device_id = (event.device_id or "").strip() or None
        if self._is_loss_event(event):
            with self._hotplug_lock:
                identity = self._active_source_identities.pop(source_type, None)
                if identity is None:
                    identity = ActiveSourceIdentity(source_type, event_device_id, event.friendly_name, event.flow)
                lost_identity = ActiveSourceIdentity(
                    type=identity.type,
                    device_id=event_device_id or identity.device_id,
                    friendly_name=event.friendly_name or identity.friendly_name,
                    flow=event.flow or identity.flow,
                    is_active=False,
                    lost_at=now,
                )
                self._lost_source_identities[source_type] = lost_identity
                remaining = len(self._active_source_identities)


            with self._hotplug_lock:
                if remaining > 0:
                    result = RecoveryResult(RecoveryOutcome.DEGRADED, source_type, event_device_id, "Recording source lost; continuing with remaining recording source")
                    self._emit_recovery_result(result)
                    logger.warning("Recording hotplug partial degradation: %s", result.as_diagnostics())
                    return result

            
            result = RecoveryResult(RecoveryOutcome.TOTAL_LOSS, source_type, event_device_id, "Capture source lost: all active capture sources lost", True)
            logger.error("Recording hotplug total device loss: %s", result.as_diagnostics())
            self._emit_recovery_result(result)
            self._set_error("Capture source lost: all active capture sources lost. Reconnect a device and retry recording recovery.", is_recoverable=True)
            return result
        return result

        if self._is_reconnect_event(event):
            with self._hotplug_lock:
                lost_identity = self._lost_source_identities.get(source_type)
            if lost_identity is None:
                result = RecoveryResult(RecoveryOutcome.IGNORED, source_type, event_device_id, "Reconnect event for source that was not lost")
                self._emit_recovery_result(result)
                return result
            elapsed = now - (lost_identity.lost_at or now)
            event_friendly_name = (event.friendly_name or "").strip() or None
            same_device = (
                (lost_identity.device_id and event_device_id and lost_identity.device_id == event_device_id)
                or (lost_identity.friendly_name and event_friendly_name and lost_identity.friendly_name == event_friendly_name)
                or (not lost_identity.device_id and not event_device_id)
            )
            if same_device and elapsed <= _HOTPLUG_RECOVERY_WINDOW_SECONDS:
                recovered = ActiveSourceIdentity(
                    type=lost_identity.type,
                    device_id=event_device_id or lost_identity.device_id,
                    friendly_name=event.friendly_name or lost_identity.friendly_name,
                    flow=event.flow or lost_identity.flow,
                    is_active=True,
                    lost_at=None,
                )
                with self._hotplug_lock:
                    self._active_source_identities[source_type] = recovered
                    self._lost_source_identities.pop(source_type, None)
                with self._state_lock:
                    self._error = None
                    self._state = ControllerState.RECORDING
                result = RecoveryResult(RecoveryOutcome.AUTO_RECOVERED, source_type, recovered.device_id, "Recording source reappeared within recovery window")
                logger.info("Recording hotplug auto-recovered: %s", result.as_diagnostics())
                self._emit_recovery_result(result)
                state_cb = self.on_state_change
                if state_cb:
                    try:
                        state_cb(ControllerState.RECORDING)
                    except Exception as exc:
                        logger.error("State change callback failed: %s", exc)
                return result

            result = RecoveryResult(RecoveryOutcome.MANUAL_RETRY_REQUIRED, source_type, event_device_id, "Recovery window expired; manual retry required")
            logger.warning("Recording hotplug recovery window expired: %s", result.as_diagnostics())
            self._emit_recovery_result(result)
            self._set_error("Recording device reconnected after the automatic recovery window. Retry recording recovery manually.", is_recoverable=True)
            return result

        result = RecoveryResult(RecoveryOutcome.IGNORED, source_type, event_device_id, "Device event type does not affect recording")
        self._emit_recovery_result(result)
        return result

    def retry_recovery(self, *, now: Optional[float] = None) -> RecoveryResult:
        """Manual recovery entry point for UI retry/fallback controls."""
        del now  # reserved for future endpoint retry/fallback timing
        with self._hotplug_lock:
            if not self._lost_source_identities:
                result = RecoveryResult(RecoveryOutcome.IGNORED, message="No lost recording sources to recover")
                self._last_recovery_result = result
                return result
            for source_type, identity in list(self._lost_source_identities.items()):
                self._active_source_identities[source_type] = ActiveSourceIdentity(
                    type=identity.type,
                    device_id=identity.device_id,
                    friendly_name=identity.friendly_name,
                    flow=identity.flow,
                    is_active=True,
                    lost_at=None,
                )
            self._lost_source_identities.clear()
        with self._state_lock:
            self._error = None
            self._state = ControllerState.RECORDING
        result = RecoveryResult(RecoveryOutcome.MANUAL_RECOVERED, message="Manual recording recovery retry accepted")
        logger.info("Recording hotplug manual recovery accepted")
        self._emit_recovery_result(result)
        state_cb = self.on_state_change
        if state_cb:
            try:
                state_cb(ControllerState.RECORDING)
            except Exception as exc:
                logger.error("State change callback failed: %s", exc)
        return result

    def _hotplug_diagnostics(self) -> Dict[str, Any]:
        with self._hotplug_lock:
            active = [identity.as_diagnostics() for identity in self._active_source_identities.values()]
            lost = [identity.as_diagnostics() for identity in self._lost_source_identities.values()]
            last_result = self._last_recovery_result.as_diagnostics() if self._last_recovery_result else None
        return {
            "monitor_active": bool(self._hotplug_monitor_active),
            "active_source_count": len(active),
            "lost_source_count": len(lost),
            "active_sources": active,
            "lost_sources": lost,
            "last_result": last_result,
        }

    def start(
        self,
        selected_sources: Set[str],
        *,
        fake_path: Optional[str] = None,
        fake_denoise: bool = False,
        fake_loop: bool = False,
    ) -> Optional[ControllerError]:
        """Start recording from selected sources.

        Args:
            selected_sources: Set of source types ('mic', 'system', 'fake')
            fake_path: Path to WAV file for fake source (required when
                'fake' is in selected_sources).
            fake_denoise: Whether to apply denoising to the fake source.
                Only meaningful when 'fake' is in selected_sources.
            fake_loop: Whether to loop the fake source WAV file.

        Returns:
            ControllerError if start failed, None on success
        """
        # Defer (but don't cancel) any in-flight post-processing from a
        # previous recording.  The queue's idle-wait gate already prevents
        # new jobs from starting while recording is active.  Cancelling
        # here would discard a completed diarization and leave the previous
        # recording stuck at "(processing speakers...)".
        with self._state_lock:
            if self._post_processor is not None:
                self._post_process_job_id = None

        # Validate state (read snapshot under lock)
        with self._state_lock:
            current_state = self._state

        # Validate state
        if current_state in (ControllerState.RECORDING, ControllerState.STARTING):
            return self._set_error("Already recording", is_recoverable=True)

        if current_state == ControllerState.STOPPING:
            return self._set_error("Cannot start while stopping", is_recoverable=True)

        # Validate sources
        if not selected_sources:
            return self._set_error(
                "No audio source selected. Enable microphone or system audio.",
                is_recoverable=True
            )

        # Clear any previous error
        self.clear_error()
        self._set_state(ControllerState.STARTING)
        logger.debug("Starting recording...")

        try:
            # Initialize transcription if enabled
            if self.enable_transcription:
                logger.debug("Initializing transcription...")
                error = self._init_transcription()
                if error:
                    # Log warning but continue with recording
                    logger.warning("Transcription not available: %s", error.message)

            # Build source configs
            source_configs = self._build_source_configs(
                selected_sources,
                fake_path=fake_path,
                fake_denoise=fake_denoise,
                fake_loop=fake_loop,
            )

            if not source_configs:
                return self._set_error(
                    "No valid audio sources configured",
                    is_recoverable=True
                )

            # Read denoising settings from persisted config with safe fallbacks
            denoise_enabled = False  # safe default (disabled due to spectral gate artifacts)
            denoise_provider = "spectral_gate"  # safe default
            denoise_budget_ms = 200.0  # safe default
            try:
                settings = self._config_manager.get_settings()
                ts = settings.transcription

                # Validate enabled - must be actual bool
                raw_enabled = ts.microphone_denoising_enabled
                denoise_enabled = raw_enabled if isinstance(raw_enabled, bool) else False

                # Validate provider - must be a non-empty string in the allowed set
                raw_provider = ts.microphone_denoising_provider
                from meetandread.audio.denoising import VALID_PROVIDER_NAMES
                if isinstance(raw_provider, str) and raw_provider in VALID_PROVIDER_NAMES:
                    denoise_provider = raw_provider
                else:
                    if raw_provider:
                        logger.warning(
                            "Invalid denoising provider '%s' in config, "
                            "falling back to '%s'",
                            raw_provider, denoise_provider,
                        )

                # Validate budget - must be a positive number
                raw_budget = ts.microphone_denoising_latency_budget_ms
                if isinstance(raw_budget, (int, float)) and raw_budget > 0:
                    denoise_budget_ms = float(raw_budget)
                else:
                    if raw_budget is not None and raw_budget != 200:
                        logger.warning(
                            "Invalid denoising latency budget %r in config, "
                            "falling back to %.0fms",
                            raw_budget, denoise_budget_ms,
                        )
            except Exception as exc:
                logger.warning(
                    "Failed to read denoising config, using defaults: %s", exc
                )

            # Tag mic sources with denoise=True when denoising is enabled
            if denoise_enabled:
                for sc in source_configs:
                    if sc.type == "mic" and sc.denoise is None:
                        sc.denoise = True

            # If fake_denoise was requested, force denoising on at the session
            # level regardless of persisted config (test harness override)
            if fake_denoise:
                denoise_enabled = True

            # Create and start session
            config = SessionConfig(
                sources=source_configs,
                enable_microphone_denoising=denoise_enabled,
                denoising_provider_name=denoise_provider if denoise_enabled else None,
                denoising_latency_budget_ms=denoise_budget_ms,
                on_frames_dropped=self._on_session_frames_dropped,
                on_error=self._on_session_error,
            )
            logger.info(
                "Denoising config: enabled=%s provider=%s budget=%.0fms",
                denoise_enabled, denoise_provider, denoise_budget_ms,
            )
            # Wire audio callback to feed transcription processor
            if self.enable_transcription and self._transcription_processor:
                config.on_audio_frame = self.feed_audio_for_transcription
                logger.debug("Audio callback wired to transcription processor")

            self._snapshot_active_sources(source_configs)
            self._session = AudioSession()
            self._session.start(config)
            logger.debug("Audio session started")

            # Start transcription if available
            if self._transcription_processor:
                logger.debug("Starting transcription processor...")
                logger.debug("Transcription processor exists: %s", self._transcription_processor is not None)
                logger.debug("Processor on_result callback: %s", self._transcription_processor.on_result is not None)
                self._transcription_processor.start()
                logger.debug("Transcription processor started")

            self._audio_chunks_fed = 0
            self._reset_live_speaker_state()
            self._set_state(ControllerState.RECORDING)
            self._start_hotplug_monitor()
            logger.info("Recording started successfully")
            return None

        except NoSourcesError as e:
            return self._set_error(f"No sources: {e}", is_recoverable=True)
        except AudioSourceError as e:
            return self._set_error(f"Audio device error: {e}", is_recoverable=True)
        except SessionError as e:
            return self._set_error(f"Session error: {e}", is_recoverable=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return self._set_error(f"Unexpected error: {e}", is_recoverable=False)

    def stop(self) -> Optional[ControllerError]:
        """Stop recording and finalize to WAV.

        This is non-blocking - finalization happens on a worker thread.

        Returns:
            ControllerError if stop cannot be initiated, None if stop started
        """
        with self._state_lock:
            current_state = self._state
        if current_state != ControllerState.RECORDING:
            return self._set_error("Not currently recording", is_recoverable=True)

        logger.info("[STOP-TIMER] stop() called on thread: %s",
                    threading.current_thread().name)
        self._stop_hotplug_monitor()
        self._set_state(ControllerState.STOPPING)

        # Run stop/finalize in worker thread to avoid blocking UI
        self._worker_thread = threading.Thread(
            target=self._stop_worker,
            daemon=True,
            name="RecordingStopWorker"
        )
        self._worker_thread.start()
        logger.info("[STOP-TIMER] stop() returning, worker started")

        return None

    def cancel_post_processing(self) -> None:
        """Cancel any in-flight post-processing job (idempotent, safe).

        Called when a new recording is about to start so the queue's
        idle-wait gate and any running job are stopped.  Exceptions are
        logged and swallowed so they cannot leave the controller in a
        bad state.
        """
        try:
            if self._post_processor is None:
                return
            if self._post_process_job_id is not None:
                logger.info(
                    "Cancelling post-processing job %s (new recording starting)",
                    self._post_process_job_id,
                )
                self._post_processor.cancel_job(
                    self._post_process_job_id,
                    reason="new recording starting",
                )
                self._post_process_job_id = None
            # Also cancel whatever the queue worker might be running right now
            self._post_processor.cancel_current_job(
                reason="new recording starting",
            )
        except Exception as exc:
            logger.warning(
                "cancel_post_processing error (non-fatal): %s", exc,
            )

    def _run_diarization_for_postprocess(self, wav_path: Path) -> "DiarizationResult":
        """Run diarization in the context of post-processing.

        This is the callback passed to PostProcessingQueue so diarization
        runs in the queue's worker thread instead of blocking the stop
        worker.  Delegates to the existing _run_diarization method but
        returns the result object instead of discarding it.

        Args:
            wav_path: Path to the saved WAV file.

        Returns:
            DiarizationResult from the diarizer.
        """
        # We need the result object back from _run_diarization, but
        # the current method stores it internally.  Call the diarizer
        # directly to get the return value.
        try:
            import numpy as np
            from meetandread.speaker.diarizer import Diarizer
            from meetandread.speaker.signatures import VoiceSignatureStore
            from meetandread.audio.storage.paths import get_recordings_dir
        except ImportError:
            logger.warning(
                "sherpa-onnx not installed - speaker diarization skipped. "
                "Install sherpa-onnx to enable speaker identification."
            )
            return None

        try:
            settings = self._config_manager.get_settings()
            speaker_cfg = settings.speaker

            if not speaker_cfg.enabled:
                logger.info("Speaker diarization disabled in settings - skipped")
                return None

            logger.info("Running speaker diarization on %s (post-process)", wav_path.name)

            # (1) Run diarization
            diarizer = Diarizer(
                clustering_threshold=speaker_cfg.clustering_threshold,
                min_duration_on=speaker_cfg.min_duration_on,
                min_duration_off=speaker_cfg.min_duration_off,
            )
            result = diarizer.diarize_subprocess(wav_path)

            if not result.succeeded:
                logger.error(
                    "Diarization failed for %s: %s", wav_path.name, result.error
                )
                return None

            # --- Degraded-result fallback: 0 speakers ---
            if result.num_speakers == 0:
                logger.warning(
                    "Diarization returned 0 speakers for %s - falling back to "
                    "single-speaker labeling",
                    wav_path.name,
                )
                self._fallback_single_speaker_labeling(result)
                return result

            if not result.segments:
                logger.info("No speaker segments detected in %s", wav_path.name)
                return None

            logger.info(
                "Diarized %s: %d segments, %d speakers",
                wav_path.name, len(result.segments), result.num_speakers,
            )

            # --- Clean up noisy over-segmentation ---
            from meetandread.speaker.diarizer import cleanup_diarization_segments
            pre_cleanup_count = len(result.segments)
            result.segments = cleanup_diarization_segments(result.segments)
            if len(result.segments) != pre_cleanup_count:
                logger.info(
                    "Diarization cleanup: %d -> %d segments",
                    pre_cleanup_count, len(result.segments),
                )

            # (2) Save diarization embeddings to signature store and match
            db_path = get_recordings_dir() / "speaker_signatures.db"
            with VoiceSignatureStore(db_path=db_path) as store:
                for label, sig in result.signatures.items():
                    emb = np.asarray(sig.embedding, dtype=np.float32) if not isinstance(sig.embedding, np.ndarray) else sig.embedding
                    match = store.find_match(
                        emb,
                        threshold=speaker_cfg.confidence_threshold,
                    )
                    if match:
                        result.matches[label] = match
                    else:
                        display_label = result.speaker_label_for(label)
                        store.save_signature(
                            display_label,
                            emb,
                            averaged_from_segments=sig.num_segments,
                        )

            # (3) Tag transcript words with speaker labels (realtime store)
            if self._transcript_store:
                self._apply_speaker_labels(result)

            # Store result for pin-to-name UX
            self._last_diarization_result = result

            return result

        except Exception as exc:
            logger.error(
                "Speaker diarization error for %s: %s",
                wav_path.name, exc, exc_info=True,
            )
            return None

    def _stop_worker(self) -> None:
        """Worker thread that handles stop and finalization.

        Immediately sets IDLE so the user can start a new recording within
        ~1 second. Heavy finalization (thread joins, WAV conversion,
        transcript save) is deferred to a separate finalizer thread so
        CPU-bound work (Whisper inference, WAV encoding) does not hold
        the GIL and starve the UI thread.
        """
        t0 = _time.monotonic()
        logger.info("[STOP-TIMER] _stop_worker entered on thread: %s",
                    threading.current_thread().name)

        # Capture references to current session/processor before IDLE
        # allows a new recording to replace them.
        old_session = self._session
        old_processor = self._transcription_processor
        old_store = self._transcript_store

        # Signal transcription processor to stop (non-blocking).
        # The processing thread checks _is_running each loop iteration
        # and exits after completing the current Whisper inference.
        if old_processor:
            old_processor._is_running = False
            if hasattr(old_processor, '_stop_event'):
                old_processor._stop_event.set()
            if self._transcription_processor is old_processor:
                self._transcription_processor = None
            logger.info("[STOP-TIMER] processor signaled stop: %.1fms",
                        (_time.monotonic() - t0) * 1000)

        # Signal audio session to stop (non-blocking).
        # The consumer thread will drain remaining frames and exit.
        if hasattr(old_session, '_stop_event'):
            old_session._stop_event.set()
            logger.info("[STOP-TIMER] session stop event set: %.1fms",
                        (_time.monotonic() - t0) * 1000)

        # Move to IDLE immediately - no joins, no blocking.
        self._set_state(ControllerState.IDLE)
        logger.info("[STOP-TIMER] IDLE set, total stop_worker: %.1fms",
                    (_time.monotonic() - t0) * 1000)

        # Spawn a low-priority finalizer thread for the heavy work.
        # This thread joins the processing and consumer threads, finalizes
        # the WAV, saves the transcript, and schedules post-processing.
        # It runs independently of the controller's state.
        def _finalize():
            ft0 = _time.monotonic()
            try:
                # (1) Wait for transcription processor thread to finish.
                if old_processor and hasattr(old_processor, '_processing_thread'):
                    if old_processor._processing_thread:
                        logger.info("[STOP-TIMER] joining processor thread...")
                        old_processor._processing_thread.join(timeout=30.0)
                        logger.info("[STOP-TIMER] processor joined: %.2fs",
                                    _time.monotonic() - ft0)

                # (2) Stop audio session (join consumer + finalize WAV).
                logger.debug("Finalizing audio session...")
                wav_path = old_session.stop()
                with self._state_lock:
                    self._last_wav_path = wav_path
                logger.info("[STOP-TIMER] session.stop() done (WAV): %.2fs, path=%s",
                            _time.monotonic() - ft0, wav_path)

                # (3) Save transcript if available (before post-processing)
                transcript_path = None
                if old_store and self._last_wav_path:
                    # Commit any remaining live phrase words before saving
                    old_store.commit_live_phrase()
                    logger.info(
                        "Saving transcript (%d words)...",
                        old_store.get_word_count(),
                    )
                    transcript_path = self._save_transcript(store=old_store)
                    with self._state_lock:
                        self._last_transcript_path = transcript_path
                    logger.info("[STOP-TIMER] transcript saved: %.2fs",
                                _time.monotonic() - ft0)

                # (4) Schedule post-processing with stronger model + diarization
                if self._post_processor and self._last_wav_path and old_store:
                    from meetandread.audio.storage.paths import get_transcripts_dir
                    job = self._post_processor.schedule_post_process(
                        audio_file=self._last_wav_path,
                        realtime_transcript=old_store,
                        output_dir=get_transcripts_dir()
                    )
                    with self._state_lock:
                        self._post_process_job_id = job.job_id
                    logger.info("[STOP-TIMER] post-process scheduled: %.2fs",
                                _time.monotonic() - ft0)
                else:
                    logger.warning(
                        "Post-processing NOT scheduled: processor=%s, wav=%s, store=%s",
                        "exists" if self._post_processor else "None",
                        self._last_wav_path,
                        "exists" if old_store else "None",
                    )

                # (5) Notify completion — triggers history refresh on UI thread
                logger.info("[STOP-TIMER] firing on_recording_complete: %.2fs",
                            _time.monotonic() - ft0)
                if self.on_recording_complete:
                    try:
                        self.on_recording_complete(wav_path, transcript_path)
                    except Exception as e:
                        logger.error("Recording complete callback failed: %s", e)
                logger.info("[STOP-TIMER] finalizer done: %.2fs",
                            _time.monotonic() - ft0)

            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.error("Finalizer error: %s", e)
                with self._state_lock:
                    self._error = ControllerError(
                        f"Finalization warning: {e}", is_recoverable=True,
                    )

        finalizer = threading.Thread(
            target=_finalize,
            daemon=True,
            name="RecordingFinalizer",
        )
        self._finalizer_thread = finalizer
        finalizer.start()
        # _stop_worker returns immediately - IDLE is already set

    def shutdown(self, timeout: float = 10.0) -> None:
        """Idempotent exit-only shutdown that waits for finalization.

        This is ONLY for application exit.  It initiates stop() when
        recording, waits briefly for the stop worker to spawn the
        finalizer thread, then joins the finalizer with a bounded
        timeout.  Normal stop() remains non-blocking per D045.

        Safe to call multiple times, from any state, and from any
        thread.  Never raises — all exceptions are logged and swallowed
        so the quit path is never compromised.

        Args:
            timeout: Maximum seconds to wait for the finalizer thread.
                Clamped to [1.0, 60.0].
        """
        # Clamp timeout to sane range
        timeout = max(1.0, min(60.0, float(timeout)))
        logger.info(
            "shutdown() called: state=%s, timeout=%.1fs",
            self._state.name, timeout,
        )

        # Idempotency guard: if already shut down, return immediately.
        # We check _finalizer_thread == None AND state is IDLE/ERROR
        # as a signal that stop was already completed or never started.
        with self._state_lock:
            current_state = self._state

        # If recording or stopping, initiate stop first
        if current_state in (ControllerState.RECORDING, ControllerState.STOPPING):
            logger.info("shutdown(): initiating stop (state=%s)", current_state.name)
            try:
                self.stop()
            except Exception as exc:
                logger.warning("shutdown(): stop() raised %s, continuing", exc)

            # Wait for stop worker to spawn the finalizer thread
            # Poll with short sleeps — the stop worker sets IDLE and spawns
            # the finalizer very quickly (< 1 second per D045).
            worker_deadline = _time.monotonic() + 2.0
            while _time.monotonic() < worker_deadline:
                with self._state_lock:
                    if self._state == ControllerState.IDLE:
                        break
                _time.sleep(0.05)

            # Small additional wait to allow the stop worker to assign
            # self._finalizer_thread (it sets IDLE first, then spawns).
            _time.sleep(0.1)

        # Join the finalizer thread if it exists and is alive
        finalizer = self._finalizer_thread
        if finalizer is not None and finalizer.is_alive():
            logger.info(
                "shutdown(): waiting for finalizer thread (timeout=%.1fs)",
                timeout,
            )
            finalizer.join(timeout=timeout)
            if finalizer.is_alive():
                logger.warning(
                    "shutdown(): finalizer did not complete within %.1fs — "
                    "proceeding with exit (WAV/transcript may be incomplete)",
                    timeout,
                )
            else:
                logger.info("shutdown(): finalizer completed successfully")
        else:
            logger.info("shutdown(): no active finalizer to wait for")

        logger.info("shutdown() complete")

    def _init_transcription(self) -> Optional[ControllerError]:
        """Initialize transcription components.

        HYBRID TRANSCRIPTION:
        - Uses AccumulatingTranscriptionProcessor for real-time
          * 60s window for context
          * Updates every 2 seconds
          * 3s silence detection for phrase breaks
        - Post-processing uses stronger model (scheduled on stop)

        Returns:
            ControllerError if initialization failed, None on success
        """
        try:
            # Get transcription settings from config
            settings = self._config_manager.get_settings()

            # HYBRID: Always use tiny for real-time (fastest)
            # Post-processing will use stronger model
            realtime_model = settings.transcription.realtime_model_size
            logger.debug("Initializing accumulating transcription with %s model", realtime_model)

            # Create transcript store
            self._transcript_store = TranscriptStore()
            self._transcript_store.start_recording()
            logger.debug("Transcript store initialized")

            # Create accumulating transcription processor
            # Configuration optimized for meetings:
            # - 60s window for good context
            # - 2s update frequency for responsiveness
            # - 3s silence timeout for natural turn-taking
            self._transcription_processor = AccumulatingTranscriptionProcessor(
                model_size=realtime_model,
                window_size=60.0,  # 60 seconds of context
                update_frequency=2.0,  # Update every 2 seconds
                silence_timeout=3.0  # 3 seconds of silence = phrase complete
            )

            # Load model (tiny takes 1-2 seconds)
            logger.info("Loading %s model for real-time transcription...", realtime_model)
            self._transcription_processor.load_model(
                progress_callback=lambda p: logger.info("Loading %s model: %d%%", realtime_model, p)
            )
            logger.info("%s model loaded successfully", realtime_model)

            # Wire up the phrase result callback
            self._transcription_processor.on_result = self._on_phrase_result
            logger.debug("Transcription result callback wired")

            # Initialize post-processing queue (for after recording stops)
            if settings.transcription.enable_postprocessing:
                logger.debug("Initializing post-processing queue")
                self._post_processor = PostProcessingQueue(
                    settings=settings,
                    on_progress=self._on_post_process_progress,
                    on_complete=self._on_post_process_complete_callback,
                    is_recording_callback=lambda: self.is_recording(),
                    diarize_callback=self._run_diarization_for_postprocess,
                    apply_speaker_labels_callback=lambda store, result: self._apply_speaker_labels(result, store),
                )
                self._post_processor.start()

            return None

        except Exception as e:
            import traceback
            traceback.print_exc()
            return ControllerError(
                message=f"Failed to initialize transcription: {e}",
                is_recoverable=True
            )

    def _on_phrase_result(self, result: SegmentResult) -> None:
        """Handle segment result from accumulating transcription processor.

        Args:
            result: SegmentResult with text, confidence, and completion status
        """
        logger.debug("Segment: %r [conf: %d%%, final: %s, idx: %s]",
                     result.text[:40], result.confidence, result.is_final,
                     result.segment_index)

        # Attempt live speaker matching (conservative; attaches name only
        # for high-confidence known-speaker matches)
        try:
            name = self._try_live_speaker_match()
            if name is not None:
                result.speaker_id = name
        except Exception as _lsm_exc:
            # Never block phrase result delivery — log and continue
            logger.debug(
                "Live speaker match error (phrase delivery continues): "
                "error_class=%s",
                type(_lsm_exc).__name__,
            )

        # Convert SegmentResult to Word objects for storage
        if self._transcript_store:
            words = self._segment_to_words(result)
            if words:
                if result.phrase_start:
                    # New phrase — commit any previous live phrase,
                    # then start fresh live buffer
                    self._transcript_store.commit_live_phrase()
                    self._transcript_store.set_live_phrase_words(words)
                    logger.debug("New phrase: %d words (total: %d)",
                                 len(words),
                                 self._transcript_store.get_word_count())
                elif result.is_final:
                    # Final transcription — commit the live phrase
                    self._transcript_store.set_live_phrase_words(words)
                    self._transcript_store.commit_live_phrase()
                    logger.debug("Final phrase: %d words (total: %d)",
                                 len(words),
                                 self._transcript_store.get_word_count())
                else:
                    # Re-transcription — replace the live phrase buffer
                    self._transcript_store.set_live_phrase_words(words)
                    logger.debug("Updated phrase: %d words (total: %d)",
                                 len(words),
                                 self._transcript_store.get_word_count())

        # Notify UI callback
        if self.on_phrase_result:
            try:
                self.on_phrase_result(result)
            except Exception as e:
                logger.error("Segment result callback failed: %s", e)

    def _segment_to_words(self, result: SegmentResult) -> List[Word]:
        """Convert a SegmentResult to Word objects.

        Args:
            result: SegmentResult from accumulating processor

        Returns:
            List of Word objects for storage
        """
        words = []
        text_parts = result.text.split()

        if not text_parts:
            return words

        # Distribute timing across words
        duration = result.end_time - result.start_time
        word_duration = duration / len(text_parts) if text_parts else 0

        for i, word_text in enumerate(text_parts):
            word_start = result.start_time + (i * word_duration)
            word_end = word_start + word_duration

            word = Word(
                text=word_text,
                start_time=word_start,
                end_time=word_end,
                confidence=result.confidence,
                speaker_id=None
            )
            words.append(word)

        return words

    def _on_post_process_progress(self, job_id: str, progress: int) -> None:
        """Handle post-processing progress updates.

        Args:
            job_id: The job identifier
            progress: Progress percentage (0-100)
        """
        logger.debug("Post-processing job %s: %d%%", job_id, progress)

    def _on_post_process_complete_callback(self, job_id: str, result: dict) -> None:
        """Handle post-processing completion (success or failure).

        Args:
            job_id: The job identifier
            result: Result dictionary with transcript_path, etc.
                On failure, contains 'error' and 'status' keys.
        """
        is_failure = result.get("status") == "failed"

        if is_failure:
            logger.error(
                "Post-processing job %s FAILED: %s",
                job_id, result.get("error", "unknown"),
            )
        else:
            logger.info(
                "Post-processing job %s completed: %s words (realtime: %s)",
                job_id,
                result.get("word_count", "?"),
                result.get("realtime_word_count", "?"),
            )

        # --- Store diarization result from background processing ---
        diarization_result = result.get("diarization_result")
        if diarization_result is not None:
            with self._state_lock:
                self._last_diarization_result = diarization_result
            logger.info(
                "Post-processing job %s delivered diarization result (%d speakers)",
                job_id,
                getattr(diarization_result, "num_speakers", 0),
            )

        # --- Auto-WER calculation (skip on failure) ---
        if not is_failure:
            self._compute_and_store_wer(result)

        if self.on_post_process_complete:
            transcript_path_str = result.get('transcript_path')
            if transcript_path_str and isinstance(transcript_path_str, str):
                transcript_path = Path(transcript_path_str)
                try:
                    self.on_post_process_complete(job_id, transcript_path)
                except Exception as e:
                    logger.error("Post-process complete callback failed: %s", e)
            elif is_failure:
                # Still notify UI so it can clear the "(processing speakers...)" indicator
                try:
                    self.on_post_process_complete(job_id, None)
                except Exception as e:
                    logger.error("Post-process failure callback failed: %s", e)

    def _compute_and_store_wer(self, result: dict) -> None:
        """Compute WER between realtime and post-processed transcripts and append to file.

        Extracts realtime words from the in-memory TranscriptStore, reads the
        post-processed words from the saved .md file's metadata JSON footer,
        calculates WER via calculate_wer(), and appends a 'wer' field to the
        file's metadata JSON footer.

        Args:
            result: Post-processing result dict with 'transcript_path' key.
        """
        try:
            from meetandread.performance.wer import calculate_wer

            # Gather realtime text from the in-memory store
            realtime_text = ""
            if self._transcript_store:
                words = self._transcript_store.get_all_words()
                realtime_text = " ".join(w.text for w in words)

            # Read post-processed text from the saved .md file
            transcript_path_str = result.get('transcript_path')
            if not transcript_path_str:
                return

            transcript_path = Path(transcript_path_str)
            if not transcript_path.exists():
                logger.warning("Cannot compute WER: transcript file not found: %s", transcript_path)
                return

            content = transcript_path.read_text(encoding="utf-8")
            footer_marker = "\n---\n\n<!-- METADATA: "
            marker_idx = content.find(footer_marker)
            if marker_idx == -1:
                logger.warning("Cannot compute WER: no metadata footer in %s", transcript_path)
                return

            # Parse metadata JSON
            import json
            metadata_text = content[marker_idx + len(footer_marker):]
            if metadata_text.strip().endswith(" -->"):
                metadata_text = metadata_text.strip()[:-len(" -->")]
            data = json.loads(metadata_text)

            # Extract post-processed words
            postproc_words = data.get("words", [])
            postproc_text = " ".join(w.get("text", "") for w in postproc_words)

            if not realtime_text.strip() and not postproc_text.strip():
                logger.info("Both transcripts empty - skipping WER calculation")
                return

            wer_value = calculate_wer(realtime_text, postproc_text)
            logger.info(
                "Auto-WER for %s: %.3f (realtime: %d words, postproc: %d words)",
                transcript_path.name, wer_value,
                len(realtime_text.split()) if realtime_text else 0,
                len(postproc_words),
            )

            # Append WER to the metadata and rewrite the file
            data["wer"] = wer_value

            # Rebuild the file: markdown body + updated metadata footer
            md_body = content[:marker_idx]
            updated_json = json.dumps(data, indent=2)
            new_content = md_body + footer_marker + updated_json + " -->\n"
            from meetandread.utils.file_utils import atomic_write
            atomic_write(transcript_path, new_content)

            # Store WER value for UI access
            with self._state_lock:
                self._last_wer = wer_value

        except Exception as exc:
            logger.error("Auto-WER computation failed: %s", exc, exc_info=True)

    def feed_audio_for_transcription(self, audio_chunk) -> None:
        """Feed audio chunk to transcription processor.

        This is called from the audio capture consumer thread
        to provide audio data for transcription.

        Args:
            audio_chunk: Audio samples as float32 numpy array
        """
        if self._transcription_processor and self._state == ControllerState.RECORDING:
            self._transcription_processor.feed_audio(audio_chunk)

            # Debug logging
            self._audio_chunks_fed += 1
            if self._audio_chunks_fed % 100 == 0:
                stats = self._transcription_processor.get_stats()
                logger.debug("Fed %d audio chunks, buffer: %.1fs", self._audio_chunks_fed, stats.get("buffer_duration", 0))

        # Buffer raw PCM for live speaker matching (only while recording)
        if self._state == ControllerState.RECORDING and audio_chunk is not None:
            try:
                import numpy as np
                chunk = audio_chunk
                if isinstance(chunk, np.ndarray):
                    # Convert float32 to int16 PCM bytes for the live buffer
                    clamped = np.clip(chunk, -1.0, 1.0)
                    pcm_int16 = (clamped * 32767).astype(np.int16)
                    pcm_bytes = pcm_int16.tobytes()
                    with self._buffer_lock:
                        self._live_audio_buffer.extend(pcm_bytes)
                        # Trim to rolling window
                        if len(self._live_audio_buffer) > self._live_max_buffer_bytes:
                            excess = len(self._live_audio_buffer) - self._live_max_buffer_bytes
                            del self._live_audio_buffer[:excess]
            except Exception as _pcm_exc:
                # Non-critical; matching simply won't have audio
                logger.debug(
                    "PCM buffer write for live matching failed "
                    "(matching degraded): error_class=%s",
                    type(_pcm_exc).__name__,
                )

    # ------------------------------------------------------------------
    # Live speaker matching (conservative, for CC overlay)
    # ------------------------------------------------------------------

    def _ensure_live_extractor(self) -> bool:
        """Lazily initialize the speaker embedding extractor for live matching.

        Returns True if the extractor is available (or was successfully
        created), False otherwise.  Failures are sticky for the session.
        """
        if self._live_extractor_available is True:
            return True
        if self._live_extractor_available is False:
            return False

        try:
            import sherpa_onnx
            from meetandread.speaker.model_downloader import ensure_embedding_model

            emb_path = ensure_embedding_model()
            if not emb_path or not emb_path.exists():
                self._live_extractor_available = False
                self._live_last_status = "model_unavailable"
                logger.info("Live speaker matching disabled: embedding model not found")
                return False

            config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(emb_path),
            )
            if not config.validate():
                self._live_extractor_available = False
                self._live_last_status = "model_unavailable"
                logger.info("Live speaker matching disabled: extractor config invalid")
                return False

            self._live_extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
            self._live_extractor_available = True
            self._live_last_status = "enabled"
            logger.info("Live speaker matching enabled")
            return True

        except (ImportError, Exception) as exc:
            self._live_extractor_available = False
            self._live_last_error_class = type(exc).__name__
            self._live_last_error_message = str(exc)[:_SANITIZED_STATUS_MAX_LENGTH]
            self._live_last_status = "extractor_error"
            logger.info(
                "Live speaker matching disabled: %s", type(exc).__name__
            )
            return False

    def _try_live_speaker_match(self) -> Optional[str]:
        """Attempt to match buffered live audio against known speakers.

        Returns the matched speaker name for high-confidence matches,
        or None.  Thread-safe via the extractor lock.  Never raises -
        all failures degrade to None.
        """
        now = _time.monotonic()

        # Rate-limit: don't attempt more than once per interval
        with self._buffer_lock:
            last_ts = self._live_last_attempt_ts
            buffer_len = len(self._live_audio_buffer)
            min_bytes = self._live_min_audio_bytes
        if last_ts is not None and now - last_ts < self._live_attempt_interval:
            return None

        # Check audio buffer has enough data
        if buffer_len < min_bytes:
            with self._buffer_lock:
                self._live_last_status = "insufficient_audio"
            return None

        # Check speaker settings
        try:
            settings = self._config_manager.get_settings()
            if not settings.speaker.enabled:
                self._live_last_status = "disabled"
                return None
        except Exception as _cfg_exc:
            logger.debug(
                "Config read failed during speaker matching, "
                "defaulting to enabled: error_class=%s",
                type(_cfg_exc).__name__,
            )

        # Lazy-init extractor (non-blocking check)
        if not self._ensure_live_extractor():
            return None

        # Serialize extraction to avoid concurrent sherpa-onnx use
        if not self._live_extractor_lock.acquire(blocking=False):
            return None

        try:
            with self._buffer_lock:
                self._live_last_attempt_ts = now
                self._live_match_attempts += 1
                raw_bytes = bytes(self._live_audio_buffer)

            # Convert int16 PCM buffer to mono float32 numpy array
            import numpy as np
            pcm_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
            audio_float32 = pcm_int16.astype(np.float32) / 32768.0

            # Create stream and compute embedding
            stream = self._live_extractor.create_stream()
            stream.accept_waveform(16000, audio_float32)
            stream.input_finished()

            if not self._live_extractor.is_ready(stream):
                with self._buffer_lock:
                    self._live_last_status = "insufficient_audio"
                    self._live_match_fallbacks += 1
                return None

            embedding = self._live_extractor.compute(stream)

            if embedding is None or len(embedding) == 0:
                with self._buffer_lock:
                    self._live_last_status = "no_match"
                    self._live_match_fallbacks += 1
                return None

            # Look up against known signatures
            from meetandread.speaker.signatures import VoiceSignatureStore
            from meetandread.audio.storage.paths import get_recordings_dir

            db_path = get_recordings_dir() / "speaker_signatures.db"
            try:
                with VoiceSignatureStore(db_path=db_path) as store:
                    match = store.find_match(
                        embedding,
                        threshold=_LIVE_SPEAKER_MATCH_THRESHOLD,
                    )
            except Exception as store_exc:
                with self._buffer_lock:
                    self._live_last_error_class = type(store_exc).__name__
                    self._live_last_error_message = str(store_exc)[:_SANITIZED_STATUS_MAX_LENGTH]
                    self._live_last_status = "store_error"
                    self._live_match_fallbacks += 1
                return None

            if match is None or match.confidence != "high":
                with self._buffer_lock:
                    self._live_last_status = (
                        "high_confidence_match_without_name"
                        if match is not None and match.confidence != "high"
                        else "no_match"
                    )
                    self._live_match_fallbacks += 1
                return None

            # High-confidence match found
            with self._buffer_lock:
                self._live_match_hits += 1
                self._live_last_status = "matched"
            return match.name

        except Exception as exc:
            with self._buffer_lock:
                self._live_last_error_class = type(exc).__name__
                self._live_last_error_message = str(exc)[:_SANITIZED_STATUS_MAX_LENGTH]
                self._live_last_status = "extractor_error"
                self._live_match_fallbacks += 1
            logger.debug(
                "Live speaker matching error: %s", type(exc).__name__
            )
            return None

        finally:
            self._live_extractor_lock.release()

    def _reset_live_speaker_state(self) -> None:
        """Reset live speaker matching state for a new recording."""
        with self._buffer_lock:
            self._live_audio_buffer = bytearray()
            self._live_match_attempts = 0
            self._live_match_hits = 0
            self._live_match_fallbacks = 0
            self._live_last_status = "disabled"
            self._live_last_error_class = None
            self._live_last_error_message = None
            self._live_last_attempt_ts = None
        # Keep extractor available across recordings for efficiency
        # (don't recreate it for each recording session)

    def get_live_audio_samples(
        self, duration_seconds: float = 1.5
    ) -> "np.ndarray":
        """Return recent live audio as a normalized float32 NumPy array.

        Copies the most recent *duration_seconds* of audio from the internal
        live buffer, converts from int16 PCM to float32 in approximately
        [-1.0, 1.0], and returns a **snapshot** that does not alias the
        internal buffer.  Empty or invalid states return an empty float32
        array - this method never raises.

        Thread-safe: takes a consistent snapshot of the buffer under
        _buffer_lock so concurrent writes from the audio callback thread
        do not cause torn reads.

        Args:
            duration_seconds: How many seconds of recent audio to return.
                Clamped to a positive value; capped by the rolling max
                buffer window.

        Returns:
            NumPy float32 array of normalized samples (one sample per
            16 kHz int16 frame).  Empty ``ndarray(dtype=float32)`` when
            no audio is available.
        """
        import numpy as np

        # Guard: non-positive duration → empty
        if duration_seconds <= 0:
            return np.ndarray(0, dtype=np.float32)

        # Clamp requested duration to the rolling max buffer window
        max_duration = self._live_max_buffer_bytes / (16000 * 2)
        duration_seconds = min(duration_seconds, max_duration)

        # Calculate requested byte count (int16 = 2 bytes per sample)
        requested_bytes = int(duration_seconds * 16000 * 2)

        # Take a consistent snapshot under buffer lock
        with self._buffer_lock:
            buf_snapshot = bytes(self._live_audio_buffer) if self._live_audio_buffer else b""

        if not buf_snapshot:
            return np.ndarray(0, dtype=np.float32)

        # Take only the most recent bytes, up to what's available
        available = min(requested_bytes, len(buf_snapshot))
        raw = buf_snapshot[-available:]

        # Align to whole int16 samples (2 bytes each) - drop trailing odd byte
        aligned_len = (len(raw) // 2) * 2
        if aligned_len == 0:
            return np.ndarray(0, dtype=np.float32)

        # Convert and normalize to float32 in [-1, 1]
        pcm_int16 = np.frombuffer(raw[:aligned_len], dtype=np.int16)
        normalized = pcm_int16.astype(np.float32) / 32768.0

        # Return a copy so the caller never aliases the internal buffer
        return normalized.copy()

    def _run_diarization(self, wav_path: Path) -> None:
        """Run speaker diarization on the saved WAV and tag transcript words.

        Post-processing step executed AFTER the WAV is saved and BEFORE the
        transcript is saved. Gracefully degrades if sherpa-onnx is not
        installed - logs a warning and returns without tagging.

        Args:
            wav_path: Path to the saved WAV file.
        """
        try:
            import numpy as np
            from meetandread.speaker.diarizer import Diarizer
            from meetandread.speaker.signatures import VoiceSignatureStore
            from meetandread.audio.storage.paths import get_recordings_dir
        except ImportError:
            logger.warning(
                "sherpa-onnx not installed - speaker diarization skipped. "
                "Install sherpa-onnx to enable speaker identification."
            )
            return

        try:
            settings = self._config_manager.get_settings()
            speaker_cfg = settings.speaker

            if not speaker_cfg.enabled:
                logger.info("Speaker diarization disabled in settings - skipped")
                return

            logger.info("Running speaker diarization on %s", wav_path.name)

            # (1) Run diarization
            diarizer = Diarizer(
                clustering_threshold=speaker_cfg.clustering_threshold,
                min_duration_on=speaker_cfg.min_duration_on,
                min_duration_off=speaker_cfg.min_duration_off,
            )
            logger.info(
                "Diarization config: clustering=%.2f min_duration_on=%.2f min_duration_off=%.2f",
                speaker_cfg.clustering_threshold,
                speaker_cfg.min_duration_on,
                speaker_cfg.min_duration_off,
            )
            result = diarizer.diarize(wav_path)

            if not result.succeeded:
                logger.error(
                    "Diarization failed for %s: %s", wav_path.name, result.error
                )
                return

            # --- Degraded-result fallback: 0 speakers ---
            if result.num_speakers == 0:
                logger.warning(
                    "Diarization returned 0 speakers for %s - falling back to "
                    "single-speaker labeling",
                    wav_path.name,
                )
                self._fallback_single_speaker_labeling(result)
                return

            if not result.segments:
                logger.info("No speaker segments detected in %s", wav_path.name)
                return

            logger.info(
                "Diarized %s: %d segments, %d speakers",
                wav_path.name, len(result.segments), result.num_speakers,
            )

            # --- Implausible speaker count warning ---
            if result.num_speakers > 8:
                logger.warning(
                    "Diarization detected %d speakers for %s - implausible "
                    "count, continuing with cleanup and labeling",
                    result.num_speakers, wav_path.name,
                )

            # --- Clean up noisy over-segmentation ---
            from meetandread.speaker.diarizer import cleanup_diarization_segments
            pre_cleanup_count = len(result.segments)
            result.segments = cleanup_diarization_segments(result.segments)
            if len(result.segments) != pre_cleanup_count:
                logger.info(
                    "Diarization cleanup in controller: %d -> %d segments "
                    "(gap_threshold=%.2fs, short_threshold=%.2fs)",
                    pre_cleanup_count, len(result.segments),
                    0.2, 0.5,
                )

            # (2) Save diarization embeddings to signature store and match
            #     against known identities.
            #
            #     Every new speaker gets a raw-label profile (SPK_N) so the
            #     user can later link it to an identity via the history
            #     dialog.  If the embedding already matches a known
            #     identity, we skip the raw save and record the match
            #     instead.
            db_path = get_recordings_dir() / "speaker_signatures.db"
            with VoiceSignatureStore(db_path=db_path) as store:
                for label, sig in result.signatures.items():
                    emb = np.asarray(sig.embedding, dtype=np.float32) if not isinstance(sig.embedding, np.ndarray) else sig.embedding
                    match = store.find_match(
                        emb,
                        threshold=speaker_cfg.confidence_threshold,
                    )
                    if match:
                        result.matches[label] = match
                        logger.debug(
                            "Matched %s -> '%s' (score=%.4f, confidence=%s)",
                            label, match.name, match.score, match.confidence,
                        )
                    else:
                        # Save raw profile so it can be linked later.
                        # Use the same display label (SPK_N) that
                        # _apply_speaker_labels will assign to words.
                        display_label = result.speaker_label_for(label)
                        store.save_signature(
                            display_label,
                            emb,
                            averaged_from_segments=sig.num_segments,
                        )
                        logger.debug(
                            "Saved raw profile '%s' to signature store (no known match)",
                            display_label,
                        )

            # (3) Tag transcript words with speaker labels
            self._apply_speaker_labels(result)

            # Store result for pin-to-name UX
            self._last_diarization_result = result

        except Exception as exc:
            logger.error(
                "Speaker diarization error for %s: %s",
                wav_path.name, exc, exc_info=True,
            )

    def _apply_speaker_labels(self, result: "DiarizationResult", transcript_store: Optional[TranscriptStore] = None) -> None:
        """Tag transcript store words with speaker IDs from diarization.

        Strategy:

        1. **Midpoint overlap**: For each word, check if its midpoint
           falls within a diarization segment. Assign the segment's
           speaker label.

        2. **Single-speaker fill**: When only 1 speaker is detected,
           the diarization segment often covers only part of the audio
           (pyannote is conservative on short recordings). In this
           case, assign ALL words to that single speaker regardless
           of segment boundaries.

        3. **Nearest-speaker fill**: For multi-speaker recordings,
           words that fall in gaps between segments are assigned to
           the temporally nearest segment's speaker.

        Args:
            result: A successful DiarizationResult with segments and matches.
            transcript_store: Optional TranscriptStore to apply labels to.
                When None (default), uses self._transcript_store.
        """
        store = transcript_store or self._transcript_store
        if store is None:
            raise RuntimeError(
                "Cannot apply speaker labels: no transcript store available"
            )

        words = store.get_all_words()
        if not words:
            return

        # Build a mapping from raw label -> display label
        label_map: dict[str, str] = {}
        for seg in result.segments:
            raw = seg.speaker
            if raw not in label_map:
                label_map[raw] = result.speaker_label_for(raw)

        if not label_map:
            return

        # --- Pass 1: midpoint overlap ---
        tagged_count = 0
        for word in words:
            word_mid = (word.start_time + word.end_time) / 2
            for seg in result.segments:
                if seg.start <= word_mid <= seg.end:
                    word.speaker_id = label_map[seg.speaker]
                    tagged_count += 1
                    break

        # --- Pass 2: fill untagged words ---
        if tagged_count < len(words):
            if len(label_map) == 1:
                # Single speaker: assign all words to that speaker
                single_label = next(iter(label_map.values()))
                for word in words:
                    if word.speaker_id is None:
                        word.speaker_id = single_label
                        tagged_count += 1
            else:
                # Multi-speaker: assign to nearest segment's speaker
                sorted_segments = sorted(result.segments, key=lambda s: s.start)
                for word in words:
                    if word.speaker_id is not None:
                        continue
                    word_mid = (word.start_time + word.end_time) / 2
                    nearest_label = min(
                        sorted_segments,
                        key=lambda s: min(
                            abs(word_mid - s.start),
                            abs(word_mid - s.end),
                        ),
                    ).speaker
                    word.speaker_id = label_map[nearest_label]
                    tagged_count += 1

        logger.info(
            "Tagged %d/%d words with speaker labels (%d speakers)",
            tagged_count, len(words), len(label_map),
        )

    def _fallback_single_speaker_labeling(self, result: "DiarizationResult") -> None:
        """Convert a degraded diarization result into single-speaker labeling.

        Creates a conservative fallback segment spanning the full transcript
        duration so all words receive a single speaker label. Does not
        fabricate embeddings - signatures remain empty.

        Args:
            result: A DiarizationResult with 0 speakers or no usable segments.
        """
        from meetandread.speaker.models import SpeakerSegment

        words = self._transcript_store.get_all_words()
        if not words:
            return

        duration = result.duration_seconds
        if duration <= 0:
            # Derive from transcript word spans
            duration = max(w.end_time for w in words)

        fallback_segment = SpeakerSegment(
            start=0.0,
            end=duration,
            speaker="spk0",
        )
        result.segments = [fallback_segment]
        result.num_speakers = 1

        self._apply_speaker_labels(result)

        logger.info(
            "Applied single-speaker fallback for %s (%.1fs, %d words)",
            getattr(self._last_wav_path, "name", "unknown"),
            duration,
            len(words),
        )

    def _speaker_matches_metadata(self) -> dict:
        """Build a raw-label-keyed speaker matches map for transcript persistence.

        Returns an empty dict when there is no successful diarization result.
        Otherwise collects every unique raw label from detected segments and
        serialises matched ``SpeakerMatch`` objects as
        ``{"identity_name": ..., "score": ..., "confidence": ...}``; detected-
        but-unmatched labels are serialised as ``None``.

        Returns:
            Dict mapping raw labels (e.g. ``"spk0"``) to match dicts or None.
        """
        result = self._last_diarization_result
        if result is None or not getattr(result, "succeeded", False):
            return {}

        segments = getattr(result, "segments", [])
        if not segments:
            return {}

        matches_map: dict = {}
        # Collect all unique raw labels from segments
        raw_labels: set[str] = set()
        for seg in segments:
            raw_labels.add(seg.speaker)

        for label in sorted(raw_labels):
            match = getattr(result, "matches", {}).get(label)
            if match is not None:
                matches_map[label] = {
                    "identity_name": match.name,
                    "score": match.score,
                    "confidence": match.confidence,
                }
            else:
                matches_map[label] = None

        return matches_map

    def _save_transcript(self, store: Optional["TranscriptStore"] = None) -> Optional[Path]:
        """Save transcript to file.

        Persists speaker match metadata alongside transcript words so
        downstream consumers can resolve raw diarization labels without
        re-running diarization.

        Args:
            store: TranscriptStore to save. When None (default), uses
                self._transcript_store.

        Returns:
            Path to saved transcript file, or None if no transcript
        """
        transcript_store = store or self._transcript_store
        if not transcript_store or not self._last_wav_path:
            return None

        try:
            from meetandread.audio.storage.paths import get_transcripts_dir

            # Create transcript filename based on WAV filename
            wav_stem = self._last_wav_path.stem
            transcripts_dir = get_transcripts_dir()
            transcript_path = transcripts_dir / f"{wav_stem}.md"

            # Build speaker matches metadata from last diarization result
            speaker_matches = self._speaker_matches_metadata()

            # Save as markdown with metadata
            transcript_store.save_to_file(
                transcript_path,
                speaker_matches=speaker_matches,
            )

            return transcript_path

        except Exception as e:
            logger.error("Failed to save transcript: %s", e)
            return None

    def _build_source_configs(
        self,
        selected_sources: Set[str],
        *,
        fake_path: Optional[str] = None,
        fake_denoise: bool = False,
        fake_loop: bool = False,
    ) -> List[SourceConfig]:
        """Build SourceConfig list from selected source types.

        Args:
            selected_sources: Set of source type strings.
            fake_path: Path to WAV file (required for 'fake' source).
            fake_denoise: Apply denoising to the fake source.
            fake_loop: Loop the fake source WAV file.

        Returns:
            List of SourceConfig objects for valid sources.
        """
        configs = []

        for source_type in selected_sources:
            source_type = source_type.lower().strip()

            if source_type == 'mic':
                configs.append(SourceConfig(type='mic', gain=1.0))
            elif source_type == 'system':
                configs.append(SourceConfig(type='system', gain=0.8))
            elif source_type == 'fake':
                if not fake_path:
                    logger.warning("Fake source requested without fake_path - skipping")
                    continue
                configs.append(SourceConfig(
                    type='fake',
                    fake_path=fake_path,
                    loop=fake_loop,
                    denoise=True if fake_denoise else None,
                ))

        return configs

    def get_last_recording_path(self) -> Optional[Path]:
        """Get path to the most recently completed recording (thread-safe snapshot)."""
        with self._state_lock:
            return self._last_wav_path

    def get_last_transcript_path(self) -> Optional[Path]:
        """Get path to the most recently completed transcript (thread-safe snapshot)."""
        with self._state_lock:
            return self._last_transcript_path

    def get_transcript_store(self) -> Optional[TranscriptStore]:
        """Get the current transcript store (for UI access during recording)."""
        return self._transcript_store

    def pin_speaker_name(self, raw_label: str, name: str) -> None:
        """Pin a user-chosen name to a speaker and save the voice signature.

        After pinning, re-checks all unmatched speakers against the updated
        signature store, then updates transcript word labels.

        Args:
            raw_label: Raw speaker label from diarization (e.g. "spk0").
            name: User-chosen display name for this speaker.
        """
        if not self._last_diarization_result or not self._last_transcript_path:
            logger.warning(
                "Cannot pin speaker '%s': no diarization result available",
                raw_label,
            )
            return

        result = self._last_diarization_result
        if not result.succeeded or raw_label not in result.signatures:
            logger.warning(
                "Cannot pin speaker '%s': no signature found in diarization result",
                raw_label,
            )
            return

        sig = result.signatures[raw_label]

        # Save or update the voice signature in the store
        from meetandread.audio.storage.paths import get_recordings_dir
        db_path = get_recordings_dir() / "speaker_signatures.db"
        try:
            from meetandread.speaker.signatures import VoiceSignatureStore
            with VoiceSignatureStore(db_path=db_path) as store:
                existing = store.find_match(sig.embedding, threshold=0.99)
                if existing and existing.name == name:
                    # Already saved - update the embedding average
                    store.update_signature(name, sig.embedding)
                else:
                    store.save_signature(name, sig.embedding, sig.num_segments)

                logger.info("Saved voice signature for '%s' (was %s)", name, raw_label)

                # Update the in-memory result mapping
                from meetandread.speaker.models import SpeakerMatch
                result.matches[raw_label] = SpeakerMatch(
                    name=name, score=1.0, confidence="high",
                )

                # Re-check all unmatched speakers against updated store
                for label, label_sig in result.signatures.items():
                    if label in result.matches:
                        continue  # Already matched (including the just-pinned one)
                    match = store.find_match(
                        label_sig.embedding,
                        threshold=self._config_manager.get_settings().speaker.confidence_threshold,
                    )
                    if match:
                        result.matches[label] = match
                        logger.info(
                            "Re-checked %s -> '%s' (score=%.4f)",
                            label, match.name, match.score,
                        )

            # Re-apply speaker labels to transcript words
            if self._transcript_store:
                self._apply_speaker_labels(result)

        except Exception as exc:
            logger.error("Failed to pin speaker '%s': %s", name, exc, exc_info=True)

    def get_speaker_names(self) -> dict:
        """Return current speaker label mapping from the last diarization.

        Returns:
            Dict mapping raw labels (e.g. "spk0") to display names
            (e.g. "Alice" or "SPK_0").
        """
        if not self._last_diarization_result:
            return {}
        result = self._last_diarization_result
        names = {}
        seen_labels = set()
        # From segments, collect all unique raw labels
        if hasattr(result, 'segments'):
            for seg in result.segments:
                if seg.speaker not in seen_labels:
                    seen_labels.add(seg.speaker)
                    names[seg.speaker] = result.speaker_label_for(seg.speaker)
        return names

    def get_last_wer(self) -> Optional[float]:
        """Return the WER value from the last auto-WER computation (thread-safe snapshot).

        Returns:
            WER as float (0.0-1.0+) or None if not yet computed.
        """
        with self._state_lock:
            return self._last_wer

    def get_diagnostics(self) -> dict:
        """Return sanitized controller diagnostics for testing/inspection.

        Thread-safe: takes consistent snapshots of state fields and buffer
        counters under their respective locks.

        Exposes controller state, recording paths, session stats (including
        denoising stats), VAD/transcription stats when present, and
        diarization result metadata. Does NOT expose raw audio, transcript
        text, embeddings, or secrets.

        Returns:
            Dict of sanitized diagnostic key/value pairs.
        """
        # Take consistent snapshot of state fields
        with self._state_lock:
            state = self._state
            last_wav = self._last_wav_path
            last_transcript = self._last_transcript_path
            error = self._error

        diag: dict = {
            "state": state.name,
            "last_wav_path": str(last_wav) if last_wav else None,
            "last_transcript_path": str(last_transcript) if last_transcript else None,
        }

        # Session stats
        try:
            stats = self._session.get_stats()
            diag["session"] = {
                "frames_recorded": stats.frames_recorded,
                "frames_dropped": stats.frames_dropped,
                "duration_seconds": stats.duration_seconds,
                "source_stats": stats.source_stats,
                "session_error": str(self._session.get_error()) if self._session.get_error() else None,
                "denoising": {
                    "provider": stats.denoising.provider,
                    "enabled": stats.denoising.enabled,
                    "active": stats.denoising.active,
                    "processed_frame_count": stats.denoising.processed_frame_count,
                    "fallback_count": stats.denoising.fallback_count,
                    "avg_latency_ms": stats.denoising.avg_latency_ms,
                    "max_latency_ms": stats.denoising.max_latency_ms,
                    "budget_exceeded_count": stats.denoising.budget_exceeded_count,
                    "last_error_class": stats.denoising.last_error_class,
                },
            }
        except Exception:
            logger.debug("Diagnostics: session stats unavailable")
        # Device hot-plug recovery diagnostics (sanitized - no audio/transcript/secrets)
        diag["hotplug"] = self._hotplug_diagnostics()


        # Transcription / VAD stats
        if self._transcription_processor:
            try:
                ts = self._transcription_processor.get_stats()
                diag["transcription"] = {
                    "buffer_duration": ts.get("buffer_duration"),
                    "segments_processed": ts.get("segments_processed"),
                }
                # VAD stats if available
                vad = getattr(self._transcription_processor, "get_vad_stats", None)
                if callable(vad):
                    vs = vad()
                    diag["vad"] = vs
            except Exception:
                logger.debug("Diagnostics: transcription stats unavailable")

        # Transcript store stats
        if self._transcript_store:
            try:
                words = self._transcript_store.get_all_words()
                diag["transcript"] = {
                    "word_count": len(words),
                    "words_with_speaker": sum(1 for w in words if w.speaker_id is not None),
                }
            except Exception:
                logger.debug("Diagnostics: transcript store stats unavailable")

        # Diarization result metadata
        with self._state_lock:
            diarization_result = self._last_diarization_result
        if diarization_result:
            try:
                result = self._last_diarization_result
                raw_labels: set = set()
                if hasattr(result, "segments"):
                    for seg in result.segments:
                        raw_labels.add(seg.speaker)
                matches = getattr(result, "matches", {})
                matched_labels = {lbl for lbl in raw_labels if lbl in matches}
                diag["diarization"] = {
                    "succeeded": result.succeeded,
                    "num_speakers": result.num_speakers,
                    "segment_count": len(result.segments),
                    "error": result.error,
                    "match_count": len(matched_labels),
                    "matched_label_count": len(matched_labels),
                    "labels": sorted(raw_labels),
                }
            except Exception:
                logger.debug("Diagnostics: diarization stats unavailable")

        # Live speaker matching diagnostics (sanitized - no names/embeddings)
        with self._buffer_lock:
            live_buffer_seconds = round(
                len(self._live_audio_buffer) / (16000 * 2), 1
            )
            live_attempts = self._live_match_attempts
            live_hits = self._live_match_hits
            live_fallbacks = self._live_match_fallbacks
            live_status = self._live_last_status
            live_err_class = self._live_last_error_class
            live_err_msg = self._live_last_error_message
            live_last_ts = self._live_last_attempt_ts
        with self._state_lock:
            live_ext_avail = self._live_extractor_available
            live_store_avail = self._live_store_available
        diag["live_speaker_matching"] = {
            "enabled": live_ext_avail is True,
            "extractor_available": live_ext_avail,
            "store_available": live_store_avail,
            "audio_buffer_seconds": live_buffer_seconds,
            "attempts": live_attempts,
            "matches": live_hits,
            "fallbacks": live_fallbacks,
            "last_status": live_status,
            "last_error_class": live_err_class,
            "last_error_message": live_err_msg,
            "last_attempt_ts": live_last_ts,
        }

        # Finalizer thread info (for shutdown diagnostics)
        finalizer_thread = self._finalizer_thread
        diag["finalizer"] = {
            "alive": finalizer_thread is not None and finalizer_thread.is_alive(),
            "name": getattr(finalizer_thread, "name", None) if finalizer_thread else None,
        }

        # Error info
        if error:
            diag["error"] = {
                "message": error.message,
                "is_recoverable": error.is_recoverable,
            }

        return diag