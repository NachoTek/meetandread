"""Recording controller - UI-friendly wrapper around AudioSession.

Provides non-blocking recording control with proper error handling
and state management for UI integration.
"""

import threading
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional, Set, Callable, List

from metamemory.audio import (
    AudioSession,
    SessionConfig,
    SourceConfig,
    SessionState,
    SessionError,
    NoSourcesError,
)
from metamemory.audio.capture import AudioSourceError


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
    
    Example:
        controller = RecordingController()
        controller.on_state_change = lambda state: print(f"State: {state}")
        controller.on_error = lambda err: print(f"Error: {err.message}")
        
        # Start recording
        error = controller.start({'mic', 'system'})
        if error:
            print(f"Failed to start: {error.message}")
        
        # Stop recording (non-blocking)
        controller.stop(lambda path: print(f"Saved to: {path}"))
    """
    
    def __init__(self):
        self._session = AudioSession()
        self._state = ControllerState.IDLE
        self._error: Optional[ControllerError] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._last_wav_path: Optional[Path] = None
        
        # Callbacks
        self.on_state_change: Optional[Callable[[ControllerState], None]] = None
        self.on_error: Optional[Callable[[ControllerError], None]] = None
        self.on_recording_complete: Optional[Callable[[Path], None]] = None
    
    def _set_state(self, state: ControllerState) -> None:
        """Update state and notify listeners."""
        self._state = state
        if self.on_state_change:
            self.on_state_change(state)
    
    def _set_error(self, message: str, is_recoverable: bool = True) -> ControllerError:
        """Set error state and notify listeners."""
        self._error = ControllerError(message, is_recoverable)
        self._set_state(ControllerState.ERROR)
        if self.on_error:
            self.on_error(self._error)
        return self._error
    
    def clear_error(self) -> None:
        """Clear error state and return to idle."""
        self._error = None
        if self._state == ControllerState.ERROR:
            self._set_state(ControllerState.IDLE)
    
    def get_state(self) -> ControllerState:
        """Get current controller state."""
        # Sync with underlying session state if needed
        session_state = self._session.get_state()
        if session_state == SessionState.RECORDING and self._state != ControllerState.RECORDING:
            self._set_state(ControllerState.RECORDING)
        return self._state
    
    def get_error(self) -> Optional[ControllerError]:
        """Get current error if any."""
        return self._error
    
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._state == ControllerState.RECORDING
    
    def is_busy(self) -> bool:
        """Check if controller is busy (starting, stopping, etc.)."""
        return self._state in (ControllerState.STARTING, ControllerState.STOPPING)
    
    def start(self, selected_sources: Set[str]) -> Optional[ControllerError]:
        """Start recording from selected sources.
        
        Args:
            selected_sources: Set of source types ('mic', 'system', 'fake')
        
        Returns:
            ControllerError if start failed, None on success
        """
        # Validate state
        if self._state in (ControllerState.RECORDING, ControllerState.STARTING):
            return self._set_error("Already recording", is_recoverable=True)
        
        if self._state == ControllerState.STOPPING:
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
        
        try:
            # Build source configs
            source_configs = self._build_source_configs(selected_sources)
            
            if not source_configs:
                return self._set_error(
                    "No valid audio sources configured",
                    is_recoverable=True
                )
            
            # Create and start session
            config = SessionConfig(sources=source_configs)
            self._session = AudioSession()
            self._session.start(config)
            
            self._set_state(ControllerState.RECORDING)
            return None
            
        except NoSourcesError as e:
            return self._set_error(f"No sources: {e}", is_recoverable=True)
        except AudioSourceError as e:
            return self._set_error(f"Audio device error: {e}", is_recoverable=True)
        except SessionError as e:
            return self._set_error(f"Session error: {e}", is_recoverable=True)
        except Exception as e:
            return self._set_error(f"Unexpected error: {e}", is_recoverable=False)
    
    def stop(self, on_complete: Optional[Callable[[Path], None]] = None) -> Optional[ControllerError]:
        """Stop recording and finalize to WAV.
        
        This is non-blocking - finalization happens on a worker thread.
        
        Args:
            on_complete: Callback when finalization completes (receives wav path)
        
        Returns:
            ControllerError if stop cannot be initiated, None if stop started
        """
        if self._state != ControllerState.RECORDING:
            return self._set_error("Not currently recording", is_recoverable=True)
        
        self._set_state(ControllerState.STOPPING)
        
        # Store callback
        if on_complete:
            self.on_recording_complete = on_complete
        
        # Run stop/finalize in worker thread to avoid blocking UI
        self._worker_thread = threading.Thread(
            target=self._stop_worker,
            daemon=True,
        )
        self._worker_thread.start()
        
        return None
    
    def _stop_worker(self) -> None:
        """Worker thread that handles stop and finalization."""
        try:
            wav_path = self._session.stop()
            self._last_wav_path = wav_path
            
            self._set_state(ControllerState.IDLE)
            
            # Notify completion
            if self.on_recording_complete:
                self.on_recording_complete(wav_path)
                
        except Exception as e:
            self._set_error(f"Failed to finalize recording: {e}", is_recoverable=False)
    
    def _build_source_configs(self, selected_sources: Set[str]) -> List[SourceConfig]:
        """Build SourceConfig list from selected source types."""
        configs = []
        
        for source_type in selected_sources:
            source_type = source_type.lower().strip()
            
            if source_type == 'mic':
                configs.append(SourceConfig(type='mic', gain=1.0))
            elif source_type == 'system':
                configs.append(SourceConfig(type='system', gain=0.8))
            elif source_type == 'fake':
                # For testing - would need fake_path in real usage
                # This is handled specially in the widget
                pass
        
        return configs
    
    def get_last_recording_path(self) -> Optional[Path]:
        """Get path to the most recently completed recording."""
        return self._last_wav_path
