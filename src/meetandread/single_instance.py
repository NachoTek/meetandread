"""Single-instance guard via a named Windows mutex.

Prevents two copies of the (frozen) app from running concurrently. Uses
a named kernel mutex created through ``ctypes`` — no extra dependencies.
The handle is stored in a module-level global and intentionally never
closed: the OS releases the kernel object when the owning process dies,
so there is no stale-lock cleanup path.
"""

from __future__ import annotations

import ctypes
import sys

ERROR_ALREADY_EXISTS = 183

# Module-level: must stay alive for the process lifetime, otherwise the
# OS destroys the mutex and the lock silently releases.
_lock_handle = None


def _release_lock_for_tests() -> None:
    """Test seam: drop the held handle (not used by application code)."""
    global _lock_handle
    if _lock_handle is not None:
        try:
            ctypes.windll.kernel32.CloseHandle(_lock_handle)
        except Exception:
            pass
        _lock_handle = None


def acquire_single_instance_lock(name: str = "meetandread") -> bool:
    """Try to become the sole running instance.

    On Windows, creates (or opens) a named mutex
    ``Global\\<name>_single_instance``. If the mutex already exists
    (another instance is running), returns False. Otherwise keeps the
    handle alive for the process lifetime and returns True. The OS
    reclaims the mutex when the process exits — even on crash — so no
    stale-lock cleanup exists or is needed.

    On non-Windows platforms, always returns True without touching
    ``ctypes.windll``.

    Args:
        name: Base name for the mutex (defaults to "meetandread").

    Returns:
        True if this process holds the single-instance lock (or the
        platform does not support one); False if another instance
        already holds it or the mutex could not be created/opened
        (caller must refuse to start).
    """
    global _lock_handle

    if sys.platform != "win32":
        return True

    from ctypes import wintypes

    k32 = ctypes.windll.kernel32
    k32.CreateMutexW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
    k32.CreateMutexW.restype = wintypes.HANDLE
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    k32.CloseHandle.restype = wintypes.BOOL
    k32.GetLastError.restype = wintypes.DWORD

    handle = k32.CreateMutexW(None, False, f"Global\\{name}_single_instance")
    if k32.GetLastError() == ERROR_ALREADY_EXISTS:
        if handle:
            k32.CloseHandle(handle)
        return False

    if not handle:
        return False

    _lock_handle = handle
    return True


__all__ = ["acquire_single_instance_lock"]
