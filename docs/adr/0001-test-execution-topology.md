# ADR 0001: Test execution topology — WSL + Windows interop

- **Date:** 2026-08-13
- **Status:** Accepted

## Context

MeetAndRead is a Windows desktop widget. Its audio capture stack targets
Windows: system/loopback audio uses WASAPI via `pyaudiowpatch` (Windows-only),
and microphone/system capture uses `sounddevice`, which wraps the native
PortAudio C library (`libportaudio2`). Development happens inside WSL, where
these Windows native libraries are not present — and cannot be: a Linux process
cannot load Windows PE DLLs (ABI mismatch: ELF vs PE).

The test suite therefore contains two distinct populations:

1. **Pure-logic tests** — Transcript Footer parsing, queue state machines,
   frame-drop counting (with mocks), UI helpers. These need no native backend.
2. **Native/integration tests** — anything that exercises the real audio stack,
   including tests that spawn the CLI as a subprocess. A subprocess starts a
   fresh interpreter, so it imports the real `sounddevice` and raises
   `OSError: PortAudio library not found` when PortAudio is absent.

`tests/conftest.py` already stubs `sounddevice` at import time so the app's
pure-Python layers are importable on non-Windows. That stub makes the
pure-logic layer testable everywhere — but it does **not** cross into spawned
subprocesses, so native/integration tests cannot be satisfied under the WSL
Linux interpreter. As a result, a pre-push hook that ran the whole suite under
WSL Linux Python was falsely blocked on the missing PortAudio library, even
though the code was correct.

The project's Windows venv (`.venv/Scripts/python.exe`) has all native
dependencies installed (PyQt6, `sounddevice` + PortAudio, `pyaudiowpatch`) and
is invokable from WSL via Windows interop. Running the native/integration
suite under it makes those tests pass against the real target binaries.

## Decision

Adopt a **two-layer test topology**, each layer run by the interpreter that can
validly satisfy its dependencies:

| Layer | Interpreter | Selection |
|---|---|---|
| Pure logic | WSL Linux Python (`python3`) | `-m "not windows"` |
| Native / integration | Windows venv via WSL interop (`.venv/Scripts/python.exe`) | full suite, or `-m windows` |

Mechanism:

- A `windows` pytest marker identifies tests that must run against the real
  Windows native audio stack.
- `tests/conftest.py` auto-skips `windows`-marked tests when
  `sys.platform != "win32"`, so the WSL Linux run is **green (skipped)** rather
  than red (errored) on missing libraries.
- A `Makefile` exposes one target per layer (`make test`, `make test-windows`,
  `make test-native`) so the correct interpreter is used without memorising
  incantations.
- Because the product is a Windows application, the authoritative pass
  (pre-push / CI) runs the full suite under the Windows venv interpreter.

## Consequences

**Positive**

- Native tests run against the binaries users actually run — real coverage, not
  a Linux approximation that does not exist in production.
- Fast pure-logic feedback loop in WSL with Linux Python (no native deps, fast
  in-memory `/tmp` I/O).
- Missing-library failures stop masquerading as code failures; the `windows`
  marker makes platform intent explicit and self-documenting.

**Negative / costs**

- Two interpreters to keep dependency-synchronised (Linux site-packages for the
  logic layer; Windows venv for the native layer). Dependency drift is possible
  and should be checked.
- `/mnt/c` filesystem I/O is slower under WSL for the Windows-layer run.
- Each interpreter uses its own OS temp by default, so `tmp_path` works
  out of the box. Tests that **hardcode POSIX paths** or **share paths across
  the two interpreters** must use `--basetemp` on a Windows-visible dir.
- GUI tests render to different display surfaces (WSLg for Linux Qt vs the
  Windows desktop for Windows Qt). Keep GUI tests on one layer to avoid
  comparing across surfaces.

## Alternatives considered

- **Install `libportaudio2` in WSL and run everything under Linux Python.**
  Rejected: this tests a Linux build that does not exist in production; it does
  not exercise the Windows target, and WASAPI loopback (`pyaudiowpatch`) is
  Windows-only regardless.
- **`--no-verify` to bypass the failing tests.** Rejected: hides the real gap
  and provides no Windows coverage.
- **ctypes-load Windows DLLs into the Linux interpreter.** Impossible: ABI
  mismatch (ELF vs PE).

## References

- `tests/conftest.py` — existing `sounddevice` import stub for the pure-logic
  layer; new `windows`-marker auto-skip.
- `Makefile` — per-layer runner targets.
- `.git/hooks/pre-push` — authoritative pass should target the Windows venv.
