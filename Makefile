# MeetAndRead test runner.
# Topology defined in docs/adr/0001-test-execution-topology.md:
#   - Pure-logic layer:   WSL Linux Python (no native deps) -> `make test`
#   - Native/integration: Windows .venv via WSL interop      -> `make test-windows`
#
# Override interpreters via env, e.g.:
#   make test PY=python
#   make test-windows WIN_PY="C:/path/to/python.exe"

PY ?= python3
WIN_PY ?= .venv/Scripts/python.exe

# Default pytest selection: skip slow tests. The logic layer additionally
# deselects `windows`-marked tests (they are also auto-skipped off-Windows by
# tests/conftest.py, but `-m` keeps them out of the count entirely).
LOGIC_MARKERS ?= not slow and not windows
PYTEST_OPTS ?=

.PHONY: help test test-fast test-unit test-windows test-native

help:
	@echo "Targets (see docs/adr/0001-test-execution-topology.md):"
	@echo "  make test          pure-logic layer (WSL Linux python3)"
	@echo "  make test-fast     alias for 'test'"
	@echo "  make test-unit     same logic layer, verbose one-line failures"
	@echo "  make test-windows  authoritative FULL pass (Windows .venv via interop)"
	@echo "  make test-native   only windows-marked tests, under the Windows .venv"

# Pure-logic layer — green under WSL Linux Python without any native backend.
test test-fast:
	$(PY) -m pytest -m "$(LOGIC_MARKERS)" -q $(PYTEST_OPTS)

test-unit:
	$(PY) -m pytest -m "$(LOGIC_MARKERS)" --tb=line -q $(PYTEST_OPTS)

# Authoritative full pass — real Windows native audio stack (PortAudio/WASAPI).
test-windows:
	$(WIN_PY) -m pytest -q $(PYTEST_OPTS)

# Only the Windows-native subset, under the Windows venv.
test-native:
	$(WIN_PY) -m pytest -m windows -q $(PYTEST_OPTS)
