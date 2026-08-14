## Agent skills

### Issue tracker

Issues live in GitHub Issues, managed via the `gh` CLI. See `docs/agents/issue-tracker.md`.

**Important:** `gh` requires a GitHub App token that is not available as a persistent env var in agent shell sessions. Always prefix `gh` commands with:

```
GH_TOKEN=$(bash ~/.config/gh-app/gh-app-token.sh 2>/dev/null) gh ...
```

This mirrors the `gh-app` shell function defined in `~/.bashrc`.

### Triage labels

Default five-label vocabulary (needs-triage, needs-info, ready-for-agent, ready-for-human, wontfix). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout: `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.

### Running tests

Two-layer topology — see `docs/adr/0001-test-execution-topology.md`. The app targets Windows-only native audio backends (PortAudio, WASAPI/`pyaudiowpatch`) that a Linux/WSL process cannot load, so the interpreter must match the layer:

- **Pure logic** — `make test` (or `python3 -m pytest -m "not slow and not windows"`). Fast, under WSL `python3`. `windows`-marked tests auto-skip off-Windows.
- **Native / integration** — `make test-windows` (`.venv/Scripts/python.exe -m pytest`). The authoritative pass; the only layer that exercises the real audio stack and CLI subprocess tests.

A versioned pre-push hook (`.githooks/pre-push`) gates every push on the full suite under the Windows venv. **If a push is blocked by `OSError: PortAudio library not found` or a `windows`-marked test, you are on the wrong interpreter** — switch to `make test-windows` / `.venv/Scripts/python.exe`. Do **not** reach for `--no-verify` to mask it; reserve `--no-verify` for content-free pushes (e.g. branch deletions).

Activate hooks once per clone: `git config core.hooksPath .githooks`.
