# Release Testing Guide

## Quick Start

Before pushing a release tag, test locally:

```powershell
# 1. Build the executable
pyinstaller meetandread.spec --noconfirm

# 2. Validate the build
python validate_build.py

# 3. Test manually (optional)
dist\meetandread\meetandread.exe

# 4. Smoke the Issue Reporter entry (issue #111)
dist\meetandread\issue-reporter.exe
```

If validation passes, push your tag:

```bash
git tag v0.19.2
git push origin v0.19.2
```

## What the Validation Checks

1. **Build directory exists** — Verifies PyInstaller ran successfully
2. **Required DLLs present** — Checks for pywhispercpp, sherpa-onnx, PortAudio, PyQt6, etc.
3. **Module imports work** — Imports each required module from the built exe
4. **Issue Reporter entry point** — `issue-reporter.exe` exists beside the app exe (issue #111)
5. **Assets bundled** — Verifies SVG icons and test data are included
6. **Executable launches** — Tests that the exe starts without DLL errors

## Smoke Test the Issue Reporter (issue #111)

The bundle contains two exes: `meetandread.exe` (the app) and
`issue-reporter.exe` (the Issue Reporter wizard, ADR 0003). Before
tagging a release:

1. Run `dist\meetandread\issue-reporter.exe` — the console wizard
   must reach the "describe the problem" prompt WITHOUT the app's
   subsystems (this is the startup-crash reporting path).
2. Optionally complete a wizard run end to end (describe -> launch ->
   reproduce -> stop -> review -> submit) to verify the frozen
   reporter supervises the frozen app.
3. Run `install-shortcuts.ps1` from the bundle and check both
   Start-menu shortcuts launch their exes.

The manual smoke check on a clean Windows install (no Python) is a
maintainer step — record it in the release notes.

## CI Workflow

One consolidated "CI" workflow (issue #71) with a single reusable
lint+test job:

- **Pull requests to main**: lint + full suite + PyInstaller bundle
  validation + artifact upload
- **Nightly schedule on main**: lint + full suite (safety net)
- **Tag push (v\*)**: lint + full suite, then release build published to
  the GitHub release

PyInstaller is pinned via `constraints.txt` — bump it deliberately.

## Download Test Builds

Pull-request runs upload the build as an artifact:

1. Go to Actions tab
2. Click the "CI" workflow run on your PR
3. Download `meetandread-windows`
4. Test on your machine before tagging

## Why This Matters

PyInstaller's static analysis can miss:
- ctypes-loaded libraries
- delvewheel-patched packages
- Dynamically-discovered DLLs

The validation catches these before you push a broken release.