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
4. **Assets bundled** — Verifies SVG icons and test data are included
5. **Executable launches** — Tests that the exe starts without DLL errors

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