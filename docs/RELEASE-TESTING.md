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

- **Pull requests to main**: lint + full test suite, then build + bundle validation; artifact uploaded for download
- **Nightly (main)**: lint + full test suite (safety net)
- **Tag push (v\*)**: lint + full test suite, then build + validation + published release
- **No CI on plain pushes to main** (issue #71; the pre-push git hook gates local work)

## Download Test Builds

Every pull request to main uploads the build as an artifact:

1. Go to the Actions tab
2. Click the "CI" workflow run on the pull request
3. Download `meetandread-windows`
4. Test on your machine before merging

## Why This Matters

PyInstaller's static analysis can miss:
- ctypes-loaded libraries
- delvewheel-patched packages
- Dynamically-discovered DLLs

The validation catches these before you push a broken release.