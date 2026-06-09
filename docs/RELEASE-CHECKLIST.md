# Release Checklist

## Pre-Tag Checks (Local) - DO THIS FIRST

- [ ] Run `build-and-validate.bat` OR:
  - [ ] Tests pass: `pytest -m "not slow"`
  - [ ] Lint passes: `flake8 src/ --max-line-length=120`
  - [ ] Build succeeds: `pyinstaller meetandread.spec --noconfirm`
  - [ ] Validation passes: `python validate_build.py`
- [ ] Manual test: Run `dist\meetandread\meetandread.exe` and verify it opens
- [ ] Test basic functionality: start/stop recording, check transcription works

## Create Release

- [ ] Update version in `pyproject.toml`
- [ ] Commit version change: `git commit -am "Bump version to X.Y.Z"`
- [ ] Push commit: `git push`
- [ ] Tag release: `git tag vX.Y.Z`
- [ ] Push tag: `git push origin vX.Y.Z`
- [ ] GitHub Actions will build and publish automatically

## Post-Release

- [ ] Download release artifact from GitHub Releases
- [ ] Test on a clean Windows machine (no Python)
- [ ] Update CHANGELOG.md with release notes
- [ ] Commit CHANGELOG.md: `git commit -am "Update changelog for X.Y.Z"`
- [ ] Push: `git push`

## Notes

- The `build-and-validate.bat` script does all pre-tag checks in one command
- Validation ensures all DLLs, modules, and assets are bundled correctly
- GitHub Actions only runs on tag push - not on regular pushes
- Use the CI artifacts for testing if you want to avoid local builds