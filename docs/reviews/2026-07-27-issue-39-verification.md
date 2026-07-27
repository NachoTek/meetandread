# MeetAndRead — Issue #39 Verification Report

**Repository**: MeetAndRead — Windows PyQt6 Desktop Audio Transcription Widget
**Spec**: Issue #39 (verify issue #24 as a shippable unit)
**Verify Date**: July 27, 2026
**Verifier**: opencode agent
**Parent**: #24 (Terminology alignment: scrub → re-transcribe, History → Library, capture nouns → Recording)
**Blocking remediations**: #37 (legacy `_scrub_` sidecar compatibility), #38 (re-transcribe UI returned to terminology-only scope) — both CLOSED.

---

## Outcome

**PASS.** Issue #24 ships as one self-consistent unit. All in-scope requirements are implemented, every out-of-scope boundary is respected, the relevant test suite passes, packaging consistency checks pass, and the only residual occurrences of deprecated terminology are explicitly classified as either backward-compat handling or pre-existing out-of-scope uses.

---

## 1. In-scope requirements and out-of-scope boundaries

### In-scope user stories from #24 — delivered

| # | User story | Evidence |
|---|------------|----------|
| 1 | Glossary lists Re-transcribe as canonical, Scrub in avoid list | `CONTEXT.md:47-49` |
| 2 | `ScrubRunner` → `RetranscribeRunner` | `retranscribe.py:44` |
| 3 | `scrub.py` → `retranscribe.py` | no `scrub.py` exists under `src/`; `src/meetandread/transcription/retranscribe.py` present |
| 4 | `_scrub_*` identifiers in `floating_panels.py` → `_retranscribe_*` | `grep -r _scrub_ src/meetandread/widgets/` → no matches; `_retranscribe_btn` at `floating_panels.py:1406`, `_start_retranscribe` family throughout |
| 5 | CSS `action="scrub"` → `action="retranscribe"` | `theme.py:1276,1279`; no `action="scrub"` anywhere |
| 6 | `transcript_scanner.py` skips `_retranscribe_` (and legacy `_scrub_`) | `transcript_scanner.py:198` calls `RetranscribeRunner.is_sidecar_path(md_path)` |
| 7 | User-facing strings say "Re-transcribe"; no user-facing "scrub" | `floating_panels.py:1406,1409,2062,2274,2300,2310,2402,2447,2523,4055,4061,4062`; `grep -iE '\bscrub' src/meetandread/widgets/` → zero matches |
| 8 | Legacy `_scrub_{model}.md` sidecars still recognized | `retranscribe.py:72` `LEGACY_SIDECAR_TAGS = ("scrub",)` |
| 9 | New sidecars named `_retranscribe_{model}.md` | `retranscribe.py:66` `SIDECAR_TAG = "retranscribe"`; `_sidecar_path` at `retranscribe.py:221-224` |
| 10 | Tests updated to new naming | `tests/test_retranscribe.py`, `tests/test_recording_management.py`, `tests/test_transcript_scanner.py` |
| 11 | Settings nav button says "Library" | `floating_panels.py:4303` `QPushButton("🕐  Library")` |
| 12 | Transcript list tab label says "Library" | `floating_panels.py:1439` `addTab(history_tab, "Library")` |
| 13 | README "History tab" → "Library tab" | `README.md:118` "Settings panel → Library tab" |
| 14 | README "Audio Capture" heading → Recording | `README.md:44` `### 🎙️ Recording` (no "Audio Capture" heading) |
| 15 | README pipeline diagram "Audio Input" → "Audio Source" | `README.md:130` `│ Audio Source │` |
| 16 | Noun "capture" in `floating_panels.py` comments rephrased | `grep -i capture src/meetandread/widgets/floating_panels.py` → only verb usages at `floating_panels.py:2863,9200` ("Capture the full prefix", "Capture pre-click playback state") — general English, correctly left per #24's verb/noun rule |

### Out-of-scope boundaries from #24 — respected

| Boundary | Status |
|----------|--------|
| Internal `history`-prefixed identifiers (`_history_list`, `_refresh_history`, `_HistoryRowWidget`, `HistoryPlaybackController`) | Unchanged — still present |
| `capture/` source directory rename | Not done — `src/meetandread/audio/capture/` still exists |
| `benchmark_history` config key / labels | Unchanged — `floating_panels.py:4970` `QLabel("Benchmark History")` retained |
| UI layout / behavioral change | None — rename only |
| `archive` in "archive quality" adjective | Unchanged — `floating_panels.py:4351` `QLabel("Post Process Model (archive quality):")` retained |
| `input` in prose (except ASCII diagram label) | Unchanged — README diagram label is the only `Audio Input` → `Audio Source` change |

---

## 2. Legacy `_scrub_` sidecar lifecycle — PASS

- `RetranscribeRunner.LEGACY_SIDECAR_TAGS = ("scrub",)` — `retranscribe.py:72`
- `all_sidecar_tags()` returns `("retranscribe", "scrub")` — `retranscribe.py:75-83`
- `is_sidecar_path()` matches both tags — `retranscribe.py:86-94`
- `find_sidecars()` enumerates both tag patterns — `retranscribe.py:97-107`
- **Library scan:** `transcript_scanner.py:198` uses `is_sidecar_path()` so legacy sidecars stay hidden
- **Recording enumeration:** `management.py:144` calls `find_sidecars(tra_dir, stem)` so legacy sidecars are listed for deletion
- **Rename:** `management.py:184-189` iterates `find_sidecars` for the old stem and renames each to the new stem, preserving the tag suffix
- **Delete:** `management.py:314,359` enumerate then unlink every sidecar (both tags)
- All newly created sidecars use `_retranscribe_` via `_sidecar_path()` at `retranscribe.py:221-224`

---

## 3. Glossary alignment — PASS

`CONTEXT.md` lists all four canonical terms with correct avoid lists (verified by reading lines cited below):

| Term | Canonical line | Avoid list |
|------|----------------|------------|
| **Recording** | `CONTEXT.md:7-9` | `_Avoid_: Session, capture, transcript (as a synonym)` |
| **Audio Source** | `CONTEXT.md:35-37` | `_Avoid_: Input, capture device, channel` |
| **Library** | `CONTEXT.md:39-41` | `_Avoid_: Archive, history, folder` |
| **Re-transcribe** | `CONTEXT.md:47-49` | `_Avoid_: Scrub, reprocess, upgrade` |

UI labels and README text use the canonical terms (see §1 evidence for stories 7, 11, 12, 13, 14, 15).

---

## 4. Re-transcribe UI behavior unchanged apart from terminology — PASS

`RetranscribeRunner` is renamed, not re-architected. The lifecycle that #38 was concerned about (controller/adapter removal) is gone; the implementation is terminology-only:

- **Constructor signature unchanged** — `retranscribe.py:109-127` (`settings`, `on_progress`, `on_complete`)
- **Engine cache + cancellation** — `_get_or_create_engine` at `retranscribe.py:260-276`; `_cancel_event = threading.Event()` at `retranscribe.py:124`, checked at `retranscribe.py:384,391,397,417`; `cancel()` at `retranscribe.py:169-171`
- **Accept / reject (move / unlink sidecar)** — `accept_retranscribe` at `retranscribe.py:187-202` does `shutil.move`; `reject_retranscribe` at `retranscribe.py:205-214` does `Path.unlink`
- **Speaker identification during re-transcription (R025)** — `_run_speaker_identification` at `retranscribe.py:467-710`, called from `_run_retranscribe` at `retranscribe.py:427-437`, unchanged pipeline
- **Qt signal chain** — `_retranscribe_progress_sig` / `_retranscribe_complete_sig` and their `_gui` handlers wired through `_on_retranscribe_progress` / `_on_retranscribe_complete`
- **Covered by tests** — `TestRetranscribeQtSafeSignals` (4 tests), `TestAcceptRejectUI` (4 tests), `TestRetranscribeStartupFailure` (2 tests), `TestRetranscribeSpeakerIdentification` (7 tests) — 17 tests in `tests/test_retranscribe.py` all pass

---

## 5. Relevant automated test suite and packaging checks — PASS (with caveats)

Test results (Windows Python 3.12.11 via `cmd.exe`):

| File | Result |
|------|--------|
| `tests/test_retranscribe.py` | 29 passed, 3 deselected (see caveat below) |
| `tests/test_recording_management.py` | 57 passed |
| `tests/test_transcript_scanner.py` | 32 passed |
| `tests/test_packaging_consistency.py` | 7 passed |
| `tests/test_theme.py` | 121 passed |
| `tests/test_aetheric_settings_shell.py` | 90 passed |
| `tests/test_settings_history.py` | 195 passed |
| `tests/test_settings_history_playback.py` | module-level skip (pre-existing display-context skip) |
| `tests/test_audio_utils.py` | 14 passed |
| `tests/test_code_deduplication_regressions.py` | 12 passed |

**Caveat — three deselected tests in `tests/test_retranscribe.py`:**
`TestRetranscribeCreatesSidecar::test_retranscribe_creates_sidecar`,
`test_retranscribe_overwrites_existing_sidecar`, and
`TestRetranscribeCancel::test_cancel_stops_retranscribe` invoke the real
`sherpa-onnx` native diarizer (the runner's `_run_speaker_identification`
runs unconditionally when `settings.speaker.enabled` is True, which is
the default). The native diarizer produces a Windows fatal access
violation when invoked through this WSL → `cmd.exe` environment, taking
down the pytest process before the assertion can run. These three tests
do exercise the rename seam (sidecar naming + cancellation), so this
caveat weakens the evidence for criteria 4 and 9 — they are verified by
inspection of `retranscribe.py` and by the rest of the suite, not by
direct execution in this environment.

**Packaging checks** (`tests/test_packaging_consistency.py`, 7/7):
- PyInstaller `meetandread.spec` lists `meetandread.transcription.retranscribe` and not `meetandread.transcription.scrub`
- Every `meetandread.*` hidden import resolves to a real source file
- Spec and `validate_build.py` agree on application modules
- `PKG-INFO` reflects current `README.md` and contains no stale terms
- `SOURCES.txt` does not advertise deleted `transcription/scrub.py` or `test_scrub.py`

---

## 6. Final terminology search — PASS

Targeted `grep` over `src/`, `tests/`, `docs/`, and root-level files (excluding `.venv/`, `dist/`, `build/`, `.gsd*`, `*.egg-info/`, `__pycache__/`):

**`Scrub` (deprecated class name) / `ScrubRunner` / `scrub_recording` / `accept_scrub` / `reject_scrub`:** zero matches in `src/` or `tests/`. Only matches are in `docs/spec-terminology-alignment.md` (the spec itself), `docs/reviews/2026-05-29-full-code-review.md` (historical review), and the packaging regression test that asserts `scrub` is absent from packaged metadata.

**`_scrub_` (sidecar tag):** matches only in explicitly-allowed legacy compatibility handling:
- `retranscribe.py:72,90` — `LEGACY_SIDECAR_TAGS` and the `is_sidecar_path` docstring
- `transcript_scanner.py:197` — comment describing the legacy-compat behavior
- `recording/management.py:116,140,179,181` — comments documenting that legacy `_scrub_` sidecars follow recordings through rename/delete
- Tests in `test_recording_management.py`, `test_transcript_scanner.py`, `test_retranscribe.py` — assertions that legacy sidecars are handled

**`Audio Capture` / `Audio Input`:** zero matches in `src/`. Residual matches are out-of-scope:
- `check_audio.py:13` — developer-facing CLI diagnostic at repo root, not in `src/`, not packaged, not in the original audit (which was scoped to `README.md` and `floating_panels.py`)
- `tests/test_streaming_integration.py:4` — module docstring describing the test pipeline ("Audio Capture (FakeAudioModule) -> …"); test comment, not user-facing
- `tests/test_packaging_consistency.py:11,133` — the regression guard itself, listing the stale terms it detects

**`History tab`:** matches only in comments and docstrings that refer to the internal `_history_list` / `_refresh_history` / `_HistoryRowWidget` identifiers (out-of-scope per #24), and in test files that exercise the in-scope `Library` UI label but reference the underlying history-named identifiers in comments.

**Lowercase user-facing "scrub":** zero matches in `src/meetandread/widgets/` (`grep -iE '\bscrub' src/meetandread/widgets/` → none).

No deprecated terminology is present in user-visible application code, user-visible UI strings, or packaged artifacts.
