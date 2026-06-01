# MeetAndRead — Consolidated Code Review Report

**Repository**: MeetAndRead — Windows PyQt6 Desktop Audio Transcription Widget  
**Source**: ~30K lines across 7 modules  
**Review Date**: May 29, 2026  
**Reviewers**: GPT-5.5 via Hermes (8 review sessions: 1a, 1b, 2, 3, 4, 5, 6, 7)

---

## Executive Summary

MeetAndRead is a functional Windows desktop transcription application that captures mic+system audio, transcribes with Whisper, and identifies speakers. The review identified **20 CRITICAL**, **65 HIGH**, **131 MEDIUM**, and **89 LOW** findings across all 7 modules. The most urgent concerns are **data corruption risks** from non-atomic file writes and unsynchronized concurrent state mutations, which can silently lose or corrupt user recordings and transcripts. Thread safety is the dominant cross-cutting issue — nearly every module has shared mutable state accessed without locks from multiple threads. Additional systemic risks include **silent exception swallowing** that masks real failures, **PII/sensitive data exposure** in logs and UI rendering, **blocking operations on the GUI thread** causing UI freezes, and pervasive **god-class architecture** making the codebase difficult to test and maintain. Despite these issues, the codebase demonstrates good intent in several areas: Qt signal bridges for thread safety, atomic config writes with temp-file replacement, lazy model loading, denoising provider abstraction, crash recovery patterns, and structured state enums.

---

## Severity Breakdown

| Severity | Total | [Widgets] | [Recording] | [Transcription] | [Audio] | [Speaker] | [Infra] | [Core] |
|----------|-----:|----------:|------------:|----------------:|--------:|----------:|--------:|-------:|
| CRITICAL |   20 |         3 |           4 |               3 |       2 |         4 |       3 |      1 |
| HIGH     |   65 |        15 |          10 |               8 |       8 |         9 |      10 |      7 |
| MEDIUM   |  131 |        25 |          13 |              20 |      13 |        24 |      20 |     16 |
| LOW      |   89 |        18 |          10 |              12 |      12 |        16 |       9 |     12 |
| **TOTAL**| **305**| **61** | **37** | **43** | **35** | **53** | **42** | **36** |

> **Note**: Two findings (1b-C2, 7-C2) are flagged as possible paste/concatenation artifacts and may not represent real bugs. They are included for completeness with annotations.

---

## Cross-Cutting Themes

The following systemic patterns appear across multiple modules and represent the highest-impact areas for remediation:

| Theme | Affected Modules | Finding Count |
|-------|-----------------|--------------|
| **Thread safety — unsynchronized shared state** | Recording, Audio, Speaker, Infra, Core | 12 |
| **Non-atomic file writes corrupting user data** | Widgets, Recording, Transcription, Speaker | 8 |
| **Blocking operations on GUI thread** | Widgets, Recording, Speaker, Core | 7 |
| **Silent exception swallowing** | Widgets, Recording, Transcription, Speaker, Infra | 9 |
| **PII / sensitive data in logs & UI** | Widgets, Speaker, Infra, Core | 6 |
| **God class / god module** | Widgets, Recording, Audio, Core | 5 |
| **Code duplication / maintenance drift** | Recording, Transcription, Speaker | 8 |
| **Metadata footer parsing inconsistencies** | Widgets, Recording, Speaker, Infra | 5 |
| **Path trust & boundary validation gaps** | Widgets, Recording, Transcription, Audio, Infra | 7 |
| **Cancellation does not interrupt work** | Transcription, Infra | 3 |

---

## CRITICAL Findings (20)

### CC-C1: Non-Atomic Transcript/Identity File Writes
> **Root cause**: `Path.write_text()` used without atomic temp-file-replace pattern. Crash, disk-full, or AV lock during write corrupts user transcripts.

- **1a-C1** [Widgets] `floating_panels.py` — `_link_speaker_identity_in_file`, `_try_link_identity_in_file`, `_rename_speaker_in_file` read entire `.md`, mutate JSON/body, and call `write_text()` directly. If interrupted, the canonical transcript can be truncated.
- **5-C2** [Speaker] `identity_management.py` — `rename_identity()` saves new profile, deletes old, then rewrites transcript files. Partial rewrite failure leaves store and transcripts permanently inconsistent. `_rewrite_transcripts_safely()` only raises if ALL rewrites fail.
- **3-C2** [Transcription] `post_processing_queue.py` — `_process_job()` accepts empty `[]` from engine failures. `_save_post_processed_transcript()` overwrites `{audio_file.stem}.md` in-place with empty/degraded output, destroying the realtime transcript.

### CC-C2: Unsafe Concurrent State Mutation
> **Root cause**: Shared mutable objects accessed from multiple threads without locks or synchronization.

- **2-C1** [Recording] `controller.py` — `RecordingController` mutates `_state`, `_session`, `_transcription_processor`, `_transcript_store`, `_last_wav_path`, `_last_transcript_path`, `_live_audio_buffer`, and `_post_process_job_id` from audio, stop, finalizer, and UI threads without a controller-level lock.
- **5-C4** [Speaker] `voice_signature_store.py` — SQLite connection opened with `check_same_thread=False` but no lock guards `save_signature()`, `delete_signature()`, `update_signature()`, `load_signatures()`, or `close()`. Concurrent writes can corrupt database state.
- **6-C2** [Infra] `benchmark.py` — `_is_running`, `_last_result`, `_history`, and `_thread` are mutated from both caller and background threads without a lock. Two rapid calls can start multiple benchmark threads.

### CC-C3: Background Worker Race Conditions
> **Root cause**: Background finalization/recovery uses shared controller/process state, allowing new operations to interfere with in-progress work.

- **2-C3** [Recording] `controller.py` — `_finalize()` writes `self._last_wav_path = wav_path`, but `_stop_worker()` sets controller to `IDLE` immediately. A new `start()` can begin and overwrite `_last_wav_path` before the first `_save_transcript()` runs, saving old transcript under wrong stem.
- **7-C1** [Core] `main.py` — Recovery thread started with `join(timeout=30.0)`. If recovery exceeds timeout, the app continues while the recovery thread may still be reading/writing/deleting `.part` files and mutating `recovered_files`.

### CC-C4: Qt Callbacks from Non-GUI Threads
> **Root cause**: UI-facing callbacks invoked directly from worker/audio threads, violating Qt's single-thread rule.

- **2-H1** [Recording] `controller.py` — `_set_state()`, `_set_error()`, `_on_phrase_result()`, `_on_post_process_complete_callback()`, `_on_session_frames_dropped()`, and `_finalize()` invoke callbacks from whichever thread called them.
- **6-C3** [Infra] `benchmark.py` — `on_progress` and `on_complete` are plain Python callbacks executed from the benchmark worker thread.

### Module-Specific CRITICALs

- **1a-C2** [Widgets] `floating_panels.py` — User-controlled identity names are injected into markdown/HTML without escaping (`**{identity_name}**`, `<p><a href="speaker:{speaker_label}">`). Crafted names can break rendering, create malformed links, or inject rich text into `QTextBrowser`.
- **1b-C1** [Widgets] `meetandread_widget.py` — `_exit_application()` and `closeEvent()` save geometry and call `QApplication.quit()`, but neither stops active recording, flushes pending audio, cancels post-processing, joins worker threads, nor releases audio devices. Data loss and leaked handles on exit.
- **1b-C2** [Widgets] *(possibly artifact)* — `from __future__ import annotations` appears after executable code. If this reflects actual file structure, it's a syntax error. May be a paste artifact.
- **2-C2** [Recording] `controller.py` — `_stop_worker()` sets `old_processor._is_running = False`, `old_processor._stop_event.set()`, `old_session._stop_event.set()`, and joins `old_processor._processing_thread` — violating encapsulation and coupling to private fields.
- **2-C4** [Recording] `cleanup_queue.py` — `process_pending()` docstring says "Completed operations are removed," but implementation keeps them: `self._operations = completed + still_pending`. Queue bloats; UI retry dialogs mislead.
- **3-C1** [Transcription] `engine.py` — `transcribe_chunk()` catches every `Exception`, logs "Transcription error," and returns `[]`. Callers cannot distinguish "no speech" from "model crashed," "temp file write failed," etc. Silent transcript data loss.
- **3-C3** [Transcription] `post_processing_queue.py`, `scrub_runner.py` — `audio_file`, `output_dir`, and `transcript_path` are trusted without boundary validation. Malformed persisted jobs or caller bugs can write arbitrary `.md` files. No containment check against approved directories.
- **4-C1** [Audio] `session.py` — Background `_consumer_loop()` has no top-level `try/except`. Any exception from source reads, denoising, mixing, or disk writes terminates the daemon thread silently. `stop()` still finalizes the WAV, reporting false success on a partial recording.
- **4-C2** [Audio] `pyaudiowpatch_source.py` — `PyAudioWPatchSource` documents `terminate()` in `close()`, but `AudioSession.stop()` only calls `wrapper.stop()`. The PyAudio instance is never terminated in normal shutdown — leaked handles on Windows.
- **5-C1** [Speaker] `model_downloader.py` — `ensure_segmentation_model()` tar member validation only checks `startswith("/")` and `".." in member.name` — insufficient for safe extraction. Remote content downloaded over the network and extracted with `tar.extractall()`. Path traversal and symlink attacks possible.
- **5-C3** [Speaker] `identity_management.py` — `merge_identities()` divides by `total_samples = source.num_samples + target.num_samples` without validating both are positive. Zero sum raises; negative/malformed counts produce invalid weighted embeddings.
- **6-C1** [Infra] `benchmark.py` — `BenchmarkRunner.cancel()` only sets `_is_running = False`, but `_run_benchmark()` never checks it inside the chunk loop. Long-running Whisper transcription continues despite reported cancellation.
- **7-C2** [Core] *(possibly artifact)* — Production source appears to contain multiple unrelated modules concatenated together (main.py, __main__.py, __init__.py, diagnostic scripts, test scripts, PyInstaller hook). If real, import-time side effects and duplicate `main()` calls leak into production.

---

## HIGH Findings (65)

### Theme: Thread Safety & Concurrency (12 findings)

- **1a-H5** [Widgets] `floating_panels.py` — Scrub completion updates the wrong recording if user changes history selection mid-run. `_start_scrub()` starts work for a specific `md_path`, but completion uses mutable panel state.
- **2-H4** [Recording] `controller.py` — `_live_audio_buffer` is mutated by `feed_audio_for_transcription()`, read by `_try_live_speaker_match()` and `get_live_audio_samples()`, all from different threads without a lock. `bytearray` concurrent mutation produces inconsistent snapshots.
- **3-H1** [Transcription] `post_processing_queue.py` — `PostProcessJob` fields (`status`, `progress`, `result`, `error`, `cancel_requested`) mutated from worker and cancellation/UI threads without a consistent lock. `get_job_status()` returns live mutable object.
- **3-H2** [Transcription] `streaming_pipeline.py` — `_is_running`, `_total_samples_processed`, `_engine`, and `_processing_thread` read/written across control and worker threads without locking. `stop()` can set `_processing_thread = None` while worker is in a blocking call.
- **3-H4** [Transcription] `scrub_runner.py` — `ScrubRunner` allows concurrent scrubs but has only one cancellation event and one thread handle. `scrub_recording()` overwrites `self._thread` and clears shared `_cancel_event`.
- **3-H6** [Transcription] `vad.py` — `VoiceActivityDetector._buffer`, `_stats`, `_webrtc_available`, and `_vad` mutated without locking. Concurrent `process_chunk()`/`reset()` calls can lose/duplicate frames or corrupt stats.
- **4-H3** [Audio] `session.py` — `stop()` calls `self._consumer_thread.join(timeout=5.0)` then unconditionally closes writer and finalizes WAV. If consumer thread is still alive, closing the writer while it writes produces data races and truncated WAV.
- **4-H4** [Audio] `session.py` — `_stats.frames_recorded`, `_state`, `_error`, `_denoising_disabled` read/written across main, consumer, and audio callback threads. Only `frames_dropped` increments are locked. `get_stats()` returns live mutable object.
- **4-H6** [Audio] `session.py` — `on_audio_frame` callback invoked without exception handling. A transcription buffer bug or Whisper failure terminates the consumer thread; session may still finalize and report success.
- **5-H7** [Speaker] `identity_management.py` — `merge_identities()` saves merged embedding to target before transcript rewrites. If all rewrites fail, target is already overwritten; source remains, producing duplicated/incorrect embeddings.
- **5-M15** *(promoted)* [Speaker] `voice_signature_store.py` — `update_signature()` reads embedding, computes average, writes result. Two concurrent updates can both read old count; one update overwrites the other (lost-update race).
- **7-H2** [Core] `main.py` — `recovered_files` and `recovery_error` assigned from worker thread and read on main thread without synchronization.

### Theme: Blocking Operations on GUI Thread (7 findings)

- **1a-H4** [Widgets] `floating_panels.py` — `_refresh_history()`, `_open_identity_link_dialog()`, `_render_history_transcript()`, `_delete_recording()` execute file/directory scans synchronously in UI event handlers. Large recordings folders freeze PyQt UI.
- **1b-H2** [Widgets] `meetandread_widget.py` — `_update_animations()` calls `get_live_audio_samples()` every 33ms timer tick via `if True:`, contradicting comments about throttled polling. Unnecessary CPU load and audio buffer contention.
- **1b-H3** [Widgets] `meetandread_widget.py` — `RecordButtonItem.set_waveform_samples()` performs NumPy conversions, flattening, NaN replacement, and list comprehensions every animation frame while recording.
- **2-H2** [Recording] `controller.py` — `_init_transcription()` calls `self._transcription_processor.load_model(...)` synchronously (1-2+ seconds) on the caller/UI thread before audio session starts. Freezes Windows desktop UI.
- **5-H1** [Speaker] `diarizer.py` — `subprocess.run(..., timeout=300)` is synchronous. Blocks calling thread for up to 5 minutes. If invoked from UI-adjacent worker incorrectly, still blocks.
- **7-H1** [Core] `main.py` — Recovery thread started "for UI responsiveness" but immediately `join(timeout=30.0)` is called from GUI thread before event loop. App frozen for up to 30 seconds.
- **7-H4** [Core] `main.py` — `ModelRecommender().detect_and_recommend()` and hardware detection run synchronously during startup before main widget is shown. Can hang on drivers/WMI.

### Theme: Resource Leaks & Lifecycle (6 findings)

- **1a-H6** [Widgets] `floating_panels.py` — `VoiceSignatureStore` created in `_open_identity_link_dialog()` without context manager and never closed. Leaked SQLite handles on Windows can interfere with rename/delete operations.
- **2-H5** [Recording] `controller.py` — Failed transcription initialization leaves partially initialized objects alive. `_init_transcription()` creates `TranscriptStore`, starts recording, creates processor, then may partially fail without tearing down earlier steps.
- **2-H6** [Recording] `controller.py` — Each `_init_transcription()` can create new `PostProcessingQueue.start()` without stopping previous. Multiple queue workers, duplicated callbacks, retained old controller references.
- **4-H2** [Audio] `pcm_part.py` — `PcmPartWriter.create()` uses second-level timestamp precision for stems. Two sessions started in the same second generate same stem; metadata overwrite race condition.
- **4-H7** [Audio] `fake_module.py` — Fake source `_read_loop()` can hang on full queue during stop. `stop()` clears `self._thread = None` while old daemon thread may still be running.
- **4-H8** [Audio] `sounddevice_source.py` — `SystemSource.__init__()` always calls `get_default_wasapi_loopback()` and overwrites `self._device_index`. Caller-specified `device_id` silently ignored.

### Theme: Path Trust & Validation (6 findings)

- **1a-H7** [Widgets] `floating_panels.py` — History item paths stored via `QListWidgetItem.UserRole` and trusted for read/delete/rename without canonicalization or containment checks. Symlinks or malformed names could read/delete outside intended directory.
- **2-H9** [Recording] `cleanup_queue.py` — `_process_identity_cleanup()` loops through `op.paths`, constructs `Path(path_str)`, and unlinks. No validation that path is under app-owned directory, no symlink handling, no allowlist. Arbitrary file deletion if queue tampered.
- **2-H10** [Recording] `management.py` — `_validate_stem()` exists but `enumerate_recording_files()`, `delete_recording()`, and `delete_recording_structured()` do not call it. Malformed stem with path separators can influence glob patterns.
- **4-H1** [Audio] `paths.py` — `--output-dir` and configured storage paths accepted, resolved, created, and written without containment checks. Absolute paths accepted by design; path traversal possible.
- **6-H1** [Infra] `config/manager.py` — `validate_storage_paths()` accepts arbitrary paths, creates directories, writes sentinel file without policy guard for sensitive locations, symlinks, or system directories.
- **6-H2** [Infra] `playback.py` — `load_transcript_audio()` loads `<recordings>/<stem>.wav` from any transcript `Path` without verifying the transcript belongs to the expected directory or that audio/transcript pair are related.

### Theme: Exception Swallowing & Silent Failures (5 findings)

- **1b-H6** [Widgets] `meetandread_widget.py` — Multiple blocks catch `Exception` and `pass` or only log debug: `_create_floating_panels()`, `_validate_startup_storage_paths()`, `_on_controller_state_change()`. Real config corruption silently skipped.
- **2-H7** [Recording] `controller.py` — Multiple broad `except Exception: pass` blocks in live speaker matching and diagnostics. Masks broken speaker matching, failing session stats, broken transcript store reads.
- **2-H8** [Recording] `controller.py` — `assert store is not None` in `_apply_speaker_labels()`. Assertions stripped with `python -O`; next line fails with unclear `AttributeError`.
- **5-H8** [Speaker] `diarizer.py` — Subprocess `_extract_embeddings()` catches all exceptions per speaker and does `pass`. Missing signatures produce successful diarization with no diagnostics.
- **6-M10** *(promoted)* [Infra] `benchmark.py` — `_run_benchmark()` catches all exceptions, logs only error message string without traceback. Limits diagnostics for model/transcription failures.

### Theme: PII & Sensitive Data Exposure (5 findings)

- **1a-M10** *(promoted)* [Widgets] `floating_panels.py` — Logs include `md_path`, `old_name`, `new_name`, and raw speaker labels despite "PII-safe" comments. Identity names can be personal data.
- **1b-H1** [Widgets] `meetandread_widget.py` — `_on_phrase_result()` logs `result.text[:40]` during live transcription. Recordings may contain PHI/PII, credentials, or customer data. Debug logs retained, uploaded with bug reports.
- **5-H6** [Speaker] `voice_signature_store.py` — Logs raw speaker names in multiple debug messages: "Saved signature for '%s'", "Match: '%s'", "Deleted signature for '%s'" despite PII policy elsewhere in the module.
- **6-M4** *(promoted)* [Infra] `config/persistence.py`, `playback.py` — Several log statements include full config paths, transcript/audio paths, or expected WAV paths. Exposes user names or sensitive directory structure.
- **7-M3** *(promoted)* [Core] `main.py` — `TeeOutput.write()` logs all non-empty stdout at DEBUG. Third-party libraries or app code may print file paths, device names, transcript snippets.

### Theme: Rendering & UI Correctness (6 findings)

- **1a-H1** [Widgets] `floating_panels.py` — Live transcript uses `QTextEdit` for clickable anchors, but `QTextEdit` does not expose `anchorClicked`. Speaker links are likely not clickable. Should use `QTextBrowser`.
- **1a-H2** [Widgets] `floating_panels.py` — `_replace_segment_in_display()` navigates by blocks and `NextWord`, assuming single-word segments. Multi-word segments, punctuation, timestamps, or speaker labels cause wrong replacement or corruption.
- **1b-H4** [Widgets] `meetandread_widget.py` — `FullViewportUpdate` forces complete scene repaints at 30fps. Combined with animation, glow effects, waveform rendering, and per-segment drawing, causes unnecessary CPU/GPU use.
- **1b-H5** [Widgets] `theme.py` — `current_palette()` imports `QtColorScheme` from `PyQt6.QtGui` which may not exist depending on version. `ImportError` caught silently and falls back to dark. Light theme detection broken.
- **1b-H8** [Widgets] `meetandread_widget.py` — `ControllerState.STARTING` locks lobes but doesn't update `is_recording`/visual state. If start fails before `RECORDING`, UI lobes remain locked. Dependency on controller event ordering not enforced.
- **1b-H7** [Widgets] `meetandread_widget.py` — `_validate_startup_storage_paths()` calls `cm._dirty_paths.add("storage_paths")`, coupling UI code to private config manager internals.

### Theme: Transcription & Audio Pipeline (6 findings)

- **2-H3** [Recording] `controller.py` — `_on_phrase_result()` calls `_try_live_speaker_match()`, which may lazily initialize sherpa-onnx, copy 12 seconds of PCM, create extractor stream, compute embedding, query SQLite — blocking transcription result path.
- **3-H3** [Transcription] `post_processing_queue.py`, `scrub_runner.py` — Cancellation checked only between coarse steps. `engine.load_model()`, `_load_audio_file()`, diarization, and `engine.transcribe_chunk()` cannot be interrupted. Worker joins for only 5 seconds; daemon work may continue.
- **3-H5** [Transcription] `streaming_pipeline.py` — `feed_audio()` calls `self._vad_processor.feed_audio(chunk, vad_is_speech=True)` unconditionally. VADChunkingProcessor never sees speech-to-silence transitions. Silence/noise transcribed as normal audio.
- **3-H7** [Transcription] `engine.py` — `MODEL_URLS['large']` points to `ggml-large-v3.bin` but `_get_model_path()` returns `ggml-large.bin`. Large model downloaded from v3 URL into misleading filename. Duplicate downloads or wrong cache detection.
- **3-H8** [Transcription] `engine.py` — Invalid model sizes accepted into object state and only fail later in `_download_model()`. No validation in constructors, `set_model_config()`, or `schedule_post_process()`.
- **4-H5** [Audio] `session.py` — `on_frames_dropped` callback called from audio backend callback path. Arbitrary user code can block, allocate, or deadlock from the non-UI/non-real-time audio callback thread.

### Theme: Config & Identity Integrity (5 findings)

- **5-H4** [Speaker] `identity_management.py` — `_rewrite_transcripts_safely()` returns successful list, not errors. Some files can fail silently; caller gets no failure indication. Log variable name (`rewrite_errors`) actually contains successes.
- **5-H9** [Speaker] `model_downloader.py` — Remote ONNX/tarball files downloaded without checksum, signature, or expected size validation. Compromised release, MITM, or partial file results in arbitrary model content loaded by ONNX runtime.
- **6-H3** [Infra] `config/manager.py` — `_get_all_paths()` omits many settings (`transcription.realtime_model_size`, `speaker.*`, `ui.cc_panel_geometry`, etc.). `reset_to_defaults()` relies on this list, so reset/save behavior is incomplete.
- **6-H4** [Infra] `config/manager.py` — `ConfigManager` singleton ignores injected persistence after first initialization. Later calls return existing instance with original persistence. Tests order-dependent; wrong config backend if initialization order changes.
- **6-H5** [Infra] `config/models.py` — Two real-time model fields: `model.realtime_model_size` (default `"auto"`) and `transcription.realtime_model_size` (default `"tiny"`). `ModelRecommender` writes `hardware.recommended_model`. Ambiguous source-of-truth.

### Theme: Security & Validation (4 findings)

- **6-H6** [Infra] `config/manager.py` — `isinstance(value, (int, float))` accepts `True`/`False` for fields like `confidence_threshold` and `min_chunk_size_sec` (Python `bool` is subclass of `int`).
- **6-H7** [Infra] `config/manager.py` — Settings validation is absent beyond runtime type check. Invalid values like `confidence_threshold=999`, `min_chunk_size_sec=-5`, or `realtime_model_size="not-a-model"` can be persisted.
- **6-H8** [Infra] `config/persistence.py` — Corrupted config JSON silently discarded. `load_raw()` returns `None` for invalid JSON/empty files; defaults used. Later save can overwrite the only broken-but-recoverable config.
- **6-H9** [Infra] `config/persistence.py` — Migration `4→5` unconditionally sets `microphone_denoising_enabled = False`, destroying user's explicit setting during upgrade.

### Theme: Module-Specific HIGH (3 findings)

- **7-H3** [Core] `main.py` — `sigint_handler()` and Windows console handler call `app.quit()` directly from signal/console callback context. Qt GUI operations should be marshalled into event loop.
- **7-H5** [Core] `main.py` — `main()` runs `ModelRecommender.detect_and_recommend()` then `HardwareDetector().detect()` separately — duplicating slow hardware probing and potentially producing inconsistent results.
- **7-H6** [Core] `check_audio_levels.py` — `sys.path.insert(0, 'src')` makes imports depend on CWD. Brittle and potentially unsafe if malicious/stale `src/meetandread` exists.

---

## Top 10 MEDIUM Findings (by impact)

1. **3-M5** [Transcription] `streaming_pipeline.py` — **VAD completely bypassed in real-time pipeline.** `feed_audio()` passes `vad_is_speech=True` unconditionally. All audio (including silence/noise) is transcribed. Degrades accuracy, wastes CPU, and contradicts module design intent.

2. **2-M4** [Recording] `controller.py` — **Fallback speaker labeling uses wrong transcript store.** `_fallback_single_speaker_labeling()` reads `self._transcript_store` instead of the post-processing store. If a new recording started, labels apply to the wrong transcript.

3. **2-M5** [Recording] `controller.py` — **WER computed against wrong realtime transcript.** `_compute_and_store_wer()` reads realtime text from `self._transcript_store` rather than the store associated with the completed post-processing job.

4. **4-M1** [Audio] `session.py` — **Frame accounting incorrect for multi-channel recordings.** `AudioSession` counts sample rows; `PcmPartWriter` counts individual samples. For stereo, writer double-counts frames and `max_frames` slicing is wrong.

5. **1a-M5** [Widgets] + **5-M19** [Speaker] — **Markdown replacement renames non-heading bold text.** `re.sub(re.escape(f"**{old_name}**"), ...)` across entire body can replace bold text in transcript content that is not a speaker heading. Affects both `floating_panels.py` and `identity_management.py`.

6. **1b-M12** [Widgets] `meetandread_widget.py` — **Recording state duplicated across controller and UI booleans.** `is_recording`/`is_processing` maintained manually alongside `ControllerState`. Can drift if events arrive out of order or errors occur.

7. **6-M9** [Infra] `config/persistence.py` — **Config migration v4→v5 forcibly resets user denoising setting.** Unconditionally sets `microphone_denoising_enabled = False` during upgrade, destroying user's explicit choice.

8. **5-M13** [Speaker] `voice_signature_store.py` — **cosine_similarity doesn't guard shape mismatch or NaN.** One malformed stored embedding can break all matching. `np.dot(a, b)` raises on shape mismatch; NaN/Inf propagates into scores.

9. **1a-H3** [Widgets] `floating_panels.py` — **Metadata footer parsing duplicated and inconsistent.** Six+ functions parse the transcript footer with slightly different markers: `"\\n---\\n\\n<!-- METADATA:"` vs `"\\n---\\n\\n<!-- METADATA: "`. `.rstrip(" -->\\n")` treats argument as character set, not suffix. Same issue in **2-M7** [Recording] and **5-M18** [Speaker].

10. **3-M13** [Transcription] + **3-M14** [Transcription] — **Duplicate transcript construction and audio loading logic.** `post_processing_queue.py` and `scrub_runner.py` contain near-identical `_create_post_processed_transcript()`/`_create_transcript_from_segments()` and `_load_audio_file()`. Fixes must be applied twice. High maintenance drift risk.

### Honorable Mentions (11–15)

11. **1a-M1** [Widgets] + **1b-M1** [Widgets] + **2** [Recording] + **4** [Audio] + **7** [Core] — **God class/god module pattern.** Every major module concentrates too many responsibilities: `FloatingTranscriptPanel` (UI + persistence + identity + scrub + history), `MeetAndReadWidget` (controller + config + audio + animation + error), `RecordingController` (lifecycle + transcription + diarization + persistence), `AudioSession` (sources + denoising + mixing + storage + threading), `main.py` (logging + DLL + hardware + recovery + widget + tray).

12. **2-M3** [Recording] + **5-M4** [Speaker] — **Duplicate diarization implementations.** `_run_diarization_for_postprocess()` and `_run_diarization()` in controller duplicate imports, settings, diarizer creation, cleanup, signature matching. Also `diarizer.diarize()` and `_SUBPROCESS_SCRIPT.main()` duplicate audio reading, resampling, embedding extraction. Divergence already exists.

13. **3-M1** [Transcription] `engine.py` — **Temp WAV write doesn't clip float audio before int16 conversion.** `(audio_np * 32767).astype(np.int16)` wraps values outside [-1.0, 1.0] instead of clipping. Samples of 2.0 become invalid wrapped int16 data.

14. **6-M1** [Infra] `config/manager.py` — **`get()` exposes mutable internal settings object.** Callers can bypass validation and dirty tracking: `get_config().transcription.confidence_threshold = -99`. Deep copy or controlled mutation APIs needed.

15. **3-M3** [Transcription] `engine.py` — **Model download uses `urlretrieve` without timeout, checksum, or atomic destination.** Downloads can hang indefinitely; corrupted/partial model files trusted without verification.

---

## Architecture Assessment

### Module-by-Module Assessment

**[Widgets]** (floating_panels.py, meetandread_widget.py, theme.py) — *Low cohesion, high coupling.*
Both panel files are god classes. `FloatingTranscriptPanel` combines live transcription display, history list, context menus, scrub workflow, speaker identity linking, file mutation, and HTML rendering. `MeetAndReadWidget` is simultaneously a controller coordinator, persistence caller, device checker, storage-path validator, tray integration, animation scheduler, and error presenter. The `_ControllerBridge` and `_WidgetVisualStateMachine` are good starts, but the rest needs decomposition into testable services. Theme module has caching issues (stale on OS theme change) and broken light-mode detection.

**[Recording]** (controller.py, cleanup_queue.py, management.py) — *Low cohesion, high concurrency risk.*
`RecordingController` acts as UI adapter, audio-session lifecycle manager, transcription coordinator, diarization service, speaker identity service, transcript persistence layer, WER calculator, and diagnostics endpoint — all in one class with no synchronization. The controller needs splitting into explicit services: `RecordingLifecycleService`, `TranscriptionService`, `PostProcessingService`, `DiarizationService`. File-management and cleanup-queue modules show better separation but need stricter validation.

**[Transcription]** (engine.py, post_processing_queue.py, scrub_runner.py, streaming_pipeline.py, vad.py) — *Reasonable design, poor boundary enforcement.*
Engine, real-time pipeline, post-processing queue, VAD, chunking, and scrub runner are conceptually separated. But boundaries are violated: audio loading and transcript building duplicated across post-processing and scrubbing; real-time pipeline claims VAD integration but bypasses actual VAD; engine errors silently converted to empty results; background workers mix orchestration, persistence, model lifecycle, and file overwrite policy in large methods. Highest-value refactor: extract shared `AudioLoader`, `TranscriptBuilder`, and `TranscriptWritePolicy`.

**[Audio]** (session.py, storage/, capture/, denoising.py) — *Reasonable layering, god-session problem.*
Device enumeration, capture sources, session orchestration, denoising abstraction, and storage primitives are split into separate modules. Main weakness: `AudioSession` owns source construction, denoising lifecycle, resampling, mixing, transcription callback dispatch, persistence, stats, and thread shutdown. That concentration makes error handling and concurrency fragile. Cleaner design: separate source lifecycle, audio processing/mixing, storage writing, and event dispatch behind explicit interfaces.

**[Speaker]** (diarizer.py, identity_management.py, model_downloader.py, voice_signature_store.py, models.py) — *Good module boundaries, leaky implementation seams.*
Data models, diarization runtime, storage, identity management, and model downloading are properly separated. However: subprocess runner duplicates business logic as embedded string; identity operations mutate database and transcript files without transaction boundaries; persistence/security policy inconsistent; `_cosine_similarity()` duplicated; subprocess diarization can block for 5 minutes. Biggest improvements: tested diarization worker module, transaction-like identity operation coordinator with rollback, shared PII-safe logging utility.

**[Infra]** (config/, hardware/, performance/, history/) — *Good modularity, validation gaps.*
Config, persistence, performance, hardware, and playback are logically separated. Weaknesses: validation/persistence/runtime behavior not cleanly separated; dataclasses accept invalid values; `ConfigManager.set()` only shallow type checks; global singleton creates hidden coupling; `BenchmarkRunner` uses raw threads instead of Qt integration. Schema drift and duplicate model-selection concepts need deliberate cleanup.

**[Core]** (main.py, __main__.py, diagnostic scripts, hooks) — *God-level startup coordinator.*
`main.py` concentrates logging setup, DLL validation, hardware detection, recovery, cleanup queue processing, widget creation, tray wiring, and signal handling. Startup should be an explicit pipeline with `ApplicationController` owning services. UI layer coupled to private widget methods. Diagnostic scripts are useful but should be separated from production package code. The source excerpt suggests possible file boundary issues requiring verification.

### Systemic Architecture Issues

1. **No consistent thread-safety model.** Each module independently decides whether to use locks, signals, or nothing. Some use `check_same_thread=False` and hope for the best. A shared concurrency policy and primitives package is needed.

2. **No consistent file-write policy.** Some modules use atomic temp-file-replace (config persistence), others use direct `write_text()` (transcripts, identity files). A shared `atomic_write()` utility should be mandatory for all user-data files.

3. **No consistent error-reporting model.** Some modules propagate exceptions, some return error strings, some silently swallow failures, some use structured error objects. Callers cannot make reliable decisions.

4. **No service layer.** Business logic (transcript parsing, speaker identity, diarization, post-processing) is embedded in UI classes and controller classes. Pure services with explicit inputs/outputs would enable testing without Qt widgets and filesystem state.

5. **Duplicated code across modules.** Audio loading, transcript construction, metadata parsing, diarization workflows, cosine similarity, and VAD-related code appear in 2+ locations. Divergence already exists in several cases.

---

## Recommended Remediation Priorities

### Phase 1: Fix CRITICAL Data Integrity & Safety (1–2 weeks)
> **Goal**: Prevent silent data loss and corruption.

| Priority | Finding IDs | Action |
|----------|-------------|--------|
| P1.1 | CC-C1 (1a-C1, 5-C2, 3-C2) | Create shared `atomic_write(path, content)` utility using temp-file + fsync + `os.replace()`. Apply to ALL transcript and identity file writes. |
| P1.2 | CC-C2 (2-C1, 5-C4, 6-C2) | Add `threading.RLock` to `RecordingController` and `VoiceSignatureStore`. Protect all shared mutable fields under lock. |
| P1.3 | 4-C1 | Wrap `_consumer_loop()` in top-level exception guard. Store exception, transition to ERROR state, prevent false-success finalization. |
| P1.4 | 3-C1 | Change `transcribe_chunk()` to return typed result (`ok/error/no_speech`). Post-processing must fail job on transcription error. |
| P1.5 | 4-C2 | Define formal source lifecycle interface (`start`/`stop`/`close`). `AudioSession.stop()` must call `close()` after `stop()`. |
| P1.6 | 2-C3, 7-C1 | Make finalization local: pass `wav_path` explicitly. Recovery: use `QThread` with signals, not raw thread + join. |
| P1.7 | 1a-C2, 1b-H1 | Strip/validate identity names on input. Use `html.escape()` for HTML output. Never log transcript text. |
| P1.8 | 5-C1 | Replace `tar.extractall()` with per-member extraction validating resolved paths against cache directory. |
| P1.9 | 5-C3 | Validate `num_samples > 0` before merge division. Raise `MergeError` on invalid data. |
| P1.10 | 1b-C1 | Add `controller.shutdown(timeout=...)` that stops recording, flushes audio, joins workers, releases devices before quit. |
| P1.11 | 6-C1 | Introduce `threading.Event` for benchmark cancellation. Check before each chunk. |
| P1.12 | 3-C3, 2-H9 | Add path containment validation: resolve paths, verify under approved directories, reject symlinks. |

### Phase 2: Fix HIGH Thread Safety & Performance (2–4 weeks)
> **Goal**: Eliminate crashes from cross-thread access and GUI freezes.

| Priority | Finding IDs | Action |
|----------|-------------|--------|
| P2.1 | CC-C4 (2-H1, 6-C3) | Route all UI callbacks through Qt signals or `QMetaObject.invokeMethod(QueuedConnection)`. Make `BenchmarkRunner` a `QObject` with `pyqtSignal`s. |
| P2.2 | 2-H4, 3-H1, 3-H2, 3-H4, 4-H4, 5-M15 | Add per-object or per-module locks. Return immutable snapshots from query APIs. Per-job cancellation tokens for scrub runner. |
| P2.3 | 1a-H4, 7-H1, 2-H2, 7-H4 | Move file scans to worker threads. Restore waveform throttling. Defer hardware detection until after event loop. Use async recovery with `QProgressDialog`. |
| P2.4 | 2-H2 | Pre-load Whisper models during app startup or perform in worker with progress signals. |
| P2.5 | 1a-H6, 2-H5, 2-H6, 4-H2 | Fix resource leaks: close VoiceSignatureStore with context manager, stop old PostProcessingQueue, use UUID stems. |
| P2.6 | 4-H3 | After `join(timeout)`, check `is_alive()`. If alive, do NOT finalize as successful. |
| P2.7 | 5-H1 | Run subprocess diarization from `QThread`/`QProcess` with completion signals. |
| P2.8 | 1a-H1, 1a-H2 | Use `QTextBrowser` for live transcript. Replace word-position surgery with segment model rebuild. |
| P2.9 | 1b-H6, 2-H7, 2-H8 | Catch specific exception types. Replace `assert` with explicit guards. Log `exc_info=True`. |
| P2.10 | 6-H3, 6-H4, 6-H5 | Fix config schema: derive paths from dataclass fields. Replace singleton with keyed instances. Unify model selection setting. |
| P2.11 | 6-H8, 6-H9 | Backup corrupted config. Only set migration defaults when field is missing. |

### Phase 3: Fix MEDIUM Correctness & Reliability (3–5 weeks)
> **Goal**: Fix correctness bugs and reduce maintenance burden.

| Priority | Finding IDs | Action |
|----------|-------------|--------|
| P3.1 | 3-M5 | Integrate actual VAD into `feed_audio()`. Pass real `is_speech` result. |
| P3.2 | 2-M4, 2-M5 | Pass explicit transcript store to fallback labeling and WER computation. Never use controller-global store during background processing. |
| P3.3 | 4-M1 | Define "frame" as one sample per channel. Fix `PcmPartWriter` and session accounting. |
| P3.4 | 1a-M5, 5-M19 | Restrict markdown replacement to heading lines: `(?m)^\*\*{escaped}\*\*\s*$`. |
| P3.5 | 1b-M12 | Derive UI booleans from single controller state property or use state reducer. |
| P3.6 | 1a-H3, 2-M7, 5-M18, 6-M20 | Create shared `parse_metadata_footer()` / `write_metadata_footer()` utility. Use `rfind()`. |
| P3.7 | 3-M13, 3-M14, 2-M3, 5-M4 | Extract shared `AudioLoader`, `TranscriptBuilder`, `DiarizationWorkflow`. |
| P3.8 | 3-M1, 3-M3, 5-H9 | Clip float audio before int16 conversion. Add download timeouts, checksums, atomic destination. |
| P3.9 | 5-M13, 5-M16 | Validate embedding shapes/finite values. Add `CHECK(num_samples > 0)` to SQLite schema. |
| P3.10 | 6-M1, 6-M7 | Return deep copies from `get()`. Validate `StoragePaths.from_dict()` value types. |

### Phase 4: Architecture Refactoring (4–8 weeks)
> **Goal**: Decompose god classes into testable services.

| Priority | Module | Action |
|----------|--------|--------|
| P4.1 | [Recording] | Split `RecordingController` into `RecordingLifecycleService`, `TranscriptionService`, `PostProcessingService`, `DiarizationService`. Thin Qt-safe adapter on top. |
| P4.2 | [Widgets] | Split `FloatingTranscriptPanel` into `TranscriptMetadataRepository`, `SpeakerIdentityService`, `HistoryPanel`, `LiveTranscriptView`, `ScrubController`, `TranscriptHtmlRenderer`. |
| P4.3 | [Audio] | Split `AudioSession` into `SourceLifecycleManager`, `AudioProcessingPipeline`, `StorageWriter`, `EventDispatcher`. |
| P4.4 | [Core] | Create `ApplicationController` owning `LoggingService`, `HardwareService`, `RecoveryService`, `CleanupService`, `TrayService`. Remove private method coupling. |
| P4.5 | [Speaker] | Move subprocess runner to `meetandread.speaker.diarizer_worker` module. Create identity operation coordinator with rollback support. |
| P4.6 | All | Create shared concurrency primitives package. Establish consistent file-write policy (`atomic_write()` mandatory). Standardize error-reporting model. |

### Phase 5: LOW / Polish (ongoing)
> **Goal**: Clean up code quality, remove dead code, improve maintainability.

- Remove unused imports, dead methods (`_on_panel_segment`, `_ts()`, `_defaults`), and stub code
- Standardize logging: use parameterized logging, PII-safe helpers, structured levels
- Replace `print()` with `logger` in engine, streaming pipeline, diagnostics
- Centralize magic numbers into named constants
- Fix minor bugs: `TeeOutput` flush behavior, `BudgetProgressBar` clamping, widget positioning on multi-monitor
- Improve type annotations: replace `object` with Protocols, use `Optional[DiarizationResult]`
- Clean up diagnostic scripts: consistent exit codes, proper CLI entry points, remove `sys.path` mutation

---

## Appendix: All MEDIUM Findings by Module

### [Widgets] MEDIUM (25 findings)
- **1a-M1** — God module violates separation of concerns (floating_panels.py)
- **1a-M2** — Silent exception swallowing hides real failures
- **1a-M3** — Duplicate `_norm_label` helper in multiple scopes
- **1a-M4** — Speaker matching uses inconsistent raw/display labels
- **1a-M5** — Markdown replacement renames unintended body occurrences
- **1a-M6** — HTML rendering enables malformed markup from italic regex
- **1a-M7** — Extra leading space for first segment in phrase
- **1a-M8** — Drag movement not clamped despite clamp helpers existing
- **1a-M9** — Scrub error recovery doesn't reset all scrub state
- **1a-M10** — Logging exposes file paths and speaker labels despite PII-safe comments
- **1b-M1** — Main widget violates separation of concerns
- **1b-M2** — Controller callback assignment fragile and single-consumer
- **1b-M3** — `_on_phrase_result()` comment misleading about thread safety
- **1b-M4** — QTimer instances repeatedly allocated instead of reused
- **1b-M5** — ErrorIndicatorItem visibility bypasses QGraphicsItem semantics
- **1b-M6** — Hidden ErrorIndicatorItem may still process mouse events
- **1b-M7** — `get_error_help_text()` regex operator precedence probably wrong
- **1b-M8** — Widget positioning ignores multi-monitor virtual desktop origin
- **1b-M9** — Desktop clamping treats multiple monitors as one bounding rectangle
- **1b-M10** — System-audio availability probe optimistic on all exceptions
- **1b-M11** — Storage-path validation shows blocking modal during widget construction
- **1b-M12** — Recording state split across controller and duplicated UI booleans
- **1b-M13** — `ToggleLobeItem.set_locked()` resets cursor for unavailable lobes
- **1b-M14** — Visual state machine documentation doesn't match implementation
- **1b-M15** — Theme cache doesn't respond to runtime OS theme changes

### [Recording] MEDIUM (13 findings)
- **2-M1** — `cancel_post_processing()` documentation contradicts `start()` behavior
- **2-M2** — Type annotations inaccurate for diarization callbacks returning None
- **2-M3** — Duplicate diarization implementations with diverging behavior
- **2-M4** — Fallback labeling uses wrong transcript store during post-processing
- **2-M5** — WER computed against wrong realtime transcript
- **2-M6** — File rewrite for WER metadata is non-atomic
- **2-M7** — JSON metadata parsing brittle (exact marker matching)
- **2-M8** — Progress callback uses `print()` from model loading path
- **2-M9** — `except (ImportError, Exception)` redundant, masks all failures
- **2-M10** — `delete_recording()` loses failure information
- **2-M11** — Cleanup queue atomic write not fully durable on Windows
- **2-M12** — Cleanup queue operations not thread-safe
- **2-M13** — `CleanupOperation.from_dict()` accepts arbitrary malformed types

### [Transcription] MEDIUM (20 findings)
- **3-M1** — Temp WAV write doesn't clip float audio before int16 conversion
- **3-M2** — Temp file cleanup errors silently ignored
- **3-M3** — Model download uses urlretrieve without timeout/checksum
- **3-M4** — Progress callbacks invoked from worker threads without UI dispatch
- **3-M5** — Real-time pipeline bypasses actual VAD (every chunk = speech)
- **3-M6** — `RealTimeTranscriptionProcessor.start()` doesn't require/create engine
- **3-M7** — Debug print() statements leak transcript text to stdout
- **3-M8** — Audio loading reads entire WAV into memory with struct.unpack
- **3-M9** — Resampling uses linear interpolation without anti-aliasing
- **3-M10** — Timestamp logic produces inaccurate starts/ends for first chunk
- **3-M11** — Whisper timestamp units assumed but not verified
- **3-M12** — Word-level data is not actually word-level (whole segments)
- **3-M13** — Duplicate transcript construction logic in post-processing & scrub
- **3-M14** — Duplicate audio loading/resampling logic in post-processing & scrub
- **3-M15** — Queue persistence recovery ambiguous for duplicate/empty job IDs
- **3-M16** — `queue.Queue.task_done()` never called
- **3-M17** — `cancel_job()` doesn't remove pending jobs from queue
- **3-M18** — `VADChunkingProcessor` can grow unbounded if processing blocked
- **3-M19** — VAD sanitization flattens multi-channel audio instead of mixing
- **3-M20** — `VADStats.get_stats()` returns live mutable stats object

### [Audio] MEDIUM (13 findings)
- **4-M1** — Frame accounting incorrect for multi-channel recordings
- **4-M2** — `frames_dropped` increments by callback count, not frame count
- **4-M3** — Resampler channel config wrong before downmix/normalization
- **4-M4** — `_float32_to_int16_bytes()` doesn't sanitize NaN/Inf
- **4-M5** — Denoising stats don't count fallback frames as processed
- **4-M6** — Denoising provider state not thread-safe if reused
- **4-M7** — `AudioSession.start()` cleanup leaves stale fields after failure
- **4-M8** — `SoundDeviceSource.start()` can leak stream if `.start()` fails
- **4-M9** — Device enumeration failures swallowed in CLI
- **4-M10** — Recovery derives metadata path through fragile suffix transformation
- **4-M11** — Recovery backup can overwrite previous backups
- **4-M12** — `PcmPartWriter.flush()` doesn't fsync metadata or audio
- **4-M13** — Fake audio test helper has nondeterministic behavior (no seed)

### [Speaker] MEDIUM (24 findings)
- **5-M1** — Subprocess protocol assumes no extra stdout before length prefix
- **5-M2** — Subprocess output parser ignores trailing data
- **5-M3** — Unbounded JSON payload can consume large memory
- **5-M4** — Duplicate diarization logic between in-process and subprocess
- **5-M5** — Subprocess script imports from its own defining module (circular risk)
- **5-M6** — `cleanup_diarization_segments()` absorption logic doesn't match docstring
- **5-M7** — Cleanup function accepts negative thresholds without validation
- **5-M8** — Diarizer constructor accepts invalid clustering/duration parameters
- **5-M9** — `_read_wav()` loads entire audio file into memory
- **5-M10** — Embedding extraction duplicates audio into large concatenated arrays
- **5-M11** — Dataclass immutability shallow for NumPy arrays
- **5-M12** — `SpeakerMatch.score` and embeddings not validated
- **5-M13** — `_cosine_similarity()` doesn't guard shape mismatch or NaN
- **5-M14** — Duplicate `_cosine_similarity()` implementation across modules
- **5-M15** — `VoiceSignatureStore.update_signature()` has lost-update race
- **5-M16** — Store schema doesn't validate `num_samples > 0`
- **5-M17** — Default DB path is relative; constants unused
- **5-M18** — Metadata footer parsing uses `find()` (first), not `rfind()` (last)
- **5-M19** — Markdown replacement alters non-heading bold text
- **5-M20** — Replacement value not escaped for regex replacement semantics
- **5-M21** — Identity names not normalized consistently
- **5-M22** — `delete_identity()` accepts unused `transcripts_dir` parameter
- **5-M23** — `_find_transcripts_with_label()` only checks `words[].speaker_id`
- **5-M24** — Scan and rewrite operations O(files × metadata) and repeated

### [Infra] MEDIUM (20 findings)
- **6-M1** — `ConfigManager.get()` exposes mutable internal settings object
- **6-M2** — Dirty tracking marks settings dirty even when value unchanged
- **6-M3** — Dirty tracking doesn't track "modified from defaults" as documented
- **6-M4** — Logging leaks local filesystem paths and config locations
- **6-M5** — `ModelSettings.from_dict()` uses fragile dataclass field descriptor fallback
- **6-M6** — Tuple/list conversion lacks shape/type validation for geometry
- **6-M7** — `StoragePaths.from_dict()` doesn't validate value types
- **6-M8** — Benchmark chunk size can become zero or negative
- **6-M9** — Benchmark loads entire WAV into memory
- **6-M10** — Benchmark swallows exceptions into result strings only
- **6-M11** — WER implementation has quadratic memory usage
- **6-M12** — ResourceMonitor starts QTimer without parent
- **6-M13** — ResourceMonitor callback exceptions can break polling
- **6-M14** — HardwareDetector doesn't handle psutil returning None for CPU count
- **6-M15** — Hardware warning message has malformed text (missing parenthesis)
- **6-M16** — Model recommendation documentation contradicts implementation
- **6-M17** — `prefer_accuracy` parameter accepted but ignored
- **6-M18** — Playback numeric setters can throw on invalid input
- **6-M19** — Bookmark writes not atomic
- **6-M20** — Bookmark metadata parser uses first footer marker, not last

### [Core] MEDIUM (16 findings)
- **7-M1** — Logging redirects sys.stdout globally but not stderr, never restores
- **7-M2** — File logging has no rotation or retention policy
- **7-M3** — Logs may capture sensitive transcript/audio metadata at DEBUG level
- **7-M4** — Recovery errors lose traceback and actionable context
- **7-M5** — Startup failures use `print()` instead of consistent logging
- **7-M6** — `check_critical_dlls()` only catches ImportError, missing DLLs often OSError
- **7-M7** — `check_critical_dlls()` uses QMessageBox before validating GUI state
- **7-M8** — `setup_signal_handlers()` silently ignores missing win32api
- **7-M9** — TrayIconManager stored in private widget attribute
- **7-M10** — `on_exit=widget._exit_application` binds to private method
- **7-M11** — Diagnostic tools use broad exception handling, lose tracebacks
- **7-M12** — Metadata file reads assume trusted, well-formed JSON
- **7-M13** — Test script uses sleep-based synchronization
- **7-M14** — Real microphone test uses first mic without explicit selection
- **7-M15** — Test stores recordings in TemporaryDirectory but asks user to play after cleanup
- **7-M16** — PyInstaller runtime hook mutates process PATH globally

---

## Appendix: All LOW Findings by Module

### [Widgets] LOW (18 findings)
- **1a-L1** — Late imports and `# noqa: E402` reduce readability
- **1a-L2** — Unused imports increase maintenance noise
- **1a-L3** — Dialog stores dynamic attribute on QDialog
- **1a-L4** — Broad `object` type annotations lose guarantees
- **1a-L5** — Close behavior doesn't stop all timers/workflows
- **1a-L6** — Magic numbers scattered through UI code
- **1a-L7** — `BudgetProgressBar` doesn't clamp budget marker
- **1a-L8** — `_escape_html` duplicates stdlib and misses quotes
- **1b-L1** — Comments contradict current behavior
- **1b-L2** — Dead compatibility stub remains in production
- **1b-L3** — Unused imports and duplicate local re-imports
- **1b-L4** — Magic numbers dominate layout and rendering
- **1b-L5** — `DragSurfaceItem.paint()` comment says "near-invisible" but fully transparent
- **1b-L6** — `press_time` recorded but unused
- **1b-L7** — `_waveform_frame_counter` ineffective (unused due to `if True`)
- **1b-L8** — Waveform health warning color computed but not applied
- **1b-L9** — `on_frames_dropped()` accepts floats and truncates
- **1b-L10** — Error text may overflow 14px-high indicator

### [Recording] LOW (10 findings)
- **2-L1** — Unused helper `_ts()`
- **2-L2** — Imported `DiarizationResult` not used in type annotations
- **2-L3** — `# noqa: E402` suppressions suggest module organization problems
- **2-L4** — Debug logging noisy and inconsistent
- **2-L5** — Magic constants scattered through controller
- **2-L6** — `_reset_live_speaker_state()` ends with unnecessary `pass`
- **2-L7** — `get_live_audio_samples()` uses non-idiomatic `np.ndarray(0, ...)`
- **2-L8** — `RenameResult.rolled_back` stores renamed path, not restored original
- **2-L9** — Unknown cleanup operation kinds marked completed
- **2-L10** — Public `operations` property returns shallow copy only

### [Transcription] LOW (12 findings)
- **3-L1** — `callable` used as type annotation instead of `Callable`
- **3-L2** — Configuration fields unused or misleading
- **3-L3** — Dead compatibility methods/fields remain unused
- **3-L4** — Import ordering and E402 indicate file concatenation
- **3-L5** — Logger declaration placement inconsistent
- **3-L6** — Some caught exceptions too broad and silently ignored
- **3-L7** — `average_confidence` uses float then truncates with `int()`
- **3-L8** — `AudioRingBuffer` appended to but not used for processing
- **3-L9** — `clear_completed_jobs()` only clears memory, not persisted state
- **3-L10** — `ScrubRunner.accept_scrub()` overwrites without backup
- **3-L11** — Sidecar filename uses raw `model_size`
- **3-L12** — `_record_webrtc_error()` can store unsanitized exception messages

### [Audio] LOW (12 findings)
- **4-L1** — `source_stats` defined but never populated
- **4-L2** — Redundant local logger imports shadow module logger
- **4-L3** — CLI mutates sys.path at runtime
- **4-L4** — CLI accepts negative or zero recording durations
- **4-L5** — `AudioSourceWrapper.frames_dropped` is dead state
- **4-L6** — `DenoisingStats.last_error_message` may expose unsanitized text
- **4-L7** — Denoising provider constants can drift
- **4-L8** — Type hints loose around source interfaces and callbacks
- **4-L9** — `print_device_summary()` uses print() in library module
- **4-L10** — Long lines reduce maintainability in fake module
- **4-L11** — Denoising provider contract vs session behavior inconsistent
- **4-L12** — Public API exports low-level implementation classes

### [Speaker] LOW (16 findings)
- **5-L1** — Diarizer uses `assert` for runtime state checks
- **5-L2** — `_SUBPROCESS_SCRIPT` is large raw string in production module
- **5-L3** — Comments contradict implementation in embedding duration threshold
- **5-L4** — `_FOOTER_END` defined but unused
- **5-L5** — `Dict`/`Optional` imports inconsistently modernized
- **5-L6** — Error messages include full file paths
- **5-L7** — Cache hit logging at INFO may be noisy
- **5-L8** — Possible typo in URL path name (`speaker-recongition-models`)
- **5-L9** — `SpeakerSegment.duration` can be negative
- **5-L10** — `speaker_label_for()` digit extraction can produce surprising labels
- **5-L11** — `StopIteration` caught where `ValueError` would occur
- **5-L12** — `VoiceSignatureStore.__repr__()` may expose DB path
- **5-L13** — Unused module-level logger in `speaker/__init__.py`
- **5-L14** — `_DEFAULT_DB_DIR` and `_DEFAULT_DB_NAME` unused
- **5-L15** — `cache_dir` in subprocess path ignored
- **5-L16** — `duration_seconds` may reflect pre-resample duration

### [Infra] LOW (9 findings)
- **6-L1** — Dead `_defaults` field in ConfigManager
- **6-L2** — Inconsistent logging style (f-strings vs parameterized)
- **6-L3** — `get_config_manager()` return type technically Optional
- **6-L4** — `__all__` omits `SpeakerSettings`
- **6-L5** — ResourceMonitor history method name overpromises
- **6-L6** — Benchmark default test clip comment vs WER semantics unclear
- **6-L7** — `_on_playback_state_changed()` is a no-op
- **6-L8** — `load_transcript_audio()` logs stem before type validation
- **6-L9** — Minor wording/format issues in comments and examples

### [Core] LOW (12 findings)
- **7-L1** — `setup_logging()` stores unused `log_file` with noqa
- **7-L2** — Mixed lowercase and title-case product name
- **7-L3** — Comments contradict behavior in recovery flow
- **7-L4** — `QApplication.processEvents()` used as manual repaint workaround
- **7-L5** — `TeeOutput.write()` flushes on every write
- **7-L6** — Hardware warning dialog modal and interrupts startup
- **7-L7** — Diagnostic scripts contain hardcoded magic thresholds
- **7-L8** — Long diagnostic print lines reduce readability
- **7-L9** — Diagnostic prints raw `st_mtime` timestamp
- **7-L10** — `ModelRecommender` import at module level increases startup cost
- **7-L11** — `__author__` is low-value package metadata
- **7-L12** — Diagnostic scripts lack consistent exit codes

---

*Report generated from 8 individual code review sessions. Original review session IDs: 20260529_202222_f24c33, 20260529_203804_d4420f, 20260529_202142_0343b7, 20260529_203143_338275, 20260529_203342_ea8d45, 20260529_203543_3441bc, 20260529_204004_d6c112, 20260529_204201_7d77fd.*
