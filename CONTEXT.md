# MeetAndRead

An always-on-top desktop widget that captures meeting audio (microphone and system sound), transcribes it with speaker identification, and saves formatted Markdown transcripts for later review and AI-assisted extraction of action items.

## Language

**Recording**:
The atomic artifact created by one record-to-stop cycle. Encompasses the captured audio, the derived transcript, and the saved Markdown output.
_Avoid_: Session, capture, transcript (as a synonym)

**Transcript**:
The primary output of a Recording — a formatted Markdown document with speaker-labeled text. Immediately available as a speaker-unlabeled Live Transcript when recording stops, then automatically improved in-place by post-processing (better accuracy and speaker identification) during idle time.
_Avoid_: Output, document, notes, final transcript

**Transcript Footer**:
The JSON metadata block appended to a Transcript's Markdown file, carrying its machine-readable data — words (with timing, confidence, and speaker_id), segments, speaker_matches, recording_start_time, and the Post-processing Outcome. The machine-readable twin of the Markdown body: the body is for humans to read, the footer is for the system to read back. Written and read through one canonical format, owned by `transcript_footer`.
_Avoid_: footer, metadata block, metadata section, trailer

**Live Transcript**:
The speaker-unlabeled text captured during a Recording and saved when recording stops. Available immediately, but accuracy and speaker identification are improved in-place by background post-processing. From the user's perspective, the Transcript simply gets better over time.
_Avoid_: Draft transcript, raw transcript, interim transcript

**Post-processing**:
The automatic background activity that turns a Live Transcript into the full Transcript. Re-transcribes the Recording's audio with a stronger Whisper model and applies speaker diarization, overwriting the canonical transcript file in place. Runs while idle after a recording stops; several Recordings may queue and are processed one at a time.
_Avoid_: Enhancement, refinement, finalization, second pass

**Post-processing Outcome**:
The durable terminal result of Post-processing for a Recording — Completed (it ran to completion, including zero-speaker results) or Failed (it errored, with the failing stage and reason). Carried in the Transcript Footer so it lives and dies with the Recording.
_Avoid_: Status (ambiguous with the in-flight job state), cancellation state

**Stalled**:
A Recording with no recorded Post-processing Outcome — Post-processing never ran, was lost, or was interrupted. Automatically re-queued for Post-processing when its Audio still exists and Post-processing is enabled.
_Avoid_: Manual Action Required (as a state name), stuck, incomplete

**Preempt**:
The cooperative interruption of a running Post-processing job by a higher-priority action (starting a live Recording, or a user-initiated Retry). The job steps aside within one transcription segment, returns to the front of the queue — it is not cancelled — and is shielded from further preemption until it completes. Partial transcription progress is redone, not resumed.
_Avoid_: Cancel (terminal — discards the job), pause, abort

**Audio**:
The raw captured sound of a Recording, stored as a WAV file. It is the input material from which the Transcript is derived, not an end product in itself.
_Avoid_: Recording (as a synonym — a Recording contains Audio), sound file

**Speaker**:
Anyone the diarization system detects as producing speech during a Recording. Identified by a machine label (e.g. spk0, spk1) which the user may later replace with a name.
_Avoid_: Participant, talker, voice

**Speaker Profile**:
A named identity (e.g. "David") associated with a voice signature, created when the user assigns a name to a Speaker. Stored persistently so the system can recognize the same person across Recordings.
_Avoid_: Identity, user (as a synonym — this is not an app user)

**Closed Captioning Overlay**:
A live display of in-progress transcription shown during a Recording, serving as a confidence indicator that audio is being captured and speech is being detected. Its text is saved as the initial Live Transcript when recording stops, then improved in-place by post-processing.
_Avoid_: Live transcript, real-time transcript

**Audio Source**:
The input the system captures audio from during a Recording. The two kinds are microphone (external sound picked up by a mic) and system audio (sound output by the computer, via WASAPI loopback). A Recording may use one or both.
_Avoid_: Input, capture device, channel

**Library**:
The persistent collection of all Recordings the user has made. Browsable through the UI — allows reviewing transcripts, playing back audio, and deleting Recordings. Each Recording in the Library has its Audio and Transcript accessible together.
_Avoid_: Archive, history, folder

**Source Degradation**:
A loss or reduction in audio quality from an Audio Source during an active Recording — caused by device disconnection, hot-plug events, or frame drops. The system notifies the user so they can decide whether to address the issue or accept potential data loss. Recovery is attempted automatically.
_Avoid_: Device loss, dropout, failure

**Re-transcribe**:
A user-initiated re-transcription of a Recording's audio with a different (typically stronger) Whisper model. Produces a sidecar Transcript alongside the original so the user can compare both versions side-by-side and choose which to keep. The discarded version is then removed.
_Avoid_: Scrub, reprocess, upgrade

**Retry**:
A user-initiated re-run of Post-processing for a Failed Recording, using the current default Post-processing settings. Distinct from Re-transcribe: no model picker and no sidecar — the canonical Transcript is overwritten in place. Scheduled at the front of the queue; preempts a running job (after confirmation) so it runs first. A failed Retry surfaces actively; success is quiet.
_Avoid_: Re-transcribe (different intent — model comparison), reprocess, rerun

**Feature Dependency**:
An optional installable component that powers a feature (e.g. sherpa-onnx powers Speaker identification). Checked in two tiers at startup: critical dependencies are fatal when missing; Feature Dependencies degrade instead — a dismissible banner and the Diagnostics view explain what is missing and how to fix it, and Post-processing fails with a Failed (dependency) Outcome rather than silently completing without the feature. Once the dependency imports cleanly again, dependency-failed Recordings return to Stalled and are re-queued automatically at startup.
_Avoid_: Plugin, extension, requirement (as a synonym)

**Issue Capture Mode**:
A mode the application runs in, entered at process start, in which comprehensive diagnostics are recorded for an issue the user intends to report. Entered only by being launched under the Issue Reporter — the user must start the Issue Reporter *before* reproducing the bug, though Recording is allowed inside the mode. Logging runs at DEBUG, the Interaction Trace and Resource Snapshots are recorded, and the run ends with a Diagnostics Bundle.
_Avoid_: Debug mode (ambiguous with log levels), recording mode, reproduction mode

**Issue Reporter**:
A separate process that supervises the application: it launches the app in Issue Capture Mode, monitors it, and if the app crashes it collects the diagnostics gathered so far and proceeds with submission anyway. Also runnable standalone (installer shortcut) so startup crashes are reportable. Owns the user-facing wizard flow: describe, launch, reproduce, stop, review, submit.
_Avoid_: Wizard (as a noun on its own — ambiguous), bug reporter, feedback tool

**Interaction Trace**:
The ordered record of semantic user actions taken during Issue Capture Mode — named events (button/menu/shortcut presses, panel open/close/move/resize, device selections, focus changes) with timestamps. Records what the user did and when, never what they typed: free-text entry appears only as a "text edited (N chars)" event.
_Avoid_: Click log, input capture, keystroke log

**Resource Snapshot**:
A point-in-time sample of system resource usage (RAM/CPU percentages, available RAM) recorded periodically during Issue Capture Mode. Already exists as the ResourceMonitor's snapshot; Issue Capture Mode persists the series.
_Avoid_: Metrics, telemetry

**Diagnostics Bundle**:
The single artifact produced at the end of an Issue Capture Mode run: the redacted log, the Interaction Trace, the Resource Snapshot series, and environment info (app version, OS, hardware class). What the user reviews and submits — by the Manual Submission path now; via a possible future relay. Contains no Audio and no Transcript content.
_Avoid_: Report (ambiguous with the GitHub issue), log file (the bundle contains more), package

**Redaction**:
The automatic scrubbing of the Diagnostics Bundle before it is shown to the user or submitted: usernames and home-directory paths, email addresses, and machine identifiers are rewritten; Transcript text and Recording titles are excluded outright. Runs before the review screen — nothing leaves the machine unredacted. Missing details (e.g. a specific Recording file) are requested later through GitHub during triage.
_Avoid_: Sanitization, anonymization (we do not promise anonymity)

**Manual Submission**:
The user-driven submission path for a Diagnostics Bundle: open the user's default browser on the repository's New Issue form pre-filled with the summary, and hand them the bundle file to attach (GitHub URLs cannot pre-attach files). The user files from their own GitHub account, so they are automatically subscribed to their issue.
_Avoid_: Direct submission, relay (a possible future alternative path)
