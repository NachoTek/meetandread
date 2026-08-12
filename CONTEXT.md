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
The JSON metadata block appended to a Transcript's Markdown file, carrying its machine-readable data — words (with timing, confidence, and speaker_id), segments, speaker_matches, and recording_start_time. The machine-readable twin of the Markdown body: the body is for humans to read, the footer is for the system to read back. Written and read through one canonical format, owned by `transcript_footer`.
_Avoid_: footer, metadata block, metadata section, trailer

**Live Transcript**:
The speaker-unlabeled text captured during a Recording and saved when recording stops. Available immediately, but accuracy and speaker identification are improved in-place by background post-processing. From the user's perspective, the Transcript simply gets better over time.
_Avoid_: Draft transcript, raw transcript, interim transcript

**Post-processing**:
The automatic background activity that turns a Live Transcript into the full Transcript. Re-transcribes the Recording's audio with a stronger Whisper model and applies speaker diarization, overwriting the canonical transcript file in place. Runs while idle after a recording stops; several Recordings may queue and are processed one at a time.
_Avoid_: Enhancement, refinement, finalization, second pass

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
