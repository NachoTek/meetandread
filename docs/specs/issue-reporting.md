# Spec: Issue Reporting & Diagnostics

Status: Spec text approved by owner 2026-09-05 (settled design from the grill session of 2026-09-02; test seams + cadence approved at the seam checkpoint; single-instance direction settled). Amended 2026-09-05 after automated review: the bundle's local path never enters the public prefilled issue body, and transcript exclusion is enforced at the capture boundary rather than by pattern redaction.

## Problem Statement

When MeetAndRead misbehaves, the user has no good way to tell the developers what happened. Logs exist but are an undifferentiated firehose — flat DEBUG for every run, with no levels to separate routine operation from things worth reading — so finding the moment something went wrong means digging through megabytes of noise. Nothing records what the user actually did leading up to the problem, and nothing records what the machine was doing at the time. Worst of all, when the app crashes, the evidence of the crash — the single most valuable thing to report — dies with the process. The user is left to write a vague prose description from memory, and the developer is left to guess.

## Solution

One vertical slice that makes bugs reportable end to end:

The user launches a separate **Issue Reporter** program *before* reproducing the bug. It runs a short wizard — describe, launch, reproduce, stop, review, submit — and starts the app in **Issue Capture Mode**, in which the app logs at full DEBUG, records an **Interaction Trace** of what the user did, and samples periodic **Resource Snapshots**. If the app crashes during reproduction, the reporter survives, collects the diagnostics gathered so far, and proceeds anyway — the crash itself is always reportable, including startup crashes (the reporter is installed as its own Start-menu shortcut).

When reproduction stops (or the app dies), the reporter assembles a **Diagnostics Bundle**, automatically applies **Redaction** (usernames and home-directory paths, email addresses, and machine identifiers are rewritten; transcript text and recording titles are excluded outright; no Audio, ever), and shows the user a review screen: here is what will be sent. Only then does anything leave the machine — via **Manual Submission**: the user's default browser opens on the repository's New Issue form prefilled with the summary, the bundle's path is on the clipboard, and the user attaches the bundle file by hand.

Underpinning this, the logging system is overhauled properly: real log levels, normal runs at INFO, and a full-audit pass that normalizes DEBUG instrumentation across every module in the application.

## User Stories

1. As a user hit by a bug, I want a dedicated reporter program that walks me through reproducing it, so that I don't have to figure out what information the developers need.
2. As a user hit by a bug, I want to describe the problem in my own words at the start of the flow, so that the human context isn't lost among the machine diagnostics.
3. As a user, I want the reporter to start the app for me in a special capture mode, so that diagnostics are being recorded from the very first moment of the run.
4. As a user, I want to be able to record a meeting inside capture mode, so that bugs that only strike mid-recording are still reportable.
5. As a user, I want the reporter to be something I must start *before* reproducing the bug, so that the reproduction is always fully covered by diagnostics (the tradeoff is stated plainly in the wizard).
6. As a user whose app just crashed, I want the reporter to keep going — collect what was gathered and continue to submission, so that a crash never prevents the bug from being reported.
7. As a user whose app won't even start, I want the reporter available as its own Start-menu shortcut, so that startup crashes are reportable too.
8. As a user with the app already open, I want the reporter to tell me to close it first, so that capture starts from a clean single instance.
9. As a user, I want the reporter itself to be tiny and hard to kill, so that even when everything else is broken, the reporting path still works.
10. As a user, I want the reporter to tell me clearly when to start reproducing the bug and when to stop, so that I know the flow is progressing.
11. As a user, I want a review screen showing exactly what will be sent, so that nothing leaves my machine that I haven't seen.
12. As a user, I want the bundle automatically scrubbed of my username, my home-directory paths, my email address, and machine identifiers before I ever see it, so that I'm not leaking personal data by reporting a bug.
13. As a user, I want transcript text and recording titles excluded from the bundle outright, so that meeting content never enters a bug report by accident.
14. As a user, I want it made obvious that no audio and no transcript content is included, so that I can report with confidence.
15. As a user, I want free-text I typed during reproduction to appear only as a "text edited (N chars)" event, so that my keystrokes are never captured.
16. As a user, I want the reporter to open my browser on the issue form already filled in, so that filing the report is a couple of clicks.
17. As a user, I want the bundle's file path on my clipboard — and a neutral filename with attach-by-hand instructions in the prefilled issue body, never my local path — so that attaching the bundle is trivial without publishing where my files live.
18. As a user filing from my own GitHub account, I want to be automatically subscribed to the issue I just filed, so that I hear about follow-up questions.
19. As a user asked later for more details (e.g. a specific recording file), I want that request to arrive through GitHub, so that I keep control of anything additional that leaves my machine.
20. As a user, I want normal runs of the app to log at a quieter level, so that my disk isn't filled with debug noise from every session.
21. As a developer, I want proper log levels throughout the codebase, so that I can read a normal run's log without drowning.
22. As a developer, I want capture-mode runs at full DEBUG with no per-module configuration, so that a reproduction yields maximum information — neither I nor the user has to guess which modules matter.
23. As a developer, I want DEBUG instrumentation audited and normalized across every module in the application, so that capture-mode logs cover hot spots and cold spots alike.
24. As a developer, I want an Interaction Trace of named, timestamped user actions (button/menu/shortcut presses, panel open/close/move/resize, device selections, window focus/blur), so that I can see what the user did without a screen recording.
25. As a developer, I want periodic Resource Snapshots (RAM/CPU percentages, available RAM) across the capture run, so that I can correlate the bug with resource pressure.
26. As a developer, I want the DEBUG log to start at process start in capture mode, so that startup-sequence bugs are visible.
27. As a developer, I want environment info (app version, OS, hardware class) in the bundle, so that I can rule out version- and hardware-specific causes.
28. As a developer triaging a report, I want the bundle in a predictable format (redacted log, Interaction Trace, Resource Snapshot series, environment info), so that I can read any report the same way.
29. As a developer triaging a report, I want the crash itself captured when one occurred, so that the highest-value evidence isn't lost.
30. As a maintainer, I want diagnostics to leave the machine only through the user's own hands, so that no endpoint, token, or transmission channel ships inside the app.
31. As a maintainer of a public repository, I want every bundle redacted before review and submission, so that nothing sensitive is published even by accident.

## Implementation Decisions

**Scope is one vertical slice.** Logging overhaul, full-audit DEBUG instrumentation, the Issue Reporter, Issue Capture Mode, the Diagnostics Bundle, Redaction, the review screen, and Manual Submission are delivered together.

### Logging

- Real log levels are introduced across the codebase. Normal runs log at **INFO**; Issue Capture Mode logs at full **DEBUG**.
- Level selection is **all-or-nothing**: there are no per-module overrides. The owner scrapped per-module configuration explicitly — users cannot know which modules to enable, and the goal is maximum information during reproduction.
- A **full-audit pass** normalizes DEBUG instrumentation across **all** ~40 modules (chosen over a hot-spots-only sweep), in addition to cleaning up the scattered ad-hoc levels already present.
- The existing per-run timestamped log file under the user's Documents folder is preserved in shape. The stdout tee into the root logger is preserved only as a console-mirroring convenience; it is not a transcript channel: transcript-bearing output may reach the console but must never enter the log stream, so the captured DEBUG log contains no transcript fragments in the first place. Redaction still runs on the log, but only for identifiers (usernames, home paths, email addresses, machine identifiers) — it is not, and cannot be, the transcript boundary: arbitrary meeting speech is not reliably removable by pattern or table redaction, especially from a crashed run with no complete Transcript to compare against.

### Issue Reporter (supervisor process — ADR 0003)

- The Issue Reporter is a **standalone supervisor process**, not an in-app dialog. The app cannot both be the bug and the observer.
- It launches the app with a flag that enables **Issue Capture Mode at process start**. Entering capture mode mid-process is not supported.
- It monitors the app during reproduction; if the app crashes, the reporter survives, assembles the Diagnostics Bundle from what was gathered so far, and proceeds with the flow anyway.
- It is also installed as a **separate Start-menu shortcut** so that startup crashes ("won't open at all") are reportable.
- When the reporter launches and detects the app is already running (single-instance guard), it alerts the user to close the existing instance before proceeding — a stage-level decision, may be revisited.
- The reporter process is **small and defensive**: minimal imports, its own crash handling, and no reliance on the app's subsystems (audio, Qt widget tree). If the reporter itself dies, the bundle from the run remains on disk, and the next app launch offers to resume the submission.
- It owns the user-facing wizard flow: **describe, launch, reproduce, stop, review, submit**.

### Entry discipline

- The user must start the Issue Reporter **before** reproducing the bug. Recording is allowed inside Issue Capture Mode, but only if the wizard was already started first. Mid-recording entry is not possible — this is the deliberate cost of capture-from-process-start.

### Capture-directory contract (the new seam)

- The reporter passes the app a flag naming a **capture directory**; the app in Issue Capture Mode writes every artifact there.
- Artifacts are written **append-only as they are produced** — one JSONL line per Interaction Trace event, one JSONL line per Resource Snapshot, the DEBUG log streaming from process start — so a crash mid-run loses at most the last line, never the run.
- The app writes a **completion marker** on clean exit; the reporter treats the directory's contents as the source of truth in **any** state (complete or crashed mid-run) and assembles the bundle from the directory alone.
- **Resource Snapshot cadence** (approved): reuse the ResourceMonitor building block's existing default poll interval of 2 seconds; persist **full history** (no ring buffer) — capture runs are minutes long, and crash tolerance beats truncation.
- **Interaction Trace cadence/storage** (approved): each event is written immediately as one JSONL line, never held only in memory.

### Captured content

- **Interaction Trace** — semantic and structural: named events (button/menu/shortcut presses, panel open/close/move/resize, device selections, window focus/blur changes) with timestamps. **No keystroke content, ever**: free-text entry appears only as a "text edited (N chars)" event. This vocabulary is the same one a future relay's log-format validation gate would check against, so nothing here blocks that path.
- **Resource Snapshots** — the periodic series described above, built on the existing ResourceSnapshot shape (RAM/CPU percentages, available RAM).
- **DEBUG logs from process start.**
- **Environment info**: app version, OS, hardware class.

### Privacy: Redaction + review

- The Diagnostics Bundle is **auto-redacted before the review screen**: usernames and home-directory paths, email addresses, and machine identifiers are rewritten. Transcript text and recording titles are excluded outright — enforced at the capture boundary (transcript-bearing output never enters the log stream), not by scrubbing the log afterwards.
- The bundle contains **no Audio and no Transcript content** by design.
- The user sees a review screen — "here's what will be sent" — before anything leaves the machine. Nothing unredacted is ever shown as what-will-be-sent or written into the submittable artifact.
- Missing details (e.g. a specific recording file) are requested later, during triage, through GitHub.

### Submission: Manual Submission only (ADR 0004)

- The wizard opens the user's **default browser** on the repository's New Issue form, prefilled via query params (title/body).
- GitHub query params cannot attach files, so the **bundle is saved to disk** and its **path is copied to the clipboard** and shown on the reporter's review screen. The prefilled issue body carries only a neutral bundle filename plus attach-by-hand instructions — never a local filesystem path: the issue body is public, and an unredacted path would disclose the user's home directory.
- The user files from their own GitHub account, so they are automatically subscribed to their own issue.
- No token ships inside the app and no report endpoint exists in this slice.

## Testing Decisions

**What makes a good test here:** external behavior at the seams — what the app writes into a capture directory, what the reporter assembles from it, what the review screen is offered, what the prefilled issue looks like. Internal buffering strategies, timer objects, or widget internals are not tested directly.

**The one new seam — the capture-directory contract.** Three testable edges:

1. **Launch edge**: the reporter starts the app in Issue Capture Mode pointed at a capture directory.
2. **Artifact edge**: the app writes the DEBUG log from process start, the Interaction Trace, the Resource Snapshot series, and the completion marker into that directory, append-only as events happen.
3. **Assembly edge**: the reporter's pure core is a function of a capture directory in any state (complete, or crashed mid-run) producing: assembled bundle → redaction → review-screen data → prefilled New Issue URL + clipboard text. Because this is a pure function, the entire privacy-and-submission half of the spec is tested against fixture capture directories with zero subprocesses.

**Existing seams reused, no new ones:**

- **Qt widget seam** (prior art: the widget test suite's pattern of driving real widgets): install a trace sink, drive real widgets, assert semantic named events appear and the no-keystroke-content invariant ("text edited (N chars)", never content) holds.
- **`windows`-marked CLI subprocess seam** (prior art: the CLI subprocess tests): the end-to-end supervisor flow — including the app-crashes-during-reproduction case — spawning real processes under the Windows venv per the two-layer test topology (ADR 0001).
- **Pure-logic unit seam** (prior art: the performance/monitor tests): Redaction tables (identifier rewriting), log-level normalization, snapshot-series persistence — plus negative/canary tests for the transcript boundary: feed canary transcript fragments at every capture-side source and assert none appear in the assembled bundle.

**Test topology** follows ADR 0001 unchanged: pure-logic tests run in the fast non-Windows lane; the authoritative pass, including all subprocess supervisor tests, runs under the Windows venv.

## Out of Scope

- **The relay.** A server-side submission relay (Cloudflare Worker, auto-filing via the org bot, validation gates including prompt-injection and log-format checks, quarantine, tracking codes, optional GitHub-username mention) was designed in the grill session but explicitly deferred by the owner: it earns its keep with more users. It is a potential future enhancement, to be built as a separate vertical slice. Nothing in this spec should be designed in a way that blocks it (the interaction vocabulary already matches its validation gate).
- **Entering Issue Capture Mode mid-process.** Capture mode is entered only at process start, by being launched under the Issue Reporter.
- **Automatic transmission of diagnostics of any kind.** No endpoint, no token, no auto-upload — Manual Submission is the only path in this slice.
- **An in-app reporting wizard** that restarts the app into capture mode — rejected by ADR 0003: it loses the crash itself whenever collection depends on the crashed process.

## Further Notes

- **OPEN — log retention policy.** How long log files are kept before cleanup is undecided. A value of "14 days" floated in a later round of the grill was **not approved** — do not treat it as decided. Tickets should surface retention as its own decision.
- **Approved cadence/storage decisions** (owner-approved at the seam checkpoint): Resource Snapshots at the ResourceMonitor's existing 2-second default poll interval; Interaction Trace and Resource Snapshot series persisted as append-only JSONL in the capture directory, one line per event/snapshot, written immediately; **full history, no ring buffer**.
- **Single-instance guard interaction — decided.** When the Issue Reporter launches and detects the app is already running (single-instance guard, issue #20), it alerts the user to close the existing instance before proceeding. Accepted by the owner as adequate at this stage; may be revisited later (e.g. a capture-mode bypass of the guard).
- **Frozen build.** The reporter ships in the PyInstaller build as a separate entry point with its own Start-menu shortcut; the bundle-validation workflow must cover it.
- **Repo facts.** Repository: NachoTek/meetandread (public — the reason Redaction is non-negotiable). The owner is @TerminalSausage.
