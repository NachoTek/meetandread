# Issue reporting is a supervisor, not an in-app dialog

When a user wants to report a bug, the app cannot both be the bug and the observer.
We decided the Issue Reporter is a **separate supervisor process**: it launches the
application in Issue Capture Mode, monitors it, and — critically — if the app
crashes during reproduction, the reporter survives, collects the diagnostics
gathered so far, and proceeds with submission anyway. The app itself is never
entrusted with reporting its own death. The reporter is also installed as a
standalone shortcut so startup crashes ("won't open at all") are reportable.

Rejected: an in-app wizard that restarts the app into capture mode and offers a
sentinel-based recovery on next launch — it loses the crash itself, the highest-
value bug class, whenever collection or submission depends on the crashed process.

## Consequences

- The Issue Reporter process must be deliberately small and defensive: minimal
  imports, own crash handling, no reliance on the app's subsystems (audio, Qt
  widget tree). If the reporter itself dies, the Diagnostics Bundle from the run
  remains on disk under the at-rest controls and lifecycle defined in ADR 0004
  (user-only file permissions, deletion after submit or cancel, 14-day expiry)
  and the next app launch offers to resume the submission.
- The app is launched with a flag (or equivalent) that enables Issue Capture
  Mode at process start; entering capture mode mid-process is not supported.
- Users must enter the reporter *before* reproducing the bug. Recording is
  allowed inside Issue Capture Mode, so mid-recording bugs remain reportable —
  but only if the user started the reporter first.
