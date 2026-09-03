# Diagnostics leave the machine only through Manual Submission

The Diagnostics Bundle (redacted log, Interaction Trace, Resource Snapshots,
environment info) is privacy-sensitive by construction: logs carry Windows
usernames in paths, transcript fragments may flow through the stdout tee, and
the repo is public. We decided that in this slice, the **only** submission path
is Manual Submission — the user's own browser opens on the prefilled New Issue
form and the user attaches the bundle file themselves. No token ships inside
the app, no report endpoint exists yet, and nothing is transmitted
automatically. Filing from their own account also subscribes the reporter to
their own issue for free.

A server-side relay (Cloudflare Worker, auto-filing via the org bot, with
validation gates and quarantine) was designed but deliberately deferred: a
relay earns its keep with more users than we have today. It is a potential
future feature, to be built as a separate vertical slice.

## Consequences

- Redaction (usernames/paths, emails, machine identifiers; transcript text and
  Recording titles excluded outright) runs before the review screen — nothing
  unredacted is ever shown as "what will be sent" or written into the
  submittable artifact.
- The bundle excludes Audio and Transcript content by design — enforced at
  the source by the logging overhaul (ADR 0005: no stdout tee, a named-event
  logger vocabulary, and the negative test proving transcript-like content
  cannot reach the capture log). Redaction stays defense-in-depth. Follow-up
  data (e.g. a specific Recording) is requested during triage through GitHub.
- The interaction vocabulary (named-action events, "text edited (N chars)") is
  the same vocabulary a future relay's log-format validation gate would check
  against, so nothing here blocks that path.
- Persisted bundles (reporter crash, ADR 0003) rest in a user-profile-scoped
  directory with user-only file permissions — no other account can read them
  at rest.
- Bundle lifecycle: deleted immediately when the user cancels the wizard flow
  or confirms submission in its final step; bundles that are never resumed
  expire after 14 days and are cleaned up on the next app or reporter start.
  Nothing privacy-sensitive lingers on disk without a reason.
