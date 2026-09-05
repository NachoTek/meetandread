# Diagnostics leave the machine only through Manual Submission

The Diagnostics Bundle (redacted log, Interaction Trace, Resource Snapshots,
environment info) is privacy-sensitive by construction: logs carry Windows
usernames in paths, and the repo is public. We decided that in this slice, the
**only** submission path is Manual Submission — the user's own browser opens on
the prefilled New Issue form and the user attaches the bundle file themselves.
No token ships inside the app, no report endpoint exists yet, and nothing is
transmitted automatically. Filing from their own account also subscribes the
reporter to their own issue for free.

A server-side relay (Cloudflare Worker, auto-filing via the org bot, with
validation gates and quarantine) was designed but deliberately deferred: a
relay earns its keep with more users than we have today. It is a potential
future feature, to be built as a separate vertical slice.

## Consequences

- Redaction (usernames/paths, emails, machine identifiers) runs before the
  review screen — nothing unredacted is ever shown as "what will be sent" or
  written into the submittable artifact.
- Transcript text and Recording titles never enter the bundle: the exclusion
  is enforced at the capture boundary — transcript-bearing output may reach
  the console but must never enter the log stream — and negative/canary tests
  prove no transcript text reaches the assembled bundle. Post-hoc pattern
  redaction is not a reliable boundary for arbitrary speech.
- The bundle excludes Audio and Transcript content by design; follow-up data
  (e.g. a specific Recording) is requested during triage through GitHub.
- The interaction vocabulary (named-action events, "text edited (N chars)") is
  the same vocabulary a future relay's log-format validation gate would check
  against, so nothing here blocks that path.
