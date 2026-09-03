# Logging is one global mode switch — no stdout scraping, no per-module overrides

Diagnostics for Issue Capture Mode were originally going to be mined from the
existing stdout tee (`TeeOutput` in `main.py`), which mirrors arbitrary program
output into the log at DEBUG. We decided instead on a deliberate logging
overhaul: every module instruments itself through its named logger, and the
tee is removed. What the capture log contains is what the code chose to say —
never what some code path happened to print.

- **Two modes, one switch.** Normal runs log at INFO; Issue Capture Mode logs
  at DEBUG from process start (entered by the same launch flag that enters the
  mode, ADR 0003). There is intentionally no per-module override mechanism —
  no environment variables, no settings, no per-module levels. Full audit
  means one switch; if modules could opt in and out, captures would not be
  comparable and "full" would mean nothing.
- **Deliberate instrumentation only.** Modules emit structured, named events
  through their loggers (state transitions, decisions, timings, resource
  samples) — the full-audit DEBUG sweep across all modules. The stdout tee is
  deleted: nothing enters the log from stdout by accident. A missing
  diagnostic is fixed by adding a logger call at the source, not by widening
  capture.
- **Transcript exclusion is enforced here, at the source.** The logger
  vocabulary never includes Transcript text, Audio content, or Recording
  titles — free-text user input appears only as counts ("text edited
  (N chars)", CONTEXT.md). Because the tee is gone, transcript fragments
  printed anywhere can no longer reach the capture log; Redaction (ADR 0004)
  remains defense-in-depth, not the primary control. The implementation must
  include a negative test: inject transcript-like sentinel content into
  stdout and assert it appears nowhere in the capture log the Diagnostics
  Bundle consumes.
- **Retention.** Normal-run INFO logs are kept for 14 days and cleaned up at
  startup. The capture-run DEBUG log is consumed into the Diagnostics Bundle
  at run end and then follows the bundle lifecycle (ADR 0004); it is not
  retained separately.

Rejected: keeping the tee and filtering transcript content out during
Redaction — arbitrary transcript text cannot be reliably identified by
pattern-based redaction (usernames, paths, emails catch only what they know),
so anything downstream of the firehose is a heuristic, not a guarantee.

## Consequences

- Every module gains DEBUG instrumentation — the full-audit work is a code
  sweep across the codebase, not a configuration change. New code without
  instrumentation is incomplete by definition.
- `print()` stays banned in production source (already enforced by test
  guards); stdout returns to being a human console, not a diagnostic channel.
- Ad-hoc stdout diagnostics no longer land in the log. Acceptable: they were
  unreliable, unordered, and the privacy leak this decision closes.
- Log volume in capture mode grows substantially; the Interaction Trace and
  Resource Snapshot series give it structure to be read against.
