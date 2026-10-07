# Flake Watch - nightly CI failure triage (approved 2026-10-07 session)

Every night GitHub runs the full suite at 03:00 UTC (`.github/workflows/ci.yml`
schedule). This watch reviews each run's outcome; failures get triaged into
GitHub Issues so flakes never silently become "the normal state" (the #142 /
#149 pattern: flakes that taxed every session for weeks before being filed).

## Cadence

Daily, shortly after the nightly finishes (~04:30 UTC / 00:30 EDT).

## Procedure (agent session, read-only unless filing)

1. Auth gh: `$env:GH_TOKEN = & "$env:USERPROFILE\.config\gh-app\gh-app-token.ps1"`
   (gh lives at `C:\Program Files\GitHub CLI\gh.exe`).
2. List the last night's scheduled runs:
   `gh run list --workflow ci.yml --event schedule --limit 2`
3. If the latest run is `success`: log one line to the report file (below) and stop.
4. If it failed: pull the failing job log (`gh run view <id> --log-failed`),
   classify each failing test:
   - Known & filed (grep open issues for the test name, e.g. #149 sustained-load):
     add a `+1 occurrence` comment on that issue with the run URL and timestamp.
   - New failure: file an issue (label `bug`, `needs-triage`) titled
     `Flaky test: <nodeid> <one-phrase symptom>`, body: nodeid, run URL,
     last-green SHA, failure excerpt, and whether it reproduces locally via
     `.venv\Scripts\python.exe -m pytest <nodeid> -p no:qt -q` (5x loop).
   - Product (non-test) failures: label `bug`, `P1` if it blocks the nightly.
5. Append a one-line outcome to `docs/watchlog/flake-watch.md`
   (date, run id, verdict) so trends are visible quarter-over-quarter.
   Create the log file on first entry if it does not exist yet.

## Escalation

- Same nodeid failing 3+ consecutive nights and already filed: raise priority
  label to P2 and note the streak on the issue.
- Nightly failing 5+ consecutive nights for mixed reasons: that is a broken
  environment, not flakes - flag for the weekly env-health pass (see
  `env-health.md`) and consider blocking merges.

## Notes

- The runner (CI-RUNNER-01) is not this VM; local reproduction is advisory
  evidence, not proof (see #144 session notes).
- One-off transient: `test_normal_run_creates_no_trace_file` loopback OSError
  (seen 2026-10-07 on PR #156, green on retry). File only if it recurs.
