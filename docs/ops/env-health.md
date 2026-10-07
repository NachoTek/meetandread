# Env Health - weekly agent-environment audit (approved 2026-10-07 session)

Weekly verification that the agent "pit of success" still holds. Each check
exists because it broke once and cost real session time; the file names the
issue so the motivation survives.

## Cadence

Weekly, Monday ~13:00 UTC (after the weekend's nightlies).

## Checks (agent session, worktree `mar-wt/env-health`, branch from main)

1. **Pre-push hook gates for real.** `sh .githooks/pre-push` from a WORKTREE
   exit 0 (fast lane green); a deliberately failing probe exits 1. Proven fix
   in #151; regression here silently reverts us to `--no-verify` everywhere.
2. **Central venv serves worktrees.** `.venv\Scripts\python.exe -m pytest
   --collect-only -q` from a fresh worktree collects the full suite (<60s).
   (Mechanism: `pythonpath=["src"]`; broke nothing yet, cheap to verify.)
3. **Pyright baseline is 0.** `.venv\Scripts\python.exe -m pyright` from the
   checkout root reports 0 errors (#154 baseline). If >0, the diff that
   reintroduced noise is findable via `git log --oneline -20` + rerun.
4. **AGENTS.md still matches machine reality.** Spot-check the three claims:
   gh-token one-liner works, no-WSL lane note present, PRE_PUSH_DESELECT flake
   nodeid matches the actual #149 test path (#145/#156).
5. **Docs drift.** `git log --oneline --since="7 days ago" -- AGENTS.md
   Makefile .githooks pyproject.toml` - summarize anything that changed and
   whether AGENTS.md was updated in the same window.
6. **Worktree hygiene.** `git worktree list` + `git ls-remote --heads origin`:
   no merged-but-undeleted branches; only active worktrees under mar-wt/.
   (2026-10-07 cleanup found 11 stale remote branches.)

## Output

- All green: append one line to `docs/watchlog/env-health.md` (date, verdict).
  Create the log file on first entry if it does not exist yet.
- Any red: file an issue (`bug`, `needs-triage`, title prefix "env: ...") with
  the failing check and evidence; if trivially fixable (docs drift, stale
  branches), fix directly via the standard PR protocol in AGENTS.md.

## Escalation

Two consecutive weeks red on the same check: raise to P2 and propose a CI-side
guard (e.g. a meta-test that runs the hook probe) so the check self-enforces.
