# One capture directory, one run — concurrent capture runs are impossible

Capture directories are the crash-safe source of truth for the Diagnostics Bundle:
the reporter assembles the bundle from the directory alone, in any state. A
directory holding records from two runs cannot be interpreted (whose record is
whose?), and a stale completion marker can brand a crashed peer as clean. The
risk was spotted in automated review of PR #121 (rounds 1-2), where the reviewer
reproduced two processes co-logging into one directory; fixed and codified at
owner direction 2026-09-07.

We decided on **exclusivity at entry**. A capture run atomically claims its
directory — an O_CREAT|O_EXCL claim file, `capture_run.claim`, created before
any logging — and any second start against a claimed or non-empty directory is
rejected (exit code 2). The Issue Reporter (#107) always creates a fresh
directory per run.

Rejected: reconciling concurrent writers into one directory — timestamps and
record ownership cannot disambiguate interleaved streams after a crash, and a
false clean marker for a crashed peer is unacceptable as the source of truth.
Also rejected: tolerating concurrency and degrading the bundle to "best effort".

## Consequences

- The claim file is part of the capture-directory contract consumed by #105-#110.
  Run identity comes from the claim (started_at); a held claim means the
  directory belongs to a live — or crashed-mid-run — run.
- Rejection-before-logging means a rejected start writes NOTHING into the target
  directory.
- Sequential reuse is equally refused: a fresh directory per run is the
  reporter's job.
