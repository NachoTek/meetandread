# Post-processing jobs are preemptible, not just cancellable

When a user hits Retry on a Failed recording, or starts a live Recording while a
Post-processing job is in flight, the running job must step aside promptly — not
wait out a multi-hour transcription. We decided to make **preempt** a first-class
operation of the Post-processing queue: a cancellation request that the
transcription engine honors cooperatively *per-segment* inside the transcription
loop, which returns the job to the **front of the queue** rather than terminating
it, and marks it **shielded from re-preemption** until it completes (so a
fast-failing Retry loop cannot starve a long job).

This supersedes the earlier deliberate choice (documented in a code comment at
`controller.py` `start()`) to defer-but-never-cancel in-flight jobs because
cancellation was terminal and would discard completed diarization work. Preempt
keeps the job alive; only its partial transcription progress is lost and redone.
Record-start and Retry are the two preemption triggers; the live Recording and
the user-initiated Retry always outrank background Post-processing.

## Considered Options

- Front-of-queue Retry without interruption — rejected: a 3-hour job would still
  block the Retry for 3 hours.
- Existing stage-boundary checkpoints — rejected: transcription (the dominant
  cost) is one blocking call with no interior checkpoint, so cancellation during
  it would not take effect until it finished anyway.
