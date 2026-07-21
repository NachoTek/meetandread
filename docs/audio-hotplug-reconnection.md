# Audio Hot-Plug Reconnection Hardware Acceptance

Use this procedure to verify Issue #1 against a real Windows WASAPI device. Automated fake-source tests prove the controller, session-swap, Qt bridge, and toast contracts, but they cannot prove physical device timing, Windows endpoint behavior, or the silence gap in a real WAV file.

Run both the automatic and manual-recovery scenarios. A run is accepted only when every item in [Acceptance checks](#acceptance-checks) passes.

## Hardware setup

### Required equipment

- Windows 11 with the MeetAndRead build under test.
- A removable USB microphone, USB headset, or USB audio interface that Windows exposes as an input endpoint. Use the same physical device throughout a run.
- A timer capable of measuring tenths of a second.
- A continuous, clearly visible test signal for WAV inspection. A steady low-volume tone played near the microphone is preferred; do not use sensitive conversation.
- An audio editor that displays a waveform and selection duration, such as Audacity.

The microphone path is the simplest acceptance target. The same procedure may be repeated for a removable system-audio endpoint, but record the source type in the evidence.

### Prepare the run

1. Connect the test device and confirm that Windows can record from it.
2. Launch MeetAndRead normally (`meetandread.exe`, or `python -m meetandread.main` from a configured source environment).
3. Select only the source under test. Disabling other sources ensures that disconnecting it exercises total source loss rather than degraded dual-source recording.
4. Start recording and let the continuous test signal run for at least 10 seconds. Confirm that the waveform or live captions show that frames are arriving.
5. Start application log capture if it is available. Logs are supporting evidence; the UI and final WAV remain required.
6. Note a sanitized run label and start time. Do not record the Windows endpoint ID, serial number, user profile path, or other device identifier.

## Automatic recovery within five seconds

1. While recording remains active, disconnect the selected device and start the timer when the physical connection is removed.
2. Confirm that MeetAndRead reports the disconnect and leaves recording paused rather than ending the recording. The recovery notification uses the stable recovery-toast slot and remains visible while recovery is pending.
3. Reconnect the **same physical device** before 5.0 seconds have elapsed. Record the measured disconnect-to-reconnect interval.
4. Wait for Windows to make the endpoint available and observe MeetAndRead. Do not click the record control or start a second recording.
5. Confirm that the pending notification is replaced by a `Recording recovered` success notification. The success notification must not retain a `Resume Recording` action and should dismiss after approximately five seconds.
6. Let the test signal continue for at least 10 seconds after recovery, then stop recording normally and wait for WAV finalization.
7. Play the WAV across the disconnect boundary. Confirm that pre-disconnect and post-recovery audio are in one file and that audio resumes without accelerated, duplicated, or corrupted playback.
8. Measure the disconnect-related silence gap using [WAV silence-gap inspection](#wav-silence-gap-inspection).

## Recovery-window expiry and manual resume

Use a new recording for this scenario so its timing and evidence are unambiguous.

1. Repeat the prepared-run steps and allow at least 10 seconds of baseline audio.
2. Disconnect the selected device. Keep it disconnected for at least 6.0 seconds so the five-second automatic recovery window expires.
3. Reconnect the same physical device. Confirm that recording does **not** silently claim automatic recovery.
4. Confirm that MeetAndRead shows a persistent `Recording paused` notification with a `Resume Recording` action.
5. Exercise the unavailable-device failure path once:
   - Disconnect the device again before selecting `Resume Recording`.
   - Select `Resume Recording` and confirm that MeetAndRead reports that the device is still unavailable.
   - Confirm that the paused notification remains persistent and actionable; the failed attempt must not produce a recovery-success notification.
6. Reconnect the same device, wait until Windows exposes it, and select `Resume Recording` again.
7. Confirm that the paused notification is replaced by a `Recording resumed` success notification. The replacement must have no stale action and should dismiss after approximately five seconds.
8. Continue the test signal for at least 10 seconds, then stop recording normally.
9. Confirm that one WAV contains audio from before the loss and after manual resume. Measure and report its silence gap, but do not apply the five-second automatic-gap limit to this intentionally expired scenario.

If manual resume continues to fail after Windows shows the endpoint as available, stop the run, preserve sanitized logs and aggregate diagnostics, and mark the scenario failed. Do not repeatedly start new recordings to turn a failed run into a pass.

## WAV silence-gap inspection

Inspect the finalized WAV locally; do not attach or publish it.

1. Open the WAV from `Documents/MeetAndRead/` in a waveform editor.
2. Locate the last clear test-signal sample before physical disconnect and the first clear test-signal sample after recovery.
3. Select the region between those boundaries and record its duration to the nearest 0.1 second. This is the disconnect-related silence gap.
4. Inspect several seconds on both sides to distinguish the outage from intentional silence or the test signal's fade-in/fade-out.
5. For automatic recovery, accept only a measured gap of **5.0 seconds or less**.
6. For the expired-window run, report the measured gap for reproducibility; it is expected to exceed five seconds because the operator intentionally delays and manually resumes.

A transcription gap alone is not a WAV-gap measurement. Use the waveform, not transcript timestamps, because model processing can add latency unrelated to audio capture.

## Acceptance checks

### Automatic-recovery run

- [ ] The tested source is a real Windows device, not a fake/test source.
- [ ] The same physical device is reconnected in less than 5.0 seconds; the measured interval is recorded.
- [ ] Recording does not stop and the operator does not start another recording.
- [ ] The pending disconnect/recovery notification is replaced by `Recording recovered`.
- [ ] The success notification has no retry action and dismisses after approximately five seconds.
- [ ] Audio from the replacement endpoint resumes in the same finalized WAV.
- [ ] The measured disconnect-related WAV silence gap is 5.0 seconds or less.
- [ ] Playback around the boundary has no acceleration, duplication, or corruption.

### Expired-window manual-recovery run

- [ ] The source remains unavailable for at least 6.0 seconds.
- [ ] MeetAndRead requires manual recovery and shows persistent `Recording paused` feedback with `Resume Recording`.
- [ ] A retry while the endpoint is unavailable does not claim success and leaves an actionable paused state.
- [ ] A retry after reconnection replaces the paused state with `Recording resumed` and clears the stale action.
- [ ] Frames resume without starting a new recording, and one finalized WAV contains both sides of the interruption.
- [ ] The intentionally longer WAV gap is measured and reported.

Any unchecked item is a failed acceptance run. Include the failed observation in Issue #1 rather than attaching sensitive artifacts or substituting an automated fake-source result.

## Evidence to attach

Attach one compact, sanitized result for each scenario:

- Build/version and Windows version.
- Source type (`microphone` or `system audio`) and a generic device class or model only when needed for reproducibility.
- Disconnect-to-reconnect timing and whether the automatic five-second window was met or intentionally exceeded.
- Ordered toast observations: title, persistent versus approximately five-second duration, action label presence, and whether replacement cleared the old action.
- Outcome (`automatic recovered`, `manual retry required`, `manual recovered`, or `failed`).
- Aggregate diagnostics when available: recovery window, active/lost source counts, recovery outcome, total frames received/consumed, and frame/drop counters before and after recovery.
- Final WAV duration, measured silence-gap duration, and a human playback result (`clean`, `accelerated`, `duplicated`, `corrupt`, or another short sanitized description).
- Sanitized warning/error log lines needed to explain a failure.

Do **not** attach raw audio, the WAV file, transcript text, speaker embeddings, secrets, full user paths, Windows endpoint IDs, device serial numbers, or other persistent device identifiers. Redact those values from screenshots and logs. Use aggregate counts and elapsed times rather than audio or transcript excerpts.

### Suggested Issue #1 result

```text
Build / Windows:
Scenario: automatic (<5 s) | expired + manual
Source type / generic hardware class:
Disconnect-to-reconnect interval:
Toast sequence (title, duration, action):
Recovery outcome:
Frames before / after (aggregate):
Frame drops before / after (aggregate):
WAV duration / measured gap:
Playback observation:
Result: PASS | FAIL
Sanitized failure note (if any):
```

A passing fake-source test suite is supporting regression evidence, not a substitute for these real-hardware results.
