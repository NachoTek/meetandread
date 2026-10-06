"""Diagnostics Bundle assembly, Redaction, and review-screen tests
(issue #108, docs/specs/issue-reporting.md, ADR 0004).

The assembly edge of the capture-directory seam: a PURE function of a
capture directory in any state (complete, or crashed mid-run)
producing the redacted Diagnostics Bundle. Because it is pure, the
entire privacy half of the spec is tested here against fixture
capture directories with zero subprocesses (spec, Testing Decisions;
ADR 0001 fast lane).

Covers:

- Assembly of every constituent (log, Interaction Trace, Resource
  Snapshot series, environment info, termination record) into one
  predictable-format bundle.
- Crash tolerance: no completion marker, torn final log/JSONL records
  — assembly never fails on incompleteness; every valid record before
  the truncation survives.
- Redaction: usernames and home-directory paths, email addresses,
  machine identifiers rewritten across EVERY component.
- Transcript/title exclusion enforced fail-closed: canary-seeded
  fixtures leak into any component → NO submittable artifact, never a
  raw or partially redacted fallback.
- The review screen renders the redacted artifact itself.
"""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

import meetandread.diagnostics_bundle as dbundle
from meetandread.diagnostics_bundle import (
    BUNDLE_FILE_NAME,
    COMPONENT_ORDER,
    IdentifierSet,
    assemble_bundle,
    create_reviewable_bundle,
    redact_text,
    render_review,
)
from meetandread.reporter import write_termination_record, RunOutcome
from meetandread.transcript_canary import canary_ngrams

# A fixed identifier set for deterministic redaction tests.
IDS = IdentifierSet(
    home_dir="C:\\Users\\Bob",
    username="Bob",
    machine_name="BOB-PC",
)


# ---------------------------------------------------------------------------
# Fixture capture-directory builder (zero subprocesses)
# ---------------------------------------------------------------------------


def make_capture(
    tmp_path,
    *,
    name="run-20260910_100000",
    claim=True,
    marker=False,
    log_lines=None,
    torn_log=None,
    trace=None,
    torn_trace=None,
    snapshots=None,
    torn_snapshots=None,
    environment=None,
    termination=None,
    canary_texts=None,
) -> Path:
    """Build a capture directory in any state, from plain data."""
    d = tmp_path / name
    d.mkdir(parents=True)
    if claim:
        (d / "capture_run.claim").write_text(
            '{"started_at": "2026-09-10T10:00:00"}\n', encoding="utf-8"
        )
    if marker:
        (d / "capture_complete.marker").write_text(
            '{"finished_at": "2026-09-10T10:02:30"}\n', encoding="utf-8"
        )
    if log_lines is not None or torn_log is not None:
        parts = list(log_lines or [])
        body = "".join(line + "\n" for line in parts)
        if torn_log is not None:
            body += torn_log  # no trailing newline: torn final record
        (d / "meetandread_capture_20260910_100000.log").write_text(
            body, encoding="utf-8"
        )
    if trace is not None or torn_trace is not None:
        body = "".join(
            json.dumps(event) + "\n" for event in (trace or [])
        )
        if torn_trace is not None:
            body += torn_trace
        (d / "interaction_trace.jsonl").write_text(
            body, encoding="utf-8"
        )
    if snapshots is not None or torn_snapshots is not None:
        body = "".join(
            json.dumps(snap) + "\n" for snap in (snapshots or [])
        )
        if torn_snapshots is not None:
            body += torn_snapshots
        (d / "resource_snapshots.jsonl").write_text(
            body, encoding="utf-8"
        )
    if environment is not None:
        (d / "environment.json").write_text(
            json.dumps(environment) + "\n", encoding="utf-8"
        )
    if termination is not None:
        outcome, exit_code, marker_present = termination
        write_termination_record(
            d,
            outcome=outcome,
            exit_code=exit_code,
            started_at=__import__("datetime").datetime(
                2026, 9, 10, 10, 0, 0
            ),
            ended_at=__import__("datetime").datetime(
                2026, 9, 10, 10, 2, 30
            ),
            marker_present=marker_present,
        )
    if canary_texts:
        grams = set()
        for text in canary_texts:
            grams |= canary_ngrams(text)
        body = "".join(
            json.dumps({"gram": g}) + "\n" for g in sorted(grams)
        )
        (d / "transcript_canary.jsonl").write_text(
            body, encoding="utf-8"
        )
    return d


COMPLETE = dict(
    marker=True,
    log_lines=[
        "2026-09-10 10:00:00 - DEBUG - app_startup: mode=issue_capture",
        "2026-09-10 10:01:00 - INFO - tray_icon_ready:",
    ],
    trace=[
        {"ts": "2026-09-10T10:00:05", "event": "panel_opened",
         "target": "settings_panel"},
        {"ts": "2026-09-10T10:01:10", "event": "button_pressed",
         "target": "record_button"},
    ],
    snapshots=[
        {"ts": "2026-09-10T10:00:00", "ram_percent": 55.5,
         "cpu_percent": 12.25, "available_ram_gb": 8.125,
         "total_ram_gb": 31.9},
        {"ts": "2026-09-10T10:00:02", "ram_percent": 56.0,
         "cpu_percent": 13.0, "available_ram_gb": 8.0,
         "total_ram_gb": 31.9},
    ],
    environment={"app_version": "0.19.1", "os": "Windows",
                 "hardware_class": "medium"},
    termination=(RunOutcome.CLEAN_STOP, 0, True),
)


def make_complete(tmp_path, **overrides) -> Path:
    spec = dict(COMPLETE)
    spec.update(overrides)
    return make_capture(tmp_path, **spec)


# ---------------------------------------------------------------------------
# Defensive boundary
# ---------------------------------------------------------------------------


class TestDefensiveBoundary:
    def test_importing_diagnostics_bundle_pulls_no_heavy_stack(self):
        import os as _os

        code = (
            "import sys, meetandread.diagnostics_bundle; "
            "heavy = sorted(m for m in sys.modules "
            "if m.split('.')[0] in {'PyQt6','sounddevice','numpy',"
            "'psutil','sherpa_onnx','pyaudiowpatch','comtypes',"
            "'webrtcvad','scipy','pywhispercpp'}); "
            "print(heavy)"
        )
        env = dict(_os.environ)
        env["PYTHONPATH"] = str(
            Path(dbundle.__file__).resolve().parent.parent
        )
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == "[]"


# ---------------------------------------------------------------------------
# Redaction (identifier rewriting — pure)
# ---------------------------------------------------------------------------


class TestRedaction:
    def test_home_path_backslash_rewritten(self):
        out = redact_text(
            "streaming diagnostics into C:\\Users\\Bob\\captures", IDS
        )
        assert "Bob" not in out
        assert "<redacted:home>" in out

    def test_home_path_forward_slash_rewritten(self):
        out = redact_text("log dir C:/Users/Bob/captures", IDS)
        assert "Bob" not in out
        assert "<redacted:home>" in out

    def test_home_path_case_insensitive(self):
        out = redact_text("c:\\users\\BOB\\appdata\\local", IDS)
        assert "BOB" not in out
        assert "BoB" not in out

    def test_generic_other_user_home_tree_rewritten(self):
        # A path under ANOTHER user's tree is still a home-directory
        # path with a username in it — rewritten generically.
        out = redact_text("shared from C:\\Users\\Alice\\temp", IDS)
        assert "Alice" not in out
        assert "<redacted:home>" in out

    def test_home_tree_slash_and_case_variants_rewritten(self):
        # Windows paths appear with either separator and any casing.
        for probe in (
            "C:/users/Zoe/temp/file.txt",
            "C:\\USERS\\Zoe\\temp\\file.txt",
            "c:/Users/Zoe",
            "copied from C:/Users/Alice/data",
        ):
            out = redact_text(probe, IDS)
            assert "Zoe" not in out and "Alice" not in out, probe
            assert "<redacted:home>" in out, probe

    def test_posix_home_tree_rewritten(self):
        out = redact_text("/home/alice/run.log and /Users/carol/x", IDS)
        assert "alice" not in out and "carol" not in out

    def test_email_rewritten(self):
        out = redact_text(
            "sent from bob.smith@example.com to Alice@ExAmPlE.org", IDS
        )
        assert "bob.smith@example.com" not in out
        assert "Alice@ExAmPlE.org" not in out
        assert out.count("<redacted:email>") == 2

    def test_machine_name_rewritten(self):
        out = redact_text("hostname BOB-PC resolved (bob-pc)", IDS)
        assert "BOB-PC" not in out and "bob-pc" not in out
        assert "<redacted:machine>" in out

    def test_username_word_rewritten(self):
        out = redact_text("user Bob logged in as BOB", IDS)
        assert "Bob" not in out
        assert out.count("<redacted:user>") == 2

    def test_username_substring_not_rewritten(self):
        # Word-boundary rewriting: "Bobby" and "Bobcat" are different
        # words — only the standalone username is rewritten.
        out = redact_text("Bobby saw a Bobcat", IDS)
        assert "Bobby" in out and "Bobcat" in out

    def test_email_before_username(self):
        # An email whose local part is the username must become ONE
        # email token, not a mangled mix.
        out = redact_text("contact bob@example.com", IDS)
        assert out == "contact <redacted:email>"

    def test_clean_text_unchanged(self):
        text = "2026-09-10 - DEBUG - snapshot_series_persisted: n=1"
        assert redact_text(text, IDS) == text

    def test_empty_and_none_identifiers_ignored(self):
        empty = IdentifierSet(home_dir=None, username="", machine_name="")
        text = "plain prose with no paths or identifiers"
        assert redact_text(text, empty) == text


# ---------------------------------------------------------------------------
# Assembly: complete and crashed directories
# ---------------------------------------------------------------------------


class TestAssembleComplete:
    def test_bundle_file_name_is_contract(self):
        assert BUNDLE_FILE_NAME == "diagnostics_bundle.txt"

    def test_component_order_is_the_documented_five(self):
        assert COMPONENT_ORDER == (
            "environment",
            "termination",
            "interaction_trace",
            "resource_snapshots",
            "debug_log",
        )

    def test_complete_dir_assembles_all_constituents(self, tmp_path):
        d = make_complete(tmp_path)
        result = assemble_bundle(d, identifiers=IDS)
        assert not isinstance(result, dbundle.AssemblyFailure), result
        text = result.text
        assert "app_version" in text and "0.19.1" in text
        assert "clean_stop" in text
        assert "panel_opened" in text and "record_button" in text
        assert "55.5" in text and "31.9" in text
        assert "tray_icon_ready" in text

    def test_sections_appear_in_fixed_order(self, tmp_path):
        d = make_complete(tmp_path)
        text = assemble_bundle(d, identifiers=IDS).text
        positions = [
            text.index("== environment =="),
            text.index("== termination =="),
            text.index("== interaction trace"),
            text.index("== resource snapshots"),
            text.index("== debug log"),
        ]
        assert positions == sorted(positions)

    def test_any_two_reports_read_the_same_way(self, tmp_path):
        d1 = make_complete(tmp_path, name="run-one")
        d2 = make_complete(tmp_path, name="run-two")
        t1 = assemble_bundle(d1, identifiers=IDS).text
        t2 = assemble_bundle(d2, identifiers=IDS).text
        headers1 = [ln for ln in t1.splitlines() if ln.startswith("==")]
        headers2 = [ln for ln in t2.splitlines() if ln.startswith("==")]
        assert headers1 == headers2
        assert t1.split("run: run-one")[0] == t2.split("run: run-two")[0]

    def test_bundle_carries_format_marker(self, tmp_path):
        d = make_complete(tmp_path)
        assert "diagnostics bundle v1" in assemble_bundle(
            d, identifiers=IDS
        ).text


class TestAssembleCrashed:
    def test_crashed_dir_still_assembles(self, tmp_path):
        d = make_capture(
            tmp_path,
            log_lines=COMPLETE["log_lines"],
            torn_log="2026-09-10 10:02:29 - DEBUG - snapshot_persist",
            trace=COMPLETE["trace"],
            torn_trace='{"ts": "2026-09-10T10:02:29", "event": "panel_',
            snapshots=COMPLETE["snapshots"],
            torn_snapshots='{"ts": "2026-09-10T10:02:29", "ram_pe',
            environment=COMPLETE["environment"],
            termination=(RunOutcome.CRASH, 0xC0000005, False),
        )
        result = assemble_bundle(d, identifiers=IDS)
        assert not isinstance(result, dbundle.AssemblyFailure), result
        text = result.text
        assert "crash" in text and "3221225477" in text
        # Every valid record before the truncation is preserved...
        assert "tray_icon_ready" in text
        assert "record_button" in text
        assert "55.5" in text
        # ...and the torn tails are dropped, not carried.
        assert "snapshot_persist" not in text
        assert "10:02:29" not in text  # every torn fragment's ts

    def test_markerless_run_reports_absent_marker(self, tmp_path):
        d = make_complete(tmp_path, marker=False,
                          termination=(RunOutcome.CRASH, 1, False))
        text = assemble_bundle(d, identifiers=IDS).text
        assert "completion marker: absent" in text

    def test_missing_artifacts_assemble_with_absent_markers(
        self, tmp_path
    ):
        # Startup crash: claimed dir, nothing else ever written.
        d = make_capture(tmp_path, log_lines=[], trace=[],
                         snapshots=[])
        result = assemble_bundle(d, identifiers=IDS)
        assert not isinstance(result, dbundle.AssemblyFailure), result
        assert "(not captured)" in result.text

    def test_assembly_never_fails_on_incompleteness(self, tmp_path):
        # The emptiest possible claimed directory still assembles.
        d = make_capture(tmp_path)
        result = assemble_bundle(d, identifiers=IDS)
        assert not isinstance(result, dbundle.AssemblyFailure), result


# ---------------------------------------------------------------------------
# Redaction across EVERY component
# ---------------------------------------------------------------------------


class TestRedactionAcrossComponents:
    def test_log_identifiers_redacted(self, tmp_path):
        d = make_complete(
            tmp_path,
            log_lines=[
                "2026-09-10 - DEBUG - opening C:\\Users\\Bob\\Documents",
                "2026-09-10 - DEBUG - machine BOB-PC user Bob ok",
                "2026-09-10 - DEBUG - mail bob@example.com failed",
            ],
        )
        text = assemble_bundle(d, identifiers=IDS).text
        assert "Bob" not in text.replace("<redacted", "")
        assert "BOB-PC" not in text
        assert "bob@example.com" not in text

    def test_trace_identifiers_redacted(self, tmp_path):
        d = make_complete(
            tmp_path,
            trace=[
                {"ts": "2026-09-10T10:00:05", "event": "panel_opened",
                 "target": "C:\\Users\\Bob\\panel"},
            ],
        )
        text = assemble_bundle(d, identifiers=IDS).text
        assert "Bob" not in text

    def test_termination_identifiers_redacted(self, tmp_path):
        d = make_complete(tmp_path)
        (d / "termination.json").write_text(
            json.dumps(
                {
                    "outcome": "crash",
                    "exit_code": 1,
                    "started_at": "2026-09-10T10:00:00",
                    "ended_at": "2026-09-10T10:02:30",
                    "marker_present": False,
                    "note": "died on BOB-PC for bob@example.com",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        text = assemble_bundle(d, identifiers=IDS).text
        assert "BOB-PC" not in text
        assert "bob@example.com" not in text

    def test_environment_identifiers_redacted(self, tmp_path):
        d = make_complete(
            tmp_path,
            environment={
                "app_version": "0.19.1",
                "os": "Windows on BOB-PC",
                "hardware_class": "medium",
            },
        )
        text = assemble_bundle(d, identifiers=IDS).text
        assert "BOB-PC" not in text

    def test_snapshots_scanned_too(self, tmp_path):
        # Snapshots are numeric by shape, but the check runs over
        # every component regardless — a poisoned fixture proves the
        # scan covers them.
        d = make_complete(
            tmp_path,
            canary_texts=["possum circuit breaker yacht marinade"],
            snapshots=[
                {"ts": "possum circuit breaker yacht marinade now",
                 "ram_percent": 55.5, "cpu_percent": 1.0,
                 "available_ram_gb": 8.0, "total_ram_gb": 31.9},
            ],
        )
        result = assemble_bundle(d, identifiers=IDS)
        assert isinstance(result, dbundle.AssemblyFailure)
        assert result.reason == "canary_leak"
        assert result.component == "resource_snapshots"


# ---------------------------------------------------------------------------
# Fail-closed: canary leaks and redaction failures
# ---------------------------------------------------------------------------

SECRET = "Sebastopol canary spoken words recorded"


class TestFailClosedCanary:
    def test_canary_leak_in_log_blocks_artifact(self, tmp_path):
        d = make_complete(
            tmp_path,
            canary_texts=[SECRET],
            log_lines=COMPLETE["log_lines"]
            + [f"transcribe failed near {SECRET} (chunk 3)"],
        )
        result = create_reviewable_bundle(d, identifiers=IDS)
        assert isinstance(result, dbundle.AssemblyFailure)
        assert result.reason == "canary_leak"
        assert result.component == "debug_log"
        # Fail-closed: NO submittable artifact exists.
        assert not (d / BUNDLE_FILE_NAME).exists()

    def test_canary_leak_in_trace_blocks_artifact(self, tmp_path):
        d = make_complete(
            tmp_path,
            canary_texts=[SECRET],
            trace=COMPLETE["trace"]
            + [
                {"ts": "2026-09-10T10:00:05", "event": "panel_opened",
                 "target": SECRET},
            ],
        )
        result = create_reviewable_bundle(d, identifiers=IDS)
        assert isinstance(result, dbundle.AssemblyFailure)
        assert result.component == "interaction_trace"
        assert not (d / BUNDLE_FILE_NAME).exists()

    def test_canary_leak_in_termination_blocks_artifact(self, tmp_path):
        d = make_complete(tmp_path, canary_texts=[SECRET])
        (d / "termination.json").write_text(
            json.dumps(
                {
                    "outcome": "crash",
                    "exit_code": 1,
                    "started_at": "2026-09-10T10:00:00",
                    "ended_at": None,
                    "marker_present": False,
                    "detail": f"died transcribing {SECRET}",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        result = create_reviewable_bundle(d, identifiers=IDS)
        assert isinstance(result, dbundle.AssemblyFailure)
        assert result.component == "termination"
        assert not (d / BUNDLE_FILE_NAME).exists()

    def test_canary_leak_in_environment_blocks_artifact(self, tmp_path):
        d = make_complete(
            tmp_path,
            canary_texts=[SECRET],
            environment={
                "app_version": "0.19.1",
                "os": SECRET,
                "hardware_class": "medium",
            },
        )
        result = create_reviewable_bundle(d, identifiers=IDS)
        assert isinstance(result, dbundle.AssemblyFailure)
        assert result.component == "environment"
        assert not (d / BUNDLE_FILE_NAME).exists()

    def test_canary_leak_in_description_blocks_artifact(self, tmp_path):
        # A pasted transcript fragment in the user's own description
        # is a leak like any other — checked, fail-closed.
        d = make_complete(tmp_path, canary_texts=[SECRET])
        (d / "description.txt").write_text(
            f"it printed {SECRET} on screen\n", encoding="utf-8"
        )
        result = create_reviewable_bundle(d, identifiers=IDS)
        assert isinstance(result, dbundle.AssemblyFailure)
        assert result.reason == "canary_leak"
        assert result.component == "description"
        assert not (d / BUNDLE_FILE_NAME).exists()

    def test_recording_title_canary_blocks_artifact(self, tmp_path):
        title = "board meeting confidential merger talks audio"
        d = make_complete(
            tmp_path,
            canary_texts=[title],
            log_lines=COMPLETE["log_lines"]
            + [f"rename finished: {title}"],
        )
        result = create_reviewable_bundle(d, identifiers=IDS)
        assert isinstance(result, dbundle.AssemblyFailure)
        assert result.reason == "canary_leak"
        assert not (d / BUNDLE_FILE_NAME).exists()

    def test_clean_canary_registry_does_not_block(self, tmp_path):
        d = make_complete(tmp_path, canary_texts=[SECRET])
        result = create_reviewable_bundle(d, identifiers=IDS)
        assert not isinstance(result, dbundle.AssemblyFailure), result
        assert (d / BUNDLE_FILE_NAME).exists()

    def test_failure_detail_never_contains_the_leak(self, tmp_path):
        d = make_complete(
            tmp_path,
            canary_texts=[SECRET],
            log_lines=[f"leak {SECRET} here"],
        )
        result = create_reviewable_bundle(d, identifiers=IDS)
        assert isinstance(result, dbundle.AssemblyFailure)
        assert SECRET not in result.detail
        assert "Sebastopol" not in result.detail


class TestFailClosedRedaction:
    def test_unrewritable_identifier_fails_closed(self, tmp_path):
        # A username that embeds itself in every redaction token can
        # never be scrubbed — assembly must fail closed, not ship a
        # partially redacted artifact.
        poisoned = IdentifierSet(
            home_dir="C:\\Users\\redacted",
            username="redacted",
            machine_name="redacted",
        )
        d = make_complete(
            tmp_path,
            log_lines=["user redacted logged in"],
        )
        result = create_reviewable_bundle(d, identifiers=poisoned)
        assert isinstance(result, dbundle.AssemblyFailure)
        assert result.reason == "redaction_incomplete"
        assert not (d / BUNDLE_FILE_NAME).exists()

    def test_assembly_error_fails_closed(self, tmp_path):
        d = make_complete(tmp_path)
        with patch.object(
            dbundle,
            "read_appendable_records",
            side_effect=OSError("disk unreadable"),
        ):
            result = create_reviewable_bundle(d, identifiers=IDS)
        assert isinstance(result, dbundle.AssemblyFailure)
        assert result.reason == "assembly_error"
        assert not (d / BUNDLE_FILE_NAME).exists()


# ---------------------------------------------------------------------------
# create_reviewable_bundle: the artifact write
# ---------------------------------------------------------------------------


class TestCreateReviewableBundle:
    def test_writes_bundle_into_capture_dir(self, tmp_path):
        d = make_complete(tmp_path)
        result = create_reviewable_bundle(d, identifiers=IDS)
        assert not isinstance(result, dbundle.AssemblyFailure), result
        assert result.path == d / BUNDLE_FILE_NAME
        assert result.path.exists()
        assert result.path.read_text(encoding="utf-8") == result.text

    def test_written_artifact_is_redacted(self, tmp_path):
        d = make_complete(
            tmp_path,
            log_lines=[
                "2026-09-10 - DEBUG - home C:\\Users\\Bob\\Documents",
            ],
        )
        result = create_reviewable_bundle(d, identifiers=IDS)
        assert not isinstance(result, dbundle.AssemblyFailure), result
        written = result.path.read_text(encoding="utf-8")
        assert "Bob" not in written
        assert "<redacted:home>" in written

    def test_bundle_path_never_enters_the_bundle(self, tmp_path):
        # The artifact must not embed its own absolute path (that
        # would re-leak the username inside the path).
        d = make_complete(tmp_path)
        result = create_reviewable_bundle(d, identifiers=IDS)
        assert not isinstance(result, dbundle.AssemblyFailure), result
        assert str(d) not in result.text


# ---------------------------------------------------------------------------
# Review screen: "here's what will be sent" — the redacted artifact
# ---------------------------------------------------------------------------


class TestReviewScreen:
    def _bundle(self, tmp_path):
        d = make_complete(
            tmp_path,
            log_lines=[
                "2026-09-10 - DEBUG - opening C:\\Users\\Bob\\data",
            ],
        )
        result = assemble_bundle(d, identifiers=IDS)
        assert not isinstance(result, dbundle.AssemblyFailure), result
        return d, result

    def test_review_lists_every_constituent(self, tmp_path):
        _, bundle = self._bundle(tmp_path)
        review = render_review(bundle)
        for section in COMPONENT_ORDER:
            assert section.replace("_", " ") in review.lower()

    def test_review_shows_redacted_artifact_content(self, tmp_path):
        _, bundle = self._bundle(tmp_path)
        review = render_review(bundle)
        assert "<redacted:home>" in review  # the artifact itself
        assert "Bob" not in review

    def test_review_states_privacy_guarantees(self, tmp_path):
        _, bundle = self._bundle(tmp_path)
        review = render_review(bundle)
        lowered = review.lower()
        assert "no audio" in lowered
        assert "no transcript" in lowered
        assert "redacted" in lowered

    def test_review_reports_run_outcome(self, tmp_path):
        _, bundle = self._bundle(tmp_path)
        assert "clean_stop" in render_review(bundle)

    def test_review_shows_description_when_present(self, tmp_path):
        d, _ = self._bundle(tmp_path)
        (d / "description.txt").write_text(
            "Settings panel froze the app\n", encoding="utf-8"
        )
        bundle = assemble_bundle(d, identifiers=IDS)
        assert not isinstance(bundle, dbundle.AssemblyFailure)
        review = render_review(bundle)
        assert "Settings panel froze the app" in review

    def test_review_works_without_description(self, tmp_path):
        _, bundle = self._bundle(tmp_path)
        # No description section without a description — the review
        # still renders every other constituent.
        review = render_review(bundle)
        assert "environment" in review.lower()
        assert "REVIEW" in review


# ---------------------------------------------------------------------------
# Default identifier gathering (reporter-side, injectable)
# ---------------------------------------------------------------------------


class TestDefaultIdentifiers:
    def test_gathers_from_environment(self, monkeypatch):
        monkeypatch.setenv("COMPUTERNAME", "TEST-PC-42")
        monkeypatch.setenv("USERNAME", "testuser")
        ids = dbundle.default_identifiers()
        assert ids.machine_name == "TEST-PC-42"
        assert ids.username == "testuser"
        assert ids.home_dir  # something non-empty

    def test_short_values_are_dropped(self, monkeypatch):
        monkeypatch.setenv("USERNAME", "x")
        ids = dbundle.default_identifiers()
        assert ids.username is None
