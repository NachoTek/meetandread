"""Manual Submission tests (issue #109,
docs/specs/issue-reporting.md "Submission", ADR 0004).

The submission edge of the capture-directory seam: a PURE function of
a capture directory in any state producing the prefilled New Issue
URL and the clipboard text (spec, Testing Decisions §3) —" plus the
wizard's submit step built on it. Because the core is pure, the
privacy half of submission is tested here against fixture capture
directories with zero subprocesses (ADR 0001 fast lane).

Covers:

- The prefilled issue: title and body built from the user's own
  description; the body carries the NEUTRAL bundle filename and
  attach-by-hand instructions and NEVER a local filesystem path (the
  issue body is public —" a path would disclose the home directory).
- The URL: the repository's New Issue form, title/body riding as
  properly encoded query params.
- The clipboard text: the FULL local path of the written bundle.
- Body size: GitHub caps the prefilled body, so an over-long
  description is truncated with an explicit marker (never a 414).
- The wizard's submit step: browser open + clipboard copy happen at
  the step (and ONLY there —" the browser open is the single outbound
  action, ADR 0004), the submittable file on disk is byte-identical
  to the reviewed artifact, declining the submit keeps everything on
  disk with nothing opened.
- Fail-closed review: no bundle artifact → no submit step at all.
- Defensive boundary: importing the module pulls no heavy stack.
"""

import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from typing import Optional

import meetandread.diagnostics_bundle as dbundle
import meetandread.manual_submission as msub
from meetandread.diagnostics_bundle import (
    BUNDLE_FILE_NAME,
    IdentifierSet,
    create_reviewable_bundle,
)
from meetandread.reporter import RunOutcome, write_description

# The fixture-bundle builder shared with the #108 assembly tests —"
# submission tests consume the same fixture shape (the submission
# edge is the tail of the assembly edge).
from tests.test_diagnostics_bundle import make_capture, make_complete


# A fixed identifier set so description redaction is deterministic.
IDS = IdentifierSet(
    home_dir="C:\\Users\\Bob",
    username="Bob",
    machine_name="BOB-PC",
)


def _draft(capture_dir):
    """build_submission with the success path narrowed (the happy
    tests only run against reviewed fixtures) and the fixed test
    identifier set (deterministic redaction)."""
    result = msub.build_submission(capture_dir, identifiers=IDS)
    assert isinstance(result, msub.SubmissionDraft), str(result)
    return result


def _reviewed(
    tmp_path, description: Optional[str] = "Settings panel froze"
):
    """A fixture capture directory WITH its written bundle artifact —
    the state the wizard's submit step starts from (review passed)."""
    d = make_complete(tmp_path)
    if description is not None:
        write_description(d, description)
    result = create_reviewable_bundle(d, identifiers=IDS)
    assert not hasattr(result, "reason"), (
        f"fixture failed to assemble: {result}"
    )
    return d


def _stub_wizard_seams(monkeypatch, opened, copied):
    """Point the submission IO seams at capture lists (the browser
    must never open and the clipboard must never receive the path
    during tests). The wizard resolves these at call time, so
    patching the module attributes is enough."""
    monkeypatch.setattr(
        msub, "open_new_issue_form",
        lambda url: opened.append(url) or True,
    )
    monkeypatch.setattr(
        msub, "copy_to_clipboard",
        lambda text: copied.append(text) or True,
    )


# ---------------------------------------------------------------------------
# The prefilled issue (title/body/URL/clipboard —" pure)
# ---------------------------------------------------------------------------


class TestPrefilledIssue:
    def test_title_comes_from_the_description(self, tmp_path):
        d = _reviewed(tmp_path, "Settings panel froze the app")
        draft = _draft(d)
        assert draft.title == "Settings panel froze the app"

    def test_description_fallback_title_when_none(self, tmp_path):
        d = _reviewed(tmp_path, description=None)  # no description
        draft = _draft(d)
        assert draft.title == msub.FALLBACK_TITLE

    def test_body_carries_description_and_attach_instructions(
        self, tmp_path
    ):
        d = _reviewed(tmp_path, "Settings panel froze the app")
        draft = _draft(d)
        assert "Settings panel froze the app" in draft.body
        # The attach-by-hand instruction is plain and explicit.
        assert BUNDLE_FILE_NAME in draft.body
        assert "attach" in draft.body.lower()
        assert "diagnostics_bundle.txt" in draft.body

    def test_body_never_carries_a_local_path(self, tmp_path):
        # THE privacy line (amended spec): the public prefilled body
        # must not disclose the local filesystem path.
        d = _reviewed(
            tmp_path,
            "Crashed while saving under C:\\Users\\Bob\\Documents",
        )
        draft = _draft(d)
        self._assert_no_local_paths(draft)

    def test_url_points_at_repo_new_issue_form(self, tmp_path):
        d = _reviewed(tmp_path)
        draft = _draft(d)
        parts = urlsplit(draft.url)
        assert parts.scheme == "https"
        assert parts.netloc == "github.com"
        assert parts.path == "/NachoTek/meetandread/issues/new"

    def test_url_carries_title_and_body_as_query_params(self, tmp_path):
        d = _reviewed(tmp_path, "Settings panel froze the app")
        draft = _draft(d)
        query = parse_qs(urlsplit(draft.url).query)
        assert query["title"] == [draft.title]
        assert query["body"] == [draft.body]
        # The URL is fully encoded: no raw spaces/newlines ride along.
        assert " " not in urlsplit(draft.url).query
        assert "\n" not in draft.url

    def test_clipboard_text_is_the_full_local_path(self, tmp_path):
        d = _reviewed(tmp_path)
        draft = _draft(d)
        assert draft.clipboard_text == str(
            d / BUNDLE_FILE_NAME
        )

    def test_overlong_description_is_truncated_with_marker(
        self, tmp_path
    ):
        long_desc = "freeze " * 2000  # way past the URL budget
        d = _reviewed(tmp_path, long_desc)
        draft = _draft(d)
        assert msub.TRUNCATION_MARKER in draft.title
        assert len(draft.title) <= msub.MAX_TITLE_LEN
        assert msub.TRUNCATION_MARKER in draft.body
        assert len(draft.body) <= msub.MAX_BODY_LEN
        # The truncation keeps the head of the user's words.
        assert draft.title.startswith("freeze freeze")
        # THE attach contract survives truncation: an over-long
        # description shrinks, but the neutral filename and the
        # attach-by-hand instructions always reach the prefilled
        # body (a cut tail must never delete them).
        assert BUNDLE_FILE_NAME in draft.body
        assert "attach" in draft.body.lower()

    def test_multiline_description_flattens_into_title(self, tmp_path):
        d = _reviewed(tmp_path, "line one\nline two\nline three")
        draft = _draft(d)
        assert "\n" not in draft.title
        assert "line one" in draft.title

    @staticmethod
    def _assert_no_local_paths(draft):
        lowered = (draft.title + "\n" + draft.body).lower()
        for needle in (
            "c:\\users",
            "c:/users",
            "/home/",
            "/users/",
            str(Path.home()).lower(),
        ):
            assert needle not in lowered, (
                f"local path fragment {needle!r} in the public "
                "prefilled issue"
            )


class TestRedactedDescription:
    def test_identifiers_in_description_do_not_reach_the_issue(
        self, tmp_path
    ):
        # The user TYPED their own path into the description —" the
        # public issue body must carry it redacted, same as the bundle
        # (the description line is part of both).
        d = _reviewed(
            tmp_path,
            "Crashes when saving to C:\\Users\\Bob\\Documents\\x "
            "on machine BOB-PC",
        )
        draft = _draft(d)
        text = draft.title + "\n" + draft.body
        assert "C:\\Users\\Bob" not in text
        assert "BOB-PC" not in text
        assert "<redacted:home>" in text or "<redacted:machine>" in text


class TestBuildSubmissionFailClosed:
    def test_missing_bundle_artifact_fails_closed(self, tmp_path):
        # Review never wrote the artifact (the fail-closed path), so
        # there is nothing submittable on disk: submission must not
        # offer a URL pointing at a nonexistent file.
        d = make_complete(tmp_path)
        result = msub.build_submission(d)
        assert isinstance(result, msub.SubmissionUnavailable)
        assert "diagnostics_bundle.txt" not in result.detail

    def test_unreviewed_capture_dir_fails_closed(self, tmp_path):
        # A directory in ANY state —" a nearly-empty crashed run
        # without even a termination record still fails closed
        # (no bundle artifact exists to attach).
        result = msub.build_submission(make_capture(tmp_path))
        assert isinstance(result, msub.SubmissionUnavailable)


# ---------------------------------------------------------------------------
# IO seams (browser open, clipboard copy —" injectable, default-off in tests)
# ---------------------------------------------------------------------------


class TestIOSeams:
    def test_open_new_issue_form_uses_webbrowser(self, monkeypatch):
        opened = []
        monkeypatch.setattr(
            msub.webbrowser, "open", lambda url, **k: opened.append(url)
            or True,
        )
        assert msub.open_new_issue_form("https://example/x") is True
        assert opened == ["https://example/x"]

    def test_open_new_issue_form_reports_failure(self, monkeypatch):
        monkeypatch.setattr(
            msub.webbrowser, "open", lambda url, **k: False
        )
        assert msub.open_new_issue_form("https://example/x") is False


# ---------------------------------------------------------------------------
# The wizard's submit step
# ---------------------------------------------------------------------------


class TestSubmitStep:
    def _run_wizard_to_submit(
        self, tmp_path, submit_answer, monkeypatch=None
    ):
        from tests.test_reporter_wizard import (
            Lines,
            ScriptedInput,
            STUB_APP_CLEAN,
            _stub_command,
        )

        import meetandread.reporter_wizard as wizard

        opened, copied = [], []
        _stub_wizard_seams(monkeypatch, opened, copied)

        inp = ScriptedInput(
            [
                "Settings panel froze the app",
                "",  # Press Enter to finish reproducing
                submit_answer,  # Open the issue form now? [Y/n]
            ]
        )
        out = Lines()
        run = wizard.run_wizard(
            data_base=tmp_path,
            app_command=_stub_command(STUB_APP_CLEAN),
            input_fn=inp,
            print_fn=out,
        )
        return run, out.text(), opened, copied

    def test_submit_step_opens_browser_and_copies_path(
        self, tmp_path, monkeypatch
    ):
        run, text, opened, copied = self._run_wizard_to_submit(
            tmp_path, "y", monkeypatch
        )
        assert run is not None
        # The browser opened on the prefilled New Issue form.
        assert len(opened) == 1
        assert opened[0].startswith(
            "https://github.com/NachoTek/meetandread/issues/new?"
        )
        query = parse_qs(urlsplit(opened[0]).query)
        assert query["title"] == ["Settings panel froze the app"]
        # The clipboard carries the full local path.
        assert copied == [
            str(run.capture_dir / BUNDLE_FILE_NAME)
        ]
        # The screen told the user to attach by hand.
        assert "attach" in text.lower()
        assert "clipboard" in text.lower()

    def test_declining_still_copies_path_but_opens_nothing(
        self, tmp_path, monkeypatch
    ):
        run, text, opened, copied = self._run_wizard_to_submit(
            tmp_path, "n", monkeypatch
        )
        assert run is not None
        assert opened == []
        # The clipboard copy is local and needs no approval: the
        # path is on the clipboard on EVERY branch, so attaching by
        # hand stays one paste away even without the browser.
        assert copied == [
            str(run.capture_dir / BUNDLE_FILE_NAME)
        ]
        # Everything the user needs stays on screen and on disk.
        assert (run.capture_dir / BUNDLE_FILE_NAME).is_file()
        assert BUNDLE_FILE_NAME in text

    def test_submittable_file_is_byte_identical_to_reviewed_artifact(
        self, tmp_path, monkeypatch
    ):
        run, _text, _opened, _copied = self._run_wizard_to_submit(
            tmp_path, "y", monkeypatch
        )
        assert run is not None
        # Snapshot the on-disk bytes the submit step offered BEFORE
        # any re-assembly — then re-assemble and require the fresh
        # artifact to match those exact bytes (the submit step never
        # rewrote the file the user reviewed).
        bundle_path = run.capture_dir / BUNDLE_FILE_NAME
        on_disk = bundle_path.read_bytes()
        reviewed = create_reviewable_bundle(
            run.capture_dir, identifiers=IDS
        )
        assert not hasattr(reviewed, "reason")
        assert on_disk == reviewed.text.encode(  # type: ignore[union-attr]
            "utf-8"
        )

    def test_no_submit_step_when_review_failed_closed(
        self, tmp_path, monkeypatch
    ):
        from tests.test_reporter_wizard import (
            Lines,
            ScriptedInput,
            STUB_APP_CLEAN,
            _stub_command,
        )

        import meetandread.reporter_wizard as wizard

        opened, copied = [], []
        _stub_wizard_seams(monkeypatch, opened, copied)

        # Poison the assembly: a canary leak fails the review closed
        # (same leaky seam as the #108 wizard tests).
        real_create = dbundle.create_reviewable_bundle

        def leaky_create(capture_dir, identifiers=None):
            log = (
                capture_dir
                / "meetandread_capture_20260910_100000.log"
            )
            if not log.exists():
                log.write_text(
                    "leak: Sebastopol canary spoken words recorded\n",
                    encoding="utf-8",
                )
                from meetandread.transcript_canary import canary_ngrams

                body = "".join(
                    __import__("json").dumps({"gram": g}) + "\n"
                    for g in sorted(
                        canary_ngrams(
                            "Sebastopol canary spoken words recorded"
                        )
                    )
                )
                (
                    capture_dir / "transcript_canary.jsonl"
                ).write_text(body, encoding="utf-8")
            return real_create(capture_dir, identifiers=identifiers)

        monkeypatch.setattr(
            dbundle, "create_reviewable_bundle", leaky_create
        )
        inp = ScriptedInput(["desc", ""])
        out = Lines()
        run = wizard.run_wizard(
            data_base=tmp_path,
            app_command=_stub_command(STUB_APP_CLEAN),
            input_fn=inp,
            print_fn=out,
        )
        assert run is not None
        assert not (run.capture_dir / BUNDLE_FILE_NAME).exists()
        assert "UNAVAILABLE" in out.text()
        assert opened == []  # the browser never opened


# ---------------------------------------------------------------------------
# Defensive boundary
# ---------------------------------------------------------------------------


class TestDefensiveBoundary:
    def test_importing_manual_submission_pulls_no_heavy_stack(self):
        code = (
            "import sys, meetandread.manual_submission; "
            "heavy = sorted(m for m in sys.modules "
            "if m.split('.')[0] in {'PyQt6','sounddevice','numpy',"
            "'psutil','sherpa_onnx','pyaudiowpatch','comtypes',"
            "'webrtcvad','scipy','pywhispercpp'}); "
            "print(heavy)"
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = str(
            Path(msub.__file__).resolve().parent.parent
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
