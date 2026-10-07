"""Issue Reporter wizard manual-cleanup tests (issue #112).

Fast-lane (ADR 0001): the cleanup step's seams are the same injectable
``input_fn`` / ``print_fn`` / ``data_base`` the wizard already uses —
no real hardware, no real app.

Covers:

- The listing: capture runs appear (newest first, state + bundle
  flags); a data base with no runs skips the step silently.
- Per-run and bulk delete via the scripted console answers; the
  confirm prompt guards; declining deletes nothing.
- Deleting a run removes its capture directory AND its Diagnostics
  Bundle together (the bundle lives inside the directory).
- THE safety canary (AC): deleting runs NEVER touches the user's
  Recordings, Audio, or Transcripts — asserted against a realistic
  storage tree planted next to the data base.
- Path-trickery attempts (absolute paths, traversal, the captures
  root itself, foreign directories) are refused by the pure
  ``delete_capture_runs`` seam.
"""

import sys
from datetime import datetime
from pathlib import Path

import pytest

import meetandread.reporter_wizard as wizard
from meetandread.reporter import (
    delete_capture_runs,
    list_capture_runs,
)


class ScriptedInput:
    """input_fn that replays scripted answers, recording prompts."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []

    def __call__(self, prompt: str = "") -> str:
        self.prompts.append(prompt)
        if not self.answers:
            raise AssertionError(
                f"unexpected extra input prompt: {prompt!r} "
                f"(prompts so far: {self.prompts})"
            )
        return self.answers.pop(0)


class Lines:
    """print_fn collecting output lines."""

    def __init__(self):
        self.lines = []

    def __call__(self, *args, **kwargs):
        self.lines.append(" ".join(str(a) for a in args))

    def text(self) -> str:
        return "\n".join(self.lines)


def _make_run(base: Path, name: str, *, bundle: bool = False) -> Path:
    """Plant a finished capture run (claim + marker + termination)."""
    from meetandread.reporter import RunOutcome, write_termination_record

    d = base / "captures" / name
    d.mkdir(parents=True)
    (d / "capture_run.claim").write_text(
        '{"started_at": "2026-09-09T09:00:00"}\n', encoding="utf-8"
    )
    (d / "capture_complete.marker").write_text(
        '{"finished_at": "2026-09-09T09:05:00"}\n', encoding="utf-8"
    )
    write_termination_record(
        d,
        outcome=RunOutcome.CLEAN_STOP,
        exit_code=0,
        started_at=datetime(2026, 9, 9, 9, 0, 0),
        ended_at=datetime(2026, 9, 9, 9, 5, 0),
        marker_present=True,
    )
    (d / "meetandread_capture_20260909_090000.log").write_text(
        "capture log\n", encoding="utf-8"
    )
    if bundle:
        (d / "diagnostics_bundle.txt").write_text(
            "diagnostics bundle v1\n", encoding="utf-8"
        )
    return d


# ---------------------------------------------------------------------------
# The listing seam
# ---------------------------------------------------------------------------

class TestListCaptureRuns:
    def test_lists_runs_with_state_and_bundle_flag(self, tmp_path):
        with_bundle = _make_run(tmp_path, "run-a", bundle=True)
        without = _make_run(tmp_path, "run-b")
        listings = list_capture_runs(tmp_path)
        by_name = {item.path.name: item for item in listings}
        assert by_name["run-a"].has_bundle is True
        assert by_name["run-b"].has_bundle is False
        assert by_name["run-a"].state_name == "done"
        assert by_name["run-a"].size_bytes > 0
        assert {item.path for item in listings} == {with_bundle, without}

    def test_empty_base_is_empty_listing(self, tmp_path):
        assert list_capture_runs(tmp_path) == []
        assert list_capture_runs(tmp_path / "missing") == []

    def test_foreign_directories_not_listed(self, tmp_path):
        foreign = tmp_path / "captures" / "not-a-run"
        foreign.mkdir(parents=True)
        foreign_file = foreign / "random.txt"
        foreign_file.write_text("x", encoding="utf-8")
        assert list_capture_runs(tmp_path) == []


# ---------------------------------------------------------------------------
# The wizard step (interactive flow at the injectable seams)
# ---------------------------------------------------------------------------

class TestOfferCleanupStep:
    def test_no_runs_skips_step_silently(self, tmp_path):
        inp = ScriptedInput([])
        out = Lines()
        deleted = wizard.offer_cleanup(tmp_path, inp, out)
        assert deleted == []
        assert inp.prompts == []  # never even asked
        assert "capture run(s)" not in out.text()

    def test_bulk_delete_all_runs(self, tmp_path):
        run_a = _make_run(tmp_path, "run-a", bundle=True)
        run_b = _make_run(tmp_path, "run-b")
        inp = ScriptedInput(["a", "y"])  # all; confirm
        out = Lines()
        deleted = wizard.offer_cleanup(tmp_path, inp, out)
        assert {p.name for p in deleted} == {"run-a", "run-b"}
        assert not run_a.exists()
        assert not run_b.exists()
        text = out.text()
        assert "run-a" in text and "run-b" in text
        assert "Deleted 2 capture run(s)" in text
        # The bundle flag was shown.
        assert "[bundle]" in text

    def test_per_run_delete_by_number(self, tmp_path):
        _make_run(tmp_path, "run-a")
        run_b = _make_run(tmp_path, "run-b", bundle=True)
        # Newest-first listing: run-b is entry 1, run-a is entry 2.
        inp = ScriptedInput(["1", "y"])
        out = Lines()
        deleted = wizard.offer_cleanup(tmp_path, inp, out)
        assert [p.name for p in deleted] == ["run-b"]
        assert not run_b.exists()
        assert (tmp_path / "captures" / "run-a").exists()
        # Directory AND bundle (inside it) went together.
        assert not (run_b / "diagnostics_bundle.txt").exists()

    def test_empty_answer_keeps_everything(self, tmp_path):
        run_a = _make_run(tmp_path, "run-a")
        inp = ScriptedInput([""])
        out = Lines()
        deleted = wizard.offer_cleanup(tmp_path, inp, out)
        assert deleted == []
        assert run_a.exists()
        assert "Nothing deleted" in out.text()

    def test_declining_confirm_deletes_nothing(self, tmp_path):
        run_a = _make_run(tmp_path, "run-a")
        inp = ScriptedInput(["a", "n"])  # all; decline confirm
        out = Lines()
        deleted = wizard.offer_cleanup(tmp_path, inp, out)
        assert deleted == []
        assert run_a.exists()
        assert "Nothing deleted." in out.text()

    def test_out_of_range_numbers_delete_nothing(self, tmp_path):
        run_a = _make_run(tmp_path, "run-a")
        inp = ScriptedInput(["9", "y"])
        out = Lines()
        deleted = wizard.offer_cleanup(tmp_path, inp, out)
        assert deleted == []
        assert run_a.exists()

    def test_step_reachable_in_full_wizard_flow_without_capture(
        self, tmp_path
    ):
        """AC: the cleanup step is part of the wizard startup, BEFORE
        any capture is launched — reachable by a user who just wants
        to free space. Declining continues to a normal fresh run."""
        STUB_APP = "import sys; sys.exit(0)"
        _make_run(tmp_path, "run-old")
        inp = ScriptedInput(
            [
                "",  # cleanup: keep all (Enter)
                "A problem",  # describe
                "",  # finish reproducing
                "n",  # decline submit
            ]
        )
        out = Lines()
        run = wizard.run_wizard(
            data_base=tmp_path,
            app_command=[sys.executable, "-c", STUB_APP],
            input_fn=inp,
            print_fn=out,
        )
        assert run is not None  # the fresh flow proceeded
        text = out.text()
        assert "capture run(s) from previous sessions" in text
        assert "Nothing deleted" in text
        assert "describe the problem" in text.lower()
        # The old run survived the decline.
        assert (tmp_path / "captures" / "run-old").exists()

    def test_full_flow_can_delete_then_run(self, tmp_path):
        """Bulk cleanup inside the full wizard flow, then continue."""
        STUB_APP = "import sys; sys.exit(0)"
        old = _make_run(tmp_path, "run-old", bundle=True)
        inp = ScriptedInput(
            [
                "a",  # cleanup: delete all
                "y",  # confirm
                "A problem",
                "",
                "n",
            ]
        )
        out = Lines()
        run = wizard.run_wizard(
            data_base=tmp_path,
            app_command=[sys.executable, "-c", STUB_APP],
            input_fn=inp,
            print_fn=out,
        )
        assert run is not None
        assert not old.exists()
        assert "Deleted 1 capture run(s)" in out.text()
        # The fresh run's directory exists (and was NOT offered).
        assert run.capture_dir.exists()
        assert run.capture_dir != old


# ---------------------------------------------------------------------------
# THE safety canary (AC): never touch Recordings / Audio / Transcripts
# ---------------------------------------------------------------------------

class TestStorageSafetyCanary:
    def _plant_user_tree(self, tmp_path: Path) -> Path:
        """Plant a realistic Documents/meetandread user tree (the
        Library's home: Recordings with Audio WAVs, Transcripts)
        exactly where the real app keeps it — OUTSIDE the reporter's
        data base, like production."""
        docs = tmp_path / "Documents" / "meetandread"
        rec = docs / "recordings"
        rec.mkdir(parents=True)
        (rec / "recording-20260901_100000.wav").write_bytes(
            b"RIFF fake audio data"
        )
        (rec / "recording-20260901_100000.pcm").write_bytes(b"pcm")
        transcripts = docs / "transcripts"
        transcripts.mkdir()
        (transcripts / "recording-20260901_100000.md").write_text(
            "# Transcript\n\nSPEAKER: canary transcript text\n",
            encoding="utf-8",
        )
        return docs

    def test_deleting_runs_never_touches_user_storage(self, tmp_path):
        docs = self._plant_user_tree(tmp_path)
        before = {
            p: p.read_bytes()
            for p in sorted(docs.rglob("*"))
            if p.is_file()
        }
        assert before, "canary tree must be planted"

        run_a = _make_run(tmp_path, "run-a", bundle=True)
        _make_run(tmp_path, "run-b")

        inp = ScriptedInput(["a", "y"])
        out = Lines()
        deleted = wizard.offer_cleanup(tmp_path, inp, out)

        assert len(deleted) == 2
        assert not run_a.exists()
        # THE canary: every byte of the user's storage tree is intact.
        after = {
            p: p.read_bytes()
            for p in sorted(docs.rglob("*"))
            if p.is_file()
        }
        assert after == before
        assert (docs / "recordings" / "recording-20260901_100000.wav").exists()
        assert (
            docs / "transcripts" / "recording-20260901_100000.md"
        ).exists()

    def test_real_storage_tree_untouched_when_marker_used(self, tmp_path):
        """Even with the real_storage_paths marker semantics (the
        conftest sandbox lifted), the cleanup step's delete seam
        cannot escape the data base: plant runs and assert the seam's
        own boundary directly."""
        docs = self._plant_user_tree(tmp_path)
        run = _make_run(tmp_path, "run-a")
        # Attempt to point the delete seam at the user tree.
        deleted = delete_capture_runs(
            [docs / "recordings", docs, run], tmp_path
        )
        assert deleted == [run]
        assert docs.exists()
        assert (docs / "recordings").is_dir()
        assert list(docs.rglob("*.md")), "transcripts must survive"


# ---------------------------------------------------------------------------
# Path-trickery refusal (the pure seam's boundary)
# ---------------------------------------------------------------------------

class TestDeleteSeamBoundary:
    def test_absolute_path_outside_base_refused(self, tmp_path):
        outside = tmp_path / "elsewhere" / "sneaky"
        outside.mkdir(parents=True)
        (outside / "capture_run.claim").write_text("{}", encoding="utf-8")
        assert delete_capture_runs([outside], tmp_path) == []
        assert outside.exists()

    def test_traversal_escape_refused(self, tmp_path):
        captures = tmp_path / "captures"
        captures.mkdir()
        sneaky = captures / ".." / "sneaky-run"
        sneaky.mkdir()
        (sneaky / "capture_run.claim").write_text("{}", encoding="utf-8")
        assert delete_capture_runs([sneaky], tmp_path) == []
        assert sneaky.exists()

    def test_captures_root_itself_refused(self, tmp_path):
        captures = tmp_path / "captures"
        captures.mkdir()
        assert delete_capture_runs([captures], tmp_path) == []
        assert captures.exists()

    def test_foreign_directory_inside_captures_refused(self, tmp_path):
        foreign = tmp_path / "captures" / "foreign"
        foreign.mkdir(parents=True)
        (foreign / "notes.txt").write_text("x", encoding="utf-8")
        assert delete_capture_runs([foreign], tmp_path) == []
        assert foreign.exists()

    def test_empty_target_list_deletes_nothing(self, tmp_path):
        assert delete_capture_runs([], tmp_path) == []

    def test_real_run_inside_captures_deleted(self, tmp_path):
        run = _make_run(tmp_path, "run-real", bundle=True)
        deleted = delete_capture_runs([run], tmp_path)
        assert deleted == [run]
        assert not run.exists()
        assert not (run / "diagnostics_bundle.txt").exists()
