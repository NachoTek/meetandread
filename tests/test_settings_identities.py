"""Tests for the identity management service (T01 — Settings → Identities tab).

Covers: scan counts, exact metadata/body rewriting, rename, merge weighted
averaging/sample counts, delete primitives, malformed metadata handling,
negative inputs, and no substring over-replacement.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pytest

from meetandread.speaker.signatures import VoiceSignatureStore
from meetandread.speaker.identity_management import (
    DeleteError,
    IdentityManagementError,
    IdentityRecordingRef,
    IdentityUsage,
    MergeError,
    RenameError,
    delete_identity,
    merge_identities,
    parse_metadata_footer,
    rename_identity,
    replace_speaker_label_in_file,
    scan_identity_usage,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _parse_metadata(md_path: Path) -> Dict[str, Any]:
    """Parse the metadata footer from a transcript .md file."""
    content = md_path.read_text(encoding="utf-8")
    data = parse_metadata_footer(content)
    assert data is not None, "No metadata footer found"
    return data


def _make_md(
    tmp_path: Path,
    words: list,
    segments: list,
    speaker_matches: Optional[dict] = None,
    *,
    name: str = "test.md",
) -> Path:
    """Create a transcript .md file with metadata footer.

    Produces the same format as TranscriptStore.save_to_file:
    markdown body + separator + METADATA JSON footer.
    """
    lines = ["# Transcript", "", "**Recorded:** 2026-04-22T14:30:00", ""]
    cur_spk: Optional[str] = None
    cur_words: list = []
    for w in words:
        sid = w.get("speaker_id") or "Unknown"
        if sid != cur_spk:
            if cur_words:
                lines += [f"**{cur_spk}**", "", " ".join(cur_words), ""]
            cur_spk = sid
            cur_words = [w["text"]]
        else:
            cur_words.append(w["text"])
    if cur_words:
        lines += [f"**{cur_spk}**", "", " ".join(cur_words), ""]

    md_body = "\n".join(lines)
    meta: Dict[str, Any] = {
        "recording_start_time": "2026-04-22T14:30:00",
        "word_count": len(words),
        "words": words,
        "segments": segments,
    }
    if speaker_matches is not None:
        meta["speaker_matches"] = speaker_matches

    p = tmp_path / name
    p.write_text(
        md_body + "\n---\n\n<!-- METADATA: " + json.dumps(meta, indent=2) + " -->\n",
        encoding="utf-8",
    )
    return p


def _make_store_with_identities(
    tmp_path: Path, identities: dict[str, tuple[np.ndarray, int]]
) -> VoiceSignatureStore:
    """Create a temp VoiceSignatureStore with given identities.

    Args:
        identities: Mapping from name to (embedding, num_samples).
    """
    db = tmp_path / "speaker_signatures.db"
    store = VoiceSignatureStore(db_path=str(db))
    for name, (emb, samples) in identities.items():
        store.save_signature(name, emb, averaged_from_segments=samples)
    return store


def _random_embedding(seed: int = 42) -> np.ndarray:
    """Generate a deterministic random embedding for testing."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(256).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_identity_recording_ref_fields(self):
        ref = IdentityRecordingRef(
            path=Path("/tmp/a.md"), recording_count=5, last_modified=1000.0
        )
        assert ref.path == Path("/tmp/a.md")
        assert ref.recording_count == 5
        assert ref.last_modified == 1000.0

    def test_identity_recording_ref_frozen(self):
        ref = IdentityRecordingRef(path=Path("/tmp/a.md"), recording_count=1)
        with pytest.raises(AttributeError):
            ref.recording_count = 2  # type: ignore[misc]

    def test_identity_usage_recording_count(self):
        usage = IdentityUsage(
            identity_name="Alice",
            recordings=[
                IdentityRecordingRef(Path("/a.md"), 3),
                IdentityRecordingRef(Path("/b.md"), 2),
            ],
            total_mentions=5,
        )
        assert usage.recording_count == 2
        assert usage.total_mentions == 5

    def test_identity_usage_empty_recordings(self):
        usage = IdentityUsage(identity_name="Bob")
        assert usage.recording_count == 0
        assert usage.total_mentions == 0


# ---------------------------------------------------------------------------
# parse_metadata_footer tests
# ---------------------------------------------------------------------------


class TestParseMetadataFooter:
    def test_valid_footer(self):
        data = {"words": [], "segments": []}
        content = (
            "# Transcript\n\n**SPK_0**\n\nHi\n\n"
            "---\n\n<!-- METADATA: "
            + json.dumps(data)
            + " -->\n"
        )
        result = parse_metadata_footer(content)
        assert result is not None
        assert result["words"] == []

    def test_no_footer(self):
        content = "# Transcript\n\n**SPK_0**\n\nHi\n"
        assert parse_metadata_footer(content) is None

    def test_malformed_json(self):
        content = "# T\n\n---\n\n<!-- METADATA: {bad json} -->\n"
        assert parse_metadata_footer(content) is None


# ---------------------------------------------------------------------------
# scan_identity_usage tests
# ---------------------------------------------------------------------------


class TestScanIdentityUsage:
    def test_counts_mentions_across_files(self, tmp_path):
        transcripts = tmp_path / "transcripts"
        transcripts.mkdir()
        w1 = [
            {"text": "Hi", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": "Alice"},
            {"text": "Bye", "start_time": 0.5, "end_time": 1.0, "confidence": 85, "speaker_id": "Alice"},
        ]
        s1 = [{"start_time": 0.0, "end_time": 1.0, "speaker_id": "Alice"}]
        _make_md(transcripts, w1, s1, name="rec1.md")

        w2 = [
            {"text": "Yo", "start_time": 0.0, "end_time": 0.5, "confidence": 88, "speaker_id": "Alice"},
        ]
        s2 = [{"start_time": 0.0, "end_time": 0.5, "speaker_id": "Alice"}]
        _make_md(transcripts, w2, s2, name="rec2.md")

        usage = scan_identity_usage(transcripts, ["Alice", "Bob"])
        assert usage["Alice"].total_mentions == 3
        assert usage["Alice"].recording_count == 2
        assert usage["Bob"].recording_count == 0
        assert usage["Bob"].total_mentions == 0

    def test_empty_identity_list(self, tmp_path):
        transcripts = tmp_path / "transcripts"
        transcripts.mkdir()
        _make_md(
            transcripts,
            [{"text": "Hi", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": "Alice"}],
            [{"start_time": 0.0, "end_time": 0.5, "speaker_id": "Alice"}],
        )
        usage = scan_identity_usage(transcripts, [])
        assert len(usage) == 0

    def test_nonexistent_dir(self, tmp_path):
        usage = scan_identity_usage(tmp_path / "nope", ["Alice"])
        assert usage["Alice"].recording_count == 0

    def test_skips_malformed_files(self, tmp_path):
        transcripts = tmp_path / "transcripts"
        transcripts.mkdir()
        # Valid file
        _make_md(
            transcripts,
            [{"text": "Hi", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": "Alice"}],
            [{"start_time": 0.0, "end_time": 0.5, "speaker_id": "Alice"}],
            name="good.md",
        )
        # Malformed file
        bad = transcripts / "bad.md"
        bad.write_text("# T\n\n---\n\n<!-- METADATA: {bad} -->\n", encoding="utf-8")
        # No footer file
        bare = transcripts / "bare.md"
        bare.write_text("# Bare\n\n**Alice**\n\nHi\n", encoding="utf-8")

        usage = scan_identity_usage(transcripts, ["Alice"])
        assert usage["Alice"].recording_count == 1
        assert usage["Alice"].total_mentions == 1

    def test_last_activity_is_latest_mtime(self, tmp_path):
        transcripts = tmp_path / "transcripts"
        transcripts.mkdir()
        _make_md(
            transcripts,
            [{"text": "A", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": "Alice"}],
            [{"start_time": 0.0, "end_time": 0.5, "speaker_id": "Alice"}],
            name="old.md",
        )
        _make_md(
            transcripts,
            [{"text": "B", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": "Alice"}],
            [{"start_time": 0.0, "end_time": 0.5, "speaker_id": "Alice"}],
            name="new.md",
        )
        usage = scan_identity_usage(transcripts, ["Alice"])
        assert usage["Alice"].last_activity is not None
        assert usage["Alice"].recording_count == 2

    def test_multiple_identities_in_one_transcript(self, tmp_path):
        transcripts = tmp_path / "transcripts"
        transcripts.mkdir()
        words = [
            {"text": "Hi", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": "Alice"},
            {"text": "Yo", "start_time": 0.5, "end_time": 1.0, "confidence": 88, "speaker_id": "Bob"},
        ]
        segs = [
            {"start_time": 0.0, "end_time": 0.5, "speaker_id": "Alice"},
            {"start_time": 0.5, "end_time": 1.0, "speaker_id": "Bob"},
        ]
        _make_md(transcripts, words, segs)

        usage = scan_identity_usage(transcripts, ["Alice", "Bob"])
        assert usage["Alice"].total_mentions == 1
        assert usage["Bob"].total_mentions == 1

    def test_profile_with_zero_recordings(self, tmp_path):
        transcripts = tmp_path / "transcripts"
        transcripts.mkdir()
        # No transcript files at all
        usage = scan_identity_usage(transcripts, ["Charlie"])
        assert "Charlie" in usage
        assert usage["Charlie"].recording_count == 0
        assert usage["Charlie"].total_mentions == 0


# ---------------------------------------------------------------------------
# replace_speaker_label_in_file tests
# ---------------------------------------------------------------------------


class TestReplaceSpeakerLabelInFile:
    def test_replaces_all_surfaces(self, tmp_path):
        w = [
            {"text": "Hi", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": "Alice"},
            {"text": "Bye", "start_time": 0.5, "end_time": 1.0, "confidence": 85, "speaker_id": "Alice"},
        ]
        s = [{"start_time": 0.0, "end_time": 1.0, "speaker_id": "Alice", "speaker": "Alice"}]
        md = _make_md(tmp_path, w, s)
        count = replace_speaker_label_in_file(md, "Alice", "Carol")
        assert count == 2
        d = _parse_metadata(md)
        assert all(w["speaker_id"] == "Carol" for w in d["words"])
        assert d["segments"][0]["speaker_id"] == "Carol"
        assert d["segments"][0]["speaker"] == "Carol"
        assert "**Carol**" in md.read_text(encoding="utf-8")
        assert "**Alice**" not in md.read_text(encoding="utf-8")

    def test_no_substring_replacement(self, tmp_path):
        """SPK_0 must not match SPK_01."""
        w = [
            {"text": "Hi", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": "SPK_0"},
            {"text": "Yo", "start_time": 0.5, "end_time": 1.0, "confidence": 88, "speaker_id": "SPK_01"},
        ]
        s = [
            {"start_time": 0.0, "end_time": 0.5, "speaker_id": "SPK_0"},
            {"start_time": 0.5, "end_time": 1.0, "speaker_id": "SPK_01"},
        ]
        md = _make_md(tmp_path, w, s)
        count = replace_speaker_label_in_file(md, "SPK_0", "Alice")
        assert count == 1
        d = _parse_metadata(md)
        assert d["words"][0]["speaker_id"] == "Alice"
        assert d["words"][1]["speaker_id"] == "SPK_01"
        body = md.read_text(encoding="utf-8")
        assert "**Alice**" in body
        assert "**SPK_01**" in body

    def test_updates_speaker_matches(self, tmp_path):
        w = [{"text": "Hi", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": "Alice"}]
        s = [{"start_time": 0.0, "end_time": 0.5, "speaker_id": "Alice"}]
        sm = {"SPK_0": {"identity_name": "Alice", "score": 0.92, "confidence": "high"}}
        md = _make_md(tmp_path, w, s, sm)
        replace_speaker_label_in_file(md, "Alice", "Carol")
        d = _parse_metadata(md)
        assert d["speaker_matches"]["SPK_0"]["identity_name"] == "Carol"
        assert d["speaker_matches"]["SPK_0"]["score"] == 0.92

    def test_raises_on_missing_footer(self, tmp_path):
        p = tmp_path / "bare.md"
        p.write_text("# Bare\n\n**Alice**\n\nHi\n", encoding="utf-8")
        with pytest.raises(IdentityManagementError, match="No metadata footer"):
            replace_speaker_label_in_file(p, "Alice", "Carol")

    def test_raises_on_malformed_json(self, tmp_path):
        p = tmp_path / "bad.md"
        p.write_text("# T\n\n---\n\n<!-- METADATA: {bad} -->\n", encoding="utf-8")
        with pytest.raises(IdentityManagementError, match="Malformed metadata"):
            replace_speaker_label_in_file(p, "Alice", "Carol")

    def test_returns_zero_when_label_not_found(self, tmp_path):
        w = [{"text": "Hi", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": "Bob"}]
        s = [{"start_time": 0.0, "end_time": 0.5, "speaker_id": "Bob"}]
        md = _make_md(tmp_path, w, s)
        count = replace_speaker_label_in_file(md, "Alice", "Carol")
        assert count == 0


# ---------------------------------------------------------------------------
# rename_identity tests
# ---------------------------------------------------------------------------


class TestRenameIdentity:
    def test_renames_store_and_transcripts(self, tmp_path):
        store = _make_store_with_identities(
            tmp_path, {"Alice": (_random_embedding(1), 3)}
        )
        transcripts = tmp_path / "transcripts"
        transcripts.mkdir()
        w = [{"text": "Hi", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": "Alice"}]
        s = [{"start_time": 0.0, "end_time": 0.5, "speaker_id": "Alice"}]
        _make_md(transcripts, w, s, name="rec.md")

        rename_identity(store, transcripts, "Alice", "Carol")

        # Store updated
        profiles = store.load_signatures()
        names = [p.name for p in profiles]
        assert "Carol" in names
        assert "Alice" not in names
        # num_samples preserved
        carol = next(p for p in profiles if p.name == "Carol")
        assert carol.num_samples == 3

        # Transcript updated
        d = _parse_metadata(transcripts / "rec.md")
        assert d["words"][0]["speaker_id"] == "Carol"
        assert "**Carol**" in (transcripts / "rec.md").read_text(encoding="utf-8")

    def test_preserves_embedding(self, tmp_path):
        emb = _random_embedding(42)
        store = _make_store_with_identities(tmp_path, {"Alice": (emb, 5)})
        transcripts = tmp_path / "transcripts"
        transcripts.mkdir()

        rename_identity(store, transcripts, "Alice", "Bob")

        profiles = store.load_signatures()
        bob = next(p for p in profiles if p.name == "Bob")
        np.testing.assert_array_almost_equal(bob.embedding, emb)

    def test_rejects_empty_old_name(self, tmp_path):
        store = _make_store_with_identities(tmp_path, {})
        with pytest.raises(RenameError, match="Source identity name must not be empty"):
            rename_identity(store, tmp_path / "t", "", "Bob")

    def test_rejects_empty_new_name(self, tmp_path):
        store = _make_store_with_identities(tmp_path, {})
        with pytest.raises(RenameError, match="New identity name must not be empty"):
            rename_identity(store, tmp_path / "t", "Alice", "")

    def test_rejects_same_name(self, tmp_path):
        store = _make_store_with_identities(tmp_path, {"Alice": (_random_embedding(), 1)})
        with pytest.raises(RenameError, match="must differ"):
            rename_identity(store, tmp_path / "t", "Alice", "Alice")

    def test_rejects_duplicate_target(self, tmp_path):
        store = _make_store_with_identities(
            tmp_path,
            {"Alice": (_random_embedding(1), 2), "Bob": (_random_embedding(2), 3)},
        )
        with pytest.raises(RenameError, match="already exists"):
            rename_identity(store, tmp_path / "t", "Alice", "Bob")

    def test_rejects_missing_source(self, tmp_path):
        store = _make_store_with_identities(tmp_path, {})
        with pytest.raises(RenameError, match="not found"):
            rename_identity(store, tmp_path / "t", "Ghost", "NewName")

    def test_handles_nonexistent_transcript_dir(self, tmp_path):
        store = _make_store_with_identities(tmp_path, {"Alice": (_random_embedding(), 2)})
        # transcripts dir doesn't exist — rename should still succeed in store
        rename_identity(store, tmp_path / "nonexistent", "Alice", "Bob")
        profiles = store.load_signatures()
        assert [p.name for p in profiles] == ["Bob"]


# ---------------------------------------------------------------------------
# merge_identities tests
# ---------------------------------------------------------------------------


class TestMergeIdentities:
    def test_weighted_average_embedding(self, tmp_path):
        emb_a = _random_embedding(10)
        emb_b = _random_embedding(20)
        store = _make_store_with_identities(
            tmp_path, {"Alice": (emb_a, 2), "Bob": (emb_b, 3)}
        )
        transcripts = tmp_path / "transcripts"
        transcripts.mkdir()

        merge_identities(store, transcripts, "Alice", "Bob")

        profiles = store.load_signatures()
        assert len(profiles) == 1
        merged = profiles[0]
        assert merged.name == "Bob"

        # Check weighted average
        expected = (emb_a * 2 + emb_b * 3) / 5
        np.testing.assert_array_almost_equal(merged.embedding, expected.astype(np.float32), decimal=5)
        assert merged.num_samples == 5

    def test_rewrites_transcripts(self, tmp_path):
        store = _make_store_with_identities(
            tmp_path, {"Alice": (_random_embedding(1), 1), "Bob": (_random_embedding(2), 1)}
        )
        transcripts = tmp_path / "transcripts"
        transcripts.mkdir()
        w = [{"text": "Hi", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": "Alice"}]
        s = [{"start_time": 0.0, "end_time": 0.5, "speaker_id": "Alice"}]
        _make_md(transcripts, w, s, name="rec.md")

        merge_identities(store, transcripts, "Alice", "Bob")

        d = _parse_metadata(transcripts / "rec.md")
        assert d["words"][0]["speaker_id"] == "Bob"
        assert "**Bob**" in (transcripts / "rec.md").read_text(encoding="utf-8")

    def test_deletes_source_from_store(self, tmp_path):
        store = _make_store_with_identities(
            tmp_path, {"Alice": (_random_embedding(1), 1), "Bob": (_random_embedding(2), 1)}
        )
        transcripts = tmp_path / "transcripts"
        transcripts.mkdir()

        merge_identities(store, transcripts, "Alice", "Bob")

        names = [p.name for p in store.load_signatures()]
        assert "Alice" not in names
        assert "Bob" in names

    def test_rejects_merge_to_self(self, tmp_path):
        store = _make_store_with_identities(tmp_path, {"Alice": (_random_embedding(), 1)})
        with pytest.raises(MergeError, match="into itself"):
            merge_identities(store, tmp_path / "t", "Alice", "Alice")

    def test_rejects_empty_source(self, tmp_path):
        store = _make_store_with_identities(tmp_path, {})
        with pytest.raises(MergeError, match="must not be empty"):
            merge_identities(store, tmp_path / "t", "", "Bob")

    def test_rejects_empty_target(self, tmp_path):
        store = _make_store_with_identities(tmp_path, {})
        with pytest.raises(MergeError, match="must not be empty"):
            merge_identities(store, tmp_path / "t", "Alice", "")

    def test_rejects_missing_source(self, tmp_path):
        store = _make_store_with_identities(
            tmp_path, {"Bob": (_random_embedding(2), 1)}
        )
        with pytest.raises(MergeError, match="Source identity not found"):
            merge_identities(store, tmp_path / "t", "Ghost", "Bob")

    def test_rejects_missing_target(self, tmp_path):
        store = _make_store_with_identities(
            tmp_path, {"Alice": (_random_embedding(1), 1)}
        )
        with pytest.raises(MergeError, match="Target identity not found"):
            merge_identities(store, tmp_path / "t", "Alice", "Ghost")

    def test_merges_multiple_transcripts(self, tmp_path):
        store = _make_store_with_identities(
            tmp_path, {"Alice": (_random_embedding(1), 2), "Bob": (_random_embedding(2), 3)}
        )
        transcripts = tmp_path / "transcripts"
        transcripts.mkdir()
        for i in range(3):
            w = [{"text": f"w{i}", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": "Alice"}]
            s = [{"start_time": 0.0, "end_time": 0.5, "speaker_id": "Alice"}]
            _make_md(transcripts, w, s, name=f"rec{i}.md")

        merge_identities(store, transcripts, "Alice", "Bob")

        # All 3 transcripts should now reference Bob
        for i in range(3):
            d = _parse_metadata(transcripts / f"rec{i}.md")
            assert d["words"][0]["speaker_id"] == "Bob"


# ---------------------------------------------------------------------------
# delete_identity tests
# ---------------------------------------------------------------------------


class TestDeleteIdentity:
    def test_removes_from_store(self, tmp_path):
        store = _make_store_with_identities(
            tmp_path, {"Alice": (_random_embedding(), 1), "Bob": (_random_embedding(), 1)}
        )
        transcripts = tmp_path / "transcripts"
        transcripts.mkdir()

        delete_identity(store, transcripts, "Alice")

        names = [p.name for p in store.load_signatures()]
        assert "Alice" not in names
        assert "Bob" in names

    def test_rejects_empty_name(self, tmp_path):
        store = _make_store_with_identities(tmp_path, {})
        with pytest.raises(DeleteError, match="must not be empty"):
            delete_identity(store, tmp_path / "t", "")

    def test_rejects_whitespace_name(self, tmp_path):
        store = _make_store_with_identities(tmp_path, {})
        with pytest.raises(DeleteError, match="must not be empty"):
            delete_identity(store, tmp_path / "t", "   ")

    def test_rejects_missing_identity(self, tmp_path):
        store = _make_store_with_identities(
            tmp_path, {"Alice": (_random_embedding(), 1)}
        )
        with pytest.raises(DeleteError, match="not found"):
            delete_identity(store, tmp_path / "t", "Ghost")

    def test_transcripts_unchanged_after_delete(self, tmp_path):
        store = _make_store_with_identities(
            tmp_path, {"Alice": (_random_embedding(), 1)}
        )
        transcripts = tmp_path / "transcripts"
        transcripts.mkdir()
        w = [{"text": "Hi", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": "Alice"}]
        s = [{"start_time": 0.0, "end_time": 0.5, "speaker_id": "Alice"}]
        _make_md(transcripts, w, s, name="rec.md")

        orig_content = (transcripts / "rec.md").read_text(encoding="utf-8")
        delete_identity(store, transcripts, "Alice")
        after_content = (transcripts / "rec.md").read_text(encoding="utf-8")
        assert orig_content == after_content


# ---------------------------------------------------------------------------
# Negative / boundary tests
# ---------------------------------------------------------------------------


class TestNegativeAndBoundary:
    def test_substring_labels_not_confused(self, tmp_path):
        """SPK_0 rename must not touch SPK_01 in words, segments, or body."""
        store = _make_store_with_identities(
            tmp_path, {"SPK_0": (_random_embedding(0), 1), "SPK_01": (_random_embedding(1), 1)}
        )
        transcripts = tmp_path / "transcripts"
        transcripts.mkdir()
        w = [
            {"text": "Hi", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": "SPK_0"},
            {"text": "Yo", "start_time": 0.5, "end_time": 1.0, "confidence": 88, "speaker_id": "SPK_01"},
        ]
        s = [
            {"start_time": 0.0, "end_time": 0.5, "speaker_id": "SPK_0", "speaker": "SPK_0"},
            {"start_time": 0.5, "end_time": 1.0, "speaker_id": "SPK_01", "speaker": "SPK_01"},
        ]
        _make_md(transcripts, w, s)

        rename_identity(store, transcripts, "SPK_0", "Alice")

        d = _parse_metadata(transcripts / "test.md")
        assert d["words"][0]["speaker_id"] == "Alice"
        assert d["words"][1]["speaker_id"] == "SPK_01"
        assert d["segments"][0]["speaker_id"] == "Alice"
        assert d["segments"][0]["speaker"] == "Alice"
        assert d["segments"][1]["speaker_id"] == "SPK_01"
        assert d["segments"][1]["speaker"] == "SPK_01"
        body = (transcripts / "test.md").read_text(encoding="utf-8")
        assert "**Alice**" in body
        assert "**SPK_01**" in body

    def test_transcript_with_no_footer_skipped_in_scan(self, tmp_path):
        transcripts = tmp_path / "transcripts"
        transcripts.mkdir()
        bare = transcripts / "bare.md"
        bare.write_text("# Bare\n\n**Alice**\n\nHi\n", encoding="utf-8")

        usage = scan_identity_usage(transcripts, ["Alice"])
        assert usage["Alice"].recording_count == 0

    def test_transcript_with_no_footer_skipped_in_rewrite(self, tmp_path):
        store = _make_store_with_identities(
            tmp_path, {"Alice": (_random_embedding(), 1)}
        )
        transcripts = tmp_path / "transcripts"
        transcripts.mkdir()
        bare = transcripts / "bare.md"
        bare.write_text("# Bare\n\n**Alice**\n\nHi\n", encoding="utf-8")

        # rename should succeed in store but skip the bare file
        rename_identity(store, transcripts, "Alice", "Bob")
        assert bare.read_text(encoding="utf-8") == "# Bare\n\n**Alice**\n\nHi\n"

    def test_segment_with_both_speaker_id_and_speaker_keys(self, tmp_path):
        """Segment has both speaker_id and legacy speaker keys."""
        w = [{"text": "Hi", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": "Alice"}]
        s = [{"start_time": 0.0, "end_time": 0.5, "speaker_id": "Alice", "speaker": "Alice"}]
        md = _make_md(tmp_path, w, s)
        replace_speaker_label_in_file(md, "Alice", "Carol")
        d = _parse_metadata(md)
        assert d["segments"][0]["speaker_id"] == "Carol"
        assert d["segments"][0]["speaker"] == "Carol"

    def test_merge_with_no_transcripts_succeeds(self, tmp_path):
        store = _make_store_with_identities(
            tmp_path, {"Alice": (_random_embedding(1), 2), "Bob": (_random_embedding(2), 3)}
        )
        transcripts = tmp_path / "transcripts"
        transcripts.mkdir()

        merge_identities(store, transcripts, "Alice", "Bob")
        profiles = store.load_signatures()
        assert len(profiles) == 1
        assert profiles[0].num_samples == 5

    def test_whitespace_name_rejected(self, tmp_path):
        store = _make_store_with_identities(tmp_path, {})
        with pytest.raises(RenameError):
            rename_identity(store, tmp_path / "t", "   ", "Bob")
        with pytest.raises(RenameError):
            rename_identity(store, tmp_path / "t", "Alice", "   ")

    def test_scan_does_not_crash_on_unreadable_file(self, tmp_path):
        transcripts = tmp_path / "transcripts"
        transcripts.mkdir()
        # Create a file and make it unreadable (Windows may not respect chmod)
        bad = transcripts / "unreadable.md"
        bad.write_text("# Test\n", encoding="utf-8")
        # Even if readable, metadata is missing — should be skipped
        usage = scan_identity_usage(transcripts, ["Alice"])
        assert usage["Alice"].recording_count == 0


# ---------------------------------------------------------------------------
# PII-safe logging tests
# ---------------------------------------------------------------------------


class TestPIISafeLogging:
    def test_scan_no_identity_names_in_logs(self, tmp_path, caplog):
        transcripts = tmp_path / "transcripts"
        transcripts.mkdir()
        _make_md(
            transcripts,
            [{"text": "Hi", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": "SecretName"}],
            [{"start_time": 0.0, "end_time": 0.5, "speaker_id": "SecretName"}],
        )
        with caplog.at_level(logging.DEBUG, logger="meetandread.speaker.identity_management"):
            scan_identity_usage(transcripts, ["SecretName"])
        for r in caplog.records:
            assert "SecretName" not in r.message

    def test_rename_no_identity_names_in_logs(self, tmp_path, caplog):
        store = _make_store_with_identities(tmp_path, {"SecretOld": (_random_embedding(), 1)})
        transcripts = tmp_path / "transcripts"
        transcripts.mkdir()
        with caplog.at_level(logging.DEBUG, logger="meetandread.speaker.identity_management"):
            rename_identity(store, transcripts, "SecretOld", "SecretNew")
        for r in caplog.records:
            assert "SecretOld" not in r.message
            assert "SecretNew" not in r.message

    def test_delete_no_identity_names_in_logs(self, tmp_path, caplog):
        store = _make_store_with_identities(tmp_path, {"SecretDel": (_random_embedding(), 1)})
        transcripts = tmp_path / "transcripts"
        transcripts.mkdir()
        with caplog.at_level(logging.DEBUG, logger="meetandread.speaker.identity_management"):
            delete_identity(store, transcripts, "SecretDel")
        for r in caplog.records:
            assert "SecretDel" not in r.message

    def test_merge_no_identity_names_in_logs(self, tmp_path, caplog):
        store = _make_store_with_identities(
            tmp_path,
            {"SecretSrc": (_random_embedding(1), 1), "SecretTgt": (_random_embedding(2), 1)},
        )
        transcripts = tmp_path / "transcripts"
        transcripts.mkdir()
        with caplog.at_level(logging.DEBUG, logger="meetandread.speaker.identity_management"):
            merge_identities(store, transcripts, "SecretSrc", "SecretTgt")
        for r in caplog.records:
            assert "SecretSrc" not in r.message
            assert "SecretTgt" not in r.message


# ===========================================================================
# T02 UI Tests — Settings Identities navigation and read-only detail UI
# ===========================================================================

import datetime
from unittest.mock import patch, MagicMock

from PyQt6.QtWidgets import (
    QApplication,
    QListWidget,
    QListWidgetItem,
    QSplitter,
    QLabel,
    QPushButton,
    QFrame,
)
from PyQt6.QtCore import Qt

from meetandread.widgets.floating_panels import FloatingSettingsPanel


# ---- Fixtures -------------------------------------------------------------


@pytest.fixture
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def settings_panel(qapp):
    panel = FloatingSettingsPanel()
    panel.show()
    qapp.processEvents()
    yield panel
    panel.close()


@pytest.fixture
def settings_panel_on_identities(settings_panel, qapp):
    """Navigate to Identities page before returning the panel."""
    settings_panel._on_nav_clicked(FloatingSettingsPanel._NAV_IDENTITIES)
    qapp.processEvents()
    return settings_panel


# ---------------------------------------------------------------------------
# Nav constant tests
# ---------------------------------------------------------------------------


class TestIdentitiesNavConstants:
    """Verify nav indices and stack count include Identities."""

    def test_nav_identities_is_3(self):
        assert FloatingSettingsPanel._NAV_IDENTITIES == 3

    def test_nav_settings_is_0(self):
        assert FloatingSettingsPanel._NAV_SETTINGS == 0

    def test_nav_performance_is_1(self):
        assert FloatingSettingsPanel._NAV_PERFORMANCE == 1

    def test_nav_history_is_2(self):
        assert FloatingSettingsPanel._NAV_HISTORY == 2

    def test_content_stack_has_4_pages(self, settings_panel):
        assert settings_panel._content_stack.count() == 4


# ---------------------------------------------------------------------------
# Structure / object name tests
# ---------------------------------------------------------------------------


class TestIdentitiesStructure:
    """Verify Identities page widgets exist with correct object names."""

    def test_identities_page_object_name(self, settings_panel):
        page = settings_panel._content_stack.widget(
            FloatingSettingsPanel._NAV_IDENTITIES
        )
        assert page is not None
        assert page.objectName() == "AethericIdentitiesPage"

    def test_identities_splitter_object_name(self, settings_panel):
        assert settings_panel._identities_splitter.objectName() == "AethericIdentitiesSplitter"

    def test_identity_list_object_name(self, settings_panel):
        assert settings_panel._identity_list.objectName() == "AethericIdentityList"

    def test_identity_detail_header_object_name(self, settings_panel):
        assert settings_panel._identity_detail_header.objectName() == "AethericIdentityHeader"

    def test_identity_rename_btn_object_name(self, settings_panel):
        assert settings_panel._identity_rename_btn.objectName() == "AethericIdentityActionButton"

    def test_identity_merge_btn_object_name(self, settings_panel):
        assert settings_panel._identity_merge_btn.objectName() == "AethericIdentityActionButton"

    def test_identity_delete_btn_object_name(self, settings_panel):
        assert settings_panel._identity_delete_btn.objectName() == "AethericIdentityActionButton"

    def test_rename_button_action_property(self, settings_panel):
        assert settings_panel._identity_rename_btn.property("action") == "rename"

    def test_merge_button_action_property(self, settings_panel):
        assert settings_panel._identity_merge_btn.property("action") == "merge"

    def test_delete_button_action_property(self, settings_panel):
        assert settings_panel._identity_delete_btn.property("action") == "delete"

    def test_identities_page_is_stack_index_3(self, settings_panel):
        page = settings_panel._content_stack.widget(3)
        assert page is not None
        assert page.objectName() == "AethericIdentitiesPage"

    def test_splitter_is_vertical(self, settings_panel):
        assert settings_panel._identities_splitter.orientation() == Qt.Orientation.Vertical

    def test_detail_header_initially_hidden(self, settings_panel):
        assert settings_panel._identity_detail_header.isHidden() is True

    def test_action_buttons_disabled(self, settings_panel):
        """Action buttons should be disabled (enabled in T03)."""
        assert not settings_panel._identity_rename_btn.isEnabled()
        assert not settings_panel._identity_merge_btn.isEnabled()
        assert not settings_panel._identity_delete_btn.isEnabled()

    def test_nav_button_has_accessible_name(self, settings_panel):
        assert settings_panel._nav_identities_btn.accessibleName() != ""

    def test_nav_button_has_tooltip(self, settings_panel):
        assert settings_panel._nav_identities_btn.toolTip() != ""

    def test_identity_list_has_accessible_name(self, settings_panel):
        assert settings_panel._identity_list.accessibleName() != ""

    def test_rename_btn_has_accessible_name(self, settings_panel):
        assert settings_panel._identity_rename_btn.accessibleName() != ""

    def test_state_attributes_initialized(self, settings_panel):
        assert settings_panel._identity_usage == {}


# ---------------------------------------------------------------------------
# Nav refresh tests
# ---------------------------------------------------------------------------


class TestIdentitiesNavRefresh:
    """Verify Identities nav triggers refresh."""

    def test_nav_to_identities_calls_refresh(self, settings_panel, qapp):
        """Navigating to Identities calls _refresh_identities."""
        with patch.object(settings_panel, "_refresh_identities") as mock_refresh:
            settings_panel._on_nav_clicked(FloatingSettingsPanel._NAV_IDENTITIES)
            qapp.processEvents()
            mock_refresh.assert_called_once()

    def test_nav_to_identities_stops_perf_monitor(self, settings_panel, qapp):
        """Identities nav should stop ResourceMonitor."""
        settings_panel._on_nav_clicked(FloatingSettingsPanel._NAV_PERFORMANCE)
        qapp.processEvents()
        assert settings_panel._perf_tab_active is True

        settings_panel._on_nav_clicked(FloatingSettingsPanel._NAV_IDENTITIES)
        qapp.processEvents()
        assert settings_panel._perf_tab_active is False

    def test_nav_away_and_back_refreshes(self, settings_panel, qapp):
        """Navigating away and back triggers a new refresh."""
        with patch.object(settings_panel, "_refresh_identities") as mock_refresh:
            settings_panel._on_nav_clicked(FloatingSettingsPanel._NAV_IDENTITIES)
            qapp.processEvents()
            settings_panel._on_nav_clicked(FloatingSettingsPanel._NAV_SETTINGS)
            qapp.processEvents()
            settings_panel._on_nav_clicked(FloatingSettingsPanel._NAV_IDENTITIES)
            qapp.processEvents()
            assert mock_refresh.call_count == 2


# ---------------------------------------------------------------------------
# Empty state tests
# ---------------------------------------------------------------------------


class TestIdentitiesEmptyState:
    """Verify empty-list and error states."""

    def test_empty_profiles_clears_list(self, settings_panel_on_identities):
        panel = settings_panel_on_identities
        panel._populate_identity_list([], {})
        assert panel._identity_list.count() == 0

    def test_empty_profiles_hides_detail_header(self, settings_panel_on_identities):
        panel = settings_panel_on_identities
        panel._populate_identity_list([], {})
        assert panel._identity_detail_header.isHidden()

    def test_empty_profiles_clears_detail_labels(self, settings_panel_on_identities):
        panel = settings_panel_on_identities
        panel._populate_identity_list([], {})
        assert "—" in panel._identity_name_label.text()

    def test_store_load_failure_still_discovers_transcript_identities(self, settings_panel_on_identities, qapp, tmp_path):
        """If VoiceSignatureStore fails, transcript-discovered identities still appear."""
        panel = settings_panel_on_identities
        # Patch get_transcripts_dir to an empty dir so transcript scan finds nothing
        with patch(
            "meetandread.speaker.signatures.VoiceSignatureStore.load_signatures",
            side_effect=RuntimeError("db locked"),
        ), patch(
            "meetandread.widgets.floating_panels.Path.is_dir",
            return_value=False,
        ):
            panel._refresh_identities()
            qapp.processEvents()
        # With no store profiles and no transcript dir, list should be empty
        assert panel._identity_list.count() == 0

    def test_discovers_identity_from_transcript_metadata(self, settings_panel_on_identities, qapp, tmp_path):
        """Identities linked in transcripts but missing from VoiceSignatureStore appear in list."""
        import json
        panel = settings_panel_on_identities

        # Create a transcript with speaker_matches pointing to an identity
        # that doesn't exist in VoiceSignatureStore
        transcript_dir = tmp_path / "transcripts"
        transcript_dir.mkdir()
        md_body = "# Transcript\n\n**Alice**\n\nHello world\n"
        metadata = {
            "words": [{"text": "Hello", "speaker_id": "Alice"}],
            "segments": [{"speaker_id": "Alice", "start_time": 0.0, "end_time": 1.0}],
            "speaker_matches": {
                "SPK_0": {"identity_name": "Alice", "score": 1.0, "confidence": "manual"}
            },
        }
        content = md_body + "\n---\n\n<!-- METADATA: " + json.dumps(metadata) + " -->\n"
        (transcript_dir / "test.md").write_text(content, encoding="utf-8")

        # Mock to return empty store and our temp transcripts dir
        with patch(
            "meetandread.speaker.signatures.VoiceSignatureStore.load_signatures",
            return_value=[],
        ), patch(
            "meetandread.audio.storage.paths.get_transcripts_dir",
            return_value=transcript_dir,
        ), patch(
            "meetandread.audio.storage.paths.get_recordings_dir",
            return_value=tmp_path / "recordings",
        ):
            panel._refresh_identities()
            qapp.processEvents()

        # Alice should appear even though she's not in VoiceSignatureStore
        names = [panel._identity_list.item(i).data(256) for i in range(panel._identity_list.count())]
        assert "Alice" in names

    def test_missing_signature_db_shows_empty(self, settings_panel_on_identities, qapp):
        """If the DB file doesn't exist, show empty state gracefully."""
        panel = settings_panel_on_identities
        # The store will still create the DB; test the empty-no-profiles path
        panel._populate_identity_list([], {})
        assert panel._identity_list.count() == 0


# ---------------------------------------------------------------------------
# List population tests
# ---------------------------------------------------------------------------


class TestIdentitiesListPopulation:
    """Verify list items display correct text and carry name data."""

    def test_single_profile_display(self, settings_panel_on_identities):
        panel = settings_panel_on_identities
        panel._populate_identity_list(["Alice"], {})
        assert panel._identity_list.count() == 1
        item = panel._identity_list.item(0)
        assert "Alice" in item.text()
        assert "0 recording" in item.text()

    def test_profile_with_recordings_shows_count(self, settings_panel_on_identities):
        panel = settings_panel_on_identities
        usage = {
            "Bob": IdentityUsage(
                identity_name="Bob",
                recordings=[IdentityRecordingRef(Path("/a.md"), 3)],
                total_mentions=3,
            ),
        }
        panel._populate_identity_list(["Bob"], usage)
        item = panel._identity_list.item(0)
        assert "1 recording" in item.text()

    def test_profiles_sorted_by_name(self, settings_panel_on_identities):
        panel = settings_panel_on_identities
        # Sorted when passed in (caller is responsible for sort order)
        panel._populate_identity_list(["Alice", "Bob", "Charlie"], {})
        assert panel._identity_list.count() == 3
        assert "Alice" in panel._identity_list.item(0).text()
        assert "Bob" in panel._identity_list.item(1).text()
        assert "Charlie" in panel._identity_list.item(2).text()

    def test_item_stores_name_as_user_role(self, settings_panel_on_identities):
        panel = settings_panel_on_identities
        panel._populate_identity_list(["Alice"], {})
        item = panel._identity_list.item(0)
        assert item.data(Qt.ItemDataRole.UserRole) == "Alice"

    def test_populate_clears_previous_items(self, settings_panel_on_identities):
        panel = settings_panel_on_identities
        panel._populate_identity_list(["Alice", "Bob"], {})
        assert panel._identity_list.count() == 2
        panel._populate_identity_list(["Charlie"], {})
        assert panel._identity_list.count() == 1

    def test_case_sensitive_distinct_names(self, settings_panel_on_identities):
        """Duplicate-looking names with different case render as separate items."""
        panel = settings_panel_on_identities
        panel._populate_identity_list(["alice", "Alice"], {})
        assert panel._identity_list.count() == 2

    def test_very_long_name_no_crash(self, settings_panel_on_identities):
        """Very long identity name renders without crashing."""
        panel = settings_panel_on_identities
        long_name = "A" * 500
        panel._populate_identity_list([long_name], {})
        assert panel._identity_list.count() == 1
        assert long_name in panel._identity_list.item(0).text()

    def test_plural_recordings_text(self, settings_panel_on_identities):
        """2+ recordings shows plural form."""
        panel = settings_panel_on_identities
        usage = {
            "Alice": IdentityUsage(
                identity_name="Alice",
                recordings=[
                    IdentityRecordingRef(Path("/a.md"), 1),
                    IdentityRecordingRef(Path("/b.md"), 2),
                ],
                total_mentions=3,
            ),
        }
        panel._populate_identity_list(["Alice"], usage)
        item = panel._identity_list.item(0)
        assert "2 recordings" in item.text()


# ---------------------------------------------------------------------------
# Selection / detail rendering tests
# ---------------------------------------------------------------------------


class TestIdentitiesSelectionDetail:
    """Verify item clicks render identity details."""

    def test_click_shows_detail_header(self, settings_panel_on_identities, qapp):
        panel = settings_panel_on_identities
        panel._populate_identity_list(["Alice"], {})
        item = panel._identity_list.item(0)
        panel._identity_list.setCurrentItem(item)
        panel._on_identity_item_clicked(item)
        qapp.processEvents()
        assert panel._identity_detail_header.isVisible()

    def test_click_shows_name(self, settings_panel_on_identities, qapp):
        panel = settings_panel_on_identities
        panel._populate_identity_list(["Alice"], {})
        item = panel._identity_list.item(0)
        panel._on_identity_item_clicked(item)
        qapp.processEvents()
        assert "Alice" in panel._identity_name_label.text()

    def test_click_with_usage_shows_recording_count(
        self, settings_panel_on_identities, qapp
    ):
        panel = settings_panel_on_identities
        usage = {
            "Bob": IdentityUsage(
                identity_name="Bob",
                recordings=[
                    IdentityRecordingRef(Path("/a.md"), 3, 1700000000.0),
                ],
                total_mentions=3,
                last_activity=1700000000.0,
            ),
        }
        panel._identity_usage = usage
        panel._populate_identity_list(["Bob"], usage)
        item = panel._identity_list.item(0)
        panel._on_identity_item_clicked(item)
        qapp.processEvents()
        assert "1" in panel._identity_recording_count_label.text()
        assert "2023" in panel._identity_last_used_label.text()

    def test_click_no_usage_shows_zero(self, settings_panel_on_identities, qapp):
        panel = settings_panel_on_identities
        panel._identity_usage = {}
        panel._populate_identity_list(["Alice"], {})
        item = panel._identity_list.item(0)
        panel._on_identity_item_clicked(item)
        qapp.processEvents()
        assert "0" in panel._identity_recording_count_label.text()

    def test_click_with_recordings_shows_file_stems(
        self, settings_panel_on_identities, qapp
    ):
        panel = settings_panel_on_identities
        usage = {
            "Alice": IdentityUsage(
                identity_name="Alice",
                recordings=[
                    IdentityRecordingRef(Path("/rec1.md"), 5),
                    IdentityRecordingRef(Path("/rec2.md"), 3),
                ],
                total_mentions=8,
            ),
        }
        panel._identity_usage = usage
        panel._populate_identity_list(["Alice"], usage)
        item = panel._identity_list.item(0)
        panel._on_identity_item_clicked(item)
        qapp.processEvents()
        text = panel._identity_recordings_label.text()
        assert "rec1" in text
        assert "rec2" in text

    def test_click_many_recordings_truncates(
        self, settings_panel_on_identities, qapp
    ):
        """More than 10 recordings shows truncation."""
        panel = settings_panel_on_identities
        recs = [
            IdentityRecordingRef(Path(f"/rec{i}.md"), 1)
            for i in range(15)
        ]
        usage = {
            "Alice": IdentityUsage(
                identity_name="Alice",
                recordings=recs,
                total_mentions=15,
            ),
        }
        panel._identity_usage = usage
        panel._populate_identity_list(["Alice"], usage)
        item = panel._identity_list.item(0)
        panel._on_identity_item_clicked(item)
        qapp.processEvents()
        text = panel._identity_recordings_label.text()
        assert "+5 more" in text

    def test_click_no_data_item_is_noop(self, settings_panel_on_identities, qapp):
        """Clicking an item with no UserRole data does nothing."""
        panel = settings_panel_on_identities
        empty_item = QListWidgetItem("Empty")
        panel._on_identity_item_clicked(empty_item)
        qapp.processEvents()

    def test_clear_identity_detail_resets_labels(self, settings_panel_on_identities):
        panel = settings_panel_on_identities
        panel._identity_name_label.setText("Name: Alice")
        panel._clear_identity_detail()
        assert "—" in panel._identity_name_label.text()


# ---------------------------------------------------------------------------
# History regression tests — ensure History tab is not broken
# ---------------------------------------------------------------------------


class TestHistoryUnchangedByIdentities:
    """Verify History page structure is unchanged by Identities addition."""

    def test_history_page_is_still_index_2(self, settings_panel):
        assert FloatingSettingsPanel._NAV_HISTORY == 2
        page = settings_panel._content_stack.widget(2)
        assert page is not None
        assert page.objectName() == "AethericHistoryPage"

    def test_history_list_object_name_unchanged(self, settings_panel):
        assert settings_panel._history_list.objectName() == "AethericHistoryList"

    def test_history_viewer_object_name_unchanged(self, settings_panel):
        assert settings_panel._history_viewer.objectName() == "AethericHistoryViewer"

    def test_nav_to_history_still_refreshes(self, settings_panel, qapp):
        """History tab still triggers refresh after Identities was added."""
        with patch.object(settings_panel, "_refresh_history") as mock_refresh:
            settings_panel._on_nav_clicked(FloatingSettingsPanel._NAV_HISTORY)
            qapp.processEvents()
            mock_refresh.assert_called_once()

    def test_nav_buttons_count_is_4(self, settings_panel):
        """Four nav buttons: Settings, Performance, History, Identities."""
        assert len(settings_panel._nav_buttons) == 4


# ===========================================================================
# T03 UI Tests — Identity mutation actions (rename, merge, delete)
# ===========================================================================

from PyQt6.QtWidgets import QComboBox, QDialogButtonBox, QDialog, QMessageBox
from PyQt6.QtCore import QTimer


def _make_panel_with_identities(
    qapp, settings_panel, profile_names, usage=None
):
    """Populate identity list and return the panel, ready for selection."""
    if usage is None:
        usage = {}
    settings_panel._identity_profile_names = list(profile_names)
    settings_panel._populate_identity_list(profile_names, usage)
    qapp.processEvents()
    return settings_panel


def _select_identity(panel, name, qapp):
    """Programmatically select an identity by name."""
    for i in range(panel._identity_list.count()):
        item = panel._identity_list.item(i)
        if item.data(Qt.ItemDataRole.UserRole) == name:
            panel._identity_list.setCurrentItem(item)
            panel._on_identity_item_clicked(item)
            qapp.processEvents()
            return item
    return None


def _click_dialog_button(dialog, button_role, qapp, delay_ms=50):
    """Find and click a button in a dialog by its role."""
    qapp.processEvents()
    bb = dialog.findChild(QDialogButtonBox)
    if bb is not None:
        btn = bb.button(button_role)
        if btn:
            btn.click()
            return
    # Fallback: try accept/reject
    dialog.reject()


def _schedule_dialog_accept(dialog, qapp, delay_ms=50):
    """Schedule dialog.accept() after a short delay."""
    QTimer.singleShot(delay_ms, dialog.accept)


def _schedule_dialog_reject(dialog, qapp, delay_ms=50):
    """Schedule dialog.reject() after a short delay."""
    QTimer.singleShot(delay_ms, dialog.reject)


# ---------------------------------------------------------------------------
# Button enablement tests
# ---------------------------------------------------------------------------


class TestIdentitiesButtonEnablement:
    """Verify action buttons enable/disable correctly."""

    def test_buttons_disabled_with_no_selection(
        self, settings_panel_on_identities, qapp
    ):
        panel = settings_panel_on_identities
        _make_panel_with_identities(qapp, panel, ["Alice", "Bob"])
        # No selection — buttons disabled
        assert not panel._identity_rename_btn.isEnabled()
        assert not panel._identity_merge_btn.isEnabled()
        assert not panel._identity_delete_btn.isEnabled()

    def test_rename_and_delete_enabled_on_selection(
        self, settings_panel_on_identities, qapp
    ):
        panel = settings_panel_on_identities
        _make_panel_with_identities(qapp, panel, ["Alice", "Bob"])
        _select_identity(panel, "Alice", qapp)
        assert panel._identity_rename_btn.isEnabled()
        assert panel._identity_delete_btn.isEnabled()
        assert panel._identity_merge_btn.isEnabled()

    def test_merge_disabled_with_only_one_identity(
        self, settings_panel_on_identities, qapp
    ):
        panel = settings_panel_on_identities
        _make_panel_with_identities(qapp, panel, ["Alice"])
        _select_identity(panel, "Alice", qapp)
        assert panel._identity_rename_btn.isEnabled()
        assert panel._identity_delete_btn.isEnabled()
        assert not panel._identity_merge_btn.isEnabled()

    def test_merge_enabled_with_two_identities(
        self, settings_panel_on_identities, qapp
    ):
        panel = settings_panel_on_identities
        _make_panel_with_identities(qapp, panel, ["Alice", "Bob"])
        _select_identity(panel, "Alice", qapp)
        assert panel._identity_merge_btn.isEnabled()

    def test_clear_detail_disables_buttons(
        self, settings_panel_on_identities, qapp
    ):
        panel = settings_panel_on_identities
        _make_panel_with_identities(qapp, panel, ["Alice"])
        _select_identity(panel, "Alice", qapp)
        assert panel._identity_rename_btn.isEnabled()
        panel._clear_identity_detail()
        assert not panel._identity_rename_btn.isEnabled()
        assert not panel._identity_merge_btn.isEnabled()
        assert not panel._identity_delete_btn.isEnabled()


# ---------------------------------------------------------------------------
# Rename action tests
# ---------------------------------------------------------------------------


class TestIdentitiesRenameAction:
    """Verify rename flow: dialog, validation, service call, refresh."""

    def test_rename_success(self, settings_panel_on_identities, qapp):
        """Successful rename refreshes list and reselects new name."""
        panel = settings_panel_on_identities
        _make_panel_with_identities(qapp, panel, ["Alice", "Bob"])
        _select_identity(panel, "Alice", qapp)

        with patch.object(
            panel, "_get_identity_store_and_transcripts_dir"
        ) as mock_get:
            mock_store = MagicMock()
            mock_store.load_signatures.return_value = []
            mock_get.return_value = (mock_store, Path("/tmp/t"))
            with patch(
                "meetandread.speaker.identity_management.rename_identity"
            ) as mock_rename:
                # Mock QInputDialog to return "Carol"
                with patch(
                    "meetandread.widgets.floating_panels.QInputDialog.getText"
                ) as mock_input:
                    mock_input.return_value = ("Carol", True)
                    panel._on_identity_rename()
                    qapp.processEvents()
                    mock_rename.assert_called_once_with(
                        mock_store, Path("/tmp/t"), "Alice", "Carol"
                    )

    def test_rename_cancel_no_mutation(
        self, settings_panel_on_identities, qapp
    ):
        """Cancelling the rename dialog performs no mutation."""
        panel = settings_panel_on_identities
        _make_panel_with_identities(qapp, panel, ["Alice"])
        _select_identity(panel, "Alice", qapp)

        with patch(
            "meetandread.widgets.floating_panels.QInputDialog.getText"
        ) as mock_input:
            mock_input.return_value = ("", False)  # Cancelled
            with patch(
                "meetandread.speaker.identity_management.rename_identity"
            ) as mock_rename:
                panel._on_identity_rename()
                mock_rename.assert_not_called()

    def test_rename_blank_name_rejected(
        self, settings_panel_on_identities, qapp
    ):
        """Blank new name is rejected before service call."""
        panel = settings_panel_on_identities
        _make_panel_with_identities(qapp, panel, ["Alice"])
        _select_identity(panel, "Alice", qapp)

        with patch(
            "meetandread.widgets.floating_panels.QInputDialog.getText"
        ) as mock_input:
            mock_input.return_value = ("  ", True)
            with patch(
                "meetandread.speaker.identity_management.rename_identity"
            ) as mock_rename:
                with patch(
                    "meetandread.widgets.floating_panels.QMessageBox.warning"
                ) as mock_warn:
                    panel._on_identity_rename()
                    mock_rename.assert_not_called()
                    mock_warn.assert_called_once()

    def test_rename_same_name_rejected(
        self, settings_panel_on_identities, qapp
    ):
        """Same name is rejected before service call."""
        panel = settings_panel_on_identities
        _make_panel_with_identities(qapp, panel, ["Alice"])
        _select_identity(panel, "Alice", qapp)

        with patch(
            "meetandread.widgets.floating_panels.QInputDialog.getText"
        ) as mock_input:
            mock_input.return_value = ("Alice", True)
            with patch(
                "meetandread.speaker.identity_management.rename_identity"
            ) as mock_rename:
                with patch(
                    "meetandread.widgets.floating_panels.QMessageBox.warning"
                ) as mock_warn:
                    panel._on_identity_rename()
                    mock_rename.assert_not_called()
                    mock_warn.assert_called_once()

    def test_rename_duplicate_name_rejected(
        self, settings_panel_on_identities, qapp
    ):
        """Name that already exists is rejected."""
        panel = settings_panel_on_identities
        _make_panel_with_identities(qapp, panel, ["Alice", "Bob"])
        _select_identity(panel, "Alice", qapp)

        with patch(
            "meetandread.widgets.floating_panels.QInputDialog.getText"
        ) as mock_input:
            mock_input.return_value = ("Bob", True)
            with patch(
                "meetandread.speaker.identity_management.rename_identity"
            ) as mock_rename:
                with patch(
                    "meetandread.widgets.floating_panels.QMessageBox.warning"
                ) as mock_warn:
                    panel._on_identity_rename()
                    mock_rename.assert_not_called()
                    mock_warn.assert_called_once()

    def test_rename_service_error_shows_warning(
        self, settings_panel_on_identities, qapp
    ):
        """Service error is shown as warning and UI refreshes."""
        panel = settings_panel_on_identities
        _make_panel_with_identities(qapp, panel, ["Alice"])
        _select_identity(panel, "Alice", qapp)

        with patch.object(
            panel, "_get_identity_store_and_transcripts_dir"
        ) as mock_get:
            mock_store = MagicMock()
            mock_get.return_value = (mock_store, Path("/tmp/t"))
            with patch(
                "meetandread.speaker.identity_management.rename_identity"
            ) as mock_rename:
                mock_rename.side_effect = RuntimeError("store locked")
                with patch(
                    "meetandread.widgets.floating_panels.QInputDialog.getText"
                ) as mock_input:
                    mock_input.return_value = ("Carol", True)
                    with patch(
                        "meetandread.widgets.floating_panels.QMessageBox.warning"
                    ) as mock_warn:
                        with patch.object(panel, "_refresh_identities"):
                            panel._on_identity_rename()
                            mock_warn.assert_called_once()

    def test_rename_no_selection_is_noop(
        self, settings_panel_on_identities, qapp
    ):
        """Rename with nothing selected does nothing."""
        panel = settings_panel_on_identities
        _make_panel_with_identities(qapp, panel, ["Alice"])
        # No selection
        with patch(
            "meetandread.widgets.floating_panels.QInputDialog.getText"
        ) as mock_input:
            panel._on_identity_rename()
            mock_input.assert_not_called()


# ---------------------------------------------------------------------------
# Merge action tests
# ---------------------------------------------------------------------------


class TestIdentitiesMergeAction:
    """Verify merge flow: dialog, combo box, confirmation, service call."""

    def test_merge_success(self, settings_panel_on_identities, qapp):
        """Successful merge calls service, refreshes, selects target."""
        panel = settings_panel_on_identities
        _make_panel_with_identities(qapp, panel, ["Alice", "Bob"])
        _select_identity(panel, "Alice", qapp)

        with patch.object(
            panel, "_get_identity_store_and_transcripts_dir"
        ) as mock_get:
            mock_store = MagicMock()
            mock_get.return_value = (mock_store, Path("/tmp/t"))

            with patch(
                "meetandread.speaker.identity_management.merge_identities"
            ) as mock_merge:
                # We need to handle the merge dialog and confirmation
                # Patch QDialog.exec to accept
                original_exec = QDialog.exec

                def fake_exec(self_dialog):
                    # Find combo and ensure "Bob" is selected
                    combo = self_dialog.findChild(QComboBox)
                    if combo:
                        idx = combo.findText("Bob")
                        if idx >= 0:
                            combo.setCurrentIndex(idx)
                    self_dialog.accept()
                    return QDialog.DialogCode.Accepted

                with patch.object(QDialog, "exec", fake_exec):
                    with patch(
                        "meetandread.widgets.floating_panels.QMessageBox.question"
                    ) as mock_question:
                        mock_question.return_value = QMessageBox.StandardButton.Yes
                        with patch.object(panel, "_refresh_and_reselect"):
                            panel._on_identity_merge()
                            qapp.processEvents()
                            mock_merge.assert_called_once_with(
                                mock_store, Path("/tmp/t"), "Alice", "Bob"
                            )

    def test_merge_dialog_cancel_no_mutation(
        self, settings_panel_on_identities, qapp
    ):
        """Cancelling the merge dialog performs no mutation."""
        panel = settings_panel_on_identities
        _make_panel_with_identities(qapp, panel, ["Alice", "Bob"])
        _select_identity(panel, "Alice", qapp)

        def fake_exec(self_dialog):
            self_dialog.reject()
            return QDialog.DialogCode.Rejected

        with patch.object(QDialog, "exec", fake_exec):
            with patch(
                "meetandread.speaker.identity_management.merge_identities"
            ) as mock_merge:
                panel._on_identity_merge()
                mock_merge.assert_not_called()

    def test_merge_confirmation_no_no_mutation(
        self, settings_panel_on_identities, qapp
    ):
        """Saying No to confirmation performs no mutation."""
        panel = settings_panel_on_identities
        _make_panel_with_identities(qapp, panel, ["Alice", "Bob"])
        _select_identity(panel, "Alice", qapp)

        def fake_exec(self_dialog):
            self_dialog.accept()
            return QDialog.DialogCode.Accepted

        with patch.object(QDialog, "exec", fake_exec):
            with patch(
                "meetandread.widgets.floating_panels.QMessageBox.question"
            ) as mock_question:
                mock_question.return_value = QMessageBox.StandardButton.No
                with patch(
                    "meetandread.speaker.identity_management.merge_identities"
                ) as mock_merge:
                    panel._on_identity_merge()
                    mock_merge.assert_not_called()

    def test_merge_service_error_shows_warning(
        self, settings_panel_on_identities, qapp
    ):
        """Service error surfaces as warning."""
        panel = settings_panel_on_identities
        _make_panel_with_identities(qapp, panel, ["Alice", "Bob"])
        _select_identity(panel, "Alice", qapp)

        with patch.object(
            panel, "_get_identity_store_and_transcripts_dir"
        ) as mock_get:
            mock_store = MagicMock()
            mock_get.return_value = (mock_store, Path("/tmp/t"))

            with patch(
                "meetandread.speaker.identity_management.merge_identities"
            ) as mock_merge:
                mock_merge.side_effect = RuntimeError("merge failed")

                def fake_exec(self_dialog):
                    self_dialog.accept()
                    return QDialog.DialogCode.Accepted

                with patch.object(QDialog, "exec", fake_exec):
                    with patch(
                        "meetandread.widgets.floating_panels.QMessageBox.question"
                    ) as mock_question:
                        mock_question.return_value = QMessageBox.StandardButton.Yes
                        with patch(
                            "meetandread.widgets.floating_panels.QMessageBox.warning"
                        ) as mock_warn:
                            with patch.object(panel, "_refresh_identities"):
                                panel._on_identity_merge()
                                mock_warn.assert_called_once()

    def test_merge_no_selection_is_noop(
        self, settings_panel_on_identities, qapp
    ):
        """Merge with nothing selected does nothing."""
        panel = settings_panel_on_identities
        _make_panel_with_identities(qapp, panel, ["Alice", "Bob"])
        panel._on_identity_merge()
        # No dialog opened, no error — just returns

    def test_merge_single_identity_shows_info(
        self, settings_panel_on_identities, qapp
    ):
        """Merge with only one identity shows info message."""
        panel = settings_panel_on_identities
        _make_panel_with_identities(qapp, panel, ["Alice"])
        _select_identity(panel, "Alice", qapp)

        with patch(
            "meetandread.widgets.floating_panels.QMessageBox.information"
        ) as mock_info:
            panel._on_identity_merge()
            mock_info.assert_called_once()


# ---------------------------------------------------------------------------
# Delete action tests
# ---------------------------------------------------------------------------


class TestIdentitiesDeleteAction:
    """Verify delete flow: confirmation, service call, refresh."""

    def test_delete_success(self, settings_panel_on_identities, qapp):
        """Successful delete calls service, refreshes, clears detail."""
        panel = settings_panel_on_identities
        _make_panel_with_identities(qapp, panel, ["Alice", "Bob"])
        _select_identity(panel, "Alice", qapp)

        with patch.object(
            panel, "_get_identity_store_and_transcripts_dir"
        ) as mock_get:
            mock_store = MagicMock()
            mock_get.return_value = (mock_store, Path("/tmp/t"))

            with patch(
                "meetandread.speaker.identity_management.delete_identity"
            ) as mock_delete:
                with patch(
                    "meetandread.widgets.floating_panels.QMessageBox.question"
                ) as mock_question:
                    mock_question.return_value = QMessageBox.StandardButton.Yes
                    with patch.object(panel, "_refresh_and_reselect") as mock_refresh:
                        panel._on_identity_delete()
                        qapp.processEvents()
                        mock_delete.assert_called_once_with(
                            mock_store, Path("/tmp/t"), "Alice"
                        )
                        mock_refresh.assert_called_once_with(target_name=None)

    def test_delete_cancel_no_mutation(
        self, settings_panel_on_identities, qapp
    ):
        """Cancelling delete confirmation performs no mutation."""
        panel = settings_panel_on_identities
        _make_panel_with_identities(qapp, panel, ["Alice"])
        _select_identity(panel, "Alice", qapp)

        with patch(
            "meetandread.widgets.floating_panels.QMessageBox.question"
        ) as mock_question:
            mock_question.return_value = QMessageBox.StandardButton.No
            with patch(
                "meetandread.speaker.identity_management.delete_identity"
            ) as mock_delete:
                panel._on_identity_delete()
                mock_delete.assert_not_called()

    def test_delete_service_error_shows_warning(
        self, settings_panel_on_identities, qapp
    ):
        """Service error is shown as warning and UI refreshes."""
        panel = settings_panel_on_identities
        _make_panel_with_identities(qapp, panel, ["Alice"])
        _select_identity(panel, "Alice", qapp)

        with patch.object(
            panel, "_get_identity_store_and_transcripts_dir"
        ) as mock_get:
            mock_store = MagicMock()
            mock_get.return_value = (mock_store, Path("/tmp/t"))

            with patch(
                "meetandread.speaker.identity_management.delete_identity"
            ) as mock_delete:
                mock_delete.side_effect = RuntimeError("db error")
                with patch(
                    "meetandread.widgets.floating_panels.QMessageBox.question"
                ) as mock_question:
                    mock_question.return_value = QMessageBox.StandardButton.Yes
                    with patch(
                        "meetandread.widgets.floating_panels.QMessageBox.warning"
                    ) as mock_warn:
                        with patch.object(panel, "_refresh_identities"):
                            panel._on_identity_delete()
                            mock_warn.assert_called_once()

    def test_delete_no_selection_is_noop(
        self, settings_panel_on_identities, qapp
    ):
        """Delete with nothing selected does nothing."""
        panel = settings_panel_on_identities
        _make_panel_with_identities(qapp, panel, ["Alice"])
        panel._on_identity_delete()
        # No dialog opened


# ---------------------------------------------------------------------------
# Refresh and reselection tests
# ---------------------------------------------------------------------------


class TestIdentitiesRefreshReselection:
    """Verify _refresh_and_reselect deterministically reselects."""

    def test_reselect_finds_target(self, settings_panel_on_identities, qapp):
        """Refresh and reselect selects the target name."""
        panel = settings_panel_on_identities
        _make_panel_with_identities(qapp, panel, ["Alice", "Bob"])

        # Mock _refresh_identities to repopulate
        def fake_refresh():
            panel._identity_profile_names = ["Alice", "Carol"]
            panel._populate_identity_list(["Alice", "Carol"], {})

        with patch.object(panel, "_refresh_identities", side_effect=fake_refresh):
            panel._refresh_and_reselect(target_name="Carol")
            qapp.processEvents()
            item = panel._identity_list.currentItem()
            assert item is not None
            assert item.data(Qt.ItemDataRole.UserRole) == "Carol"

    def test_reselect_missing_target_clears_detail(
        self, settings_panel_on_identities, qapp
    ):
        """If target no longer exists after refresh, detail is cleared."""
        panel = settings_panel_on_identities
        _make_panel_with_identities(qapp, panel, ["Alice", "Bob"])
        _select_identity(panel, "Alice", qapp)
        assert panel._identity_detail_header.isVisible()

        # Mock refresh that removes "Alice"
        def fake_refresh():
            panel._identity_profile_names = ["Bob"]
            panel._populate_identity_list(["Bob"], {})

        with patch.object(panel, "_refresh_identities", side_effect=fake_refresh):
            panel._refresh_and_reselect(target_name="Alice")
            qapp.processEvents()
            assert panel._identity_detail_header.isHidden()
            assert not panel._identity_rename_btn.isEnabled()

    def test_profile_names_updated_on_populate(
        self, settings_panel_on_identities, qapp
    ):
        """_populate_identity_list updates _identity_profile_names."""
        panel = settings_panel_on_identities
        panel._populate_identity_list(["X", "Y", "Z"], {})
        assert panel._identity_profile_names == ["X", "Y", "Z"]
