"""Tests for the identity-link persistence flow (T02).

Covers: metadata+body update, speaker_matches, segment key variants,
malformed inputs, signature propagation, PII-safe logging.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pytest

from meetandread.speaker.signatures import VoiceSignatureStore


def _parse_metadata(md_path: Path) -> Dict[str, Any]:
    content = md_path.read_text(encoding="utf-8")
    marker = "<!-- METADATA: "
    idx = content.find(marker)
    assert idx != -1, "No metadata footer found"
    json_str = content[idx + len(marker):]
    if json_str.rstrip().endswith(" -->"):
        json_str = json_str.rstrip()[: -len(" -->")]
    return json.loads(json_str)


def _make_md(
    tmp_path, words, segments, speaker_matches=None, *, name="test_link.md"
) -> Path:
    lines = ["# Transcript", "", "**Recorded:** 2026-04-22T14:30:00", ""]
    cur_spk = None
    cur_words = []
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


class TestMetadataAndBody:

    def test_updates_all_surfaces(self, tmp_path):
        from meetandread.widgets.floating_panels import _link_speaker_identity_in_file
        w = [
            {"text": "Hi", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": "SPK_0"},
            {"text": "Bye", "start_time": 0.5, "end_time": 1.0, "confidence": 85, "speaker_id": "SPK_0"},
            {"text": "Yo", "start_time": 1.0, "end_time": 1.5, "confidence": 88, "speaker_id": "SPK_1"},
        ]
        s = [
            {"start_time": 0.0, "end_time": 1.0, "speaker_id": "SPK_0"},
            {"start_time": 1.0, "end_time": 1.5, "speaker_id": "SPK_1"},
        ]
        md = _make_md(tmp_path, w, s, {"SPK_0": None, "SPK_1": None})
        _link_speaker_identity_in_file(md, "SPK_0", "Alice")
        d = _parse_metadata(md)
        assert d["words"][0]["speaker_id"] == "Alice"
        assert d["words"][2]["speaker_id"] == "SPK_1"
        assert d["segments"][0]["speaker_id"] == "Alice"
        assert "**Alice**" in md.read_text(encoding="utf-8")

    def test_creates_speaker_matches_when_missing(self, tmp_path):
        from meetandread.widgets.floating_panels import _link_speaker_identity_in_file
        w = [{"text": "Hi", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": "SPK_0"}]
        s = [{"start_time": 0.0, "end_time": 0.5, "speaker_id": "SPK_0"}]
        md = _make_md(tmp_path, w, s)
        _link_speaker_identity_in_file(md, "SPK_0", "Alice")
        d = _parse_metadata(md)
        m = d["speaker_matches"]["SPK_0"]
        assert m["identity_name"] == "Alice"
        assert m["score"] == 1.0
        assert m["confidence"] == "manual"

    def test_preserves_prior_score(self, tmp_path):
        from meetandread.widgets.floating_panels import _link_speaker_identity_in_file
        w = [{"text": "Hi", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": "SPK_0"}]
        s = [{"start_time": 0.0, "end_time": 0.5, "speaker_id": "SPK_0"}]
        sm = {"SPK_0": {"identity_name": "Bob", "score": 0.92, "confidence": "high"}}
        md = _make_md(tmp_path, w, s, sm)
        _link_speaker_identity_in_file(md, "SPK_0", "Alice")
        d = _parse_metadata(md)
        m = d["speaker_matches"]["SPK_0"]
        assert m["identity_name"] == "Alice"
        assert m["score"] == 0.92
        assert m["confidence"] == "high"

    def test_replaces_null_match(self, tmp_path):
        from meetandread.widgets.floating_panels import _link_speaker_identity_in_file
        w = [{"text": "Hi", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": "SPK_0"}]
        s = [{"start_time": 0.0, "end_time": 0.5, "speaker_id": "SPK_0"}]
        md = _make_md(tmp_path, w, s, {"SPK_0": None})
        _link_speaker_identity_in_file(md, "SPK_0", "Carol")
        d = _parse_metadata(md)
        assert isinstance(d["speaker_matches"]["SPK_0"], dict)
        assert d["speaker_matches"]["SPK_0"]["identity_name"] == "Carol"


class TestSegmentKeyVariants:

    def test_speaker_key(self, tmp_path):
        from meetandread.widgets.floating_panels import _link_speaker_identity_in_file
        w = [{"text": "Hi", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": "SPK_0"}]
        s = [{"start_time": 0.0, "end_time": 0.5, "speaker": "SPK_0"}]
        md = _make_md(tmp_path, w, s, name="sk.md")
        _link_speaker_identity_in_file(md, "SPK_0", "Alice")
        assert _parse_metadata(md)["segments"][0]["speaker"] == "Alice"

    def test_speaker_id_key(self, tmp_path):
        from meetandread.widgets.floating_panels import _link_speaker_identity_in_file
        w = [{"text": "Hi", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": "SPK_0"}]
        s = [{"start_time": 0.0, "end_time": 0.5, "speaker_id": "SPK_0"}]
        md = _make_md(tmp_path, w, s, name="sid.md")
        _link_speaker_identity_in_file(md, "SPK_0", "Alice")
        assert _parse_metadata(md)["segments"][0]["speaker_id"] == "Alice"

    def test_both_keys(self, tmp_path):
        from meetandread.widgets.floating_panels import _link_speaker_identity_in_file
        w = [{"text": "Hi", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": "SPK_0"}]
        s = [{"start_time": 0.0, "end_time": 0.5, "speaker_id": "SPK_0", "speaker": "SPK_0"}]
        md = _make_md(tmp_path, w, s, name="both.md")
        _link_speaker_identity_in_file(md, "SPK_0", "Alice")
        d = _parse_metadata(md)
        assert d["segments"][0]["speaker_id"] == "Alice"
        assert d["segments"][0]["speaker"] == "Alice"


class TestMalformedAndNegative:

    def test_malformed_json_no_write(self, tmp_path):
        from meetandread.widgets.floating_panels import _link_speaker_identity_in_file
        p = tmp_path / "bad.md"
        c = "# T\n\n**SPK_0**\n\nHi\n\n---\n\n<!-- METADATA: {bad} -->\n"
        p.write_text(c, encoding="utf-8")
        _link_speaker_identity_in_file(p, "SPK_0", "Alice")
        assert p.read_text(encoding="utf-8") == c

    def test_no_footer_no_write(self, tmp_path):
        from meetandread.widgets.floating_panels import _link_speaker_identity_in_file
        p = tmp_path / "bare.md"
        c = "# Bare\n\n**SPK_0**\n\nHi\n"
        p.write_text(c, encoding="utf-8")
        _link_speaker_identity_in_file(p, "SPK_0", "Alice")
        assert p.read_text(encoding="utf-8") == c

    def test_same_name_noop(self, tmp_path):
        from meetandread.widgets.floating_panels import _link_speaker_identity_in_file
        w = [{"text": "Hi", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": "SPK_0"}]
        s = [{"start_time": 0.0, "end_time": 0.5, "speaker_id": "SPK_0"}]
        md = _make_md(tmp_path, w, s)
        orig = md.read_text(encoding="utf-8")
        _link_speaker_identity_in_file(md, "SPK_0", "SPK_0")
        assert md.read_text(encoding="utf-8") == orig

    def test_empty_identity_noop(self, tmp_path):
        from meetandread.widgets.floating_panels import _link_speaker_identity_in_file
        w = [{"text": "Hi", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": "SPK_0"}]
        s = [{"start_time": 0.0, "end_time": 0.5, "speaker_id": "SPK_0"}]
        md = _make_md(tmp_path, w, s)
        orig = md.read_text(encoding="utf-8")
        _link_speaker_identity_in_file(md, "SPK_0", "")
        assert md.read_text(encoding="utf-8") == orig

    def test_whitespace_identity_noop(self, tmp_path):
        from meetandread.widgets.floating_panels import _link_speaker_identity_in_file
        w = [{"text": "Hi", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": "SPK_0"}]
        s = [{"start_time": 0.0, "end_time": 0.5, "speaker_id": "SPK_0"}]
        md = _make_md(tmp_path, w, s)
        orig = md.read_text(encoding="utf-8")
        _link_speaker_identity_in_file(md, "SPK_0", "   ")
        assert md.read_text(encoding="utf-8") == orig

    def test_substring_not_replaced(self, tmp_path):
        from meetandread.widgets.floating_panels import _link_speaker_identity_in_file
        w = [
            {"text": "Hi", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": "SPK_0"},
            {"text": "Yo", "start_time": 0.5, "end_time": 1.0, "confidence": 88, "speaker_id": "SPK_01"},
        ]
        s = [
            {"start_time": 0.0, "end_time": 0.5, "speaker_id": "SPK_0"},
            {"start_time": 0.5, "end_time": 1.0, "speaker_id": "SPK_01"},
        ]
        md = _make_md(tmp_path, w, s)
        _link_speaker_identity_in_file(md, "SPK_0", "Alice")
        d = _parse_metadata(md)
        assert d["words"][0]["speaker_id"] == "Alice"
        assert d["words"][1]["speaker_id"] == "SPK_01"
        assert "**SPK_01**" in md.read_text(encoding="utf-8")


class TestSignaturePropagation:

    def test_propagates_existing_profile(self, tmp_path):
        from meetandread.widgets.floating_panels import _link_speaker_identity_in_file
        db = tmp_path / "speaker_signatures.db"
        emb = np.random.randn(256).astype(np.float32)
        with VoiceSignatureStore(db_path=str(db)) as st:
            st.save_signature("SPK_0", emb, averaged_from_segments=3)
        w = [{"text": "Hi", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": "SPK_0"}]
        s = [{"start_time": 0.0, "end_time": 0.5, "speaker_id": "SPK_0"}]
        md = _make_md(tmp_path, w, s)
        _link_speaker_identity_in_file(md, "SPK_0", "Alice")
        with VoiceSignatureStore(db_path=str(db)) as st:
            names = [p.name for p in st.load_signatures()]
            assert "Alice" in names
            assert "SPK_0" not in names

    def test_no_crash_no_db(self, tmp_path):
        from meetandread.widgets.floating_panels import _link_speaker_identity_in_file
        w = [{"text": "Hi", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": "SPK_0"}]
        s = [{"start_time": 0.0, "end_time": 0.5, "speaker_id": "SPK_0"}]
        md = _make_md(tmp_path, w, s)
        _link_speaker_identity_in_file(md, "SPK_0", "Alice")
        assert _parse_metadata(md)["words"][0]["speaker_id"] == "Alice"

    def test_no_crash_missing_profile(self, tmp_path):
        from meetandread.widgets.floating_panels import _link_speaker_identity_in_file
        db = tmp_path / "speaker_signatures.db"
        with VoiceSignatureStore(db_path=str(db)) as st:
            st.save_signature("SPK_1", np.random.randn(256).astype(np.float32))
        w = [{"text": "Hi", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": "SPK_0"}]
        s = [{"start_time": 0.0, "end_time": 0.5, "speaker_id": "SPK_0"}]
        md = _make_md(tmp_path, w, s)
        _link_speaker_identity_in_file(md, "SPK_0", "Alice")
        with VoiceSignatureStore(db_path=str(db)) as st:
            names = [p.name for p in st.load_signatures()]
            assert "SPK_1" in names
            assert "Alice" not in names


class TestPIISafeLogging:

    def test_no_identity_name_in_logs(self, tmp_path, caplog):
        from meetandread.widgets.floating_panels import _link_speaker_identity_in_file
        w = [{"text": "Hi", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": "SPK_0"}]
        s = [{"start_time": 0.0, "end_time": 0.5, "speaker_id": "SPK_0"}]
        md = _make_md(tmp_path, w, s)
        with caplog.at_level(logging.DEBUG, logger="meetandread.widgets.floating_panels"):
            _link_speaker_identity_in_file(md, "SPK_0", "SecretName")
        for r in caplog.records:
            assert "SecretName" not in r.message

    def test_no_prior_match_in_logs(self, tmp_path, caplog):
        from meetandread.widgets.floating_panels import _link_speaker_identity_in_file
        w = [{"text": "Hi", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": "SPK_0"}]
        s = [{"start_time": 0.0, "end_time": 0.5, "speaker_id": "SPK_0"}]
        md = _make_md(tmp_path, w, s, {"SPK_0": {"identity_name": "OldName", "score": 0.9, "confidence": "high"}})
        with caplog.at_level(logging.DEBUG, logger="meetandread.widgets.floating_panels"):
            _link_speaker_identity_in_file(md, "SPK_0", "NewName")
        for r in caplog.records:
            assert "OldName" not in r.message
            assert "NewName" not in r.message

    def test_malformed_metadata_warning(self, tmp_path, caplog):
        from meetandread.widgets.floating_panels import _link_speaker_identity_in_file
        p = tmp_path / "bad.md"
        p.write_text("# T\n\n---\n\n<!-- METADATA: {bad} -->\n", encoding="utf-8")
        with caplog.at_level(logging.WARNING, logger="meetandread.widgets.floating_panels"):
            _link_speaker_identity_in_file(p, "SPK_0", "Alice")
        assert any("malformed" in r.message.lower() or "metadata" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# Integration tests: _open_identity_link_dialog + anchor flow (T03)
# ---------------------------------------------------------------------------

class TestOpenIdentityLinkDialogIntegration:
    """Integration tests for _open_identity_link_dialog helper."""

    def test_missing_md_path_returns_false(self, qapp):
        """No md_path (None) returns False — no crash."""
        from meetandread.widgets.floating_panels import _open_identity_link_dialog
        from unittest.mock import patch

        with patch("meetandread.widgets.floating_panels.SpeakerIdentityLinkDialog") as mock_dlg:
            result = _open_identity_link_dialog(None, "SPK_0", None)
        assert result is False
        mock_dlg.assert_not_called()

    def test_nonexistent_md_path_returns_false(self, qapp, tmp_path):
        """Nonexistent md_path returns False — no crash."""
        from meetandread.widgets.floating_panels import _open_identity_link_dialog
        from unittest.mock import patch

        fake_path = tmp_path / "nonexistent.md"
        with patch("meetandread.widgets.floating_panels.SpeakerIdentityLinkDialog") as mock_dlg:
            result = _open_identity_link_dialog(fake_path, "SPK_0", None)
        assert result is False
        mock_dlg.assert_not_called()

    def test_dialog_cancel_returns_false(self, qapp, tmp_path):
        """Dialog rejected → returns False, no file mutation."""
        from meetandread.widgets.floating_panels import _open_identity_link_dialog
        from unittest.mock import patch, MagicMock
        from PyQt6.QtWidgets import QDialog

        w = [{"text": "Hi", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": "SPK_0"}]
        s = [{"start_time": 0.0, "end_time": 0.5, "speaker_id": "SPK_0"}]
        md = _make_md(tmp_path, w, s)
        original = md.read_text(encoding="utf-8")

        mock_dialog = MagicMock()
        mock_dialog.exec.return_value = QDialog.DialogCode.Rejected

        with patch("meetandread.widgets.floating_panels.SpeakerIdentityLinkDialog", return_value=mock_dialog):
            result = _open_identity_link_dialog(md, "SPK_0", None)

        assert result is False
        assert md.read_text(encoding="utf-8") == original

    def test_dialog_accept_persists_identity(self, qapp, tmp_path):
        """Dialog accepted → file updated with identity."""
        from meetandread.widgets.floating_panels import _open_identity_link_dialog
        from unittest.mock import patch, MagicMock
        from PyQt6.QtWidgets import QDialog

        w = [{"text": "Hi", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": "SPK_0"}]
        s = [{"start_time": 0.0, "end_time": 0.5, "speaker_id": "SPK_0"}]
        md = _make_md(tmp_path, w, s)

        mock_dialog = MagicMock()
        mock_dialog.exec.return_value = QDialog.DialogCode.Accepted
        mock_dialog.selected_identity_name.return_value = "Alice"

        with patch("meetandread.widgets.floating_panels.SpeakerIdentityLinkDialog", return_value=mock_dialog), \
             patch("meetandread.widgets.floating_panels._propagate_identity_to_signatures"):
            result = _open_identity_link_dialog(md, "SPK_0", None)

        assert result is True
        data = _parse_metadata(md)
        assert data["words"][0]["speaker_id"] == "Alice"
        content = md.read_text(encoding="utf-8")
        assert "**Alice**" in content

    def test_dialog_accept_empty_name_returns_false(self, qapp, tmp_path):
        """Dialog accepted but selected_identity_name returns empty → no mutation."""
        from meetandread.widgets.floating_panels import _open_identity_link_dialog
        from unittest.mock import patch, MagicMock
        from PyQt6.QtWidgets import QDialog

        w = [{"text": "Hi", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": "SPK_0"}]
        s = [{"start_time": 0.0, "end_time": 0.5, "speaker_id": "SPK_0"}]
        md = _make_md(tmp_path, w, s)
        original = md.read_text(encoding="utf-8")

        mock_dialog = MagicMock()
        mock_dialog.exec.return_value = QDialog.DialogCode.Accepted
        mock_dialog.selected_identity_name.return_value = ""

        with patch("meetandread.widgets.floating_panels.SpeakerIdentityLinkDialog", return_value=mock_dialog):
            result = _open_identity_link_dialog(md, "SPK_0", None)

        assert result is False
        assert md.read_text(encoding="utf-8") == original

    def test_persistence_failure_returns_false(self, qapp, tmp_path):
        """_link_speaker_identity_in_file raises → returns False, no crash."""
        from meetandread.widgets.floating_panels import _open_identity_link_dialog
        from unittest.mock import patch, MagicMock
        from PyQt6.QtWidgets import QDialog

        w = [{"text": "Hi", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": "SPK_0"}]
        s = [{"start_time": 0.0, "end_time": 0.5, "speaker_id": "SPK_0"}]
        md = _make_md(tmp_path, w, s)

        mock_dialog = MagicMock()
        mock_dialog.exec.return_value = QDialog.DialogCode.Accepted
        mock_dialog.selected_identity_name.return_value = "Alice"

        with patch("meetandread.widgets.floating_panels.SpeakerIdentityLinkDialog", return_value=mock_dialog), \
             patch("meetandread.widgets.floating_panels._link_speaker_identity_in_file", side_effect=OSError("disk full")):
            result = _open_identity_link_dialog(md, "SPK_0", None)

        assert result is False

    def test_parses_speaker_matches_from_transcript(self, qapp, tmp_path):
        """Dialog receives speaker_matches from transcript metadata."""
        from meetandread.widgets.floating_panels import _open_identity_link_dialog
        from unittest.mock import patch, MagicMock, call
        from PyQt6.QtWidgets import QDialog

        w = [{"text": "Hi", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": "SPK_0"}]
        s = [{"start_time": 0.0, "end_time": 0.5, "speaker_id": "SPK_0"}]
        matches = {"SPK_0": {"identity_name": "Bob", "score": 0.85, "confidence": "high"}}
        md = _make_md(tmp_path, w, s, speaker_matches=matches)

        mock_dialog = MagicMock()
        mock_dialog.exec.return_value = QDialog.DialogCode.Rejected

        with patch("meetandread.widgets.floating_panels.SpeakerIdentityLinkDialog", return_value=mock_dialog) as mock_cls:
            _open_identity_link_dialog(md, "SPK_0", None)

        # Verify the dialog was constructed with the speaker_matches
        init_call = mock_cls.call_args
        assert init_call[1]["speaker_matches"] == matches or init_call[0][1] == matches

    def test_anchor_format_uses_speaker_prefix(self, qapp, tmp_path):
        """Verify anchor URL format is speaker:label (not speaker://label)."""
        from PyQt6.QtCore import QUrl

        url = QUrl("speaker:SPK_0")
        assert url.toString() == "speaker:SPK_0"
        # Ensure speaker:// is not used (QUrl normalizes host to lowercase)
        bad_url = QUrl("speaker://SPK_0")
        assert bad_url.toString() != "speaker://SPK_0"  # would be speaker://spk_0
