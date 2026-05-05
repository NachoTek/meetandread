"""Tests for SpeakerIdentityLinkDialog — history speaker identity linking UI.

Covers T01 must-haves:
- Dialog displays the current label and existing match confidence/status.
- Dialog loads identities exactly once per instance and filters in-memory.
- Dialog returns a chosen identity name for both existing and create-new flows.
- Empty and duplicate identity names are rejected before accept.
- Tests use inline fake store objects — no real untracked fixtures.
"""

import pytest

from PyQt6.QtWidgets import QApplication

from meetandread.widgets.floating_panels import SpeakerIdentityLinkDialog


# ---------------------------------------------------------------------------
# Qt application fixture (session-scoped to avoid repeated init)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def qapp():
    """Provide a QApplication for the test session."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


# ---------------------------------------------------------------------------
# Fake store helper
# ---------------------------------------------------------------------------

class FakeVoiceSignatureStore:
    """Minimal store-like object for testing without real SQLite.

    Simulates ``VoiceSignatureStore.load_signatures()`` return shape:
    a list of ``SpeakerProfile`` objects.
    """

    def __init__(self, profiles=None, *, load_error=None):
        """
        Args:
            profiles: Optional list of SpeakerProfile objects to return.
            load_error: If set, ``load_signatures()`` raises this exception.
        """
        self._profiles = profiles or []
        self._load_error = load_error

    def load_signatures(self):
        if self._load_error is not None:
            raise self._load_error
        return list(self._profiles)


def _make_profiles(*names: str):
    """Create SpeakerProfile objects with dummy embeddings."""
    import numpy as np
    from meetandread.speaker.models import SpeakerProfile
    return [
        SpeakerProfile(name=n, embedding=np.zeros(256, dtype=np.float32), num_samples=1)
        for n in names
    ]


# ---------------------------------------------------------------------------
# Test: Dialog initialization
# ---------------------------------------------------------------------------

class TestDialogInit:
    """Dialog initializes correctly with various inputs."""

    def test_shows_current_label(self, qapp):
        """Dialog must display the current raw speaker label."""
        store = FakeVoiceSignatureStore()
        dlg = SpeakerIdentityLinkDialog(
            current_label="SPK_0",
            speaker_matches={},
            store=store,
        )
        # Find the label showing current speaker
        label_text = dlg._current_label_label.text()
        assert "SPK_0" in label_text

    def test_loads_identities_into_list(self, qapp):
        """Dialog must load all store identities into the list widget."""
        profiles = _make_profiles("Alice", "Bob", "Charlie")
        store = FakeVoiceSignatureStore(profiles)
        dlg = SpeakerIdentityLinkDialog(
            current_label="SPK_0",
            speaker_matches={},
            store=store,
        )
        list_widget = dlg._identity_list
        assert list_widget.count() == 3
        names = [list_widget.item(i).text() for i in range(list_widget.count())]
        assert names == ["Alice", "Bob", "Charlie"]

    def test_empty_store_shows_no_identities(self, qapp):
        """Dialog must work with an empty store — no identities listed."""
        store = FakeVoiceSignatureStore()
        dlg = SpeakerIdentityLinkDialog(
            current_label="SPK_0",
            speaker_matches={},
            store=store,
        )
        assert dlg._identity_list.count() == 0

    def test_store_load_failure_shows_empty_list(self, qapp):
        """Dialog must handle store load failure gracefully — empty list, no crash."""
        store = FakeVoiceSignatureStore(load_error=RuntimeError("db locked"))
        dlg = SpeakerIdentityLinkDialog(
            current_label="SPK_0",
            speaker_matches={},
            store=store,
        )
        assert dlg._identity_list.count() == 0
        # Status message should indicate an issue (without PII)
        assert dlg._status_label.text() != ""

    def test_malformed_profile_name_skipped(self, qapp):
        """Profiles with empty names must be silently skipped."""
        import numpy as np
        from meetandread.speaker.models import SpeakerProfile
        profiles = [
            SpeakerProfile(name="Alice", embedding=np.zeros(256, dtype=np.float32)),
            SpeakerProfile(name="", embedding=np.zeros(256, dtype=np.float32)),
            SpeakerProfile(name="Bob", embedding=np.zeros(256, dtype=np.float32)),
        ]
        store = FakeVoiceSignatureStore(profiles)
        dlg = SpeakerIdentityLinkDialog(
            current_label="SPK_0",
            speaker_matches={},
            store=store,
        )
        names = [dlg._identity_list.item(i).text() for i in range(dlg._identity_list.count())]
        assert names == ["Alice", "Bob"]


# ---------------------------------------------------------------------------
# Test: Existing match display
# ---------------------------------------------------------------------------

class TestExistingMatch:
    """Dialog shows match status when speaker_matches contains a valid match."""

    def test_shows_match_status_for_matched_speaker(self, qapp):
        """When speaker_matches has a match, dialog shows its confidence."""
        store = FakeVoiceSignatureStore()
        matches = {
            "SPK_0": {
                "identity_name": "Alice",
                "score": 0.92,
                "confidence": "high",
            }
        }
        dlg = SpeakerIdentityLinkDialog(
            current_label="SPK_0",
            speaker_matches=matches,
            store=store,
        )
        label_text = dlg._current_label_label.text()
        assert "Alice" in label_text
        assert "high" in label_text.lower() or "0.92" in label_text

    def test_no_match_shows_unmatched(self, qapp):
        """When speaker_matches has None for the label, shows no match."""
        store = FakeVoiceSignatureStore()
        matches = {"SPK_0": None}
        dlg = SpeakerIdentityLinkDialog(
            current_label="SPK_0",
            speaker_matches=matches,
            store=store,
        )
        label_text = dlg._current_label_label.text()
        # Should show the label but indicate no match
        assert "SPK_0" in label_text

    def test_missing_matches_treated_as_unmatched(self, qapp):
        """When speaker_matches is missing or malformed, treated as unmatched."""
        store = FakeVoiceSignatureStore()
        dlg = SpeakerIdentityLinkDialog(
            current_label="SPK_0",
            speaker_matches={},  # No entry for SPK_0
            store=store,
        )
        label_text = dlg._current_label_label.text()
        assert "SPK_0" in label_text

    def test_non_dict_matches_treated_as_unmatched(self, qapp):
        """Non-dict speaker_matches is treated as unmatched."""
        store = FakeVoiceSignatureStore()
        dlg = SpeakerIdentityLinkDialog(
            current_label="SPK_0",
            speaker_matches="not a dict",  # type: ignore
            store=store,
        )
        label_text = dlg._current_label_label.text()
        assert "SPK_0" in label_text

    def test_malformed_match_entry_treated_as_unmatched(self, qapp):
        """A non-dict match entry for the label is treated as unmatched."""
        store = FakeVoiceSignatureStore()
        matches = {"SPK_0": "not a dict"}  # type: ignore
        dlg = SpeakerIdentityLinkDialog(
            current_label="SPK_0",
            speaker_matches=matches,
            store=store,
        )
        label_text = dlg._current_label_label.text()
        assert "SPK_0" in label_text


# ---------------------------------------------------------------------------
# Test: Filtering
# ---------------------------------------------------------------------------

class TestFiltering:
    """Filter box narrows the identity list without re-querying the store."""

    def test_filter_narrows_list(self, qapp):
        """Typing in the filter narrows the visible identities."""
        profiles = _make_profiles("Alice", "Bob", "Charlie")
        store = FakeVoiceSignatureStore(profiles)
        dlg = SpeakerIdentityLinkDialog(
            current_label="SPK_0",
            speaker_matches={},
            store=store,
        )
        # Simulate typing "ali" in the filter
        dlg._filter_edit.setText("ali")
        visible_names = [
            dlg._identity_list.item(i).text()
            for i in range(dlg._identity_list.count())
            if not dlg._identity_list.item(i).isHidden()
        ]
        assert visible_names == ["Alice"]

    def test_filter_case_insensitive(self, qapp):
        """Filter is case-insensitive."""
        profiles = _make_profiles("Alice", "Bob")
        store = FakeVoiceSignatureStore(profiles)
        dlg = SpeakerIdentityLinkDialog(
            current_label="SPK_0",
            speaker_matches={},
            store=store,
        )
        dlg._filter_edit.setText("ALI")
        visible_names = [
            dlg._identity_list.item(i).text()
            for i in range(dlg._identity_list.count())
            if not dlg._identity_list.item(i).isHidden()
        ]
        assert visible_names == ["Alice"]

    def test_filter_no_matches_shows_empty(self, qapp):
        """Filter with no matches hides all items."""
        profiles = _make_profiles("Alice", "Bob")
        store = FakeVoiceSignatureStore(profiles)
        dlg = SpeakerIdentityLinkDialog(
            current_label="SPK_0",
            speaker_matches={},
            store=store,
        )
        dlg._filter_edit.setText("xyz")
        visible_names = [
            dlg._identity_list.item(i).text()
            for i in range(dlg._identity_list.count())
            if not dlg._identity_list.item(i).isHidden()
        ]
        assert visible_names == []

    def test_clear_filter_shows_all(self, qapp):
        """Clearing the filter shows all identities again."""
        profiles = _make_profiles("Alice", "Bob")
        store = FakeVoiceSignatureStore(profiles)
        dlg = SpeakerIdentityLinkDialog(
            current_label="SPK_0",
            speaker_matches={},
            store=store,
        )
        dlg._filter_edit.setText("ali")
        dlg._filter_edit.setText("")
        visible_names = [
            dlg._identity_list.item(i).text()
            for i in range(dlg._identity_list.count())
            if not dlg._identity_list.item(i).isHidden()
        ]
        assert visible_names == ["Alice", "Bob"]


# ---------------------------------------------------------------------------
# Test: Selecting existing identity
# ---------------------------------------------------------------------------

class TestSelectExisting:
    """Selecting an existing identity returns its name."""

    def test_selected_identity_name_returns_chosen(self, qapp):
        """After selecting an item, selected_identity_name returns it."""
        profiles = _make_profiles("Alice", "Bob")
        store = FakeVoiceSignatureStore(profiles)
        dlg = SpeakerIdentityLinkDialog(
            current_label="SPK_0",
            speaker_matches={},
            store=store,
        )
        # Select first item (Alice)
        dlg._identity_list.setCurrentRow(0)
        assert dlg.selected_identity_name() == "Alice"

    def test_selected_identity_name_none_when_nothing_selected(self, qapp):
        """When nothing is selected, selected_identity_name returns None."""
        profiles = _make_profiles("Alice", "Bob")
        store = FakeVoiceSignatureStore(profiles)
        dlg = SpeakerIdentityLinkDialog(
            current_label="SPK_0",
            speaker_matches={},
            store=store,
        )
        # Nothing selected
        assert dlg.selected_identity_name() is None


# ---------------------------------------------------------------------------
# Test: Create new identity
# ---------------------------------------------------------------------------

class TestCreateNew:
    """Create-new path with duplicate/empty validation."""

    def test_create_new_identity_returns_name(self, qapp):
        """Setting a new name in the create-new field returns it."""
        store = FakeVoiceSignatureStore()
        dlg = SpeakerIdentityLinkDialog(
            current_label="SPK_0",
            speaker_matches={},
            store=store,
        )
        dlg._new_name_edit.setText("Eve")
        assert dlg.selected_identity_name() == "Eve"

    def test_create_new_takes_priority_over_selection(self, qapp):
        """If both a list item is selected and create-new has text, create-new wins."""
        profiles = _make_profiles("Alice")
        store = FakeVoiceSignatureStore(profiles)
        dlg = SpeakerIdentityLinkDialog(
            current_label="SPK_0",
            speaker_matches={},
            store=store,
        )
        dlg._identity_list.setCurrentRow(0)
        dlg._new_name_edit.setText("Eve")
        assert dlg.selected_identity_name() == "Eve"

    def test_empty_create_new_ignored(self, qapp):
        """Empty create-new text is treated as no input."""
        profiles = _make_profiles("Alice")
        store = FakeVoiceSignatureStore(profiles)
        dlg = SpeakerIdentityLinkDialog(
            current_label="SPK_0",
            speaker_matches={},
            store=store,
        )
        dlg._new_name_edit.setText("  ")
        # Should fall through to list selection (nothing selected → None)
        assert dlg.selected_identity_name() is None


# ---------------------------------------------------------------------------
# Test: Duplicate rejection
# ---------------------------------------------------------------------------

class TestDuplicateRejection:
    """Duplicate identity names must be rejected."""

    def test_duplicate_create_new_rejected(self, qapp):
        """Creating a new name that duplicates an existing identity is flagged."""
        profiles = _make_profiles("Alice", "Bob")
        store = FakeVoiceSignatureStore(profiles)
        dlg = SpeakerIdentityLinkDialog(
            current_label="SPK_0",
            speaker_matches={},
            store=store,
        )
        dlg._new_name_edit.setText("Alice")
        assert dlg._is_duplicate_name("Alice") is True
        # Validation message should indicate duplicate
        assert dlg._validate_selection() is False

    def test_case_insensitive_duplicate(self, qapp):
        """Duplicate check is case-insensitive."""
        profiles = _make_profiles("Alice")
        store = FakeVoiceSignatureStore(profiles)
        dlg = SpeakerIdentityLinkDialog(
            current_label="SPK_0",
            speaker_matches={},
            store=store,
        )
        assert dlg._is_duplicate_name("alice") is True
        assert dlg._is_duplicate_name("ALICE") is True

    def test_non_duplicate_accepted(self, qapp):
        """A genuinely new name is not a duplicate."""
        profiles = _make_profiles("Alice")
        store = FakeVoiceSignatureStore(profiles)
        dlg = SpeakerIdentityLinkDialog(
            current_label="SPK_0",
            speaker_matches={},
            store=store,
        )
        assert dlg._is_duplicate_name("Eve") is False


# ---------------------------------------------------------------------------
# Test: Accept/Reject dialog result
# ---------------------------------------------------------------------------

class TestDialogResult:
    """Dialog accept/reject with valid/invalid selections."""

    def test_accept_with_valid_selection(self, qapp):
        """Dialog accepts when a valid identity is selected."""
        profiles = _make_profiles("Alice")
        store = FakeVoiceSignatureStore(profiles)
        dlg = SpeakerIdentityLinkDialog(
            current_label="SPK_0",
            speaker_matches={},
            store=store,
        )
        dlg._identity_list.setCurrentRow(0)
        assert dlg._validate_selection() is True

    def test_reject_with_no_selection(self, qapp):
        """Dialog rejects validation when nothing selected and no create-new."""
        store = FakeVoiceSignatureStore()
        dlg = SpeakerIdentityLinkDialog(
            current_label="SPK_0",
            speaker_matches={},
            store=store,
        )
        assert dlg._validate_selection() is False

    def test_accept_with_valid_new_name(self, qapp):
        """Dialog accepts when a valid new name is entered."""
        store = FakeVoiceSignatureStore()
        dlg = SpeakerIdentityLinkDialog(
            current_label="SPK_0",
            speaker_matches={},
            store=store,
        )
        dlg._new_name_edit.setText("Eve")
        assert dlg._validate_selection() is True

    def test_reject_with_duplicate_new_name(self, qapp):
        """Dialog rejects validation when new name duplicates existing."""
        profiles = _make_profiles("Alice")
        store = FakeVoiceSignatureStore(profiles)
        dlg = SpeakerIdentityLinkDialog(
            current_label="SPK_0",
            speaker_matches={},
            store=store,
        )
        dlg._new_name_edit.setText("Alice")
        assert dlg._validate_selection() is False


# ---------------------------------------------------------------------------
# Test: Loads identities only once
# ---------------------------------------------------------------------------

class TestSingleLoad:
    """Store is queried exactly once per dialog instance."""

    def test_load_signatures_called_once(self, qapp):
        """load_signatures should be called exactly once during construction."""
        call_count = 0

        class CountingStore(FakeVoiceSignatureStore):
            def load_signatures(self):
                nonlocal call_count
                call_count += 1
                return super().load_signatures()

        store = CountingStore()
        dlg = SpeakerIdentityLinkDialog(
            current_label="SPK_0",
            speaker_matches={},
            store=store,
        )
        # Filter shouldn't cause additional loads
        dlg._filter_edit.setText("ali")
        dlg._filter_edit.setText("")
        dlg._filter_edit.setText("bob")
        assert call_count == 1


class TestExtraIdentityNames:
    """Verify extra_identity_names are merged into the dialog list."""

    def test_extra_names_appear_in_list(self, qapp):
        """Transcript-discovered identities appear alongside store profiles."""
        store = FakeVoiceSignatureStore(profiles=_make_profiles("Alice"))
        dlg = SpeakerIdentityLinkDialog(
            current_label="SPK_0",
            speaker_matches={},
            store=store,
            extra_identity_names={"Bob", "Carol"},
        )
        names = [dlg._identity_list.item(i).text() for i in range(dlg._identity_list.count())]
        assert "Alice" in names
        assert "Bob" in names
        assert "Carol" in names

    def test_extra_names_deduplicated_with_store(self, qapp):
        """If extra name already in store, don't show it twice."""
        store = FakeVoiceSignatureStore(profiles=_make_profiles("Alice"))
        dlg = SpeakerIdentityLinkDialog(
            current_label="SPK_0",
            speaker_matches={},
            store=store,
            extra_identity_names={"Alice", "Bob"},
        )
        names = [dlg._identity_list.item(i).text() for i in range(dlg._identity_list.count())]
        assert names.count("Alice") == 1
        assert "Bob" in names

    def test_extra_names_work_with_none_store(self, qapp):
        """Extra names still populate list when store is None."""
        dlg = SpeakerIdentityLinkDialog(
            current_label="SPK_0",
            speaker_matches={},
            store=None,
            extra_identity_names={"Alice"},
        )
        names = [dlg._identity_list.item(i).text() for i in range(dlg._identity_list.count())]
        assert "Alice" in names

    def test_extra_names_empty_when_none(self, qapp):
        """Passing no extra names is the same as empty set."""
        store = FakeVoiceSignatureStore(profiles=_make_profiles("Alice"))
        dlg = SpeakerIdentityLinkDialog(
            current_label="SPK_0",
            speaker_matches={},
            store=store,
        )
        assert dlg._identity_list.count() == 1

    def test_extra_names_sorted_alphabetically(self, qapp):
        """Extra names are sorted alongside store names."""
        store = FakeVoiceSignatureStore()
        dlg = SpeakerIdentityLinkDialog(
            current_label="SPK_0",
            speaker_matches={},
            store=store,
            extra_identity_names={"Zara", "Alice", "Bob"},
        )
        names = [dlg._identity_list.item(i).text() for i in range(dlg._identity_list.count())]
        assert names == ["Alice", "Bob", "Zara"]

    def test_extra_names_selectable(self, qapp):
        """User can select a transcript-discovered identity from the list."""
        store = FakeVoiceSignatureStore()
        dlg = SpeakerIdentityLinkDialog(
            current_label="SPK_0",
            speaker_matches={},
            store=store,
            extra_identity_names={"Alice"},
        )
        dlg._identity_list.setCurrentRow(0)
        assert dlg.selected_identity_name() == "Alice"
