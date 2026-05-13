"""
Floating panels - separate windows for transcript, CC overlay, and settings.

These panels are free-floating independent windows that can be positioned
anywhere on screen. They are not clipped by the main widget bounds.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QTextEdit, QLabel, QFrame, QHBoxLayout, QPushButton,
    QInputDialog, QApplication, QTabWidget, QListWidget, QListWidgetItem,
    QSplitter, QTextBrowser, QProgressBar, QComboBox, QMenu, QMessageBox,
    QDialog, QDialogButtonBox, QSizePolicy, QSizeGrip, QStackedWidget,
    QCheckBox, QLineEdit, QSlider,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QUrl, QPoint
from PyQt6.QtGui import QColor, QFont, QTextCharFormat, QTextCursor, QPainter, QPen, QMouseEvent
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import html as _html_module
import json
import re
import time

from meetandread.transcription.confidence import get_confidence_color, get_confidence_legend
from meetandread.hardware.detector import HardwareDetector
from meetandread.hardware.recommender import ModelRecommender


def clamp_to_screen(widget: QWidget, pos: QPoint) -> QPoint:
    """Clamp *pos* so that *widget* stays within the visible desktop area.

    Allows sliding between monitors but prevents the widget from being
    fully dragged off-screen.  At least 50 px must remain visible on
    any edge so the user can always grab it back.

    Returns *pos* unchanged if screen geometry cannot be resolved (test
    environments, headless CI, etc.).
    """
    screens = QApplication.screens()
    if not screens:
        return pos
    try:
        min_x = min(s.geometry().x() for s in screens)
        min_y = min(s.geometry().y() for s in screens)
        max_x = max(s.geometry().x() + s.geometry().width() for s in screens)
        max_y = max(s.geometry().y() + s.geometry().height() for s in screens)
    except (AttributeError, TypeError):
        return pos
    w, h = widget.width(), widget.height()
    margin = 50
    px, py = pos.x(), pos.y()
    x = max(min_x - w + margin, min(px, max_x - margin))
    y = max(min_y - h + margin, min(py, max_y - margin))
    return QPoint(x, y)


def ensure_on_screen(widget: QWidget) -> None:
    """Move *widget* into visible desktop bounds if it is currently off-screen."""
    try:
        widget.move(clamp_to_screen(widget, widget.pos()))
    except (TypeError, AttributeError):
        pass
from meetandread.performance.monitor import ResourceMonitor, ResourceSnapshot
from meetandread.performance.benchmark import BenchmarkRunner, BenchmarkResult
from meetandread.performance.wer import calculate_wer
from meetandread.widgets.theme import (
    current_palette, DARK_PALETTE,
    panel_base_css, glass_panel_css, title_css, header_button_css, tab_widget_css,
    text_area_css, status_label_css, splitter_css, list_widget_css,
    detail_header_css, action_button_css, context_menu_css, dialog_css,
    badge_css, legend_overlay_css, info_label_css,
    progress_bar_css, separator_css, combo_box_css,
    aetheric_settings_shell_css, aetheric_sidebar_css, aetheric_title_bar_css, aetheric_nav_button_css,
    aetheric_placeholder_css, aetheric_combo_box_css,
    aetheric_checkbox_css,
    aetheric_history_list_css, aetheric_history_viewer_css,
    aetheric_history_splitter_css, aetheric_history_header_css,
    aetheric_history_action_button_css,
    aetheric_playback_toolbar_css,
    aetheric_cc_overlay_css,
    AETHERIC_RED,
    AETHERIC_BORDER_DARK,
    AETHERIC_SETTINGS_BG,
    AETHERIC_RADIUS,
    ARROW_UP_SVG,
    ARROW_DOWN_SVG,
    CHECKMARK_SVG,
)

import logging

logger = logging.getLogger(__name__)

# Lazy import — avoids QtMultimedia DLL issues at module level
# HistoryPlaybackController is imported inside FloatingSettingsPanel methods.


# ---------------------------------------------------------------------------
# TexturedSizeGrip — Windows-style diagonal grip lines in a triangle
# ---------------------------------------------------------------------------

class TexturedSizeGrip(QSizeGrip):
    """QSizeGrip that paints a textured dot-triangle in the bottom-right corner.

    Draws dots in a grid pattern, keeping only those within a right triangle
    whose hypotenuse runs from the top-right to bottom-left. Produces the
    classic Windows resize-handle appearance.
    """

    def __init__(self, parent: QWidget, color: str = "white") -> None:
        super().__init__(parent)
        from PyQt6.QtGui import QColor
        self._color = QColor(color)

    def paintEvent(self, event) -> None:  # noqa: N802
        from PyQt6.QtGui import QPainter, QRadialGradient, QBrush, QColor

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()

        # Dot grid that fills a right triangle in the bottom-right corner.
        # A dot at grid position (col, row) is drawn when col + row < cols,
        # producing a triangle that's dense at the corner and tapers off.
        cols = 5
        dot_r = 1.2       # dot radius
        spacing = 2.8      # px between dot centers
        margin = 2         # px inset from widget edges

        for col in range(cols):
            for row in range(cols):
                if col + row >= cols:
                    continue  # outside the triangle

                cx = w - margin - col * spacing
                cy = h - margin - row * spacing

                if cx < margin or cy < margin:
                    continue

                # Soft dot via radial gradient
                grad = QRadialGradient(cx, cy, dot_r)
                grad.setColorAt(0, self._color)
                grad.setColorAt(1, QColor(0, 0, 0, 0))
                painter.setBrush(QBrush(grad))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(
                    int(cx - dot_r), int(cy - dot_r),
                    int(dot_r * 2), int(dot_r * 2),
                )

        painter.end()


def _escape_html(text: str) -> str:
    """Escape HTML special characters for safe embedding in innerHTML."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )


# ---------------------------------------------------------------------------
# Speaker color palette — deterministic colors for up to 8 speakers
# ---------------------------------------------------------------------------
SPEAKER_COLORS: Dict[str, str] = {
    "SPK_0": "#4FC3F7",  # Light blue
    "SPK_1": "#FF8A65",  # Orange
    "SPK_2": "#AED581",  # Light green
    "SPK_3": "#CE93D8",  # Purple
    "SPK_4": "#FFD54F",  # Amber
    "SPK_5": "#F48FB1",  # Pink
    "SPK_6": "#4DD0E1",  # Cyan
    "SPK_7": "#FFB74D",  # Deep orange
}
_DEFAULT_SPEAKER_COLOR = "#90A4AE"  # Blue grey fallback


def speaker_color(label: str) -> str:
    """Return a deterministic color hex string for a speaker label."""
    return SPEAKER_COLORS.get(label, _DEFAULT_SPEAKER_COLOR)


@dataclass
class Phrase:
    """A phrase (line) of transcript with its segments."""
    segments: List[str]  # Text of each segment
    confidences: List[int]  # Confidence of each segment
    is_final: bool  # True if phrase is complete
    speaker_id: Optional[str] = None  # Speaker label for this phrase


def _strip_confidence_percentages(text: str) -> str:
    """Remove confidence percentage markers like `` (73%)`` from transcript text.

    Older recordings were saved with ``to_markdown(include_confidence=True)``
    which appends ``(NN%)`` to low-confidence words.  This strips those markers
    so the history view always shows clean text regardless of when the file
    was saved.

    Pattern: a space followed by ``(``, 1-3 digits, ``%``, ``)`` — but NOT
    parenthesised speaker labels like ``(Empty recording)`` which lack ``%``.
    """
    return re.sub(r" \(\d{1,3}%\)", "", text)


# ---------------------------------------------------------------------------
# SpeakerIdentityLinkDialog — identity selection for history speaker labels
# ---------------------------------------------------------------------------

class SpeakerIdentityLinkDialog(QDialog):
    """Dialog for linking a raw speaker label to an existing or new identity.

    Loads known identities from a ``VoiceSignatureStore`` (or store-like
    object) exactly once on construction, then filters the in-memory list
    per keystroke.  Returns the chosen identity name via
    ``selected_identity_name()`` without performing any file I/O.

    Args:
        current_label: The raw speaker label being linked (e.g. ``"SPK_0"``).
        speaker_matches: Dict mapping raw labels to match dicts or ``None``.
        store: An object with a ``load_signatures()`` method returning
               a list of ``SpeakerProfile`` objects.
        parent: Optional parent widget.
    """

    def __init__(
        self,
        current_label: str,
        speaker_matches: dict,
        store: object,
        extra_identity_names: Optional[set] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._current_label = current_label
        self._identity_names: List[str] = []  # cached names from store
        self._selected_name: Optional[str] = None
        self._extra_identity_names = extra_identity_names or set()

        # --- Window setup ---
        self.setWindowTitle("Link Speaker Identity")
        self.setMinimumSize(360, 400)
        self.setMaximumSize(500, 600)

        p = current_palette()
        self.setStyleSheet(dialog_css(p))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)

        # --- Current label + match status ---
        self._current_label_label = QLabel()
        self._current_label_label.setWordWrap(True)
        self._current_label_label.setStyleSheet(
            f"font-weight: bold; font-size: 13px; color: {p.info};"
        )
        self._set_label_text(speaker_matches)
        layout.addWidget(self._current_label_label)

        # --- Status label (for errors/info) ---
        self._status_label = QLabel("")
        self._status_label.setStyleSheet(f"font-size: 11px; color: {p.text_secondary};")
        layout.addWidget(self._status_label)

        # --- Filter edit ---
        self._filter_edit = QLineEdit()
        self._filter_edit.setPlaceholderText("Filter identities...")
        self._filter_edit.setStyleSheet(
            f"background: {p.surface}; color: {p.text}; "
            f"border: 1px solid {p.border}; border-radius: 4px; padding: 4px 8px;"
        )
        self._filter_edit.textChanged.connect(self._on_filter_changed)
        layout.addWidget(self._filter_edit)

        # --- Identity list ---
        self._identity_list = QListWidget()
        self._identity_list.setStyleSheet(list_widget_css(p))
        self._identity_list.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self._identity_list)

        # --- Create-new section ---
        new_label = QLabel("Or create a new identity:")
        new_label.setStyleSheet(f"font-size: 11px; color: {p.text_secondary};")
        layout.addWidget(new_label)

        self._new_name_edit = QLineEdit()
        self._new_name_edit.setPlaceholderText("New identity name...")
        self._new_name_edit.setStyleSheet(
            f"background: {p.surface}; color: {p.text}; "
            f"border: 1px solid {p.border}; border-radius: 4px; padding: 4px 8px;"
        )
        layout.addWidget(self._new_name_edit)

        # --- Validation label ---
        self._validation_label = QLabel("")
        self._validation_label.setStyleSheet(f"font-size: 11px; color: {p.danger};")
        layout.addWidget(self._validation_label)

        # --- Button box ---
        self._button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
        )
        self._button_box.setStyleSheet(action_button_css(p, "dialog"))
        self._button_box.accepted.connect(self._on_accept)
        self._button_box.rejected.connect(self.reject)
        layout.addWidget(self._button_box)

        # --- Load identities (exactly once) ---
        self._load_identities(store)

    # ------------------------------------------------------------------
    # Identity loading
    # ------------------------------------------------------------------

    def _load_identities(self, store: object) -> None:
        """Load identity names from the store and extra sources, then populate list.

        On store failure, still populates from extra_identity_names so
        create-new remains usable alongside transcript-discovered names.
        """
        self._identity_names = []

        # Load from VoiceSignatureStore
        if store is not None:
            try:
                profiles = store.load_signatures()
                for profile in profiles:
                    name = getattr(profile, "name", None)
                    if name and isinstance(name, str) and name.strip():
                        self._identity_names.append(name.strip())
            except Exception:
                self._status_label.setText("Could not load identities from store.")

        # Merge transcript-discovered names (won't have embeddings)
        store_names = set(self._identity_names)
        for name in self._extra_identity_names:
            if name not in store_names:
                self._identity_names.append(name)

        # Populate list widget (sorted)
        self._identity_names.sort(key=lambda n: n.lower())
        for name in self._identity_names:
            self._identity_list.addItem(name)

    # ------------------------------------------------------------------
    # Label/match display
    # ------------------------------------------------------------------

    def _set_label_text(self, speaker_matches: dict) -> None:
        """Set the current label display with match status if available."""
        label_text = f"Speaker: [{self._current_label}]"

        # Safely extract match info
        match_info = None
        if isinstance(speaker_matches, dict):
            match_info = speaker_matches.get(self._current_label)

        if match_info is not None and isinstance(match_info, dict):
            identity = match_info.get("identity_name", "")
            score = match_info.get("score", 0)
            confidence = match_info.get("confidence", "")
            if identity:
                label_text += (
                    f"  \u2192  Currently matched: {identity} "
                    f"(confidence: {confidence}, score: {score:.2f})"
                )
            else:
                label_text += "  (no current match)"
        else:
            label_text += "  (no current match)"

        self._current_label_label.setText(label_text)

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def _on_filter_changed(self, text: str) -> None:
        """Filter the identity list based on the filter edit text.

        Uses in-memory filtering \u2014 no additional store queries.
        """
        filter_text = text.strip().lower()
        for i in range(self._identity_list.count()):
            item = self._identity_list.item(i)
            if not filter_text:
                item.setHidden(False)
            else:
                item.setHidden(filter_text not in item.text().lower())

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def selected_identity_name(self) -> Optional[str]:
        """Return the selected or created identity name.

        Priority:
        1. Create-new field (if non-empty and non-whitespace)
        2. Selected list item
        3. None
        """
        new_name = self._new_name_edit.text().strip()
        if new_name:
            return new_name
        current = self._identity_list.currentItem()
        if current is not None:
            return current.text()
        return None

    def _on_item_double_clicked(self, item: QListWidgetItem) -> None:
        """Accept the dialog on double-click of a list item."""
        if self._validate_selection():
            self.accept()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _is_duplicate_name(self, name: str) -> bool:
        """Check whether *name* duplicates an existing identity (case-insensitive)."""
        return name.lower() in (n.lower() for n in self._identity_names)

    def _validate_selection(self) -> bool:
        """Validate the current selection / create-new input.

        Returns True if a valid identity is selected/entered and is not
        a duplicate.  Updates the validation label with an appropriate
        message on failure.
        """
        new_name = self._new_name_edit.text().strip()

        # If create-new is populated, validate it
        if new_name:
            if self._is_duplicate_name(new_name):
                self._validation_label.setText(
                    "This name already exists as an identity."
                )
                return False
            self._validation_label.setText("")
            return True

        # Otherwise, check list selection
        current = self._identity_list.currentItem()
        if current is not None:
            self._validation_label.setText("")
            return True

        self._validation_label.setText("Select an identity or enter a new name.")
        return False

    def _on_accept(self) -> None:
        """Handle OK button \u2014 validate before accepting."""
        if self._validate_selection():
            self.accept()


# ---------------------------------------------------------------------------
# Module-level identity-link persistence helper
# ---------------------------------------------------------------------------


def _link_speaker_identity_in_file(
    md_path: Path, raw_label: str, identity_name: str
) -> None:
    """Link a raw speaker label to an identity in transcript metadata and body.

    Updates the transcript .md file so that *raw_label* (e.g. ``SPK_0``) is
    replaced with *identity_name* (e.g. ``Alice``) in the JSON metadata
    (words, segments), the markdown body headings, and the
    ``speaker_matches`` map.  Propagates to the ``VoiceSignatureStore``
    best-effort.

    PII-safe: identity names are never logged.

    Args:
        md_path: Path to the transcript .md file.
        raw_label: Raw diarization label to replace (e.g. ``SPK_0``).
        identity_name: Chosen identity name (e.g. ``Alice``).

    Leaves the file unchanged when:
        - *identity_name* is empty/whitespace
        - *identity_name* equals *raw_label*
        - metadata footer is missing or contains malformed JSON
    """
    if not identity_name or not identity_name.strip():
        return

    identity_name = identity_name.strip()

    if identity_name == raw_label:
        return

    content = md_path.read_text(encoding="utf-8")

    footer_marker = "\n---\n\n<!-- METADATA:"
    marker_idx = content.find(footer_marker)
    if marker_idx == -1:
        logger.warning("No metadata footer found — cannot link identity")
        return

    md_body = content[:marker_idx]
    after_marker = content[marker_idx + len(footer_marker):]
    space_before_json = ""
    if after_marker.startswith(" "):
        space_before_json = " "
        after_marker = after_marker[1:]

    metadata_text = after_marker
    if metadata_text.strip().endswith(" -->"):
        metadata_text = metadata_text.strip()[: -len(" -->")]

    try:
        data = json.loads(metadata_text)
    except json.JSONDecodeError:
        logger.warning("Malformed metadata — leaving file unchanged")
        return

    # Update words — __unknown__ matches speaker_id == None
    words_updated = 0
    matching_label = None if raw_label == "__unknown__" else raw_label
    for word in data.get("words", []):
        if word.get("speaker_id") == matching_label:
            word["speaker_id"] = identity_name
            words_updated += 1

    # Update segments — handle both speaker_id and speaker keys
    segments_updated = 0
    for seg in data.get("segments", []):
        if seg.get("speaker_id") == matching_label:
            seg["speaker_id"] = identity_name
            segments_updated += 1
        if seg.get("speaker") == matching_label:
            seg["speaker"] = identity_name
            # Don't double-count if both keys matched same segment
            if seg.get("speaker_id") != matching_label:
                segments_updated += 1

    # Update markdown body — exact **label** replacement only
    display_label = "Unknown Speaker" if raw_label == "__unknown__" else raw_label
    updated_body = re.sub(
        re.escape(f"**{display_label}**"),
        f"**{identity_name}**",
        md_body,
    )

    # Update or create speaker_matches
    if "speaker_matches" not in data:
        data["speaker_matches"] = {}

    match_key = "__unknown__" if raw_label == "__unknown__" else raw_label
    existing = data["speaker_matches"].get(match_key)
    if isinstance(existing, dict) and "score" in existing and "confidence" in existing:
        # Preserve prior score/confidence, update identity_name
        data["speaker_matches"][match_key] = {
            "identity_name": identity_name,
            "score": existing["score"],
            "confidence": existing["confidence"],
        }
    else:
        # No prior match or null — use manual-link sentinel
        data["speaker_matches"][match_key] = {
            "identity_name": identity_name,
            "score": 1.0,
            "confidence": "manual",
        }

    # Rebuild file
    updated_json = json.dumps(data, indent=2)
    new_content = (
        updated_body + footer_marker + space_before_json + updated_json + " -->\n"
    )
    md_path.write_text(new_content, encoding="utf-8")

    logger.info(
        "Linked identity for raw label in %s (%d words, %d segments)",
        md_path, words_updated, segments_updated,
    )

    # Best-effort signature propagation (PII-safe)
    # Skip for __unknown__ — no raw label with an embedding in the signature store
    if raw_label != "__unknown__":
        _propagate_identity_to_signatures(md_path, raw_label, identity_name)


def _propagate_identity_to_signatures(
    md_path: Path, raw_label: str, identity_name: str
) -> None:
    """Propagate an identity link to the VoiceSignatureStore (best-effort).

    If a raw speaker profile exists in the signature DB, saves it under the
    identity name and deletes the raw entry.

    When *raw_label* is ``"__unknown__"`` (the Unknown Speaker sentinel),
    resolves the actual SPK_N label(s) from the transcript metadata so the
    correct embedding can be propagated.

    Never crashes; all diagnostics are PII-sanitized.
    """
    try:
        from meetandread.speaker.signatures import VoiceSignatureStore
    except ImportError:
        logger.info("VoiceSignatureStore unavailable — skipping propagation")
        return

    db_path = md_path.parent / "speaker_signatures.db"
    if not db_path.exists():
        try:
            from meetandread.audio.storage.paths import get_recordings_dir
            default_db = get_recordings_dir() / "speaker_signatures.db"
            if default_db.exists():
                db_path = default_db
            else:
                logger.info("No signature database found — skipping propagation")
                return
        except Exception:
            logger.info("No signature database found — skipping propagation")
            return

    # Resolve __unknown__ to actual SPK_N label(s) from the transcript
    resolved_labels: list[str] = []
    if raw_label == "__unknown__":
        resolved_labels = _resolve_unknown_speaker_labels(md_path)
        if not resolved_labels:
            logger.info("No SPK labels found in transcript for __unknown__ — skipping propagation")
            return
    else:
        resolved_labels = [raw_label]

    try:
        with VoiceSignatureStore(db_path=str(db_path)) as store:
            profiles = store.load_signatures()
            profile_map = {p.name: p for p in profiles}

            for label in resolved_labels:
                old_profile = profile_map.get(label)
                if old_profile is None:
                    logger.info("Raw speaker '%s' not found in signature store — skipping", label)
                    continue

                store.save_signature(
                    identity_name,
                    old_profile.embedding,
                    averaged_from_segments=old_profile.num_samples,
                )
                store.delete_signature(label)

                logger.info("Propagated identity link to signature store")
    except Exception as exc:
        logger.warning("Failed to propagate identity link to signature store: %s", exc)


def _resolve_unknown_speaker_labels(md_path: Path) -> list[str]:
    """Resolve __unknown__ to the actual SPK_N labels used in a transcript.

    Reads the transcript metadata and collects all unique non-None speaker IDs
    from words.  For a single-speaker recording with __unknown__ as the match
    key, this returns the single SPK_N label that was assigned during
    diarization.

    Returns:
        List of SPK_N labels found in the transcript, or empty list.
    """
    import json

    try:
        content = md_path.read_text(encoding="utf-8")
        marker = "\n---\n\n<!-- METADATA: "
        idx = content.find(marker)
        if idx < 0:
            return []
        data = json.loads(content[idx + len(marker) :].rstrip(" -->\n"))

        # Collect all unique non-None speaker_ids from words
        labels: set[str] = set()
        for w in data.get("words", []):
            sid = w.get("speaker_id")
            if sid is not None:
                labels.add(sid)
        return sorted(labels)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Shared identity-link dialog helper for history anchor handlers
# ---------------------------------------------------------------------------


def _open_identity_link_dialog(
    md_path: Path, raw_label: str, parent_widget: QWidget
) -> bool:
    """Open SpeakerIdentityLinkDialog and persist the chosen identity.

    Shared by both FloatingTranscriptPanel and FloatingSettingsPanel anchor
    handlers so identity-link behaviour stays consistent.

    Returns True if a link was performed (file updated), False otherwise
    (cancel, validation failure, missing path, etc.).

    PII-safe: identity names are never logged here (delegated to
    ``_link_speaker_identity_in_file``).
    """
    if md_path is None or not md_path.exists():
        logger.warning("No transcript file selected for identity link")
        return False

    # Parse speaker_matches from the transcript metadata
    speaker_matches: dict = {}
    try:
        content = md_path.read_text(encoding="utf-8")
        footer_marker = "\n---\n\n<!-- METADATA:"
        marker_idx = content.find(footer_marker)
        if marker_idx != -1:
            metadata_text = content[marker_idx + len(footer_marker):]
            if metadata_text.strip().endswith(" -->"):
                metadata_text = metadata_text.strip()[: -len(" -->")]
            data = json.loads(metadata_text)
            speaker_matches = data.get("speaker_matches") or {}
    except (OSError, json.JSONDecodeError):
        pass  # empty matches is fine — dialog still usable for create-new

    # Construct VoiceSignatureStore (best-effort, dialog still works without)
    store = None
    try:
        from meetandread.speaker.signatures import VoiceSignatureStore

        db_path = md_path.parent / "speaker_signatures.db"
        try:
            from meetandread.audio.storage.paths import get_recordings_dir
            default_db = get_recordings_dir() / "speaker_signatures.db"
            # Prefer the default recordings-dir DB (authoritative location)
            db_path = default_db
        except Exception:
            pass

        store = VoiceSignatureStore(db_path=str(db_path))
    except Exception:
        pass  # dialog works with None store — create-new path still available

    # Discover additional identity names from transcript speaker_matches
    # that were linked without embeddings (won't appear in VoiceSignatureStore).
    transcript_identity_names: set = set()
    try:
        from meetandread.audio.storage.paths import get_transcripts_dir
        from meetandread.speaker.identity_management import parse_metadata_footer

        transcripts_dir = get_transcripts_dir()
        if transcripts_dir.is_dir():
            for tmd_path in transcripts_dir.glob("*.md"):
                try:
                    tcontent = tmd_path.read_text(encoding="utf-8")
                except OSError:
                    continue
                tdata = parse_metadata_footer(tcontent)
                if tdata is None:
                    continue
                for _label, match_info in tdata.get("speaker_matches", {}).items():
                    if isinstance(match_info, dict):
                        tname = match_info.get("identity_name")
                        if tname:
                            transcript_identity_names.add(tname)
    except Exception:
        pass  # transcript discovery is best-effort

    # Use display-friendly label for dialog, keep sentinel for file updates
    display_label = "Unknown Speaker" if raw_label == "__unknown__" else raw_label

    dialog = SpeakerIdentityLinkDialog(
        current_label=display_label,
        speaker_matches=speaker_matches,
        store=store,
        extra_identity_names=transcript_identity_names,
        parent=parent_widget,
    )

    if dialog.exec() != QDialog.DialogCode.Accepted:
        return False

    identity_name = dialog.selected_identity_name()
    if not identity_name or not identity_name.strip():
        return False

    try:
        _link_speaker_identity_in_file(md_path, raw_label, identity_name)
    except Exception as exc:
        logger.error("Failed to link identity in %s: %s", md_path, exc)
        return False

    return True


class FloatingTranscriptPanel(QWidget):
    """
    Floating transcript panel that appears outside the main widget.
    
    Features:
    - Separate window (not clipped by main widget bounds)
    - Shows transcript with confidence-based coloring
    - Auto-scrolls to show latest text
    - Can be manually toggled
    """
    
    # Signals
    closed = pyqtSignal()  # Emitted when user closes panel
    segment_ready = pyqtSignal(str, int, int, bool, bool, object)  # text, confidence, segment_index, is_final, phrase_start, speaker_id
    speaker_name_pinned = pyqtSignal(str, str)  # raw_speaker_label, user_chosen_name

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        # Speaker label display mapping: raw_label -> display_name
        # e.g. {"spk0": "Alice", "spk1": "SPK_1"}
        self._speaker_names: Dict[str, str] = {}
        # Raw speaker labels that have been pinned by the user
        self._pinned_speakers: set = set()
        
        # Window settings (FloatingTranscriptPanel)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool  # Don't show in taskbar
        )
        
        # Glass pair: translucent background matching the widget's glass aesthetic
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        
        # Size — resizable with min/max bounds
        self.setMinimumSize(350, 300)
        self.setMaximumSize(800, 900)
        
        # Style — applied via _apply_theme() at end of __init__
        
        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)
        
        # Header with title and close button
        header_layout = QHBoxLayout()
        header_layout.setSpacing(5)
        
        # Title bar (clickable for dragging)
        self._title_label = QLabel("Live Transcript")
        header_layout.addWidget(self._title_label)
        header_layout.addStretch()
        
        # Legend toggle button (?)
        self._legend_btn = QPushButton("?")
        self._legend_btn.setFixedSize(24, 24)
        self._legend_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._legend_btn.setToolTip("Confidence legend")
        self._legend_btn.setCheckable(True)
        self._legend_btn.clicked.connect(self._toggle_legend)
        header_layout.addWidget(self._legend_btn)
        
        # Close button
        self._close_btn = QPushButton("×")
        self._close_btn.setFixedSize(24, 24)
        self._close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._close_btn.setToolTip("Close panel")
        self._close_btn.clicked.connect(self.hide_panel)
        header_layout.addWidget(self._close_btn)
        
        layout.addLayout(header_layout)
        
        # ------------------------------------------------------------------
        # Tab widget — Live and History tabs
        # ------------------------------------------------------------------
        self._tab_widget = QTabWidget()
        layout.addWidget(self._tab_widget)

        # ------------------------------------------------------------------
        # Live tab — existing transcript display
        # ------------------------------------------------------------------
        live_tab = QWidget()
        live_layout = QVBoxLayout(live_tab)
        live_layout.setContentsMargins(0, 0, 0, 0)
        live_layout.setSpacing(2)

        # Text edit for transcript
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setFrameShape(QFrame.Shape.NoFrame)
        # Handle anchor clicks on speaker labels (signal only on QTextBrowser)
        self.text_edit.setMouseTracking(True)
        if hasattr(self.text_edit, "anchorClicked"):
            self.text_edit.anchorClicked.connect(self._on_anchor_clicked)
        live_layout.addWidget(self.text_edit)

        # Status label
        self.status_label = QLabel("Ready")
        live_layout.addWidget(self.status_label)

        self._tab_widget.addTab(live_tab, "Live")

        # ------------------------------------------------------------------
        # History tab — recording list and transcript viewer
        # ------------------------------------------------------------------
        history_tab = QWidget()
        history_layout = QVBoxLayout(history_tab)
        history_layout.setContentsMargins(0, 0, 0, 0)
        history_layout.setSpacing(0)

        self._splitter = QSplitter(Qt.Orientation.Vertical)

        # Top: recording list
        self._history_list = QListWidget()
        self._history_list.itemClicked.connect(self._on_history_item_clicked)
        # Enable context menu on history items
        self._history_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._history_list.customContextMenuRequested.connect(self._on_history_context_menu)
        self._splitter.addWidget(self._history_list)

        # Bottom section: detail header bar + transcript viewer
        viewer_container = QWidget()
        viewer_layout = QVBoxLayout(viewer_container)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        viewer_layout.setSpacing(0)

        # Detail header bar with Delete button (hidden until selection)
        self._detail_header = QFrame()
        detail_header_layout = QHBoxLayout(self._detail_header)
        detail_header_layout.setContentsMargins(6, 2, 6, 2)
        detail_header_layout.setSpacing(4)

        detail_header_layout.addStretch()

        self._scrub_btn = QPushButton("🔄 Scrub")
        self._scrub_btn.setFixedHeight(26)
        self._scrub_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._scrub_btn.setToolTip("Re-transcribe with a different model")
        self._scrub_btn.clicked.connect(self._on_scrub_clicked)
        detail_header_layout.addWidget(self._scrub_btn)

        self._delete_btn = QPushButton("🗑 Delete")
        self._delete_btn.setFixedHeight(26)
        self._delete_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._delete_btn.setToolTip("Delete this recording")
        self._delete_btn.clicked.connect(self._on_delete_btn_clicked)
        detail_header_layout.addWidget(self._delete_btn)

        self._detail_header.hide()
        viewer_layout.addWidget(self._detail_header)

        # Transcript viewer (read-only, supports anchor clicks)
        self._history_viewer = QTextBrowser()
        self._history_viewer.setReadOnly(True)
        self._history_viewer.setFrameShape(QFrame.Shape.NoFrame)
        self._history_viewer.setPlaceholderText("Select a recording to view its transcript")
        self._history_viewer.setOpenExternalLinks(False)
        self._history_viewer.setOpenLinks(False)
        self._history_viewer.anchorClicked.connect(self._on_history_anchor_clicked)
        viewer_layout.addWidget(self._history_viewer)

        self._splitter.addWidget(viewer_container)

        # 40% list / 60% viewer
        self._splitter.setSizes([160, 240])

        history_layout.addWidget(self._splitter)
        self._tab_widget.addTab(history_tab, "History")

        # Connect tab change to refresh history when switching to it
        self._tab_widget.currentChanged.connect(self._on_tab_changed)

        # Track the currently-viewed history transcript path (for rename)
        self._current_history_md_path: Optional[Path] = None

        # Scrub state
        self._scrub_runner: Optional[object] = None  # ScrubRunner instance
        self._scrub_model_size: Optional[str] = None  # model being scrubbed
        self._scrub_sidecar_path: Optional[str] = None  # expected sidecar path
        self._scrub_original_html: Optional[str] = None  # original transcript HTML
        self._is_scrubbing: bool = False  # True while scrub is in progress
        self._is_comparison_mode: bool = False  # True when showing side-by-side
        
        # Dragging
        self._dragging = False
        self._drag_pos = None
        
        # Track phrases (each phrase is a line)
        self.phrases: List[Phrase] = []
        self.current_phrase_idx = -1
        
        # Auto-scroll timer
        self.scroll_timer = QTimer(self)
        self.scroll_timer.timeout.connect(self._scroll_to_bottom)
        
        # Auto-scroll pause mechanism
        self._auto_scroll_paused = False
        self._pause_timer = QTimer(self)
        self._pause_timer.setSingleShot(True)
        self._pause_timer.timeout.connect(self._resume_auto_scroll)
        self._last_scroll_value = 0
        self._is_at_bottom = True
        
        # Pending content count for badge when auto-scroll is paused
        self._pending_content_count: int = 0
        
        # Connect to scrollbar value changed signal to detect manual scroll
        self.text_edit.verticalScrollBar().valueChanged.connect(self._on_scroll_value_changed)
        
        # Confidence legend overlay (initially hidden)
        self._create_legend_overlay()
        
        # New-content badge (initially hidden)
        self._create_new_content_badge()
        
        # Recording duration tracking
        self._recording_start_time: Optional[float] = None
        self._duration_timer = QTimer(self)
        self._duration_timer.setInterval(1000)  # 1-second tick
        self._duration_timer.timeout.connect(self._update_duration)
        
        # Empty state tracking
        self._has_content: bool = False
        
        # Glass pair opacity — matches the widget's glass aesthetic
        # 0.87 = translucent idle (desktop visible behind), 1.0 = active/recording
        self._glass_idle_opacity = 0.87
        self._glass_active_opacity = 1.0
        self._is_glass_active = False
        self.setWindowOpacity(self._glass_idle_opacity)

        # Apply initial theme to all widgets
        self._apply_theme()

        # Connect to desktop theme changes for live re-theming
        try:
            from PyQt6.QtGui import QGuiApplication
            hints = QGuiApplication.styleHints()
            if hints is not None:
                hints.colorSchemeChanged.connect(lambda: self._apply_theme())
        except (ImportError, RuntimeError):
            pass

        self._show_empty_state()
    
    # ------------------------------------------------------------------
    # Confidence legend overlay
    # ------------------------------------------------------------------

    def _create_legend_overlay(self) -> None:
        """Build the confidence legend overlay positioned over the text edit area."""
        self._legend_overlay = QFrame(self.text_edit)
        self._legend_overlay.setFixedSize(220, 140)

        # Layout
        overlay_layout = QVBoxLayout(self._legend_overlay)
        overlay_layout.setContentsMargins(10, 8, 10, 8)
        overlay_layout.setSpacing(4)

        # Title
        self._legend_title = QLabel("Confidence Levels")
        overlay_layout.addWidget(self._legend_title)

        # Separator
        self._legend_sep = QFrame()
        self._legend_sep.setFrameShape(QFrame.Shape.HLine)
        self._legend_sep.setFixedHeight(1)
        overlay_layout.addWidget(self._legend_sep)

        # Legend rows from canonical source
        self._legend_range_labels: list = []
        self._legend_desc_labels: list = []
        for item in get_confidence_legend():
            row = QHBoxLayout()
            row.setSpacing(6)

            # Color swatch
            swatch = QLabel()
            swatch.setFixedSize(14, 14)
            swatch.setStyleSheet(
                f"background-color: {item.color}; border-radius: 3px; border: none;"
            )
            row.addWidget(swatch)

            # Range text
            range_label = QLabel(item.range_str)
            range_label.setFixedWidth(50)
            row.addWidget(range_label)
            self._legend_range_labels.append(range_label)

            # Description
            desc_label = QLabel(item.description)
            row.addWidget(desc_label)
            self._legend_desc_labels.append(desc_label)

            row.addStretch()
            overlay_layout.addLayout(row)

        # Position bottom-right of text_edit
        self._position_legend_overlay()
        self._legend_overlay.hide()

    # ------------------------------------------------------------------
    # New-content badge (auto-scroll pause indicator)
    # ------------------------------------------------------------------

    def _create_new_content_badge(self) -> None:
        """Build the '↓ N new' badge that appears when auto-scroll is paused."""
        self._new_content_badge = QPushButton("↓ 0 new", self.text_edit)
        self._new_content_badge.setFixedSize(120, 32)
        self._new_content_badge.setCursor(Qt.CursorShape.PointingHandCursor)
        self._new_content_badge.clicked.connect(self._on_badge_clicked)
        self._position_new_content_badge()
        self._new_content_badge.hide()

        # Resize grip — direct child of panel (not in layout) so it stays at bottom-right
        self._resize_grip = TexturedSizeGrip(self)
        self._resize_grip.setFixedSize(16, 16)
        self._resize_grip.setCursor(Qt.CursorShape.SizeFDiagCursor)
        self._resize_grip.show()

    def _position_new_content_badge(self) -> None:
        """Position the badge at bottom-center of the text edit."""
        if not hasattr(self, '_new_content_badge'):
            return
        te = self.text_edit
        badge = self._new_content_badge
        x = (te.width() - badge.width()) // 2
        y = te.height() - badge.height() - 8
        badge.move(max(x, 0), max(y, 0))

    def _on_badge_clicked(self) -> None:
        """Handle badge click: resume auto-scroll and hide badge."""
        self._auto_scroll_paused = False
        self._pause_timer.stop()
        self._pending_content_count = 0
        self._new_content_badge.hide()
        self.status_label.setText("Recording...")
        self._scroll_to_bottom()

    def _position_legend_overlay(self) -> None:
        """Position the legend overlay in the bottom-right corner of the text edit."""
        if not hasattr(self, '_legend_overlay'):
            return
        te = self.text_edit
        x = te.width() - self._legend_overlay.width() - 8
        y = te.height() - self._legend_overlay.height() - 8
        self._legend_overlay.move(max(x, 0), max(y, 0))

    def _toggle_legend(self) -> None:
        """Toggle the confidence legend overlay visibility."""
        visible = self._legend_btn.isChecked()
        if visible:
            self._position_legend_overlay()
        self._legend_overlay.setVisible(visible)

    def resizeEvent(self, event) -> None:
        """Reposition overlays and resize grip on resize."""
        if hasattr(self, '_legend_overlay') and self._legend_overlay.isVisible():
            self._position_legend_overlay()
        if hasattr(self, '_new_content_badge') and self._new_content_badge.isVisible():
            self._position_new_content_badge()
        if hasattr(self, '_resize_grip'):
            self._resize_grip.move(
                self.width() - self._resize_grip.width(),
                self.height() - self._resize_grip.height(),
            )
        super().resizeEvent(event)
    
    def show_panel(self) -> None:
        """Show the panel with a 150ms fade-in and start auto-scroll."""
        self._set_glass_active(True)
        self._start_fade_in()
        self.scroll_timer.start(100)  # Scroll check every 100ms
        self._recording_start_time = time.time()
        self._duration_timer.start()
        self._update_duration()  # Show "Recording · 00:00" immediately
        # Reset badge state on panel show
        self._pending_content_count = 0
        if hasattr(self, '_new_content_badge'):
            self._new_content_badge.hide()
    
    def hide_panel(self) -> None:
        """Hide the panel with a 150ms fade-out."""
        self._set_glass_active(False)
        self.scroll_timer.stop()
        self._duration_timer.stop()
        self._recording_start_time = None
        self._start_fade_out()

    def _set_glass_active(self, active: bool) -> None:
        """Transition glass opacity between idle (0.87) and active (1.0).

        Args:
            active: True for recording/active state, False for idle.
        """
        self._is_glass_active = active
        target = self._glass_active_opacity if active else self._glass_idle_opacity
        self.setWindowOpacity(target)

    def _update_duration(self) -> None:
        """Update the status label with elapsed recording duration (mm:ss)."""
        if self._recording_start_time is not None:
            elapsed = int(time.time() - self._recording_start_time)
            mins = f"{elapsed // 60:02d}"
            secs = f"{elapsed % 60:02d}"
            self.status_label.setText(f"Recording · {mins}:{secs}")

    def _show_empty_state(self) -> None:
        """Show a friendly placeholder in the transcript area when no content exists."""
        if not self._has_content:
            p = current_palette()
            self.text_edit.setHtml(
                f'<div style="color: {p.text_tertiary}; text-align: center; margin-top: 80px;">'
                'Transcription will appear here...'
                '</div>'
            )

    # ------------------------------------------------------------------
    # Adaptive theme
    # ------------------------------------------------------------------

    def _apply_theme(self) -> None:
        """Apply theme-aware stylesheets to all panel widgets.

        Idempotent and cheap — just re-sets stylesheets from the current
        palette.  Called once at end of __init__ and on desktop theme change.
        """
        p = current_palette()
        self._current_palette = p

        # Panel base
        self.setStyleSheet(glass_panel_css(p, "FloatingTranscriptPanel"))

        # Header widgets
        self._title_label.setStyleSheet(title_css(p))
        self._legend_btn.setStyleSheet(header_button_css(p, "legend"))
        self._close_btn.setStyleSheet(header_button_css(p, "close"))

        # Tabs
        self._tab_widget.setStyleSheet(tab_widget_css(p))

        # Live tab — text area and status
        self.text_edit.setStyleSheet(text_area_css(p))
        self.status_label.setStyleSheet(status_label_css(p))

        # History tab — splitter, list, detail header, buttons, viewer
        self._splitter.setStyleSheet(splitter_css(p))
        self._history_list.setStyleSheet(list_widget_css(p))
        self._detail_header.setStyleSheet(detail_header_css(p))
        self._scrub_btn.setStyleSheet(action_button_css(p, "scrub"))
        self._delete_btn.setStyleSheet(action_button_css(p, "delete"))
        self._history_viewer.setStyleSheet(text_area_css(p))

        # Legend overlay
        legend_styles = legend_overlay_css(p)
        self._legend_overlay.setStyleSheet(legend_styles["overlay"])
        if hasattr(self, "_legend_title"):
            self._legend_title.setStyleSheet(legend_styles["title"])
        if hasattr(self, "_legend_sep"):
            self._legend_sep.setStyleSheet(legend_styles["separator"])
        for lbl in getattr(self, "_legend_range_labels", []):
            lbl.setStyleSheet(legend_styles["range_label"])
        for lbl in getattr(self, "_legend_desc_labels", []):
            lbl.setStyleSheet(legend_styles["desc_label"])

        # Badge
        self._new_content_badge.setStyleSheet(badge_css(p))

        # Resize grip — draws its own textured triangle via paintEvent

        # Re-render empty state with updated text colour
        if not self._has_content:
            self._show_empty_state()

        scheme_name = "dark" if p is DARK_PALETTE else "light"
        logger.info("Applied %s theme to FloatingTranscriptPanel", scheme_name)

    # ------------------------------------------------------------------
    # Fade transition helpers
    # ------------------------------------------------------------------

    _FADE_DURATION_MS = 150
    _FADE_STEP_MS = 10
    _FADE_STEPS = _FADE_DURATION_MS // _FADE_STEP_MS  # 15

    def _start_fade_in(self) -> None:
        """Animate window opacity from 0 → 1 over 150ms, then show."""
        if hasattr(self, "_fade_timer") and self._fade_timer.isActive():
            self._fade_timer.stop()
        # Re-apply theme on show (picks up any desktop theme change while hidden)
        self._apply_theme()
        self.setWindowOpacity(0.0)
        self.show()
        self.raise_()
        self.activateWindow()
        self._fade_step = 0
        self._fade_direction = 1  # 1 = fading in
        self._fade_timer = QTimer(self)
        self._fade_timer.timeout.connect(self._on_fade_tick)
        self._fade_timer.start(self._FADE_STEP_MS)

    def _start_fade_out(self) -> None:
        """Animate window opacity from 1 → 0 over 150ms, then hide."""
        if hasattr(self, "_fade_timer") and self._fade_timer.isActive():
            self._fade_timer.stop()
        self.setWindowOpacity(1.0)
        self._fade_step = 0
        self._fade_direction = -1  # -1 = fading out
        self._fade_timer = QTimer(self)
        self._fade_timer.timeout.connect(self._on_fade_tick)
        self._fade_timer.start(self._FADE_STEP_MS)

    def _on_fade_tick(self) -> None:
        """Process one step of a fade animation."""
        self._fade_step += 1
        progress = self._fade_step / self._FADE_STEPS
        if self._fade_direction == 1:
            self.setWindowOpacity(min(progress, 1.0))
        else:
            self.setWindowOpacity(max(1.0 - progress, 0.0))
        if self._fade_step >= self._FADE_STEPS:
            self._fade_timer.stop()
            if self._fade_direction == -1:
                self.hide()
                self.setWindowOpacity(1.0)  # Reset for next show
    
    def toggle_panel(self) -> None:
        """Toggle panel visibility."""
        if self.isVisible():
            self.hide_panel()
        else:
            self.show_panel()
    
    def clear(self) -> None:
        """Clear all transcript content."""
        self.text_edit.clear()
        self.phrases.clear()
        self.current_phrase_idx = -1
        self._has_content = False
        self._show_empty_state()

    def update_segment(self, text: str, confidence: int, segment_index: int, is_final: bool = False, phrase_start: bool = False, speaker_id: Optional[str] = None) -> None:
        """
        Update a single segment. Each segment is part of a phrase (line).

        Args:
            text: Transcribed text for this segment
            confidence: Confidence score (0-100)
            segment_index: Position of this segment in current phrase
            is_final: If True, this phrase is complete
            phrase_start: If True, start a new phrase (new line)
            speaker_id: Optional speaker label for this phrase
        """
        if text.strip() == "[BLANK_AUDIO]":
            return
        
        # Clear empty-state placeholder on first real content
        if not self._has_content:
            self._has_content = True
            self.text_edit.clear()
        
        # Start new phrase if needed
        if phrase_start or self.current_phrase_idx < 0:
            # Insert new block before creating phrase structure
            cursor = self.text_edit.textCursor()
            if self.current_phrase_idx >= 0:
                cursor.insertBlock()  # New paragraph after previous phrase
            
            self.phrases.append(Phrase(segments=[], confidences=[], is_final=False, speaker_id=speaker_id))
            self.current_phrase_idx = len(self.phrases) - 1
            
            # Insert speaker label if provided
            if speaker_id:
                display_name = self._display_speaker_for(speaker_id)
                self._insert_speaker_label(display_name)
        
        # Get current phrase
        phrase = self.phrases[self.current_phrase_idx]
        phrase.is_final = is_final
        
        # Update or add segment to internal tracking
        if segment_index < len(phrase.segments):
            # Update existing segment - REPLACE IN PLACE
            phrase.segments[segment_index] = text
            phrase.confidences[segment_index] = confidence

            # Find and replace just this segment in display
            self._replace_segment_in_display(self.current_phrase_idx, segment_index, text, confidence)
        else:
            # Add new segment - APPEND TO CURRENT LINE
            phrase.segments.append(text)
            phrase.confidences.append(confidence)

            # Append segment to display with proper formatting
            self._append_segment_to_display(text, confidence)
        
        # Auto-scroll
        self._scroll_to_bottom()
        
        # If auto-scroll is paused, increment pending badge
        if self._auto_scroll_paused:
            self._pending_content_count += 1
            if hasattr(self, '_new_content_badge'):
                self._new_content_badge.setText(f"↓ {self._pending_content_count} new")
                self._new_content_badge.show()
                self._position_new_content_badge()
    
    # ------------------------------------------------------------------
    # Speaker label management
    # ------------------------------------------------------------------

    def set_speaker_names(self, names: Dict[str, str]) -> None:
        """Set the speaker label display mapping and rebuild the transcript.

        Args:
            names: Mapping from raw speaker labels (e.g. "spk0") to display
                   names (e.g. "Alice" or "SPK_0").
        """
        self._speaker_names = dict(names)
        self._rebuild_display()

    def get_speaker_names(self) -> Dict[str, str]:
        """Return the current speaker label mapping."""
        return dict(self._speaker_names)

    def pin_speaker_name(self, raw_label: str, name: str) -> None:
        """Pin a user-chosen name to a speaker label and refresh display.

        Args:
            raw_label: Raw speaker label (e.g. "spk0").
            name: User-chosen display name.
        """
        self._speaker_names[raw_label] = name
        self._pinned_speakers.add(raw_label)
        self._rebuild_display()

    # ------------------------------------------------------------------
    # Speaker label helpers
    # ------------------------------------------------------------------

    def _display_speaker_for(self, raw_or_display_label: str) -> str:
        """Resolve a label to its display form.

        The transcript store may contain display labels like "Alice" or
        "SPK_0" that were set by _apply_speaker_labels. We need to find
        the original raw label to check for user pins.
        """
        # Direct hit in speaker names
        if raw_or_display_label in self._speaker_names:
            return self._speaker_names[raw_or_display_label]
        # Search by value — the store has display labels, we need to
        # check if any raw label maps to this display label
        for raw, display in self._speaker_names.items():
            if display == raw_or_display_label:
                return display
        return raw_or_display_label

    def _raw_label_for_display(self, display_label: str) -> Optional[str]:
        """Find the raw label that maps to a display label.

        Returns None if no mapping found (label may be a raw label itself).
        """
        for raw, display in self._speaker_names.items():
            if display == display_label:
                return raw
        # The display label might itself be a raw label
        if display_label in self._speaker_names:
            return display_label
        return None

    def _prompt_speaker_name(self, current_label: str) -> None:
        """Show an input dialog for the user to name a speaker.

        If the user provides a name, emits speaker_name_pinned signal.
        """
        parent = self.parent() if self.parent() else self
        name, ok = QInputDialog.getText(
            parent,
            "Name Speaker",
            f"Enter a name for {current_label}:",
            text=current_label if not current_label.startswith("SPK_") else "",
        )
        if ok and name.strip():
            raw_label = self._raw_label_for_display(current_label)
            if raw_label is None:
                raw_label = current_label
            self.speaker_name_pinned.emit(raw_label, name.strip())

    def _on_anchor_clicked(self, url) -> None:
        """Handle clicks on speaker label anchors in the transcript."""
        link = url.toString() if hasattr(url, 'toString') else str(url)
        if link.startswith("speaker://"):
            speaker_id = link[len("speaker://"):]
            self._prompt_speaker_name(speaker_id)

    # ------------------------------------------------------------------
    # History tab methods
    # ------------------------------------------------------------------

    def _on_tab_changed(self, index: int) -> None:
        """Refresh history when switching to the History tab."""
        if index == 1:  # History tab
            self._refresh_history()

    def _refresh_history(self) -> None:
        """Re-scan recordings and repopulate the history list."""
        try:
            from meetandread.transcription.transcript_scanner import scan_recordings
        except ImportError:
            logger.warning("transcript_scanner not available — cannot populate history")
            return
        self._populate_history_list(scan_recordings())

    def _populate_history_list(self, recordings: list) -> None:
        """Populate the history QListWidget from a list of RecordingMeta.

        Args:
            recordings: List of RecordingMeta objects (expected sorted newest-first).
        """
        self._history_list.clear()
        for meta in recordings:
            # Format display date from ISO timestamp
            display_date = meta.recording_time
            if display_date:
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(display_date)
                    display_date = dt.strftime("%Y-%m-%d %H:%M")
                except (ValueError, TypeError):
                    pass  # keep raw string

            if meta.word_count == 0:
                display_text = f"{display_date} | (Empty recording)"
            else:
                display_text = (
                    f"{display_date} | {meta.word_count} words"
                    f" | {meta.speaker_count} speakers"
                )

            item = QListWidgetItem(display_text)
            item.setData(Qt.ItemDataRole.UserRole, str(meta.path))
            self._history_list.addItem(item)

    def _on_history_item_clicked(self, item: QListWidgetItem) -> None:
        """Load and display the transcript for the clicked history item."""
        # Reset comparison mode when switching items
        if self._is_comparison_mode:
            self._hide_scrub_accept_reject()

        md_path_str = item.data(Qt.ItemDataRole.UserRole)
        if not md_path_str:
            return
        md_path = Path(md_path_str)
        if not md_path.exists():
            self._current_history_md_path = None
            self._history_viewer.setPlainText(f"(File not found: {md_path})")
            self._detail_header.show()
            return

        self._current_history_md_path = md_path
        self._detail_header.show()
        html = self._render_history_transcript(md_path)
        if html is not None:
            self._history_viewer.setHtml(html)
        else:
            # Fallback: display raw markdown without anchors
            try:
                content = md_path.read_text(encoding="utf-8")
            except OSError as exc:
                self._history_viewer.setPlainText(f"(Error reading file: {exc})")
                return
            footer_marker = "\n---\n\n<!-- METADATA:"
            marker_idx = content.find(footer_marker)
            if marker_idx != -1:
                content = content[:marker_idx]
            self._history_viewer.setMarkdown(_strip_confidence_percentages(content))

    # ------------------------------------------------------------------
    # History delete functionality
    # ------------------------------------------------------------------

    def _on_history_context_menu(self, pos) -> None:
        """Show context menu on history list items.

        Args:
            pos: Position relative to the history list widget.
        """
        item = self._history_list.itemAt(pos)
        if item is None:
            return

        menu = QMenu(self._history_list)
        p = current_palette()
        menu.setStyleSheet(context_menu_css(p, accent_color=p.danger))

        scrub_action = menu.addAction("🔄  Scrub Recording")
        delete_action = menu.addAction("🗑  Delete Recording")
        scrub_action.triggered.connect(lambda: self._on_scrub_clicked())
        delete_action.triggered.connect(lambda: self._delete_recording(item))
        menu.exec(self._history_list.mapToGlobal(pos))

    def _on_delete_btn_clicked(self) -> None:
        """Handle Delete button click in the detail header."""
        current = self._history_list.currentItem()
        if current is None:
            return
        self._delete_recording(current)

    def _delete_recording(self, item: QListWidgetItem) -> None:
        """Delete a recording after user confirmation.

        Extracts the .md path from the item, enumerates associated files,
        shows a confirmation dialog, and performs the deletion.

        Args:
            item: The QListWidgetItem representing the recording.
        """
        md_path_str = item.data(Qt.ItemDataRole.UserRole)
        if not md_path_str:
            return

        md_path = Path(md_path_str)
        stem = md_path.stem  # filename without .md

        # Build a human-readable name from the item display text
        recording_name = item.text().split("|")[0].strip()

        # Enumerate files to show count in confirmation
        try:
            from meetandread.recording.management import enumerate_recording_files, delete_recording
            files = enumerate_recording_files(stem)
        except Exception as exc:
            logger.error("Failed to enumerate recording files: %s", exc)
            files = []

        file_count = len(files)

        # Show confirmation dialog
        parent = self.parent() if self.parent() else self
        reply = QMessageBox.question(
            parent,
            "Delete Recording",
            f"Delete '{recording_name}'?\n\n"
            f"This will permanently remove {file_count} file{'s' if file_count != 1 else ''}.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # Perform deletion
        try:
            count, deleted = delete_recording(stem)
            logger.info(
                "Deleted recording '%s': %d files removed",
                recording_name, count,
            )
        except Exception as exc:
            logger.error("Failed to delete recording '%s': %s", recording_name, exc)
            QMessageBox.warning(
                parent,
                "Delete Failed",
                f"Could not delete recording '{recording_name}'.\n\n{exc}",
            )
            return

        # Clear viewer state
        self._current_history_md_path = None
        self._history_viewer.clear()
        self._history_viewer.setPlaceholderText("Select a recording to view its transcript")
        self._detail_header.hide()

        # Refresh the history list
        self._refresh_history()

    # ------------------------------------------------------------------
    # Scrub (re-transcribe) functionality
    # ------------------------------------------------------------------

    def _on_scrub_clicked(self) -> None:
        """Handle Scrub button / context-menu click.

        Validates that the selected recording has a WAV file, shows a model
        picker dialog, then starts ScrubRunner in a background thread.
        """
        if self._is_scrubbing:
            return

        current = self._history_list.currentItem()
        if current is None:
            return

        md_path_str = current.data(Qt.ItemDataRole.UserRole)
        if not md_path_str:
            return
        md_path = Path(md_path_str)
        stem = md_path.stem

        # Check for WAV file
        try:
            from meetandread.audio.storage.paths import get_recordings_dir
            wav_path = get_recordings_dir() / f"{stem}.wav"
        except Exception:
            wav_path = md_path.parent.parent / "recordings" / f"{stem}.wav"

        if not wav_path.exists():
            parent = self.parent() if self.parent() else self
            QMessageBox.information(
                parent,
                "Cannot Scrub",
                "Cannot scrub — audio file missing.\n\n"
                "The original .wav recording file is required for re-transcription.",
            )
            return

        # Show model picker dialog
        dialog = self._create_scrub_dialog()
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        model_size = dialog._model_combo.currentData()
        if not model_size:
            return

        # Start the scrub
        self._start_scrub(wav_path, md_path, model_size)

    def _create_scrub_dialog(self) -> QDialog:
        """Create the model picker dialog for scrub.

        Returns a QDialog with a QComboBox showing all 5 Whisper models
        with WER from benchmark_history. Default selection is the current
        post-process model from config.
        """
        dialog = QDialog(self)
        dialog.setWindowTitle("Scrub Recording")
        dialog.setFixedSize(340, 180)
        p = current_palette()
        dialog.setStyleSheet(dialog_css(p))

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        # Title label
        title_label = QLabel("Re-transcribe with a different model:")
        title_label.setStyleSheet(f"font-weight: bold; color: {p.info}; font-size: 13px;")
        layout.addWidget(title_label)

        # Model combo
        combo = QComboBox()
        combo.setStyleSheet(combo_box_css(p, accent_color=p.info))

        # Populate with models + WER
        try:
            from meetandread.config import get_config
            _cfg = get_config()
            _bench_history = _cfg.transcription.benchmark_history
            _default_model = _cfg.transcription.postprocess_model_size
        except Exception:
            _bench_history = {}
            _default_model = "base"

        _model_order = ["tiny", "base", "small", "medium", "large"]
        _select_idx = 0
        for _i, _mn in enumerate(_model_order):
            _entry = _bench_history.get(_mn)
            if _entry and "wer" in _entry:
                _wer_pct = _entry["wer"] * 100
                _item_text = f"{_mn} — WER: {_wer_pct:.1f}%"
            else:
                _item_text = f"{_mn} (not benchmarked)"
            combo.addItem(_item_text, _mn)
            if _mn == _default_model:
                _select_idx = _i
        combo.setCurrentIndex(_select_idx)

        layout.addWidget(combo)
        dialog._model_combo = combo  # store reference for caller

        layout.addStretch()

        # OK / Cancel buttons
        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
        )
        btn_box.setStyleSheet(action_button_css(p, "dialog"))
        btn_box.accepted.connect(dialog.accept)
        btn_box.rejected.connect(dialog.reject)
        layout.addWidget(btn_box)

        return dialog

    def _start_scrub(self, wav_path: Path, md_path: Path, model_size: str) -> None:
        """Start a ScrubRunner background re-transcription.

        Args:
            wav_path: Path to the source .wav audio file.
            md_path: Path to the canonical transcript .md file.
            model_size: Whisper model size (e.g. "small").
        """
        from meetandread.transcription.scrub import ScrubRunner

        # Store state
        self._scrub_model_size = model_size
        self._is_scrubbing = True
        self._is_comparison_mode = False

        # Save original transcript HTML for comparison later
        self._scrub_original_html = self._history_viewer.toHtml()

        # Disable scrub button and update text
        self._scrub_btn.setEnabled(False)
        self._scrub_btn.setText("Scrubbing... 0%")

        # Create and start runner
        self._scrub_runner = ScrubRunner(
            settings=self._get_app_settings(),
            on_progress=self._on_scrub_progress,
            on_complete=self._on_scrub_complete,
        )
        self._scrub_sidecar_path = self._scrub_runner.scrub_recording(
            wav_path, md_path, model_size,
        )

    def _get_app_settings(self):
        """Get the current AppSettings from config."""
        try:
            from meetandread.config import get_config
            return get_config()
        except Exception:
            from meetandread.config.models import AppSettings
            return AppSettings()

    def _on_scrub_progress(self, pct: int) -> None:
        """Update scrub button text with progress percentage.

        Called from the ScrubRunner background thread — uses
        QTimer.singleShot to marshal the update to the GUI thread.
        """
        QTimer.singleShot(0, lambda: self._scrub_btn.setText(f"Scrubbing... {pct}%"))

    def _on_scrub_complete(self, sidecar_path: str, error: Optional[str]) -> None:
        """Handle scrub completion — show comparison or error.

        Called from the ScrubRunner background thread — schedules the
        heavy UI work on the GUI thread via QTimer.singleShot.
        """
        # Use QTimer to run on GUI thread
        QTimer.singleShot(0, lambda: self._handle_scrub_complete(sidecar_path, error))

    def _handle_scrub_complete(self, sidecar_path: str, error: Optional[str]) -> None:
        """Process scrub completion on the GUI thread."""
        self._is_scrubbing = False
        self._scrub_btn.setEnabled(True)
        self._scrub_btn.setText("🔄 Scrub")

        if error:
            parent = self.parent() if self.parent() else self
            QMessageBox.warning(
                parent,
                "Scrub Failed",
                f"Re-transcription failed:\n\n{error}",
            )
            logger.error("Scrub failed: %s", error)
            return

        # Show side-by-side comparison
        self._show_scrub_comparison(sidecar_path)

    def _show_scrub_comparison(self, sidecar_path: str) -> None:
        """Show side-by-side comparison of original vs scrubbed transcript.

        Renders both transcripts in a split view with Accept/Reject buttons.

        Args:
            sidecar_path: Path to the sidecar .md file with scrub result.
        """
        sidecar = Path(sidecar_path)
        if not sidecar.exists():
            logger.warning("Sidecar not found for comparison: %s", sidecar_path)
            return

        self._is_comparison_mode = True
        self._scrub_sidecar_path = sidecar_path

        # Build the comparison view as HTML in the history viewer
        original_text = self._extract_transcript_body(
            self._current_history_md_path
        )
        scrubbed_text = self._extract_transcript_body(sidecar)

        # Build HTML with two-column layout
        html = f"""
        <html>
        <head><style>
            body {{ margin: 0; padding: 4px; background-color: #2a2a2a; color: #fff; font-size: 12px; }}
            .comparison {{ display: flex; gap: 8px; }}
            .column {{ flex: 1; }}
            .column-header {{
                font-weight: bold;
                padding: 4px 8px;
                border-radius: 4px 4px 0 0;
                font-size: 11px;
                text-align: center;
            }}
            .original .column-header {{ background-color: #37474F; color: #B0BEC5; }}
            .scrubbed .column-header {{ background-color: #1B5E20; color: #A5D6A7; }}
            .content {{
                padding: 6px 8px;
                background-color: #333;
                border-radius: 0 0 4px 4px;
                min-height: 50px;
                line-height: 1.4;
                white-space: pre-wrap;
                word-wrap: break-word;
            }}
        </style></head>
        <body>
        <div class="comparison">
            <div class="column original">
                <div class="column-header">Original</div>
                <div class="content">{_escape_html(original_text)}</div>
            </div>
            <div class="column scrubbed">
                <div class="column-header">Scrubbed ({_escape_html(self._scrub_model_size or "?")})</div>
                <div class="content">{_escape_html(scrubbed_text)}</div>
            </div>
        </div>
        </body></html>
        """

        self._history_viewer.setHtml(html)

        # Show Accept/Reject buttons instead of normal header buttons
        self._show_scrub_accept_reject()

    def _show_scrub_accept_reject(self) -> None:
        """Replace the scrub button with Accept/Reject during comparison mode."""
        self._scrub_btn.hide()

        # Create Accept button
        if not hasattr(self, '_scrub_accept_btn'):
            self._scrub_accept_btn = QPushButton("✓ Accept")
            self._scrub_accept_btn.setFixedHeight(26)
            p = current_palette()
            self._scrub_accept_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {p.surface};
                    color: {p.accent};
                    border: 1px solid {p.accent};
                    border-radius: 4px;
                    padding: 2px 10px;
                    font-size: 11px;
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    background-color: {p.surface_hover};
                    border-color: {p.accent};
                }}
                QPushButton:pressed {{
                    background-color: {p.surface};
                }}
            """)
            self._scrub_accept_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self._scrub_accept_btn.clicked.connect(self._on_scrub_accept)

            # Create Reject button
            self._scrub_reject_btn = QPushButton("✗ Reject")
            self._scrub_reject_btn.setFixedHeight(26)
            self._scrub_reject_btn.setStyleSheet(action_button_css(p, "delete"))
            self._scrub_reject_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self._scrub_reject_btn.clicked.connect(self._on_scrub_reject)

            # Insert into detail header layout (before delete button)
            header_layout = self._detail_header.layout()
            delete_idx = header_layout.indexOf(self._delete_btn)
            header_layout.insertWidget(delete_idx, self._scrub_accept_btn)
            header_layout.insertWidget(delete_idx + 1, self._scrub_reject_btn)
        else:
            self._scrub_accept_btn.show()
            self._scrub_reject_btn.show()

    def _hide_scrub_accept_reject(self) -> None:
        """Hide Accept/Reject buttons and show the scrub button again."""
        if hasattr(self, '_scrub_accept_btn'):
            self._scrub_accept_btn.hide()
        if hasattr(self, '_scrub_reject_btn'):
            self._scrub_reject_btn.hide()
        self._scrub_btn.show()
        self._is_comparison_mode = False

    def _on_scrub_accept(self) -> None:
        """Accept the scrub result — promote sidecar to canonical transcript."""
        if self._current_history_md_path is None or self._scrub_model_size is None:
            return

        try:
            from meetandread.transcription.scrub import ScrubRunner
            ScrubRunner.accept_scrub(
                self._current_history_md_path, self._scrub_model_size,
            )
            logger.info(
                "Accepted scrub: %s model %s",
                self._current_history_md_path, self._scrub_model_size,
            )
        except FileNotFoundError:
            parent = self.parent() if self.parent() else self
            QMessageBox.warning(
                parent, "Accept Failed",
                "Sidecar file not found. It may have been deleted.",
            )
            self._hide_scrub_accept_reject()
            return
        except Exception as exc:
            parent = self.parent() if self.parent() else self
            QMessageBox.warning(
                parent, "Accept Failed", f"Could not accept scrub result:\n\n{exc}",
            )
            self._hide_scrub_accept_reject()
            return

        # Refresh the viewer with the updated transcript
        self._hide_scrub_accept_reject()
        self._refresh_after_scrub()

    def _on_scrub_reject(self) -> None:
        """Reject the scrub result — delete the sidecar file."""
        if self._current_history_md_path is None or self._scrub_model_size is None:
            return

        try:
            from meetandread.transcription.scrub import ScrubRunner
            ScrubRunner.reject_scrub(
                self._current_history_md_path, self._scrub_model_size,
            )
            logger.info(
                "Rejected scrub: %s model %s",
                self._current_history_md_path, self._scrub_model_size,
            )
        except Exception as exc:
            logger.warning("Error rejecting scrub: %s", exc)

        # Restore original view
        self._hide_scrub_accept_reject()
        self._refresh_after_scrub()

    def _refresh_after_scrub(self) -> None:
        """Refresh the history list and viewer after accept/reject.

        After accept the canonical transcript changes (word count may differ),
        so the recording list must be repopulated.  After reject the list is
        refreshed as well (harmless, ensures consistency).  The previously
        selected item is re-selected so the user stays on the same recording.
        """
        md_path = self._current_history_md_path

        # Refresh the history list (word count may have changed after accept)
        self._refresh_history()

        # Re-select the item that was being viewed
        if md_path is not None:
            self._reselect_history_item(md_path)

        # Refresh the viewer content
        if md_path is not None and md_path.exists():
            html = self._render_history_transcript(md_path)
            if html is not None:
                self._history_viewer.setHtml(html)
            else:
                try:
                    content = md_path.read_text(encoding="utf-8")
                except OSError:
                    content = ""
                footer_marker = "\n---\n\n<!-- METADATA:"
                marker_idx = content.find(footer_marker)
                if marker_idx != -1:
                    content = content[:marker_idx]
                self._history_viewer.setMarkdown(_strip_confidence_percentages(content))
        else:
            self._history_viewer.clear()
            self._history_viewer.setPlaceholderText(
                "Select a recording to view its transcript",
            )

    def _reselect_history_item(self, md_path: Path) -> None:
        """Re-select a history list item by its transcript path.

        Called after the list is repopulated so the user stays on the
        same recording they were viewing.

        Args:
            md_path: Path to the transcript .md file to re-select.
        """
        md_str = str(md_path)
        for i in range(self._history_list.count()):
            item = self._history_list.item(i)
            if item and item.data(Qt.ItemDataRole.UserRole) == md_str:
                self._history_list.setCurrentItem(item)
                return
        logger.debug("Could not re-select history item for %s", md_path)

    @staticmethod
    def _extract_transcript_body(md_path: Optional[Path]) -> str:
        """Extract the markdown body (before METADATA footer) from a transcript.

        Args:
            md_path: Path to the transcript .md file.

        Returns:
            The markdown body text, or an error message string.
        """
        if md_path is None or not md_path.exists():
            return "(file not found)"
        try:
            content = md_path.read_text(encoding="utf-8")
        except OSError as exc:
            return f"(error reading file: {exc})"

        footer_marker = "\n---\n\n<!-- METADATA:"
        marker_idx = content.find(footer_marker)
        if marker_idx != -1:
            content = content[:marker_idx]
        return content.strip()

    # ------------------------------------------------------------------
    # History transcript rendering with clickable speaker anchors
    # ------------------------------------------------------------------

    def _render_history_transcript(self, md_path: Path) -> Optional[str]:
        """Render a transcript .md file as HTML with clickable speaker anchors.

        Reads the .md file, parses the JSON metadata footer to get speakers,
        and returns HTML where each speaker label is an anchor tag with
        format ``speaker://{speaker_id}``.

        Args:
            md_path: Path to the transcript .md file.

        Returns:
            HTML string for the viewer, or None if no metadata is found.
        """
        try:
            content = md_path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.error("Failed to read transcript for rendering: %s: %s", md_path, exc)
            return None

        # Split markdown body from JSON footer
        footer_marker = "\n---\n\n<!-- METADATA:"
        marker_idx = content.find(footer_marker)
        if marker_idx == -1:
            return None

        md_body = _strip_confidence_percentages(content[:marker_idx])

        # Parse metadata to find speakers
        metadata_text = content[marker_idx + len(footer_marker):]
        if metadata_text.strip().endswith(" -->"):
            metadata_text = metadata_text.strip()[:-len(" -->")]

        try:
            data = json.loads(metadata_text)
        except json.JSONDecodeError as exc:
            logger.warning("Malformed metadata in %s: %s", md_path, exc)
            return None

        # Collect unique speaker IDs from words (None counts as "Unknown Speaker")
        speakers = []
        seen = set()
        has_unknown = False
        for word in data.get("words", []):
            sid = word.get("speaker_id")
            if sid is not None and sid not in seen:
                seen.add(sid)
                speakers.append(sid)
            elif sid is None:
                has_unknown = True

        # Build HTML with clickable speaker anchors
        # The markdown body has lines like "**SPK_0**" — make them anchors
        html_lines = []
        for line in md_body.splitlines():
            # Match speaker label lines: **SpeakerName**
            match = re.match(r"^\*\*(.+?)\*\*\s*$", line)
            if match:
                speaker_label = match.group(1)
                if speaker_label in seen:
                    color = speaker_color(speaker_label)
                    html_lines.append(
                        f'<p><a href="speaker:{speaker_label}" '
                        f'style="color:{color}; font-weight:bold; text-decoration:none;">'
                        f'[{speaker_label}]</a></p>'
                    )
                elif speaker_label == "Unknown Speaker" and has_unknown:
                    # Make "Unknown Speaker" clickable so user can assign an identity
                    color = "#888888"
                    html_lines.append(
                        f'<p><a href="speaker:__unknown__" '
                        f'style="color:{color}; font-weight:bold; text-decoration:none;">'
                        f'[{speaker_label}]</a></p>'
                    )
                else:
                    html_lines.append(f"<p><b>{speaker_label}</b></p>")
            else:
                # Regular line — escape HTML and preserve whitespace
                escaped = (
                    line.replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                )
                # Preserve leading spaces and convert newlines
                if escaped.strip():
                    # Convert markdown italic markers
                    escaped = re.sub(r"\*(.+?)\*", r"<i>\1</i>", escaped)
                    html_lines.append(f"<p>{escaped}</p>")
                elif not escaped:
                    html_lines.append("<br>")

        return "\n".join(html_lines)

    # ------------------------------------------------------------------
    # Speaker rename in history transcripts
    # ------------------------------------------------------------------

    def _on_history_anchor_clicked(self, url: QUrl) -> None:
        """Handle clicks on speaker label anchors in the history viewer.

        Extracts the speaker_id from the ``speaker:{id}`` URL, opens the
        identity-link dialog, and persists the chosen identity.
        """
        link = url.toString()
        prefix = "speaker:"
        if not link.startswith(prefix):
            return

        raw_label = link[len(prefix):]
        if not raw_label:
            return

        parent = self.parent() if self.parent() else self
        md_path = self._current_history_md_path

        linked = _open_identity_link_dialog(md_path, raw_label, parent)
        if not linked:
            return

        # Refresh the viewer
        html = self._render_history_transcript(md_path)
        if html is not None:
            self._history_viewer.setHtml(html)
        else:
            self._history_viewer.setPlainText("(Error refreshing after link)")

    def _rename_speaker_in_file(
        self, md_path: Path, old_name: str, new_name: str
    ) -> None:
        """Rename a speaker in a transcript .md file.

        Updates both the JSON metadata (words and segments arrays) and the
        markdown body speaker labels.

        Args:
            md_path: Path to the transcript .md file.
            old_name: Current speaker name to replace.
            new_name: New speaker name.
        """
        content = md_path.read_text(encoding="utf-8")

        # Split into markdown body and JSON footer
        footer_marker = "\n---\n\n<!-- METADATA:"
        marker_idx = content.find(footer_marker)
        if marker_idx == -1:
            raise ValueError(f"No metadata footer found in {md_path}")

        md_body = content[:marker_idx]
        # Capture the full prefix including the space before JSON
        # e.g. "\n---\n\n<!-- METADATA: "
        after_marker = content[marker_idx + len(footer_marker):]
        space_before_json = ""
        if after_marker.startswith(" "):
            space_before_json = " "
            after_marker = after_marker[1:]

        metadata_text = after_marker
        if metadata_text.strip().endswith(" -->"):
            metadata_text = metadata_text.strip()[:-len(" -->")]

        data = json.loads(metadata_text)

        # Update words array
        words_updated = 0
        for word in data.get("words", []):
            if word.get("speaker_id") == old_name:
                word["speaker_id"] = new_name
                words_updated += 1

        # Update segments array
        segments_updated = 0
        for segment in data.get("segments", []):
            if segment.get("speaker_id") == old_name:
                segment["speaker_id"] = new_name
                segments_updated += 1

        # Rebuild markdown body: replace speaker labels
        # Speaker labels appear as **OldName** on their own line
        updated_body = re.sub(
            re.escape(f"**{old_name}**"),
            f"**{new_name}**",
            md_body,
        )

        # Rebuild the file
        updated_json = json.dumps(data, indent=2)
        new_content = (
            updated_body + footer_marker + space_before_json + updated_json + " -->\n"
        )

        md_path.write_text(new_content, encoding="utf-8")

        logger.info(
            "Renamed speaker '%s' -> '%s' in %s (%d words, %d segments updated)",
            old_name, new_name, md_path, words_updated, segments_updated,
        )

    def _propagate_rename_to_signatures(
        self, md_path: Path, old_name: str, new_name: str
    ) -> None:
        """Propagate a speaker rename to the VoiceSignatureStore.

        If the old speaker name has a saved embedding in the signature
        database (located in the same directory as the transcript file),
        saves the embedding under the new name and deletes the old entry.

        Args:
            md_path: Path to the transcript file (used to locate the DB).
            old_name: Current speaker name.
            new_name: New speaker name.
        """
        try:
            from meetandread.speaker.signatures import VoiceSignatureStore
        except ImportError:
            logger.warning(
                "VoiceSignatureStore not available — skipping rename propagation"
            )
            return

        db_path = md_path.parent / "speaker_signatures.db"
        if not db_path.exists():
            # Try the default data directory
            from meetandread.audio.storage.paths import get_recordings_dir
            default_db = get_recordings_dir() / "speaker_signatures.db"
            if default_db.exists():
                db_path = default_db
            else:
                logger.info(
                    "No signature database found — speaker '%s' not in store",
                    old_name,
                )
                return

        with VoiceSignatureStore(db_path=str(db_path)) as store:
            # Find the old speaker's profile
            profiles = store.load_signatures()
            old_profile = None
            for profile in profiles:
                if profile.name == old_name:
                    old_profile = profile
                    break

            if old_profile is None:
                logger.info(
                    "Speaker '%s' not found in signature store — no propagation needed",
                    old_name,
                )
                return

            # Save under new name, delete old
            store.save_signature(
                new_name,
                old_profile.embedding,
                averaged_from_segments=old_profile.num_samples,
            )
            store.delete_signature(old_name)

            logger.info(
                "Propagated rename '%s' -> '%s' to signature store at %s",
                old_name, new_name, db_path,
            )

    def _rebuild_display(self) -> None:
        """Rebuild the entire text display from stored phrases."""
        if self.text_edit is None:
            return
        self.text_edit.clear()
        for i, phrase in enumerate(self.phrases):
            if i > 0:
                cursor = self.text_edit.textCursor()
                cursor.insertBlock()

            # Write speaker label if known
            if phrase.speaker_id:
                display_name = self._display_speaker_for(phrase.speaker_id)
                self._insert_speaker_label(display_name)

            # Write phrase text with confidence coloring
            for seg_idx, (seg_text, seg_conf) in enumerate(zip(phrase.segments, phrase.confidences)):
                self._append_segment_to_display(seg_text, seg_conf, add_space=(seg_idx > 0))

    def _insert_speaker_label(self, speaker_id: str) -> None:
        """Insert a clickable speaker label at the current cursor position."""
        cursor = self.text_edit.textCursor()
        
        # Prepend elapsed timestamp if recording is active
        if self._recording_start_time is not None:
            elapsed = int(time.time() - self._recording_start_time)
            mins = f"{elapsed // 60:02d}"
            secs = f"{elapsed % 60:02d}"
            ts_fmt = QTextCharFormat()
            ts_fmt.setForeground(QColor("#666666"))
            ts_fmt.setFontWeight(QFont.Weight.Normal)
            cursor.insertText(f"[{mins}:{secs}] ", ts_fmt)
        
        color = speaker_color(speaker_id)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        fmt.setFontWeight(QFont.Weight.Bold)
        fmt.setAnchor(True)
        fmt.setAnchorHref(f"speaker://{speaker_id}")
        label_text = f"[{speaker_id}] "
        cursor.insertText(label_text, fmt)

    def _append_segment_to_display(self, text: str, confidence: int, add_space: bool = False) -> None:
        """Append a segment to the current line with proper formatting."""
        cursor = self.text_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        # Add space between segments
        if add_space or (self.phrases and self.current_phrase_idx >= 0
                         and self.phrases[self.current_phrase_idx].segments
                         and len(self.phrases[self.current_phrase_idx].segments) > 0):
            cursor.insertText(" ")

        # Determine color based on confidence
        color = self._get_confidence_color(confidence)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        fmt.setFontWeight(QFont.Weight.Normal)

        cursor.insertText(text, fmt)

    def _replace_segment_in_display(self, phrase_idx: int, segment_idx: int, text: str, confidence: int) -> None:
        """Replace a specific segment in the display without rebuilding everything."""
        cursor = self.text_edit.textCursor()
        
        # Move to start of document
        cursor.movePosition(QTextCursor.MoveOperation.Start)
        
        # Navigate to correct phrase block
        for _ in range(phrase_idx):
            cursor.movePosition(QTextCursor.MoveOperation.NextBlock)
        
        # Navigate to correct segment within phrase
        # Segments are separated by spaces, so we move by words
        for _ in range(segment_idx):
            # Move past text and space
            cursor.movePosition(QTextCursor.MoveOperation.NextWord)
            # Skip the space between segments (if not last)
            if _ < segment_idx - 1:
                cursor.movePosition(QTextCursor.MoveOperation.NextCharacter)
        
        # Select the segment text
        cursor.movePosition(QTextCursor.MoveOperation.StartOfWord)
        # Find end of this segment (either space or block end)
        if segment_idx < len(self.phrases[phrase_idx].segments) - 1:
            # Segment is followed by space
            cursor.movePosition(QTextCursor.MoveOperation.EndOfWord, QTextCursor.MoveMode.KeepAnchor)
        else:
            # Last segment - select to end of block
            cursor.movePosition(QTextCursor.MoveOperation.EndOfBlock, QTextCursor.MoveMode.KeepAnchor)
        
        # Replace with new text and formatting
        color = self._get_confidence_color(confidence)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        fmt.setFontWeight(QFont.Weight.Normal)
        cursor.insertText(text, fmt)
    
    def _get_confidence_color(self, confidence: int) -> str:
        """Get color based on confidence score — delegates to canonical thresholds."""
        return get_confidence_color(confidence)
    
    def _near_bottom_threshold(self) -> int:
        """Return a proportional bottom-detection threshold in pixels.
        
        Uses 10% of the scrollbar page step, with a floor of 10px to
        avoid degenerate cases on very small viewports. This replaces
        all hardcoded pixel thresholds for bottom detection.
        """
        return max(10, int(self.text_edit.verticalScrollBar().pageStep() * 0.1))
    
    def _on_scroll_value_changed(self, value: int) -> None:
        """
        Detect manual scroll and pause/resume auto-scroll.
        
        Called when scrollbar value changes. If user scrolls up from bottom,
        pause auto-scroll for 10 seconds to allow reading. If the user scrolls
        back to the bottom while paused, resume auto-scroll immediately.
        """
        scrollbar = self.text_edit.verticalScrollBar()
        maximum = scrollbar.maximum()
        threshold = self._near_bottom_threshold()
        
        if maximum > 0 and value < maximum - threshold:
            # User scrolled up — pause auto-scroll
            if not self._auto_scroll_paused:
                self._auto_scroll_paused = True
                self._pause_timer.start(10000)  # 10 seconds
                self.status_label.setText("Auto-scroll paused (10s)")
        elif self._auto_scroll_paused and maximum > 0 and value >= maximum - threshold:
            # User scrolled back to bottom while paused — resume
            self._auto_scroll_paused = False
            self._pause_timer.stop()
            self._pending_content_count = 0
            if hasattr(self, '_new_content_badge'):
                self._new_content_badge.hide()
            self.status_label.setText("Recording...")
            self._scroll_to_bottom()
        
        # Update tracking
        self._last_scroll_value = value
        self._is_at_bottom = (value >= maximum - threshold)
    
    def _resume_auto_scroll(self) -> None:
        """Resume auto-scroll after pause timer expires."""
        self._auto_scroll_paused = False
        self._pending_content_count = 0
        if hasattr(self, '_new_content_badge'):
            self._new_content_badge.hide()
        self.status_label.setText("Recording...")
        # Immediately scroll to bottom to catch up
        self._scroll_to_bottom()
    
    def _scroll_to_bottom(self) -> None:
        """Auto-scroll to show latest text."""
        # Don't scroll if auto-scroll is paused
        if self._auto_scroll_paused:
            return
        
        scrollbar = self.text_edit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def save_to_file(self, filepath: str) -> None:
        """Save transcript to file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("# Transcription\n\n")
            for i, phrase in enumerate(self.phrases):
                text = " ".join(phrase.segments)
                avg_conf = sum(phrase.confidences) // len(phrase.confidences) if phrase.confidences else 0
                f.write(f"{i+1}. [{avg_conf}%] {text}\n")
    
    def closeEvent(self, event) -> None:
        """Handle close event."""
        self.closed.emit()
        event.accept()
    
    def mousePressEvent(self, event):
        """Start dragging."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
    
    def mouseMoveEvent(self, event):
        """Handle dragging."""
        if self._dragging and self._drag_pos is not None:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()
    
    def mouseReleaseEvent(self, event):
        """Stop dragging."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            event.accept()


# Settings panel (similar floating approach)


class BudgetProgressBar(QProgressBar):
    """Progress bar that draws a red vertical line at the budget threshold."""

    def __init__(self, budget_percent: float = 80.0, parent=None):
        super().__init__(parent)
        self._budget_percent = budget_percent

    def paintEvent(self, event):
        """Paint the progress bar then overlay a red budget marker."""
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setPen(QPen(QColor(255, 60, 60, 220), 2))
        x = int(self.width() * self._budget_percent / 100.0)
        painter.drawLine(x, 0, x, self.height())
        painter.end()


# ---------------------------------------------------------------------------
# CCOverlayPanel — compact closed-caption overlay for live transcript
# ---------------------------------------------------------------------------

class CCOverlayPanel(QWidget):
    """Compact draggable/resizable CC-style overlay for live transcript text.

    Frameless, translucent, always-on-top window that displays real-time
    transcription text during recording.  Designed to be a lightweight
    surface with no history, tabs, or status chrome.

    Shell methods:
        show_panel()         — show with fade-in
        hide_panel(immediate) — hide, optionally deferred via fade
        toggle_panel()       — toggle visibility
        clear()              — reset text and content state

    Observability:
        objectName()       → "AethericCCOverlay"
        text_edit.objectName() → "AethericCCText"
        isVisible()        → panel visibility state
        _has_content       → whether any transcript text has been received
        phrases            → list of Phrase objects
        current_phrase_idx → index of the active phrase being built
    """

    # Signals
    segment_ready = pyqtSignal(str, int, int, bool, bool, object)  # text, confidence, segment_index, is_final, phrase_start, speaker_id

    # Fade constants (matching FloatingTranscriptPanel)
    _FADE_DURATION_MS = 150
    _FADE_STEP_MS = 10
    _FADE_STEPS = _FADE_DURATION_MS // _FADE_STEP_MS  # 15

    # Delay before fade-out after recording stops (1.5 seconds)
    CC_FADE_DELAY_MS = 1500

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.setObjectName("AethericCCOverlay")

        # Track content state
        self._has_content: bool = False

        # --- Phrase tracking for live transcript ---
        self.phrases: List[Phrase] = []
        self.current_phrase_idx: int = -1

        # --- Speaker display name mapping ---
        self._speaker_names: Dict[str, str] = {}

        # --- Delayed fade-out timer ---
        # After recording stops, the overlay stays visible for CC_FADE_DELAY_MS
        # before starting the 150 ms fade-out.  show_panel() cancels both.
        self._fade_delay_timer = QTimer(self)
        self._fade_delay_timer.setSingleShot(True)
        self._fade_delay_timer.timeout.connect(self._on_delay_elapsed)

        # --- Window flags: frameless, tool (no taskbar), always on top ---
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )

        # Semi-transparent overlay: translucent window + styled background
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)

        # --- Compact size constraints (48px font, max 2 visible lines) ---
        # --- Size: max width = widest display width, height free-form ---
        from PyQt6.QtWidgets import QApplication
        screens = QApplication.screens()
        max_w = max(s.geometry().width() for s in screens) if screens else 1920
        self.setMinimumSize(400, 140)
        self.setMaximumSize(max_w, 16777215)  # Width capped to widest display, height free
        self.resize(600, 180)

        # --- Layout: text edit fills the panel ---
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # -- Drag state (panel-level, all areas except resize grip) --
        self._dragging: bool = False
        self._drag_pos: Optional[QPoint] = None

        self.text_edit = QTextEdit()
        self.text_edit.setObjectName("AethericCCText")
        self.text_edit.setReadOnly(True)
        # Disable interaction so mouse events propagate to panel for drag
        self.text_edit.setEnabled(False)
        self.text_edit.setFrameShape(QFrame.Shape.NoFrame)
        self.text_edit.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.text_edit.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        layout.addWidget(self.text_edit)

        # --- Resize grip: direct child of panel (MEM083 pattern) ---
        self._resize_grip = TexturedSizeGrip(self)
        self._resize_grip.setFixedSize(16, 16)
        self._resize_grip.setCursor(Qt.CursorShape.SizeFDiagCursor)
        self._resize_grip.show()

        # --- Drag state ---
        self._dragging: bool = False
        self._drag_pos: Optional[QPoint] = None

        # Apply initial theme
        self._apply_theme()

        # Restore persisted geometry (position + size)
        self._restore_geometry()

        logger.debug("CCOverlayPanel created (parent=%s)", type(parent).__name__ if parent else "None")

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------

    def _apply_theme(self) -> None:
        """Apply Aetheric CC overlay styling."""
        p = current_palette()
        self.setStyleSheet(aetheric_cc_overlay_css(p))
        # Grip draws its own textured triangle via paintEvent

    # ------------------------------------------------------------------
    # Paint — manual semi-transparent background (QSS alpha doesn't work
    # on Windows frameless translucent windows)
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:
        from PyQt6.QtGui import QPainter, QColor, QPen
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        # Semi-transparent dark background
        painter.setBrush(QColor(30, 29, 30, 179))  # 70% opacity
        painter.setPen(QPen(QColor(255, 255, 255, 30), 1))
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 12, 12)
        painter.end()

    # ------------------------------------------------------------------
    # Resize — reposition grip (MEM083 pattern)
    # ------------------------------------------------------------------

    def resizeEvent(self, event) -> None:
        """Reposition resize grip to bottom-right corner."""
        if hasattr(self, "_resize_grip"):
            self._resize_grip.move(
                self.width() - self._resize_grip.width(),
                self.height() - self._resize_grip.height(),
            )
        self.save_geometry()
        super().resizeEvent(event)

    # ------------------------------------------------------------------
    # Drag handlers — entire panel is draggable except resize grip
    # ------------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Start drag on left-button press (anywhere except resize grip)."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Move panel with mouse during drag, clamped to screen boundaries."""
        if self._dragging and self._drag_pos is not None:
            raw_pos = event.globalPosition().toPoint() - self._drag_pos
            self.move(clamp_to_screen(self, raw_pos))
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """End drag on left-button release."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self._drag_pos = None
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    # ------------------------------------------------------------------
    # Shell methods
    # ------------------------------------------------------------------

    def show_panel(self) -> None:
        """Show the panel with a fade-in animation.

        Cancels any pending delayed hide or in-progress fade-out so that
        recording restarts are seamless.
        """
        self.cancel_delayed_hide()
        self._start_fade_in()

    def hide_panel(self, immediate: bool = False) -> None:
        """Hide the panel, optionally immediately without fade.

        Args:
            immediate: If True, hide instantly. If False, fade out.
        """
        self.save_geometry()
        if immediate:
            self.hide()
            self.setWindowOpacity(1.0)
        else:
            self._start_fade_out()

    def toggle_panel(self) -> None:
        """Toggle panel visibility."""
        if self.isVisible():
            self.hide_panel()
        else:
            self.show_panel()

    def start_delayed_hide(self) -> None:
        """Schedule a delayed fade-out after CC_FADE_DELAY_MS.

        The overlay stays visible with its final text during the delay
        period, then fades out over 150 ms.  Calling this while a delay
        or fade is already active restarts the delay cleanly.

        Logs a concise lifecycle event without transcript bodies.
        """
        self.cancel_delayed_hide()
        self._fade_delay_timer.start(self.CC_FADE_DELAY_MS)
        logger.debug(
            "CC overlay: delayed hide scheduled (%d ms), content=%s",
            self.CC_FADE_DELAY_MS,
            self._has_content,
        )

    def cancel_delayed_hide(self) -> None:
        """Cancel any pending delayed hide and stop in-progress fade-out.

        Safe to call when no timer is active — no-op in that case.
        """
        if self._fade_delay_timer.isActive():
            self._fade_delay_timer.stop()
            logger.debug("CC overlay: delayed hide cancelled")
        if hasattr(self, "_fade_timer") and self._fade_timer.isActive():
            self._fade_timer.stop()
            # Restore full opacity if we interrupted a fade-out mid-way
            if self._fade_direction == -1:
                self.setWindowOpacity(1.0)
            logger.debug("CC overlay: in-progress fade cancelled, opacity restored")

    def _on_delay_elapsed(self) -> None:
        """Callback when the fade-delay timer fires — starts the fade-out."""
        logger.debug(
            "CC overlay: delay elapsed, starting fade-out, content=%s",
            self._has_content,
        )
        self._start_fade_out()

    def clear(self) -> None:
        """Reset overlay text, phrases, and content state."""
        self.text_edit.clear()
        self.phrases.clear()
        self.current_phrase_idx = -1
        self._has_content = False

    # ------------------------------------------------------------------
    # Geometry persistence
    # ------------------------------------------------------------------

    def _restore_geometry(self) -> None:
        """Restore CC panel position and size from config."""
        try:
            from meetandread.config import get_config
            geom = get_config("ui.cc_panel_geometry")
            if geom is not None and len(geom) == 4:
                x, y, w, h = geom
                self.resize(w, h)
                self.move(x, y)
                ensure_on_screen(self)
                logger.debug("Restored CC panel geometry: (%d, %d, %d, %d)", x, y, w, h)
        except Exception as e:
            logger.warning("Failed to restore CC panel geometry: %s", e)

    def save_geometry(self) -> None:
        """Save CC panel position and size to config."""
        try:
            from meetandread.config import set_config, save_config
            if self.isVisible():
                set_config("ui.cc_panel_geometry", (self.x(), self.y(), self.width(), self.height()))
                save_config()
                logger.debug("Saved CC panel geometry: (%d, %d, %d, %d)",
                             self.x(), self.y(), self.width(), self.height())
        except Exception as e:
            logger.warning("Failed to save CC panel geometry: %s", e)

    # ------------------------------------------------------------------
    # Live transcript rendering — TV closed-caption style
    # ------------------------------------------------------------------

    # Maximum phrases kept in memory.  Old phrases scroll off on each
    # new phrase start, keeping the display small and updates fast.
    # ~8 phrases ≈ 30 seconds of speech at ~4 s per phrase.
    _MAX_VISIBLE_PHRASES = 8

    def update_segment(self, text: str, confidence: int, segment_index: int,
                       is_final: bool = False, phrase_start: bool = False,
                       speaker_id: Optional[str] = None) -> None:
        """Update a single transcript segment in the CC overlay.

        Behaviour is modelled on TV closed captions:
        - New phrases appear as new lines
        - Old phrases scroll off as new ones arrive
        - Only the most recent phrases are kept visible
        - The display always shows the latest text

        The processor re-transcribes the accumulated buffer every ~2 s and
        re-emits ALL segments (not just new ones).  This means segment_index
        0, 1, 2, ... arrive in order for every transcription pass.  When a
        re-transcription returns fewer segments than before, stale tail
        segments are trimmed so the display always reflects the latest state.

        Args:
            text: Transcribed text for this segment.
            confidence: Confidence score (0–100).
            segment_index: Position of this segment in the current phrase.
            is_final: If True, this phrase is complete.
            phrase_start: If True, start a new phrase (new line).
            speaker_id: Optional speaker label for this phrase.
        """
        if text.strip() == "[BLANK_AUDIO]":
            return

        if not self._has_content:
            self._has_content = True

        # Start new phrase if needed
        if phrase_start or self.current_phrase_idx < 0:
            # Prune old phrases to keep the list bounded
            if len(self.phrases) >= self._MAX_VISIBLE_PHRASES:
                self.phrases = self.phrases[-(self._MAX_VISIBLE_PHRASES - 1):]
                self.current_phrase_idx = len(self.phrases) - 1

            self.phrases.append(
                Phrase(segments=[], confidences=[], is_final=False, speaker_id=speaker_id)
            )
            self.current_phrase_idx = len(self.phrases) - 1

        # Update current phrase
        phrase = self.phrases[self.current_phrase_idx]
        phrase.is_final = is_final
        if speaker_id and not phrase.speaker_id:
            phrase.speaker_id = speaker_id

        # Update or add segment
        if segment_index < len(phrase.segments):
            phrase.segments[segment_index] = text
            phrase.confidences[segment_index] = confidence
        else:
            phrase.segments.append(text)
            phrase.confidences.append(confidence)

        # If this is the first segment of a re-transcription batch, trim
        # any stale tail segments from the previous pass.  Whisper may
        # return fewer segments after re-transcribing with more context.
        if segment_index == 0 and not phrase_start:
            # Keep only the segments we're about to update — but we don't
            # know the final count yet.  Trim conservatively: if segment 0
            # arrives again and the phrase had N > 0 segments, trim to 1
            # (just this segment).  Subsequent segments will extend it.
            if len(phrase.segments) > 1:
                phrase.segments = phrase.segments[:1]
                phrase.confidences = phrase.confidences[:1]

        # Re-render the display
        self._render()

    def _render(self) -> None:
        """Rebuild the display from scratch — TV closed-caption style.

        Joins the most recent phrases into plain text and sets it on the
        QTextEdit in one operation.  No cursor manipulation, no incremental
        updates — just a clean rebuild each time.  This is safe because
        the transcription update frequency (~2 s) makes full rebuilds
        trivially cheap for a small number of lines.
        """
        visible = self.phrases[-self._MAX_VISIBLE_PHRASES:]
        lines: List[str] = []
        for phrase in visible:
            text = " ".join(phrase.segments).strip()
            if not text:
                continue
            if phrase.speaker_id:
                name = self._display_speaker_for(phrase.speaker_id)
                text = f"[{name}] {text}"
            lines.append(text)
        self.text_edit.setPlainText("\n".join(lines))
        sb = self.text_edit.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ------------------------------------------------------------------
    # Speaker name management
    # ------------------------------------------------------------------

    def set_speaker_names(self, names: Dict[str, str]) -> None:
        """Set the speaker display name mapping and refresh the display."""
        self._speaker_names = dict(names)
        self._render()

    def get_speaker_names(self) -> Dict[str, str]:
        """Return a copy of the current speaker name mapping."""
        return dict(self._speaker_names)

    def _display_speaker_for(self, raw_label: str) -> str:
        """Resolve a raw speaker label to its display form."""
        if raw_label in self._speaker_names:
            return self._speaker_names[raw_label]
        return raw_label

    def _get_confidence_color(self, confidence: int) -> str:
        """Get text colour — uniform grey for CC-style display (no confidence colouring)."""
        return "#b4b4b4"  # rgb(180, 180, 180)

    # ------------------------------------------------------------------
    # Font size
    # ------------------------------------------------------------------

    def set_font_size(self, size_px: int) -> None:
        """Apply font size to the CC text display immediately.

        Preserves the monospace font-family from the theme CSS when
        updating font-size.

        Args:
            size_px: Font size in pixels (clamped to 16–96).
        """
        size_px = max(16, min(96, size_px))
        current_ss = self.text_edit.styleSheet()
        if current_ss:
            new_ss = re.sub(r'font-size:\s*\d+px', f'font-size: {size_px}px', current_ss)
            self.text_edit.setStyleSheet(new_ss)
        else:
            from meetandread.widgets.theme import AETHERIC_CC_FONT_FAMILY
            self.text_edit.setStyleSheet(
                f"font-family: {AETHERIC_CC_FONT_FAMILY}; font-size: {size_px}px;"
            )
        logger.debug("CC overlay font size set to %dpx", size_px)

    # ------------------------------------------------------------------
    # Fade helpers (matching FloatingTranscriptPanel pattern)
    # ------------------------------------------------------------------

    def _start_fade_in(self) -> None:
        """Animate opacity 0 → 1, then show."""
        if hasattr(self, "_fade_timer") and self._fade_timer.isActive():
            self._fade_timer.stop()
        self._apply_theme()
        self.setWindowOpacity(0.0)
        self.show()
        self.raise_()
        self.activateWindow()
        self._fade_step = 0
        self._fade_direction = 1
        self._fade_timer = QTimer(self)
        self._fade_timer.timeout.connect(self._on_fade_tick)
        self._fade_timer.start(self._FADE_STEP_MS)

    def _start_fade_out(self) -> None:
        """Animate opacity 1 → 0, then hide."""
        if hasattr(self, "_fade_timer") and self._fade_timer.isActive():
            self._fade_timer.stop()
        self.setWindowOpacity(1.0)
        self._fade_step = 0
        self._fade_direction = -1
        self._fade_timer = QTimer(self)
        self._fade_timer.timeout.connect(self._on_fade_tick)
        self._fade_timer.start(self._FADE_STEP_MS)

    def _on_fade_tick(self) -> None:
        """Process one step of a fade animation."""
        self._fade_step += 1
        progress = self._fade_step / self._FADE_STEPS
        if self._fade_direction == 1:
            self.setWindowOpacity(min(progress, 1.0))
        else:
            self.setWindowOpacity(max(1.0 - progress, 0.0))
        if self._fade_step >= self._FADE_STEPS:
            self._fade_timer.stop()
            if self._fade_direction == -1:
                self.hide()
                self.setWindowOpacity(1.0)
                logger.debug(
                    "CC overlay: fade-out complete, hidden, content=%s",
                    self._has_content,
                )


class FloatingSettingsPanel(QWidget):
    """Frameless Aetheric Glass settings shell with sidebar navigation.

    Hosts Settings, Performance, and History sections in a left sidebar +
    right content stack layout. No internal title bar or close button —
    the shell is closed via the widget's settings affordance or hide_panel().
    """

    closed = pyqtSignal()
    model_changed = pyqtSignal(str)  # Emit model name when changed
    cc_font_size_changed = pyqtSignal(int)  # Emit font size in px when changed

    # Nav page indices — correspond to QStackedWidget indices
    _NAV_SETTINGS = 0
    _NAV_PERFORMANCE = 1
    _NAV_HISTORY = 2
    _NAV_IDENTITIES = 3

    def __init__(self, parent: Optional[QWidget] = None,
                 controller: object = None, tray_manager: object = None,
                 main_widget: object = None):
        super().__init__(parent)
        self.setObjectName("AethericSettingsShell")
        
        # Store optional references
        self._controller = controller
        self._tray_manager = tray_manager
        self._main_widget = main_widget

        # -- Performance backend instances (wired in T03) --
        self._resource_monitor = ResourceMonitor(
            poll_interval_ms=2000,
            cpu_warning_percent=80.0,
            ram_warning_percent=85.0,
            on_snapshot=self._on_resource_snapshot,
            on_warning=self._on_resource_warning,
        )

        # -- Metrics refresh timer (updates recording metrics every 2s) --
        self._metrics_timer = QTimer(self)
        self._metrics_timer.setInterval(2000)
        self._metrics_timer.timeout.connect(self._refresh_recording_metrics)

        # -- Benchmark state --
        self._benchmark_runner: Optional[BenchmarkRunner] = None
        self._benchmark_history: List[dict] = []  # last 5 results as dicts

        # -- Track whether Performance page is active --
        self._perf_tab_active = False

        # Window settings
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )

        # Translucent background so rounded corner pixels are transparent.
        # Inner widgets (title bar, sidebar, content stack) provide solid bg.
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # Size — wider to accommodate sidebar + content stack
        self.setMinimumSize(545, 425)
        self.resize(900, 600)

        # ------------------------------------------------------------------
        # Root layout: vertical (title bar on top, then sidebar + content)
        # ------------------------------------------------------------------
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        # ------------------------------------------------------------------
        # Title bar — 25px drag handle with "Meet and Read" label
        # ------------------------------------------------------------------
        self._title_bar = QWidget()
        self._title_bar.setObjectName("AethericTitleBar")
        self._title_bar.setFixedHeight(25)
        self._title_bar.setCursor(Qt.CursorShape.OpenHandCursor)
        title_bar_layout = QHBoxLayout(self._title_bar)
        title_bar_layout.setContentsMargins(12, 0, 12, 0)
        title_label = QLabel("Meet and Read")
        title_label.setObjectName("AethericTitleLabel")
        title_bar_layout.addWidget(title_label)
        title_bar_layout.addStretch()

        # Close button in title bar
        self._title_close_btn = QPushButton("×")
        self._title_close_btn.setObjectName("AethericTitleCloseButton")
        self._title_close_btn.setFixedSize(20, 20)
        self._title_close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._title_close_btn.setToolTip("Close settings")
        self._title_close_btn.setStyleSheet("""
            QPushButton {
                color: rgba(255, 255, 255, 120);
                background: transparent;
                border: none;
                border-radius: 4px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                color: #ff5545;
                background: rgba(255, 85, 69, 40);
            }
        """)
        self._title_close_btn.clicked.connect(self.hide_panel)
        title_bar_layout.addWidget(self._title_close_btn)

        outer_layout.addWidget(self._title_bar)

        # -- Drag state for title bar --
        self._title_dragging: bool = False
        self._title_drag_pos: Optional[QPoint] = None
        self._title_bar.mousePressEvent = self._title_bar_mouse_press
        self._title_bar.mouseMoveEvent = self._title_bar_mouse_move
        self._title_bar.mouseReleaseEvent = self._title_bar_mouse_release

        # ------------------------------------------------------------------
        # Body layout: horizontal sidebar + content stack
        # ------------------------------------------------------------------
        body_layout = QHBoxLayout()
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        # ------------------------------------------------------------------
        # Left sidebar
        # ------------------------------------------------------------------
        self._sidebar = QWidget()
        self._sidebar.setObjectName("AethericSidebar")
        self._sidebar.setFixedWidth(160)
        sidebar_layout = QVBoxLayout(self._sidebar)
        sidebar_layout.setContentsMargins(12, 16, 12, 12)
        sidebar_layout.setSpacing(6)

        # Navigation buttons
        self._nav_buttons: List[QPushButton] = []

        # Settings nav
        self._nav_settings_btn = QPushButton("⚙  Settings")
        self._nav_settings_btn.setObjectName("AethericNavButton")
        self._nav_settings_btn.setCheckable(True)
        self._nav_settings_btn.setChecked(True)
        self._nav_settings_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._nav_settings_btn.setProperty("nav_id", "settings")
        self._nav_settings_btn.clicked.connect(lambda: self._on_nav_clicked(self._NAV_SETTINGS))
        sidebar_layout.addWidget(self._nav_settings_btn)
        self._nav_buttons.append(self._nav_settings_btn)

        # Performance nav
        self._nav_performance_btn = QPushButton("📊  Performance")
        self._nav_performance_btn.setObjectName("AethericNavButton")
        self._nav_performance_btn.setCheckable(True)
        self._nav_performance_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._nav_performance_btn.setProperty("nav_id", "performance")
        self._nav_performance_btn.clicked.connect(lambda: self._on_nav_clicked(self._NAV_PERFORMANCE))
        sidebar_layout.addWidget(self._nav_performance_btn)
        self._nav_buttons.append(self._nav_performance_btn)

        # History nav (placeholder — S02 builds the real list)
        self._nav_history_btn = QPushButton("🕐  History")
        self._nav_history_btn.setObjectName("AethericNavButton")
        self._nav_history_btn.setCheckable(True)
        self._nav_history_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._nav_history_btn.setProperty("nav_id", "history")
        self._nav_history_btn.clicked.connect(lambda: self._on_nav_clicked(self._NAV_HISTORY))
        sidebar_layout.addWidget(self._nav_history_btn)
        self._nav_buttons.append(self._nav_history_btn)

        # Identities nav — T02 wires the real tab
        self._nav_identities_btn = QPushButton("👤  Identities")
        self._nav_identities_btn.setObjectName("AethericNavButton")
        self._nav_identities_btn.setCheckable(True)
        self._nav_identities_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._nav_identities_btn.setToolTip("Manage stored voice identities")
        self._nav_identities_btn.setAccessibleName("Identities tab")
        self._nav_identities_btn.setProperty("nav_id", "identities")
        self._nav_identities_btn.clicked.connect(lambda: self._on_nav_clicked(self._NAV_IDENTITIES))
        sidebar_layout.addWidget(self._nav_identities_btn)
        self._nav_buttons.append(self._nav_identities_btn)

        sidebar_layout.addStretch()

        body_layout.addWidget(self._sidebar)

        # ------------------------------------------------------------------
        # Right content stack (QStackedWidget replacing QTabWidget)
        # ------------------------------------------------------------------
        self._content_stack = QStackedWidget()
        self._content_stack.setObjectName("AethericContentStack")
        body_layout.addWidget(self._content_stack, 1)
        outer_layout.addLayout(body_layout)

        # ------------------------------------------------------------------
        # Page 0: Settings — model selection + hardware info
        # ------------------------------------------------------------------
        settings_page = QWidget()
        settings_layout = QVBoxLayout(settings_page)
        settings_layout.setContentsMargins(12, 16, 12, 12)
        settings_layout.setSpacing(5)

        # Model selection — Live Model dropdown
        self._live_model_label = QLabel("Live Model (real-time display):")
        settings_layout.addWidget(self._live_model_label)

        self._live_model_combo = QComboBox()
        self._live_model_combo.setObjectName("AethericComboBox")
        self._populate_model_dropdown(self._live_model_combo, "realtime_model_size")
        self._live_model_combo.currentIndexChanged.connect(self._on_live_model_changed)
        settings_layout.addWidget(self._live_model_combo)

        # Model selection — Post Process Model dropdown
        self._postprocess_model_label = QLabel("Post Process Model (archive quality):")
        settings_layout.addWidget(self._postprocess_model_label)

        self._postprocess_model_combo = QComboBox()
        self._postprocess_model_combo.setObjectName("AethericComboBox")
        self._populate_model_dropdown(self._postprocess_model_combo, "postprocess_model_size")
        self._postprocess_model_combo.currentIndexChanged.connect(self._on_postprocess_model_changed)
        settings_layout.addWidget(self._postprocess_model_combo)

        # Hardware detection section
        self._hardware_label = QLabel("Hardware:")
        settings_layout.addWidget(self._hardware_label)

        self.hardware_detector = HardwareDetector()
        self.model_recommender = ModelRecommender()

        ram_value = self.hardware_detector.get_ram_gb()
        cpu_cores = self.hardware_detector.get_cpu_cores()
        cpu_freq = self.hardware_detector.get_cpu_frequency()
        recommended = self.model_recommender.get_recommendation()

        self._ram_label = QLabel(f"RAM: {ram_value:.1f} GB")
        self._cpu_info_label = QLabel(f"CPU: {cpu_cores} cores @ {cpu_freq:.1f} GHz")
        self._rec_label = QLabel(f"Recommended: {recommended}")

        settings_layout.addWidget(self._ram_label)
        settings_layout.addWidget(self._cpu_info_label)
        settings_layout.addWidget(self._rec_label)

        # Separator
        denois_sep = QFrame()
        denois_sep.setFrameShape(QFrame.Shape.HLine)
        denois_sep.setObjectName("AethericSeparator")
        settings_layout.addWidget(denois_sep)

        # Noise filter toggle
        self._noise_filter_checkbox = QCheckBox("Background Noise Filter")
        self._noise_filter_checkbox.setObjectName("AethericCheckBox")
        self._noise_filter_checkbox.setCursor(Qt.CursorShape.PointingHandCursor)
        self._noise_filter_checkbox.setStyleSheet(aetheric_checkbox_css(current_palette()))
        self._noise_filter_checkbox.setToolTip(
            "Enable spectral gate noise reduction on microphone input.\n"
            "May cause audio artifacts on some hardware."
        )
        # Restore from config
        try:
            from meetandread.config import get_config
            _denoise_enabled = get_config("transcription.microphone_denoising_enabled")
            self._noise_filter_checkbox.setChecked(bool(_denoise_enabled))
        except Exception:
            self._noise_filter_checkbox.setChecked(False)
        self._noise_filter_checkbox.stateChanged.connect(self._on_noise_filter_toggled)
        settings_layout.addWidget(self._noise_filter_checkbox)

        self._noise_filter_note = QLabel(
            "⚠ Experimental — may cause clicking artifacts"
        )
        self._noise_filter_note.setStyleSheet("color: #FF9800; font-size: 10px;")
        self._noise_filter_note.setWordWrap(True)
        settings_layout.addWidget(self._noise_filter_note)

        # Separator before CC settings
        cc_sep = QFrame()
        cc_sep.setFrameShape(QFrame.Shape.HLine)
        cc_sep.setObjectName("AethericSeparator")
        settings_layout.addWidget(cc_sep)

        # CC overlay font size
        cc_font_row = QHBoxLayout()
        cc_font_row.setSpacing(8)
        cc_font_label = QLabel("CC Font Size:")
        cc_font_label.setStyleSheet("color: #E0E0E0; font-size: 12px;")
        cc_font_row.addWidget(cc_font_label)

        from PyQt6.QtWidgets import QSpinBox
        self._cc_font_spin = QSpinBox()
        self._cc_font_spin.setObjectName("AethericSpinBox")
        self._cc_font_spin.setRange(16, 96)
        self._cc_font_spin.setSuffix(" px")
        self._cc_font_spin.setSingleStep(4)
        self._cc_font_spin.setStyleSheet(f"""
            QSpinBox {{
                color: #E0E0E0;
                background-color: #1e1d1e;
                border: 1px solid rgba(255, 255, 255, 30);
                border-radius: 8px;
                padding: 4px 28px 4px 8px;
                font-size: 12px;
                min-width: 80px;
                min-height: 24px;
            }}
            QSpinBox:hover {{
                border-color: #ff5545;
            }}
            QSpinBox::up-button {{
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 20px;
                border: none;
                border-left: 1px solid rgba(255, 255, 255, 30);
                border-top-right-radius: 8px;
            }}
            QSpinBox::up-button:hover {{
                background-color: rgba(255, 255, 255, 20);
            }}
            QSpinBox::up-arrow {{
                image: url({ARROW_UP_SVG});
                width: 12px;
                height: 12px;
            }}
            QSpinBox::down-button {{
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 20px;
                border: none;
                border-left: 1px solid rgba(255, 255, 255, 30);
                border-bottom-right-radius: 8px;
            }}
            QSpinBox::down-button:hover {{
                background-color: rgba(255, 255, 255, 20);
            }}
            QSpinBox::down-arrow {{
                image: url({ARROW_DOWN_SVG});
                width: 12px;
                height: 12px;
            }}
        """)
        self._cc_font_spin.setCursor(Qt.CursorShape.PointingHandCursor)
        # Restore from config
        try:
            from meetandread.config import get_config
            _cc_size = get_config("transcription.cc_font_size")
            if isinstance(_cc_size, (int, float)) and 16 <= _cc_size <= 96:
                self._cc_font_spin.setValue(int(_cc_size))
            else:
                self._cc_font_spin.setValue(48)
        except Exception:
            self._cc_font_spin.setValue(48)
        self._cc_font_spin.valueChanged.connect(self._on_cc_font_size_changed)
        cc_font_row.addWidget(self._cc_font_spin)
        cc_font_row.addStretch()
        settings_layout.addLayout(cc_font_row)

        # CC auto-open checkbox
        self._cc_auto_open_checkbox = QCheckBox("Auto-open CC overlay on recording")
        self._cc_auto_open_checkbox.setObjectName("AethericCheckBox")
        self._cc_auto_open_checkbox.setCursor(Qt.CursorShape.PointingHandCursor)
        self._cc_auto_open_checkbox.setStyleSheet(aetheric_checkbox_css(current_palette()))
        self._cc_auto_open_checkbox.setToolTip(
            "When enabled, the CC overlay panel opens automatically when recording starts."
        )
        # Restore from config
        try:
            from meetandread.config import get_config
            _auto_open = get_config("transcription.cc_auto_open")
            self._cc_auto_open_checkbox.setChecked(_auto_open if isinstance(_auto_open, bool) else True)
        except Exception:
            self._cc_auto_open_checkbox.setChecked(True)
        self._cc_auto_open_checkbox.stateChanged.connect(self._on_cc_auto_open_toggled)
        settings_layout.addWidget(self._cc_auto_open_checkbox)

        # Waveform visualization checkbox
        self._waveform_checkbox = QCheckBox("Show waveform visualization during recording")
        self._waveform_checkbox.setObjectName("AethericCheckBox")
        self._waveform_checkbox.setCursor(Qt.CursorShape.PointingHandCursor)
        self._waveform_checkbox.setStyleSheet(aetheric_checkbox_css(current_palette()))
        self._waveform_checkbox.setToolTip(
            "When disabled, the recording button shows a minimal pulse animation "
            "instead of the real-time waveform."
        )
        # Restore from config
        try:
            from meetandread.config import get_config
            _waveform_on = get_config("ui.waveform_enabled")
            self._waveform_checkbox.setChecked(
                _waveform_on if isinstance(_waveform_on, bool) else True
            )
        except Exception:
            self._waveform_checkbox.setChecked(True)
        self._waveform_checkbox.stateChanged.connect(self._on_waveform_toggled)
        settings_layout.addWidget(self._waveform_checkbox)

        settings_layout.addStretch()
        self._content_stack.addWidget(settings_page)

        # ------------------------------------------------------------------
        # Page 1: Performance — live resource monitoring + benchmarks
        # ------------------------------------------------------------------
        perf_page = QWidget()
        perf_layout = QVBoxLayout(perf_page)
        perf_layout.setContentsMargins(12, 16, 12, 12)
        perf_layout.setSpacing(6)

        # --- Resource Usage Section ---
        self._resource_header = QLabel("Resource Usage")
        perf_layout.addWidget(self._resource_header)

        # RAM bar
        ram_row = QHBoxLayout()
        ram_row.setSpacing(6)
        self._ram_lbl = QLabel("RAM:")
        self._ram_lbl.setFixedWidth(36)
        ram_row.addWidget(self._ram_lbl)
        self._ram_bar = BudgetProgressBar(budget_percent=85.0)
        self._ram_bar.setRange(0, 100)
        self._ram_bar.setValue(0)
        self._ram_bar.setFormat("%v%")
        ram_row.addWidget(self._ram_bar)
        perf_layout.addLayout(ram_row)

        # CPU bar
        cpu_row = QHBoxLayout()
        cpu_row.setSpacing(6)
        self._cpu_lbl = QLabel("CPU:")
        self._cpu_lbl.setFixedWidth(36)
        cpu_row.addWidget(self._cpu_lbl)
        self._cpu_bar = BudgetProgressBar(budget_percent=80.0)
        self._cpu_bar.setRange(0, 100)
        self._cpu_bar.setValue(0)
        self._cpu_bar.setFormat("%v%")
        cpu_row.addWidget(self._cpu_bar)
        perf_layout.addLayout(cpu_row)

        # Resource warning indicator (hidden by default)
        self._resource_warning = QLabel("⚠ Low Resource Warning")
        self._resource_warning.hide()
        perf_layout.addWidget(self._resource_warning)

        # Separator
        self._perf_sep = QFrame()
        self._perf_sep.setFrameShape(QFrame.Shape.HLine)
        perf_layout.addWidget(self._perf_sep)

        # --- Recording Metrics Section ---
        self._rec_metrics_header = QLabel("Recording Metrics")
        perf_layout.addWidget(self._rec_metrics_header)

        self._metric_model = QLabel("Model: Not recording")
        perf_layout.addWidget(self._metric_model)

        self._metric_buffer = QLabel("Buffer: Not recording")
        perf_layout.addWidget(self._metric_buffer)

        self._metric_count = QLabel("Transcriptions: Not recording")
        perf_layout.addWidget(self._metric_count)

        self._metric_throughput = QLabel("Throughput: Not recording")
        perf_layout.addWidget(self._metric_throughput)

        # Separator
        self._perf_sep2 = QFrame()
        self._perf_sep2.setFrameShape(QFrame.Shape.HLine)
        perf_layout.addWidget(self._perf_sep2)

        # --- WER Display ---
        self._wer_label = QLabel("Last recording WER: —")
        perf_layout.addWidget(self._wer_label)

        # --- Model Selector for Benchmark ---
        model_row = QHBoxLayout()
        model_row.setSpacing(6)
        self._bench_model_lbl = QLabel("Benchmark Model:")
        model_row.addWidget(self._bench_model_lbl)

        self._benchmark_model_combo = QComboBox()
        self._benchmark_model_combo.setObjectName("AethericComboBox")

        # Populate with all 5 models, default to current live model
        try:
            from meetandread.config import get_config
            _cfg = get_config()
            _default_bench_model = _cfg.transcription.realtime_model_size
            _bench_history = _cfg.transcription.benchmark_history
        except Exception:
            _default_bench_model = "tiny"
            _bench_history = {}

        _model_order = ["tiny", "base", "small", "medium", "large"]
        _select_idx = 0
        for _i, _mn in enumerate(_model_order):
            _entry = _bench_history.get(_mn)
            if _entry and "wer" in _entry:
                _wer_pct = _entry["wer"] * 100
                _item_text = f"{_mn} — WER: {_wer_pct:.1f}%"
            else:
                _item_text = f"{_mn} (not benchmarked)"
            self._benchmark_model_combo.addItem(_item_text, _mn)
            if _mn == _default_bench_model:
                _select_idx = _i
        self._benchmark_model_combo.setCurrentIndex(_select_idx)
        model_row.addWidget(self._benchmark_model_combo, 1)
        perf_layout.addLayout(model_row)

        # --- Benchmark Button ---
        self._benchmark_btn = QPushButton("Run Benchmark")
        self._benchmark_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        perf_layout.addWidget(self._benchmark_btn)

        # --- Benchmark History ---
        self._history_header = QLabel("Benchmark History")
        perf_layout.addWidget(self._history_header)

        self._benchmark_history_label = QLabel("No benchmarks yet")
        self._benchmark_history_label.setWordWrap(True)
        perf_layout.addWidget(self._benchmark_history_label)

        perf_layout.addStretch()
        self._content_stack.addWidget(perf_page)

        # ------------------------------------------------------------------
        # Page 2: History — recording list, transcript viewer, scrub/delete
        # ------------------------------------------------------------------
        history_page = QWidget()
        history_page.setObjectName("AethericHistoryPage")
        history_layout = QVBoxLayout(history_page)
        history_layout.setContentsMargins(6, 8, 6, 6)
        history_layout.setSpacing(0)

        self._history_splitter = QSplitter(Qt.Orientation.Vertical)
        self._history_splitter.setObjectName("AethericHistorySplitter")

        # Top: recording list
        self._history_list = QListWidget()
        self._history_list.setObjectName("AethericHistoryList")
        self._history_list.itemClicked.connect(self._on_history_item_clicked)
        self._history_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._history_list.customContextMenuRequested.connect(self._on_history_context_menu)
        self._history_splitter.addWidget(self._history_list)

        # Bottom: detail header bar + transcript viewer
        viewer_container = QWidget()
        viewer_layout = QVBoxLayout(viewer_container)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        viewer_layout.setSpacing(0)

        # Detail header bar with playback controls + Scrub/Delete buttons (hidden until selection)
        self._history_detail_header = QFrame()
        self._history_detail_header.setObjectName("AethericHistoryHeader")
        detail_header_layout = QHBoxLayout(self._history_detail_header)
        detail_header_layout.setContentsMargins(6, 2, 6, 2)
        detail_header_layout.setSpacing(4)

        # -- Playback controls (left side, before stretch) --
        self._playback_play_btn = QPushButton("▶")
        self._playback_play_btn.setObjectName("AethericHistoryPlaybackButton")
        self._playback_play_btn.setProperty("playback_action", "play_pause")
        self._playback_play_btn.setAccessibleName("Play or pause audio")
        self._playback_play_btn.setAccessibleDescription("Toggle audio playback for the selected transcript recording")
        self._playback_play_btn.setFixedHeight(26)
        self._playback_play_btn.setFixedWidth(32)
        self._playback_play_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._playback_play_btn.setToolTip("Play / Pause audio")
        self._playback_play_btn.setEnabled(False)
        self._playback_play_btn.clicked.connect(self._on_playback_play_clicked)
        detail_header_layout.addWidget(self._playback_play_btn)

        self._playback_speed_combo = QComboBox()
        self._playback_speed_combo.setObjectName("AethericHistoryPlaybackSpeedCombo")
        self._playback_speed_combo.setAccessibleName("Playback speed")
        self._playback_speed_combo.setAccessibleDescription("Select audio playback speed from 0.25x to 2x")
        self._playback_speed_combo.setFixedHeight(26)
        self._playback_speed_combo.setFixedWidth(68)
        self._playback_speed_combo.setToolTip("Playback speed")
        self._playback_speed_combo.setEnabled(False)
        for rate_label in ["0.25x", "0.5x", "0.75x", "1x", "1.25x", "1.5x", "2x"]:
            self._playback_speed_combo.addItem(rate_label)
        # Default to "1x" (index 3)
        self._playback_speed_combo.setCurrentIndex(3)
        self._playback_speed_combo.currentIndexChanged.connect(self._on_playback_speed_changed)
        detail_header_layout.addWidget(self._playback_speed_combo)

        self._playback_volume_label = QLabel("🔊")
        self._playback_volume_label.setObjectName("AethericHistoryPlaybackVolumeIcon")
        self._playback_volume_label.setAccessibleName("Volume icon")
        self._playback_volume_label.setFixedWidth(18)
        self._playback_volume_label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)
        detail_header_layout.addWidget(self._playback_volume_label)

        self._playback_volume_slider = QSlider(Qt.Orientation.Horizontal)
        self._playback_volume_slider.setObjectName("AethericHistoryPlaybackVolumeSlider")
        self._playback_volume_slider.setAccessibleName("Volume control")
        self._playback_volume_slider.setAccessibleDescription("Adjust audio volume from 0 to 100 percent")
        self._playback_volume_slider.setFixedHeight(22)
        self._playback_volume_slider.setFixedWidth(70)
        self._playback_volume_slider.setRange(0, 100)
        self._playback_volume_slider.setValue(80)
        self._playback_volume_slider.setToolTip("Volume")
        self._playback_volume_slider.setEnabled(False)
        self._playback_volume_slider.valueChanged.connect(self._on_playback_volume_changed)
        detail_header_layout.addWidget(self._playback_volume_slider)

        self._playback_status_label = QLabel("")
        self._playback_status_label.setObjectName("AethericHistoryPlaybackStatusLabel")
        self._playback_status_label.setAccessibleName("Audio playback status")
        self._playback_status_label.setAccessibleDescription("Current status of audio playback or error message")
        self._playback_status_label.setFixedHeight(20)
        self._playback_status_label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        detail_header_layout.addWidget(self._playback_status_label)

        # Progress slider for transport position
        self._playback_progress_slider = QSlider(Qt.Orientation.Horizontal)
        self._playback_progress_slider.setObjectName("AethericHistoryPlaybackProgressSlider")
        self._playback_progress_slider.setAccessibleName("Playback position")
        self._playback_progress_slider.setAccessibleDescription("Seek within the audio recording by dragging or clicking the progress bar")
        self._playback_progress_slider.setFixedHeight(22)
        self._playback_progress_slider.setMinimumWidth(100)
        self._playback_progress_slider.setMaximumWidth(200)
        self._playback_progress_slider.setRange(0, 1000)
        self._playback_progress_slider.setValue(0)
        self._playback_progress_slider.setToolTip("Playback position")
        self._playback_progress_slider.setEnabled(False)
        self._playback_progress_slider.sliderPressed.connect(self._on_progress_slider_pressed)
        self._playback_progress_slider.sliderReleased.connect(self._on_progress_slider_released)
        self._playback_progress_slider.valueChanged.connect(self._on_progress_slider_value_changed)
        detail_header_layout.addWidget(self._playback_progress_slider)

        # Skip backward button (⏪ -5s)
        self._playback_skip_back_btn = QPushButton("⏪")
        self._playback_skip_back_btn.setObjectName("AethericHistoryPlaybackSkipBackButton")
        self._playback_skip_back_btn.setProperty("playback_action", "skip_back")
        self._playback_skip_back_btn.setAccessibleName("Skip backward 5 seconds")
        self._playback_skip_back_btn.setAccessibleDescription("Skip backward by 5 seconds in the audio recording")
        self._playback_skip_back_btn.setFixedHeight(26)
        self._playback_skip_back_btn.setFixedWidth(32)
        self._playback_skip_back_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._playback_skip_back_btn.setToolTip("Skip back 5 seconds")
        self._playback_skip_back_btn.setEnabled(False)
        self._playback_skip_back_btn.clicked.connect(self._on_playback_skip_back_clicked)
        detail_header_layout.addWidget(self._playback_skip_back_btn)

        # Skip forward button (⏩ +5s)
        self._playback_skip_fwd_btn = QPushButton("⏩")
        self._playback_skip_fwd_btn.setObjectName("AethericHistoryPlaybackSkipFwdButton")
        self._playback_skip_fwd_btn.setProperty("playback_action", "skip_fwd")
        self._playback_skip_fwd_btn.setAccessibleName("Skip forward 5 seconds")
        self._playback_skip_fwd_btn.setAccessibleDescription("Skip forward by 5 seconds in the audio recording")
        self._playback_skip_fwd_btn.setFixedHeight(26)
        self._playback_skip_fwd_btn.setFixedWidth(32)
        self._playback_skip_fwd_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._playback_skip_fwd_btn.setToolTip("Skip forward 5 seconds")
        self._playback_skip_fwd_btn.setEnabled(False)
        self._playback_skip_fwd_btn.clicked.connect(self._on_playback_skip_fwd_clicked)
        detail_header_layout.addWidget(self._playback_skip_fwd_btn)

        # Bookmark button (🔖)
        self._bookmark_add_btn = QPushButton("🔖")
        self._bookmark_add_btn.setObjectName("AethericHistoryBookmarkButton")
        self._bookmark_add_btn.setProperty("playback_action", "bookmark_add")
        self._bookmark_add_btn.setAccessibleName("Add bookmark at current position")
        self._bookmark_add_btn.setAccessibleDescription("Save a playback bookmark at the current audio position")
        self._bookmark_add_btn.setFixedHeight(26)
        self._bookmark_add_btn.setFixedWidth(32)
        self._bookmark_add_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._bookmark_add_btn.setToolTip("Add bookmark (M)")
        self._bookmark_add_btn.setEnabled(False)
        self._bookmark_add_btn.clicked.connect(self._on_bookmark_add_clicked)
        detail_header_layout.addWidget(self._bookmark_add_btn)

        # Bookmark list combo/dropdown
        self._bookmark_combo = QComboBox()
        self._bookmark_combo.setObjectName("AethericHistoryBookmarkCombo")
        self._bookmark_combo.setAccessibleName("Bookmarks")
        self._bookmark_combo.setAccessibleDescription("Navigate to a saved bookmark or manage bookmarks")
        self._bookmark_combo.setFixedHeight(26)
        self._bookmark_combo.setFixedWidth(120)
        self._bookmark_combo.setToolTip("Bookmarks")
        self._bookmark_combo.setEnabled(False)
        self._bookmark_combo.addItem("No bookmarks")
        self._bookmark_combo.currentIndexChanged.connect(self._on_bookmark_combo_changed)
        detail_header_layout.addWidget(self._bookmark_combo)

        # Bookmark delete button — removes selected bookmark by created_at
        self._bookmark_delete_btn = QPushButton("✕")
        self._bookmark_delete_btn.setObjectName("AethericHistoryBookmarkDeleteButton")
        self._bookmark_delete_btn.setProperty("playback_action", "bookmark_delete")
        self._bookmark_delete_btn.setAccessibleName("Delete selected bookmark")
        self._bookmark_delete_btn.setAccessibleDescription("Delete the currently selected bookmark from this transcript")
        self._bookmark_delete_btn.setFixedHeight(26)
        self._bookmark_delete_btn.setFixedWidth(26)
        self._bookmark_delete_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._bookmark_delete_btn.setToolTip("Delete selected bookmark")
        self._bookmark_delete_btn.setEnabled(False)
        self._bookmark_delete_btn.clicked.connect(self._on_bookmark_delete_clicked)
        detail_header_layout.addWidget(self._bookmark_delete_btn)

        # Drag state flag for progress slider
        self._is_dragging_progress_slider = False

        # Bookmark manager state — per-transcript BookmarkManager, created on
        # selection.  _bookmark_items stores (created_at, position_ms) tuples
        # parallel to the combo items for navigation lookup.
        self._bookmark_manager: Optional[object] = None  # BookmarkManager
        self._bookmark_items: List[tuple] = []  # [(created_at, position_ms), ...]

        detail_header_layout.addStretch()

        self._scrub_btn = QPushButton("🔄 Scrub")
        self._scrub_btn.setObjectName("AethericHistoryActionButton")
        self._scrub_btn.setProperty("action", "scrub")
        self._scrub_btn.setFixedHeight(26)
        self._scrub_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._scrub_btn.setToolTip("Re-transcribe with a different model")
        self._scrub_btn.clicked.connect(self._on_scrub_clicked)
        detail_header_layout.addWidget(self._scrub_btn)

        self._delete_btn = QPushButton("🗑 Delete")
        self._delete_btn.setObjectName("AethericHistoryActionButton")
        self._delete_btn.setProperty("action", "delete")
        self._delete_btn.setFixedHeight(26)
        self._delete_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._delete_btn.setToolTip("Delete this recording")
        self._delete_btn.clicked.connect(self._on_delete_btn_clicked)
        detail_header_layout.addWidget(self._delete_btn)

        self._history_detail_header.hide()
        viewer_layout.addWidget(self._history_detail_header)

        # Transcript viewer (read-only, supports anchor clicks)
        self._history_viewer = QTextBrowser()
        self._history_viewer.setObjectName("AethericHistoryViewer")
        self._history_viewer.setReadOnly(True)
        self._history_viewer.setFrameShape(QFrame.Shape.NoFrame)
        self._history_viewer.setPlaceholderText("Select a recording to view its transcript")
        self._history_viewer.setOpenExternalLinks(False)
        self._history_viewer.setOpenLinks(False)
        self._history_viewer.anchorClicked.connect(self._on_history_anchor_clicked)
        viewer_layout.addWidget(self._history_viewer)

        self._history_splitter.addWidget(viewer_container)

        # 40% list / 60% viewer
        self._history_splitter.setSizes([160, 240])

        history_layout.addWidget(self._history_splitter)
        self._content_stack.addWidget(history_page)

        # -- History state attributes --
        self._current_history_md_path: Optional[Path] = None

        # -- Playback helper (lazy-init on first History use) --
        self._playback_helper: Optional[object] = None  # HistoryPlaybackController

        # -- Current-word highlight state --
        # Cached list of (start_ms, end_ms) tuples for timed words in the
        # current transcript.  Populated by _extract_timed_words() and reset
        # when the history selection changes.
        self._cached_timed_words: List[tuple] = []  # [(start_ms, end_ms), ...]
        # Index of the currently highlighted word, or -1 if none.
        self._current_highlight_word_idx: int = -1
        # Timestamp (monotonic ms) of the last highlight re-render.
        self._last_highlight_update_ms: int = 0
        # Highlight throttle interval in milliseconds.
        self._HIGHLIGHT_UPDATE_INTERVAL_MS: int = 200

        # ------------------------------------------------------------------
        # Page 3: Identities — voice identity list, detail viewer
        # ------------------------------------------------------------------
        identities_page = QWidget()
        identities_page.setObjectName("AethericIdentitiesPage")
        identities_layout = QVBoxLayout(identities_page)
        identities_layout.setContentsMargins(6, 8, 6, 6)
        identities_layout.setSpacing(0)

        self._identities_splitter = QSplitter(Qt.Orientation.Vertical)
        self._identities_splitter.setObjectName("AethericIdentitiesSplitter")

        # Top: identity list
        self._identity_list = QListWidget()
        self._identity_list.setObjectName("AethericIdentityList")
        self._identity_list.setAccessibleName("Identity list")
        self._identity_list.setToolTip("Stored voice identities")
        self._identity_list.itemClicked.connect(self._on_identity_item_clicked)
        self._identities_splitter.addWidget(self._identity_list)

        # Bottom: detail header + detail fields
        identity_detail_container = QWidget()
        identity_detail_layout = QVBoxLayout(identity_detail_container)
        identity_detail_layout.setContentsMargins(0, 0, 0, 0)
        identity_detail_layout.setSpacing(0)

        # Detail header bar (with action buttons, disabled until T03)
        self._identity_detail_header = QFrame()
        self._identity_detail_header.setObjectName("AethericIdentityHeader")
        _id_header_layout = QHBoxLayout(self._identity_detail_header)
        _id_header_layout.setContentsMargins(6, 2, 6, 2)
        _id_header_layout.setSpacing(4)

        _id_header_layout.addStretch()

        self._identity_rename_btn = QPushButton("✏  Rename")
        self._identity_rename_btn.setObjectName("AethericIdentityActionButton")
        self._identity_rename_btn.setProperty("action", "rename")
        self._identity_rename_btn.setFixedHeight(26)
        self._identity_rename_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._identity_rename_btn.setToolTip("Rename this identity")
        self._identity_rename_btn.setAccessibleName("Rename identity")
        self._identity_rename_btn.setEnabled(False)
        self._identity_rename_btn.clicked.connect(self._on_identity_rename)
        _id_header_layout.addWidget(self._identity_rename_btn)

        self._identity_merge_btn = QPushButton("🔗  Merge")
        self._identity_merge_btn.setObjectName("AethericIdentityActionButton")
        self._identity_merge_btn.setProperty("action", "merge")
        self._identity_merge_btn.setFixedHeight(26)
        self._identity_merge_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._identity_merge_btn.setToolTip("Merge into another identity")
        self._identity_merge_btn.setAccessibleName("Merge identity")
        self._identity_merge_btn.setEnabled(False)
        self._identity_merge_btn.clicked.connect(self._on_identity_merge)
        _id_header_layout.addWidget(self._identity_merge_btn)

        self._identity_delete_btn = QPushButton("🗑  Delete")
        self._identity_delete_btn.setObjectName("AethericIdentityActionButton")
        self._identity_delete_btn.setProperty("action", "delete")
        self._identity_delete_btn.setFixedHeight(26)
        self._identity_delete_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._identity_delete_btn.setToolTip("Delete this identity")
        self._identity_delete_btn.setAccessibleName("Delete identity")
        self._identity_delete_btn.setEnabled(False)
        self._identity_delete_btn.clicked.connect(self._on_identity_delete)
        _id_header_layout.addWidget(self._identity_delete_btn)

        self._identity_detail_header.hide()
        identity_detail_layout.addWidget(self._identity_detail_header)

        # Detail fields (read-only labels)
        self._identity_detail_widget = QWidget()
        self._identity_detail_widget.setObjectName("AethericIdentityDetailWidget")
        _detail_fields_layout = QVBoxLayout(self._identity_detail_widget)
        _detail_fields_layout.setContentsMargins(8, 8, 8, 8)
        _detail_fields_layout.setSpacing(4)

        self._identity_name_label = QLabel("Name: —")
        self._identity_name_label.setObjectName("AethericIdentityDetailLabel")
        _detail_fields_layout.addWidget(self._identity_name_label)

        self._identity_sample_count_label = QLabel("Samples: —")
        self._identity_sample_count_label.setObjectName("AethericIdentityDetailLabel")
        _detail_fields_layout.addWidget(self._identity_sample_count_label)

        self._identity_recording_count_label = QLabel("Recordings: —")
        self._identity_recording_count_label.setObjectName("AethericIdentityDetailLabel")
        _detail_fields_layout.addWidget(self._identity_recording_count_label)

        self._identity_last_used_label = QLabel("Last used: —")
        self._identity_last_used_label.setObjectName("AethericIdentityDetailLabel")
        _detail_fields_layout.addWidget(self._identity_last_used_label)

        self._identity_recordings_label = QLabel("Associated recordings: —")
        self._identity_recordings_label.setObjectName("AethericIdentityDetailLabel")
        self._identity_recordings_label.setWordWrap(True)
        _detail_fields_layout.addWidget(self._identity_recordings_label)

        _detail_fields_layout.addStretch()

        identity_detail_layout.addWidget(self._identity_detail_widget)

        self._identities_splitter.addWidget(identity_detail_container)

        # 40% list / 60% detail
        self._identities_splitter.setSizes([160, 240])

        identities_layout.addWidget(self._identities_splitter)
        self._content_stack.addWidget(identities_page)

        # -- Identities state attributes --
        self._identity_usage: Dict[str, Any] = {}  # name -> IdentityUsage
        self._identity_profile_names: List[str] = []  # sorted identity names

        # Scrub state
        self._scrub_runner: Optional[object] = None
        self._scrub_model_size: Optional[str] = None
        self._scrub_sidecar_path: Optional[str] = None
        self._scrub_original_html: Optional[str] = None
        self._is_scrubbing: bool = False
        self._is_comparison_mode: bool = False

        # ------------------------------------------------------------------
        # Resize grip — direct child of panel, positioned at bottom-right
        # ------------------------------------------------------------------
        self._resize_grip = TexturedSizeGrip(self)
        self._resize_grip.setFixedSize(16, 16)
        self._resize_grip.setCursor(Qt.CursorShape.SizeFDiagCursor)
        self._resize_grip.show()

        # Wire benchmark button
        self._benchmark_btn.clicked.connect(self._on_benchmark_clicked)

        # Dragging
        self._dragging = False
        self._drag_pos = None

        # Apply initial theme to all widgets
        self._apply_theme()

        # Connect to desktop theme changes for live re-theming
        try:
            from PyQt6.QtGui import QGuiApplication
            hints = QGuiApplication.styleHints()
            if hints is not None:
                hints.colorSchemeChanged.connect(lambda: self._apply_theme())
        except (ImportError, RuntimeError):
            pass

    def resizeEvent(self, event) -> None:
        """Reposition resize grip on resize."""
        if hasattr(self, '_resize_grip'):
            self._resize_grip.move(
                self.width() - self._resize_grip.width(),
                self.height() - self._resize_grip.height(),
            )
        super().resizeEvent(event)

    def paintEvent(self, event) -> None:
        """Draw gradient glow on the sidebar's right inner edge."""
        from PyQt6.QtGui import QPainter, QLinearGradient, QColor
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if hasattr(self, '_sidebar'):
            sx = self._sidebar.x() + self._sidebar.width()
            sy = self._sidebar.y()
            sh = self._sidebar.height()
            gradient = QLinearGradient(sx, sy, sx + 5, sy)
            gradient.setColorAt(0, QColor(255, 85, 69, 50))
            gradient.setColorAt(1, QColor(255, 85, 69, 0))
            painter.fillRect(sx, sy, 5, sh, gradient)

        painter.end()
        super().paintEvent(event)

    # ------------------------------------------------------------------
    # Adaptive theme
    # ------------------------------------------------------------------

    def _apply_theme(self) -> None:
        """Apply Aetheric Glass theme to all settings panel widgets.

        Idempotent and cheap — just re-sets stylesheets from the current
        palette.  Called once at end of __init__ and on desktop theme change.
        """
        p = current_palette()
        self._current_palette = p

        # Panel shell — Aetheric Glass
        self.setStyleSheet(aetheric_settings_shell_css(p))

        # Title bar
        self._title_bar.setStyleSheet(aetheric_title_bar_css(p))

        # Sidebar
        self._sidebar.setStyleSheet(aetheric_sidebar_css(p))

        # Nav buttons
        nav_css = aetheric_nav_button_css(p)
        for btn in self._nav_buttons:
            btn.setStyleSheet(nav_css)

        # Content stack — solid bg (shell is transparent for rounded corners)
        self._content_stack.setStyleSheet(
            f"QStackedWidget {{ background-color: {AETHERIC_SETTINGS_BG}; border: none; border-bottom-right-radius: {AETHERIC_RADIUS}; }}"
        )

        # Settings page — labels and combos
        self._live_model_label.setStyleSheet(status_label_css(p))
        self._live_model_combo.setStyleSheet(aetheric_combo_box_css(p))
        self._postprocess_model_label.setStyleSheet(status_label_css(p))
        self._postprocess_model_combo.setStyleSheet(aetheric_combo_box_css(p))
        self._hardware_label.setStyleSheet(status_label_css(p))
        self._ram_label.setStyleSheet(info_label_css(p))
        self._cpu_info_label.setStyleSheet(info_label_css(p))
        self._rec_label.setStyleSheet(
            f"QLabel {{ font-weight: bold; color: {p.accent}; }}"
        )

        # Performance page — section headers
        self._resource_header.setStyleSheet(
            f"QLabel {{ color: {p.accent}; font-weight: bold; font-size: 12px; padding: 2px; }}"
        )

        # RAM/CPU bar labels
        self._ram_lbl.setStyleSheet(info_label_css(p))
        self._cpu_lbl.setStyleSheet(info_label_css(p))

        # Progress bars — semantic chunk colors stay constant
        self._ram_bar.setStyleSheet(progress_bar_css(p, "#ff5545"))
        self._cpu_bar.setStyleSheet(progress_bar_css(p, "#2196F3"))

        # Resource warning — semantic orange stays but text colour adapts
        self._resource_warning.setStyleSheet(
            f"QLabel {{ color: #FF9800; font-size: 11px; font-weight: bold; padding: 2px; }}"
        )

        # Separators
        self._perf_sep.setStyleSheet(separator_css(p))
        self._perf_sep2.setStyleSheet(separator_css(p))

        # Recording metrics header
        self._rec_metrics_header.setStyleSheet(
            f"QLabel {{ color: {p.accent}; font-weight: bold; font-size: 12px; padding: 2px; }}"
        )

        # Metric labels
        self._metric_model.setStyleSheet(info_label_css(p))
        self._metric_buffer.setStyleSheet(info_label_css(p))
        self._metric_count.setStyleSheet(info_label_css(p))
        self._metric_throughput.setStyleSheet(info_label_css(p))

        # WER label
        self._wer_label.setStyleSheet(
            f"QLabel {{ color: {p.text}; font-size: 12px; font-weight: bold; padding: 2px; }}"
        )

        # Benchmark section
        self._bench_model_lbl.setStyleSheet(info_label_css(p))
        self._benchmark_model_combo.setStyleSheet(aetheric_combo_box_css(p))
        self._benchmark_btn.setStyleSheet(action_button_css(p, "benchmark"))
        self._history_header.setStyleSheet(status_label_css(p))
        self._benchmark_history_label.setStyleSheet(
            f"QLabel {{ color: {p.text_disabled}; font-size: 11px; padding: 2px; }}"
        )

        # History page — Aetheric scoped styles
        history_page = self._content_stack.widget(self._NAV_HISTORY)
        if history_page is not None:
            history_page.setStyleSheet(
                "QWidget#AethericHistoryPage { background-color: transparent; }"
            )
        self._history_splitter.setStyleSheet(aetheric_history_splitter_css(p))
        self._history_list.setStyleSheet(aetheric_history_list_css(p))
        self._history_detail_header.setStyleSheet(aetheric_history_header_css(p))
        history_btn_css = aetheric_history_action_button_css(p)
        self._scrub_btn.setStyleSheet(history_btn_css)
        self._delete_btn.setStyleSheet(history_btn_css)
        self._history_viewer.setStyleSheet(aetheric_history_viewer_css(p))

        # Playback controls styling (scoped Aetheric toolbar styles)
        playback_css = aetheric_playback_toolbar_css(p)
        self._playback_play_btn.setStyleSheet(playback_css["play_button"])
        self._playback_speed_combo.setStyleSheet(playback_css["speed_combo"])
        self._playback_volume_slider.setStyleSheet(playback_css["volume_slider"])
        self._playback_status_label.setStyleSheet(playback_css["status_label"])
        self._playback_volume_label.setStyleSheet(playback_css["volume_icon"])
        self._playback_skip_back_btn.setStyleSheet(playback_css["skip_button"])
        self._playback_skip_fwd_btn.setStyleSheet(playback_css["skip_button"])
        self._playback_progress_slider.setStyleSheet(playback_css["progress_slider"])
        self._bookmark_add_btn.setStyleSheet(playback_css["bookmark_button"])
        self._bookmark_combo.setStyleSheet(playback_css["bookmark_combo"])
        self._bookmark_delete_btn.setStyleSheet(playback_css["bookmark_delete_button"])

        # Identities page — reuse history CSS patterns with Identity object names
        identities_page = self._content_stack.widget(self._NAV_IDENTITIES)
        if identities_page is not None:
            identities_page.setStyleSheet(
                "QWidget#AethericIdentitiesPage { background-color: transparent; }"
            )
        self._identities_splitter.setStyleSheet(aetheric_history_splitter_css(p))
        self._identity_list.setStyleSheet(aetheric_history_list_css(p))
        self._identity_detail_header.setStyleSheet(aetheric_history_header_css(p))
        identity_btn_css = aetheric_history_action_button_css(p)
        self._identity_rename_btn.setStyleSheet(identity_btn_css)
        self._identity_merge_btn.setStyleSheet(identity_btn_css)
        self._identity_delete_btn.setStyleSheet(identity_btn_css)
        # Detail labels — theme-aware info styling
        _id_detail_css = info_label_css(p)
        self._identity_name_label.setStyleSheet(_id_detail_css)
        self._identity_sample_count_label.setStyleSheet(_id_detail_css)
        self._identity_recording_count_label.setStyleSheet(_id_detail_css)
        self._identity_last_used_label.setStyleSheet(_id_detail_css)
        self._identity_recordings_label.setStyleSheet(_id_detail_css)

        # Resize grip — draws its own textured triangle via paintEvent

        scheme_name = "dark" if p is DARK_PALETTE else "light"
        logger.info("Applied %s Aetheric theme to FloatingSettingsPanel", scheme_name)

    # -- Title bar drag handlers --

    def _title_bar_mouse_press(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._title_dragging = True
            self._title_drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            self._title_bar.setCursor(Qt.CursorShape.ClosedHandCursor)

    def _title_bar_mouse_move(self, event: QMouseEvent) -> None:
        if self._title_dragging and self._title_drag_pos is not None:
            raw_pos = event.globalPosition().toPoint() - self._title_drag_pos
            self.move(clamp_to_screen(self, raw_pos))

    def _title_bar_mouse_release(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._title_dragging = False
            self._title_drag_pos = None
            self._title_bar.setCursor(Qt.CursorShape.OpenHandCursor)

    def show_panel(self):
        """Show the panel with a 150ms fade-in and start monitoring if on Performance tab."""
        self._start_fade_in()
        # Activate monitoring if Performance tab is visible
        if self._perf_tab_active:
            self._start_resource_monitor()
            self._metrics_timer.start()
    
    def hide_panel(self):
        """Hide the panel with a 150ms fade-out and stop monitoring."""
        self._stop_resource_monitor()
        self._metrics_timer.stop()
        self._stop_playback()
        self._start_fade_out()

    # ------------------------------------------------------------------
    # Fade transition helpers
    # ------------------------------------------------------------------

    _FADE_DURATION_MS = 150
    _FADE_STEP_MS = 10
    _FADE_STEPS = _FADE_DURATION_MS // _FADE_STEP_MS  # 15

    def _start_fade_in(self) -> None:
        """Animate window opacity from 0 → 1 over 150ms, then show."""
        if hasattr(self, "_fade_timer") and self._fade_timer.isActive():
            self._fade_timer.stop()
        # Re-apply theme on show (picks up any desktop theme change while hidden)
        self._apply_theme()
        self.setWindowOpacity(0.0)
        self.show()
        self.raise_()
        self.activateWindow()
        self._fade_step = 0
        self._fade_direction = 1  # 1 = fading in
        self._fade_timer = QTimer(self)
        self._fade_timer.timeout.connect(self._on_fade_tick)
        self._fade_timer.start(self._FADE_STEP_MS)

    def _start_fade_out(self) -> None:
        """Animate window opacity from 1 → 0 over 150ms, then hide."""
        if hasattr(self, "_fade_timer") and self._fade_timer.isActive():
            self._fade_timer.stop()
        self.setWindowOpacity(1.0)
        self._fade_step = 0
        self._fade_direction = -1  # -1 = fading out
        self._fade_timer = QTimer(self)
        self._fade_timer.timeout.connect(self._on_fade_tick)
        self._fade_timer.start(self._FADE_STEP_MS)

    def _on_fade_tick(self) -> None:
        """Process one step of a fade animation."""
        self._fade_step += 1
        progress = self._fade_step / self._FADE_STEPS
        if self._fade_direction == 1:
            self.setWindowOpacity(min(progress, 1.0))
        else:
            self.setWindowOpacity(max(1.0 - progress, 0.0))
        if self._fade_step >= self._FADE_STEPS:
            self._fade_timer.stop()
            if self._fade_direction == -1:
                self.hide()
                self.setWindowOpacity(1.0)  # Reset for next show
    
    # ------------------------------------------------------------------
    # Performance tab wiring (T03)
    # ------------------------------------------------------------------

    def _on_nav_clicked(self, page_index: int) -> None:
        """Handle sidebar nav clicks — switch page and manage ResourceMonitor.

        Args:
            page_index: QStackedWidget index (_NAV_SETTINGS, _NAV_PERFORMANCE, _NAV_HISTORY).
        """
        if page_index < 0 or page_index >= self._content_stack.count():
            logger.warning("Invalid nav index %d — ignoring", page_index)
            return

        # Update checked state on nav buttons (exclusive toggle)
        for i, btn in enumerate(self._nav_buttons):
            btn.setChecked(i == page_index)

        # Switch content stack
        self._content_stack.setCurrentIndex(page_index)

        # Track performance active state
        self._perf_tab_active = (page_index == self._NAV_PERFORMANCE)

        # Start/stop monitoring based on visibility
        if self._perf_tab_active and self.isVisible():
            self._start_resource_monitor()
            self._metrics_timer.start()
            self._refresh_recording_metrics()
        else:
            self._stop_resource_monitor()
            self._metrics_timer.stop()

        # Refresh History when navigating to it
        if page_index == self._NAV_HISTORY:
            self._refresh_history()
        else:
            # Stop playback when leaving History page
            self._stop_playback()

        # Refresh Identities when navigating to it
        if page_index == self._NAV_IDENTITIES:
            self._refresh_identities()

        nav_id = self._nav_buttons[page_index].property("nav_id") if page_index < len(self._nav_buttons) else "?"
        logger.info("Settings nav changed to '%s' (index %d)", nav_id, page_index)

    def _start_resource_monitor(self) -> None:
        """Start the ResourceMonitor if not already running."""
        if not self._resource_monitor.is_running:
            self._resource_monitor.start()
            logger.info("ResourceMonitor started for Performance tab")

    def _stop_resource_monitor(self) -> None:
        """Stop the ResourceMonitor if running."""
        if self._resource_monitor.is_running:
            self._resource_monitor.stop()
            logger.info("ResourceMonitor stopped")

    def _on_resource_snapshot(self, snapshot: ResourceSnapshot) -> None:
        """Update RAM/CPU bars from a resource snapshot.

        Args:
            snapshot: ResourceSnapshot with current metrics.
        """
        self._ram_bar.setValue(int(snapshot.ram_percent))
        self._cpu_bar.setValue(int(snapshot.cpu_percent))

        # Get current palette for theme-aware progress bar styling
        p = current_palette()

        # Color-code RAM bar: green → orange → red (budget: 85% orange, 90% red)
        if snapshot.ram_percent >= 90:
            self._ram_bar.setStyleSheet(progress_bar_css(p, "#F44336"))
        elif snapshot.ram_percent >= 85:
            self._ram_bar.setStyleSheet(progress_bar_css(p, "#FF9800"))
        else:
            self._ram_bar.setStyleSheet(progress_bar_css(p, "#ff5545"))

        # Color-code CPU bar: blue → orange → red (budget: 80% orange, 90% red)
        if snapshot.cpu_percent >= 90:
            self._cpu_bar.setStyleSheet(progress_bar_css(p, "#F44336"))
        elif snapshot.cpu_percent >= 80:
            self._cpu_bar.setStyleSheet(progress_bar_css(p, "#FF9800"))
        else:
            self._cpu_bar.setStyleSheet(progress_bar_css(p, "#2196F3"))

    def _on_resource_warning(self, resource_name: str, value: float, threshold: float) -> None:
        """Show resource warning indicator and optionally send tray notification.

        Args:
            resource_name: 'ram' or 'cpu'.
            value: Current usage percentage.
            threshold: Warning threshold percentage.
        """
        self._resource_warning.setText(f"⚠ High {resource_name.upper()}: {value:.0f}% (threshold: {threshold:.0f}%)")
        self._resource_warning.show()

        # Auto-hide after 10 seconds if resource recovers
        QTimer.singleShot(10000, self._check_hide_resource_warning)

        # Send tray notification if available
        if self._tray_manager is not None:
            try:
                tray = self._tray_manager.tray_icon
                tray.showMessage(
                    "Resource Warning",
                    f"High {resource_name.upper()} usage: {value:.0f}% (threshold: {threshold:.0f}%)",
                )
            except Exception as exc:
                logger.debug("Failed to send tray notification: %s", exc)

        # Also show warning on the main widget scene
        if self._main_widget is not None:
            try:
                self._main_widget._show_resource_warning(
                    f"⚠ High {resource_name.upper()}: {value:.0f}%"
                )
            except Exception as exc:
                logger.debug("Failed to show main widget resource warning: %s", exc)

    def _check_hide_resource_warning(self) -> None:
        """Hide warning if resources are back to normal."""
        snap = self._resource_monitor.current_snapshot
        if snap is not None:
            if (snap.ram_percent < self._resource_monitor.ram_warning_percent and
                    snap.cpu_percent < self._resource_monitor.cpu_warning_percent):
                self._resource_warning.hide()

    def _refresh_recording_metrics(self) -> None:
        """Update recording metrics labels from the controller's transcription processor.

        Only updates when the controller is recording and has an active processor.
        """
        if self._controller is None:
            return

        try:
            if not self._controller.is_recording():
                self._metric_model.setText("Model: Not recording")
                self._metric_buffer.setText("Buffer: Not recording")
                self._metric_count.setText("Transcriptions: Not recording")
                self._metric_throughput.setText("Throughput: Not recording")
                return

            processor = getattr(self._controller, '_transcription_processor', None)
            if processor is None:
                return

            stats = processor.get_stats()

            # Model info
            model_size = stats.get('model_size', 'unknown')
            self._metric_model.setText(f"Model: {model_size}")

            # Buffer duration
            buffer_dur = stats.get('buffer_duration', 0)
            self._metric_buffer.setText(f"Buffer: {buffer_dur:.1f}s")

            # Transcription count
            count = stats.get('transcription_count', 0)
            self._metric_count.setText(f"Transcriptions: {count}")

            # Throughput (buffer duration / total audio processed)
            total_samples = stats.get('total_samples', 0)
            if total_samples > 0:
                audio_seconds = total_samples / 16000
                self._metric_throughput.setText(f"Throughput: {audio_seconds:.1f}s audio")
            else:
                self._metric_throughput.setText("Throughput: —")

        except Exception as exc:
            logger.debug("Error refreshing recording metrics: %s", exc)

    def update_wer_display(self, wer_value: Optional[float]) -> None:
        """Update the WER display label.

        Args:
            wer_value: WER as a float (0.0–1.0+), or None to reset.
        """
        p = current_palette()
        if wer_value is None:
            self._wer_label.setText("Last recording WER: —")
            self._wer_label.setStyleSheet(
                f"QLabel {{ color: {p.text}; font-size: 12px; font-weight: bold; padding: 2px; }}"
            )
        else:
            pct = wer_value * 100
            if wer_value <= 0.1:
                color = "#ff5545"  # red accent — excellent
            elif wer_value <= 0.3:
                color = "#FFC107"  # yellow — acceptable
            else:
                color = "#F44336"  # red — poor
            self._wer_label.setText(f"Last recording WER: {pct:.1f}%")
            self._wer_label.setStyleSheet(
                f"QLabel {{ color: {color}; font-size: 12px; font-weight: bold; padding: 2px; }}"
            )

    def _on_benchmark_clicked(self) -> None:
        """Handle 'Run Benchmark' button click.

        Creates a BenchmarkRunner using the model selected in the
        benchmark model dropdown, and runs it asynchronously.
        """
        if self._benchmark_runner and self._benchmark_runner.is_running:
            logger.info("Benchmark already running, ignoring click")
            return

        # Disable button and show progress
        self._benchmark_btn.setEnabled(False)
        self._benchmark_btn.setText("Running...")

        # Read model selection from the benchmark model combo.
        # currentData() returns the plain model name (e.g. "base"),
        # but currentText() may include WER annotation — use data.
        model_size = self._benchmark_model_combo.currentData() or "tiny"

        # Create a fresh engine for the selected model.
        engine = None
        try:
            from meetandread.transcription.engine import WhisperTranscriptionEngine
            engine = WhisperTranscriptionEngine(model_size=model_size)
            engine.load_model()
        except Exception as exc:
            logger.warning("Could not create transcription engine for benchmark: %s", exc)

        self._benchmark_runner = BenchmarkRunner(
            engine=engine,
            on_progress=self._on_benchmark_progress,
            on_complete=self._on_benchmark_complete,
        )
        self._benchmark_runner.run_async()

    def _on_benchmark_progress(self, percent: int) -> None:
        """Update benchmark button text with progress.

        Args:
            percent: Progress percentage (0-100).
        """
        self._benchmark_btn.setText(f"Running... {percent}%")

    def _on_benchmark_complete(self, result: BenchmarkResult) -> None:
        """Handle benchmark completion — persist per-model result to config and update UI.

        Args:
            result: BenchmarkResult with WER, latency, and throughput data.
        """
        # Re-enable button
        self._benchmark_btn.setEnabled(True)
        self._benchmark_btn.setText("Run Benchmark")

        if result.error:
            self._benchmark_history_label.setText(f"Benchmark failed: {result.error}")
            p = current_palette()
            self._benchmark_history_label.setStyleSheet(
                f"QLabel {{ color: {p.danger}; font-size: 11px; padding: 2px; }}"
            )
            return

        # Extract model name from result
        model_name = result.model_info.get("model_size", "unknown") if result.model_info else "unknown"

        # Format result
        wer_pct = result.wer * 100
        result_text = (
            f"{model_name}: WER {wer_pct:.1f}% | "
            f"Latency: {result.total_latency_s:.2f}s | "
            f"Speed: {result.throughput_ratio:.1f}x realtime"
        )

        # Store in local history (keep last 5)
        self._benchmark_history.append({
            "wer": result.wer,
            "latency_s": result.total_latency_s,
            "throughput": result.throughput_ratio,
            "model_info": result.model_info,
        })
        if len(self._benchmark_history) > 5:
            self._benchmark_history = self._benchmark_history[-5:]

        # Persist per-model result to config
        try:
            from meetandread.config import get_config, set_config, save_config
            settings = get_config()
            history = dict(settings.transcription.benchmark_history)

            from datetime import datetime
            history[model_name] = {
                "wer": result.wer,
                "timestamp": datetime.now().isoformat(),
            }

            set_config("transcription.benchmark_history", history)
            save_config()
            logger.info("Persisted benchmark result for model '%s' to config", model_name)
        except Exception as exc:
            logger.warning("Failed to persist benchmark result to config: %s", exc)

        # Build per-model history display
        lines = []
        for i, entry in enumerate(reversed(self._benchmark_history), 1):
            w = entry["wer"] * 100
            t = entry["throughput"]
            m = entry.get("model_info", {}).get("model_size", "unknown")
            ts = ""
            # Show timestamp from config if available
            try:
                from meetandread.config import get_config
                _cfg = get_config()
                _hist_entry = _cfg.transcription.benchmark_history.get(m)
                if _hist_entry and "timestamp" in _hist_entry:
                    ts = f" ({_hist_entry['timestamp'][:16]})"
            except Exception:
                pass
            lines.append(f"#{i} {m}: WER {w:.1f}% | Speed {t:.1f}x{ts}")

        self._benchmark_history_label.setText("\n".join(lines))
        p = current_palette()
        self._benchmark_history_label.setStyleSheet(
            f"QLabel {{ color: {p.text_tertiary}; font-size: 11px; padding: 2px; }}"
        )

        # Update the benchmark model dropdown to reflect new WER
        self._refresh_benchmark_model_combo()

        # Refresh Settings dropdowns with updated WER data
        self._refresh_dropdown_wer()

        # Also update the WER display with benchmark result
        self.update_wer_display(result.wer)

        logger.info(
            "Benchmark complete: model=%s, WER=%.3f, throughput=%.1fx, latency=%.2fs",
            model_name, result.wer, result.throughput_ratio, result.total_latency_s,
        )

    # ------------------------------------------------------------------
    # Model dropdown helpers
    # ------------------------------------------------------------------

    def _populate_model_dropdown(self, combo: QComboBox, config_key: str) -> None:
        """Populate a model dropdown with all 5 models and WER annotations.

        Reads benchmark_history from config and MODEL_SPECS for model info.
        Sets the current selection from the config value.

        Args:
            combo: QComboBox to populate.
            config_key: Config key within transcription settings
                ('realtime_model_size' or 'postprocess_model_size').
        """
        combo.blockSignals(True)
        combo.clear()

        try:
            from meetandread.config import get_config
            settings = get_config()
            benchmark_history = settings.transcription.benchmark_history
            current_model = getattr(settings.transcription, config_key, "tiny")
        except Exception:
            benchmark_history = {}
            current_model = "tiny"

        model_order = ["tiny", "base", "small", "medium", "large"]
        select_index = 0

        for i, model_name in enumerate(model_order):
            entry = benchmark_history.get(model_name)
            if entry and "wer" in entry:
                wer_pct = entry["wer"] * 100
                item_text = f"{model_name} — WER: {wer_pct:.1f}%"
            else:
                item_text = f"{model_name} (not benchmarked)"

            combo.addItem(item_text, model_name)

            if model_name == current_model:
                select_index = i

        combo.setCurrentIndex(select_index)
        combo.blockSignals(False)

    def _on_live_model_changed(self, index: int) -> None:
        """Handle Live Model dropdown selection change.

        Updates config and emits model_changed signal.
        """
        model_size = self._live_model_combo.currentData()
        if model_size is None:
            return

        try:
            from meetandread.config import set_config, save_config
            set_config("transcription.realtime_model_size", model_size)
            save_config()
        except Exception as exc:
            logger.warning("Failed to save live model selection: %s", exc)

        self.model_changed.emit(model_size)
        logger.info("Live model changed to: %s", model_size)

    def _on_postprocess_model_changed(self, index: int) -> None:
        """Handle Post Process Model dropdown selection change.

        Updates config (no model_changed signal — that's for live model only).
        """
        model_size = self._postprocess_model_combo.currentData()
        if model_size is None:
            return

        try:
            from meetandread.config import set_config, save_config
            set_config("transcription.postprocess_model_size", model_size)
            save_config()
        except Exception as exc:
            logger.warning("Failed to save post-process model selection: %s", exc)

        logger.info("Post-process model changed to: %s", model_size)

    def _on_noise_filter_toggled(self, state: int) -> None:
        """Handle Background Noise Filter checkbox toggle.

        Persists the setting immediately so the next recording picks it up.
        """
        enabled = bool(state)
        try:
            from meetandread.config import set_config, save_config
            set_config("transcription.microphone_denoising_enabled", enabled)
            save_config()
        except Exception as exc:
            logger.warning("Failed to save noise filter setting: %s", exc)
        logger.info("Background noise filter %s", "enabled" if enabled else "disabled")

    def _on_cc_font_size_changed(self, value: int) -> None:
        """Handle CC font size spinbox change.

        Persists the setting and applies it immediately to the CC overlay.
        """
        try:
            from meetandread.config import set_config, save_config
            set_config("transcription.cc_font_size", value)
            save_config()
        except Exception as exc:
            logger.warning("Failed to save CC font size: %s", exc)
        self.cc_font_size_changed.emit(value)
        logger.info("CC font size set to %dpx", value)

    def _on_cc_auto_open_toggled(self, state: int) -> None:
        """Handle CC auto-open checkbox toggle.

        Persists the setting to config immediately.
        """
        from PyQt6.QtCore import Qt
        enabled = state == Qt.CheckState.Checked.value if hasattr(Qt.CheckState, 'value') else bool(state)
        try:
            from meetandread.config import set_config, save_config
            set_config("transcription.cc_auto_open", enabled)
            save_config()
        except Exception as exc:
            logger.warning("Failed to save CC auto-open setting: %s", exc)
        logger.info("CC auto-open set to %s", enabled)

    def _on_waveform_toggled(self, state: int) -> None:
        """Handle waveform visualization checkbox toggle.

        Persists the setting to config immediately so the next recording
        paint frame picks it up without restart.
        """
        enabled = bool(state)
        try:
            from meetandread.config import set_config, save_config
            set_config("ui.waveform_enabled", enabled)
            save_config()
        except Exception as exc:
            logger.warning("Failed to save waveform setting: %s", exc)
        logger.info("Waveform visualization %s", "enabled" if enabled else "disabled")

    def _refresh_dropdown_wer(self) -> None:
        """Update all dropdown item texts with latest WER from config."""
        self._populate_model_dropdown(self._live_model_combo, "realtime_model_size")
        self._populate_model_dropdown(self._postprocess_model_combo, "postprocess_model_size")

    def _refresh_benchmark_model_combo(self) -> None:
        """Update benchmark model dropdown items with latest WER from config.

        Preserves the current model selection. On first call (empty combo),
        defaults to the current live model from config.
        """
        current_model = self._benchmark_model_combo.currentData()
        if current_model is None:
            # First call — default to current live model
            try:
                from meetandread.config import get_config
                current_model = get_config().transcription.realtime_model_size
            except Exception:
                current_model = "tiny"

        self._benchmark_model_combo.blockSignals(True)
        self._benchmark_model_combo.clear()

        try:
            from meetandread.config import get_config
            _cfg = get_config()
            _bench_history = _cfg.transcription.benchmark_history
        except Exception:
            _bench_history = {}

        _model_order = ["tiny", "base", "small", "medium", "large"]
        _select_idx = 0
        for _i, _mn in enumerate(_model_order):
            _entry = _bench_history.get(_mn)
            if _entry and "wer" in _entry:
                _wer_pct = _entry["wer"] * 100
                _item_text = f"{_mn} — WER: {_wer_pct:.1f}%"
            else:
                _item_text = f"{_mn} (not benchmarked)"
            self._benchmark_model_combo.addItem(_item_text, _mn)
            if _mn == current_model:
                _select_idx = _i
        self._benchmark_model_combo.setCurrentIndex(_select_idx)
        self._benchmark_model_combo.blockSignals(False)

    def update_benchmark_display(self, wer_by_model: dict) -> None:
        """Refresh both model dropdowns after benchmark completes.

        Writes WER results to config and refreshes dropdown text.

        Args:
            wer_by_model: Dict mapping model_size -> WER float (0.0-1.0).
        """
        try:
            from meetandread.config import get_config, set_config, save_config
            settings = get_config()
            history = dict(settings.transcription.benchmark_history)

            from datetime import datetime
            now = datetime.now().isoformat()
            for model_size, wer in wer_by_model.items():
                history[model_size] = {"wer": wer, "timestamp": now}

            set_config("transcription.benchmark_history", history)
            save_config()
        except Exception as exc:
            logger.warning("Failed to update benchmark history in config: %s", exc)

        self._refresh_dropdown_wer()

    # ------------------------------------------------------------------
    # Identities page methods
    # ------------------------------------------------------------------

    def _refresh_identities(self) -> None:
        """Load voice profiles and scan usage metadata, then populate the identity list.

        Merges identity names from two sources:
        1. VoiceSignatureStore (profiles with embeddings)
        2. Transcript speaker_matches metadata (identities linked without embeddings)

        Failure Modes:
        - VoiceSignatureStore.load_signatures() error: show transcript-discovered names only.
        - scan_identity_usage() error: show profiles with zero usage, keep tab usable.
        """
        self._identity_usage = {}
        profile_names: List[str] = []

        try:
            from meetandread.audio.storage.paths import get_recordings_dir
            from meetandread.speaker.signatures import VoiceSignatureStore

            recordings_dir = get_recordings_dir()
            db_path = recordings_dir / "speaker_signatures.db"
            store = VoiceSignatureStore(str(db_path))
            try:
                profiles = store.load_signatures()
                profile_names = sorted(
                    [p.name for p in profiles],
                    key=lambda n: n.lower(),
                )
            finally:
                store.close()
        except Exception as exc:
            logger.info("Identity tab: store load failed: %s", exc)

        # Discover additional identity names from transcript speaker_matches
        # that were linked through the history dialog but don't have embeddings yet.
        try:
            from meetandread.audio.storage.paths import get_transcripts_dir
            from meetandread.speaker.identity_management import parse_metadata_footer

            transcripts_dir = get_transcripts_dir()
            if transcripts_dir.is_dir():
                existing = set(profile_names)
                discovered = set()
                for md_path in transcripts_dir.glob("*.md"):
                    try:
                        content = md_path.read_text(encoding="utf-8")
                    except OSError:
                        continue
                    data = parse_metadata_footer(content)
                    if data is None:
                        continue
                    for _label, match_info in data.get("speaker_matches", {}).items():
                        if isinstance(match_info, dict):
                            name = match_info.get("identity_name")
                            if name and name not in existing:
                                discovered.add(name)
                if discovered:
                    profile_names = sorted(
                        set(profile_names) | discovered,
                        key=lambda n: n.lower(),
                    )
        except Exception as exc:
            logger.info("Identity tab: transcript scan failed: %s", exc)

        if not profile_names:
            self._populate_identity_list(profile_names, {})
            return

        # Scan transcript usage
        usage: Dict[str, Any] = {}
        try:
            from meetandread.audio.storage.paths import get_transcripts_dir
            from meetandread.speaker.identity_management import scan_identity_usage

            transcripts_dir = get_transcripts_dir()
            usage = scan_identity_usage(transcripts_dir, profile_names)
        except Exception as exc:
            logger.info("Identity tab: usage scan failed: %s", exc)

        self._identity_usage = usage
        self._populate_identity_list(profile_names, usage)

    def _populate_identity_list(
        self,
        profile_names: List[str],
        usage: Dict[str, Any],
    ) -> None:
        """Populate the identity list widget from profile names and usage data.

        Args:
            profile_names: Sorted list of identity names.
            usage: Mapping from name to IdentityUsage (may be partial/empty).
        """
        self._identity_profile_names = list(profile_names)
        self._identity_list.clear()

        if not profile_names:
            # Empty state — detail stays hidden
            self._identity_detail_header.hide()
            self._clear_identity_detail()
            return

        for name in profile_names:
            identity_usage = usage.get(name)
            rec_count = identity_usage.recording_count if identity_usage else 0
            display_text = f"{name}  ({rec_count} recording{'s' if rec_count != 1 else ''})"
            item = QListWidgetItem(display_text)
            item.setData(Qt.ItemDataRole.UserRole, name)
            self._identity_list.addItem(item)

    def _clear_identity_detail(self) -> None:
        """Reset identity detail fields to default state and disable action buttons."""
        self._identity_name_label.setText("Name: —")
        self._identity_sample_count_label.setText("Samples: —")
        self._identity_recording_count_label.setText("Recordings: —")
        self._identity_last_used_label.setText("Last used: —")
        self._identity_recordings_label.setText("Associated recordings: —")
        self._identity_rename_btn.setEnabled(False)
        self._identity_merge_btn.setEnabled(False)
        self._identity_delete_btn.setEnabled(False)

    def _on_identity_item_clicked(self, item: QListWidgetItem) -> None:
        """Render the selected identity's details in the detail panel."""
        name = item.data(Qt.ItemDataRole.UserRole)
        if not name:
            return

        # Enable action buttons based on selection and profile count
        self._identity_rename_btn.setEnabled(True)
        self._identity_delete_btn.setEnabled(True)
        # Merge requires at least two identities
        self._identity_merge_btn.setEnabled(len(self._identity_profile_names) >= 2)

        self._identity_detail_header.show()

        # Look up usage data
        identity_usage = self._identity_usage.get(name)

        # Look up sample count from the store
        sample_count = 0
        try:
            from meetandread.audio.storage.paths import get_recordings_dir
            from meetandread.speaker.signatures import VoiceSignatureStore

            recordings_dir = get_recordings_dir()
            db_path = recordings_dir / "speaker_signatures.db"
            store = VoiceSignatureStore(str(db_path))
            try:
                profiles = store.load_signatures()
                for p in profiles:
                    if p.name == name:
                        sample_count = p.num_samples
                        break
            finally:
                store.close()
        except Exception:
            pass  # Keep sample_count as 0

        # Update detail fields
        self._identity_name_label.setText(f"Name: {name}")
        self._identity_sample_count_label.setText(f"Samples: {sample_count}")

        if identity_usage:
            rec_count = identity_usage.recording_count
            self._identity_recording_count_label.setText(f"Recordings: {rec_count}")

            if identity_usage.last_activity:
                from datetime import datetime
                try:
                    dt = datetime.fromtimestamp(identity_usage.last_activity)
                    self._identity_last_used_label.setText(
                        f"Last used: {dt.strftime('%Y-%m-%d %H:%M')}"
                    )
                except (OSError, ValueError):
                    self._identity_last_used_label.setText("Last used: —")
            else:
                self._identity_last_used_label.setText("Last used: —")

            if identity_usage.recordings:
                rec_paths = [
                    r.path.stem for r in identity_usage.recordings[:10]
                ]
                extra = (
                    f" (+{len(identity_usage.recordings) - 10} more)"
                    if len(identity_usage.recordings) > 10
                    else ""
                )
                self._identity_recordings_label.setText(
                    f"Associated recordings: {', '.join(rec_paths)}{extra}"
                )
            else:
                self._identity_recordings_label.setText("Associated recordings: none")
        else:
            self._identity_recording_count_label.setText("Recordings: 0")
            self._identity_last_used_label.setText("Last used: —")
            self._identity_recordings_label.setText("Associated recordings: —")

    # ------------------------------------------------------------------
    # Identity mutation handlers (T03)
    # ------------------------------------------------------------------

    def _get_selected_identity_name(self) -> Optional[str]:
        """Return the name of the currently selected identity, or None."""
        item = self._identity_list.currentItem()
        if item is None:
            return None
        name = item.data(Qt.ItemDataRole.UserRole)
        return name if name else None

    def _get_identity_store_and_transcripts_dir(self):
        """Return (store, transcripts_dir) for identity mutations.

        Raises RuntimeError if either cannot be resolved.
        """
        from meetandread.audio.storage.paths import get_recordings_dir, get_transcripts_dir
        from meetandread.speaker.signatures import VoiceSignatureStore

        recordings_dir = get_recordings_dir()
        db_path = recordings_dir / "speaker_signatures.db"
        store = VoiceSignatureStore(str(db_path))
        transcripts_dir = get_transcripts_dir()
        return store, transcripts_dir

    def _refresh_and_reselect(self, target_name: Optional[str] = None) -> None:
        """Refresh the identity list and optionally reselect a name.

        Uses deterministic reselection by name, never by stale item refs.
        """
        self._refresh_identities()
        if target_name:
            for i in range(self._identity_list.count()):
                item = self._identity_list.item(i)
                if item.data(Qt.ItemDataRole.UserRole) == target_name:
                    self._identity_list.setCurrentItem(item)
                    self._on_identity_item_clicked(item)
                    return
            # target_name no longer exists (e.g. after merge source deleted)
            self._clear_identity_detail()
            self._identity_detail_header.hide()

    def _on_identity_rename(self) -> None:
        """Handle the Rename action: prompt for new name, validate, rename."""
        old_name = self._get_selected_identity_name()
        if not old_name:
            return

        new_name, ok = QInputDialog.getText(
            self,
            "Rename Identity",
            f"New name for identity:",
            text=old_name,
        )
        if not ok:
            return  # User cancelled

        new_name = new_name.strip()
        if not new_name:
            QMessageBox.warning(
                self, "Rename Failed", "Identity name must not be empty."
            )
            return

        if new_name == old_name:
            QMessageBox.warning(
                self, "Rename Failed", "New name must differ from the current name."
            )
            return

        # Check for duplicate (case-sensitive, since store is case-sensitive)
        if new_name in self._identity_profile_names:
            QMessageBox.warning(
                self, "Rename Failed",
                f"An identity named \"{new_name}\" already exists."
            )
            return

        try:
            store, transcripts_dir = self._get_identity_store_and_transcripts_dir()
            try:
                from meetandread.speaker.identity_management import rename_identity
                rename_identity(store, transcripts_dir, old_name, new_name)
            finally:
                store.close()
        except Exception as exc:
            QMessageBox.warning(
                self, "Rename Failed", f"Could not rename identity: {exc}"
            )
            self._refresh_identities()
            return

        logger.info("Identity rename completed via Settings UI")
        self._refresh_and_reselect(target_name=new_name)

    def _on_identity_merge(self) -> None:
        """Handle the Merge action: pick target, confirm, merge."""
        source_name = self._get_selected_identity_name()
        if not source_name:
            return

        # Build list of potential targets (exclude source)
        other_names = [n for n in self._identity_profile_names if n != source_name]
        if not other_names:
            QMessageBox.information(
                self, "Merge", "No other identities available to merge into."
            )
            return

        # Build a small dialog with a combo box
        dialog = QDialog(self)
        dialog.setWindowTitle("Merge Identity")
        dialog.setMinimumWidth(320)
        layout = QVBoxLayout(dialog)

        layout.addWidget(QLabel(
            f"Merge \"{source_name}\" into which identity?\n\n"
            "Transcripts and voice signatures will be updated."
        ))

        combo = QComboBox()
        combo.addItems(other_names)
        combo.setEditable(False)
        layout.addWidget(combo)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        target_name = combo.currentText()
        if not target_name or target_name == source_name:
            return

        # Confirmation
        reply = QMessageBox.question(
            self,
            "Confirm Merge",
            f"Merge \"{source_name}\" into \"{target_name}\"?\n\n"
            "This will update voice signatures and transcript references.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            store, transcripts_dir = self._get_identity_store_and_transcripts_dir()
            try:
                from meetandread.speaker.identity_management import merge_identities
                merge_identities(store, transcripts_dir, source_name, target_name)
            finally:
                store.close()
        except Exception as exc:
            QMessageBox.warning(
                self, "Merge Failed", f"Could not merge identity: {exc}"
            )
            self._refresh_identities()
            return

        logger.info("Identity merge completed via Settings UI")
        self._refresh_and_reselect(target_name=target_name)

    def _on_identity_delete(self) -> None:
        """Handle the Delete action: confirm, delete from store, refresh."""
        name = self._get_selected_identity_name()
        if not name:
            return

        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Delete voice identity \"{name}\"?\n\n"
            "The voice signature will be removed from the database. "
            "Transcript references will be preserved as historical records.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            store, transcripts_dir = self._get_identity_store_and_transcripts_dir()
            try:
                from meetandread.speaker.identity_management import delete_identity, DeleteError
                delete_identity(store, transcripts_dir, name)
            finally:
                store.close()
        except DeleteError as exc:
            # Graceful handling for double-delete or already-removed identity
            msg = str(exc)
            if "not found" in msg.lower():
                QMessageBox.information(
                    self, "Already Deleted",
                    f"Identity \"{name}\" has already been removed."
                )
            else:
                QMessageBox.warning(
                    self, "Delete Failed", f"Could not delete identity: {exc}"
                )
            self._refresh_and_reselect(target_name=None)
            return
        except Exception as exc:
            QMessageBox.warning(
                self, "Delete Failed", f"Could not delete identity: {exc}"
            )
            self._refresh_identities()
            return

        logger.info("Identity delete completed via Settings UI")
        # After delete, the identity is gone — clear detail and refresh
        self._refresh_and_reselect(target_name=None)

    # ------------------------------------------------------------------
    # History page methods (adapted from FloatingTranscriptPanel)
    # ------------------------------------------------------------------

    def refresh_history_if_visible(self) -> None:
        """Refresh the history list when the History page is currently shown.

        Public entry point for external callers (e.g. MeetAndReadWidget)
        to trigger a history refresh after recording or post-processing
        completes.  Avoids unnecessary work when the History page is not
        visible — the next navigation to it will call ``_refresh_history``
        via ``_on_nav_clicked``.
        """
        if (self._content_stack.currentIndex() == self._NAV_HISTORY
                and self.isVisible()):
            self._refresh_history()

    def _refresh_history(self) -> None:
        """Re-scan recordings and repopulate the history list."""
        try:
            from meetandread.transcription.transcript_scanner import scan_recordings
        except ImportError:
            logger.warning("transcript_scanner not available — cannot populate history")
            return
        self._populate_history_list(scan_recordings())

    def _populate_history_list(self, recordings: list) -> None:
        """Populate the history QListWidget from a list of RecordingMeta.

        Args:
            recordings: List of RecordingMeta objects (expected sorted newest-first).
        """
        self._history_list.clear()
        for meta in recordings:
            display_date = meta.recording_time
            if display_date:
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(display_date)
                    display_date = dt.strftime("%Y-%m-%d %H:%M")
                except (ValueError, TypeError):
                    pass

            if meta.word_count == 0:
                display_text = f"{display_date} | (Empty recording)"
            else:
                display_text = (
                    f"{display_date} | {meta.word_count} words"
                    f" | {meta.speaker_count} speakers"
                )

            item = QListWidgetItem(display_text)
            item.setData(Qt.ItemDataRole.UserRole, str(meta.path))
            self._history_list.addItem(item)

    def _on_history_item_clicked(self, item: QListWidgetItem) -> None:
        """Load and display the transcript for the clicked history item.

        Also loads companion audio via the playback helper, updating
        toolbar control enabled/disabled state and status text.
        """
        if self._is_comparison_mode:
            self._hide_scrub_accept_reject()

        md_path_str = item.data(Qt.ItemDataRole.UserRole)
        if not md_path_str:
            self._reset_highlight_state()
            self._bookmark_manager = None
            self._bookmark_items = []
            self._refresh_bookmark_combo()
            self._update_playback_for_no_audio()
            return
        md_path = Path(md_path_str)
        if not md_path.exists():
            self._current_history_md_path = None
            self._reset_highlight_state()
            self._bookmark_manager = None
            self._bookmark_items = []
            self._refresh_bookmark_combo()
            self._history_viewer.setPlainText(f"(File not found: {md_path})")
            self._history_detail_header.show()
            self._update_playback_for_no_audio()
            return

        self._current_history_md_path = md_path
        self._history_detail_header.show()

        # Reset highlight state and extract timed words for the new transcript
        self._reset_highlight_state()
        self._extract_timed_words(md_path)

        # Load audio for this transcript
        self._load_playback_audio(md_path)

        # Load bookmarks for this transcript
        self._load_bookmarks_for_transcript(md_path)

        html = self._render_history_transcript(md_path)
        if html is not None:
            self._history_viewer.setHtml(html)
        else:
            try:
                content = md_path.read_text(encoding="utf-8")
            except OSError as exc:
                self._history_viewer.setPlainText(f"(Error reading file: {exc})")
                return
            footer_marker = "\n---\n\n<!-- METADATA:"
            marker_idx = content.find(footer_marker)
            if marker_idx != -1:
                content = content[:marker_idx]
            self._history_viewer.setMarkdown(_strip_confidence_percentages(content))

    # ------------------------------------------------------------------
    # Playback helper management
    # ------------------------------------------------------------------

    def _ensure_playback_helper(self):
        """Lazily create the HistoryPlaybackController if not yet created.

        Returns None (and logs a warning) if QtMultimedia is unavailable
        (e.g. DLL load failure in some test environments).

        On first creation, wires player.positionChanged and
        player.durationChanged to the progress slider with throttling
        and drag-guard logic.
        """
        if self._playback_helper is None:
            try:
                from meetandread.playback.history import HistoryPlaybackController
                self._playback_helper = HistoryPlaybackController()
                # Wire player signals to progress slider (once per helper)
                self._wire_player_signals()
            except ImportError as exc:
                logger.warning("QtMultimedia unavailable — playback disabled: %s", exc)
                self._playback_helper = None
        return self._playback_helper

    def _wire_player_signals(self) -> None:
        """Connect player.positionChanged / durationChanged to the slider.

        Called once after helper creation. Guards against None helper or
        missing player.
        """
        helper = self._playback_helper
        if helper is None:
            return
        player = helper.player
        player.positionChanged.connect(self._on_player_position_changed)
        player.durationChanged.connect(self._on_player_duration_changed)
        logger.info("player_signals_wired: positionChanged, durationChanged")

    # Minimum interval (ms) between slider updates from player position
    _POSITION_UPDATE_INTERVAL_MS = 50

    def _on_player_position_changed(self, position_ms: int) -> None:
        """Update progress slider and current-word highlight from player position.

        Skips the slider update when the user is dragging the slider to avoid
        feedback loops. Throttles slider updates to at most once per 50 ms.

        Highlight updates use a separate 200ms throttle and are gated by
        active-word-change detection — ``setHtml()`` is only called when the
        active word index actually changes.  Both sliders and highlights are
        independently throttled so a skipped slider update does not block
        highlighting and vice versa.
        """
        if self._is_dragging_progress_slider:
            return

        helper = self._playback_helper
        if helper is None:
            return

        duration = helper.duration_ms
        if duration <= 0:
            # No duration known yet — can't compute slider position
            return

        now_ms = int(time.monotonic() * 1000)

        # --- Slider update (50ms throttle) ---
        last_slider = getattr(self, "_last_slider_update_ms", 0)
        if (now_ms - last_slider) >= self._POSITION_UPDATE_INTERVAL_MS:
            self._last_slider_update_ms = now_ms
            slider_value = min(1000, int(position_ms / duration * 1000))
            self._playback_progress_slider.blockSignals(True)
            self._playback_progress_slider.setValue(slider_value)
            self._playback_progress_slider.blockSignals(False)

        # --- Current-word highlight (200ms throttle + active-word gating) ---
        if self._cached_timed_words and self._current_history_md_path is not None:
            last_highlight = self._last_highlight_update_ms
            if (now_ms - last_highlight) >= self._HIGHLIGHT_UPDATE_INTERVAL_MS:
                self._last_highlight_update_ms = now_ms
                active_idx = self._find_active_word_index(position_ms)
                if active_idx != self._current_highlight_word_idx:
                    logger.debug(
                        "highlight_word_changed: index=%d position_ms=%d",
                        active_idx, position_ms,
                    )
                    self._render_highlighted_transcript(
                        self._current_history_md_path, active_idx
                    )

    def _on_player_duration_changed(self, duration_ms: int) -> None:
        """Handle duration changes from the player (media load completion).

        Resets the slider when a new media source is loaded (duration
        transitions from 0 to a positive value). Keeps the slider at
        value 0 so the user sees a clean start position.
        """
        if duration_ms > 0:
            logger.info(
                "duration_changed: duration_ms=%d, resetting slider",
                duration_ms,
            )
            # Reset slider to start on new media load
            if not self._is_dragging_progress_slider:
                self._playback_progress_slider.blockSignals(True)
                self._playback_progress_slider.setValue(0)
                self._playback_progress_slider.blockSignals(False)

    def _load_playback_audio(self, md_path: Path) -> None:
        """Load audio for the given transcript and update toolbar state."""
        helper = self._ensure_playback_helper()
        if helper is not None:
            helper.load_transcript_audio(md_path)
        self._sync_playback_controls()

    def _update_playback_for_no_audio(self) -> None:
        """Disable playback controls when no transcript is selected."""
        helper = self._ensure_playback_helper()
        if helper is not None:
            helper.load_transcript_audio(None)
        self._sync_playback_controls()

    def _sync_playback_controls(self) -> None:
        """Sync toolbar enabled/disabled state and status from the helper."""
        helper = self._playback_helper
        if helper is None:
            self._playback_play_btn.setEnabled(False)
            self._playback_speed_combo.setEnabled(False)
            self._playback_volume_slider.setEnabled(False)
            self._playback_progress_slider.setEnabled(False)
            self._playback_skip_back_btn.setEnabled(False)
            self._playback_skip_fwd_btn.setEnabled(False)
            self._bookmark_add_btn.setEnabled(False)
            self._bookmark_combo.setEnabled(False)
            self._playback_status_label.setText("")
            return

        available = helper.is_audio_available
        self._playback_play_btn.setEnabled(available)
        self._playback_speed_combo.setEnabled(available)
        self._playback_volume_slider.setEnabled(available)
        self._playback_progress_slider.setEnabled(available)
        self._playback_skip_back_btn.setEnabled(available)
        self._playback_skip_fwd_btn.setEnabled(available)

        # Bookmark add requires both audio and a selected transcript
        has_transcript = self._current_history_md_path is not None
        self._bookmark_add_btn.setEnabled(available and has_transcript)
        # Bookmark combo is enabled when there are bookmarks to navigate
        has_bookmarks = len(self._bookmark_items) > 0
        self._bookmark_combo.setEnabled(has_transcript and has_bookmarks)

        # Update play/pause button text based on helper state
        if available:
            self._playback_play_btn.setText("▶")
        else:
            self._playback_play_btn.setText("▶")

        if helper.last_error:
            self._playback_status_label.setText(helper.last_error)
            # Apply error-tinted status style
            p = current_palette()
            playback_css = aetheric_playback_toolbar_css(p)
            self._playback_status_label.setStyleSheet(playback_css["status_label_error"])
        else:
            status = helper.status_text
            self._playback_status_label.setText(status)
            # Restore normal status style
            p = current_palette()
            playback_css = aetheric_playback_toolbar_css(p)
            self._playback_status_label.setStyleSheet(playback_css["status_label"])

    def _on_playback_play_clicked(self) -> None:
        """Toggle play/pause on the playback helper."""
        helper = self._playback_helper
        if helper is None or not helper.is_audio_available:
            return
        # Check current state from the helper's player
        player = helper.player
        # Use the player's own PlaybackState enum values to avoid importing
        # QtMultimedia (which may have DLL issues in some environments)
        state = player.playbackState()
        # PlayingState == 1 in QtMultimedia
        if state == 1:  # PlayingState
            helper.pause()
            self._playback_play_btn.setText("▶")
        else:
            helper.play()
            self._playback_play_btn.setText("⏸")

    def _on_playback_speed_changed(self, index: int) -> None:
        """Route speed combo change to the playback helper."""
        helper = self._playback_helper
        if helper is None or not helper.is_audio_available:
            return
        rate_text = self._playback_speed_combo.itemText(index)
        try:
            rate = float(rate_text.replace("x", ""))
        except (ValueError, AttributeError):
            logger.warning("Invalid playback rate text: %r", rate_text)
            return
        helper.set_rate(rate)

    def _on_playback_volume_changed(self, value: int) -> None:
        """Route volume slider change to the playback helper."""
        helper = self._playback_helper
        if helper is None or not helper.is_audio_available:
            return
        normalized = value / 100.0
        helper.set_volume(normalized)

    def _on_playback_skip_back_clicked(self) -> None:
        """Skip backward 5 seconds via toolbar button."""
        helper = self._playback_helper
        if helper is None or not helper.is_audio_available:
            return
        helper.skip_backward()

    def _on_playback_skip_fwd_clicked(self) -> None:
        """Skip forward 5 seconds via toolbar button."""
        helper = self._playback_helper
        if helper is None or not helper.is_audio_available:
            return
        helper.skip_forward()

    # -- Bookmark handlers ---------------------------------------------------

    def _on_bookmark_add_clicked(self) -> None:
        """Add a bookmark at the current playback position.

        Checks: helper present, audio available, transcript selected.
        Prompts for name via QInputDialog with default "Bookmark at MM:SS".
        On cancel, no-op. On empty name, uses default. On manager error,
        shows concise status message.
        """
        helper = self._playback_helper
        if helper is None or not helper.is_audio_available:
            return
        md_path = self._current_history_md_path
        if md_path is None:
            return

        position_ms = helper.position_ms
        from meetandread.playback.bookmark import _format_position
        default_name = f"Bookmark at {_format_position(position_ms)}"

        name, ok = QInputDialog.getText(
            self,
            "Add Bookmark",
            "Bookmark name:",
            text=default_name,
        )
        if not ok:
            # User cancelled — do nothing
            return

        # Empty name → use default
        if not name or not name.strip():
            name = default_name

        try:
            from meetandread.playback.bookmark import BookmarkManager
            if self._bookmark_manager is None:
                self._bookmark_manager = BookmarkManager(md_path)
            bm = self._bookmark_manager.add(position_ms, name=name)
            logger.info(
                "bookmark_added_ui: stem=%s position_ms=%d",
                md_path.stem, position_ms,
            )
        except Exception as exc:
            logger.warning("bookmark_add_failed: stem=%s error=%s", md_path.stem, exc)
            self._playback_status_label.setText(f"Bookmark error: {exc}")
            return

        # Refresh the bookmark combo
        self._refresh_bookmark_combo()

    def _on_bookmark_combo_changed(self, index: int) -> None:
        """Navigate to the selected bookmark's position.

        Looks up the (created_at, position_ms) from _bookmark_items.
        Seeks to position_ms and plays if audio is available.
        On stale id or unavailable audio, no-ops safely.
        """
        if index < 0 or index >= len(self._bookmark_items):
            return

        created_at, position_ms = self._bookmark_items[index]
        helper = self._playback_helper
        if helper is None:
            return

        if helper.is_audio_available:
            helper.seek_to(position_ms)
            helper.play()
            logger.info(
                "bookmark_navigation_triggered: position_ms=%d",
                position_ms,
            )
        else:
            # Audio unavailable — show status, don't seek
            self._playback_status_label.setText(
                "Cannot navigate: audio unavailable"
            )
            logger.info(
                "bookmark_navigation_skipped: reason=audio_unavailable position_ms=%d",
                position_ms,
            )

    def _refresh_bookmark_combo(self) -> None:
        """Reload bookmarks from the manager and repopulate the combo.

        Safe no-op when no transcript is selected or manager is unavailable.
        """
        md_path = self._current_history_md_path
        if md_path is None:
            self._bookmark_combo.clear()
            self._bookmark_combo.addItem("No bookmarks")
            self._bookmark_combo.setEnabled(False)
            self._bookmark_items = []
            return

        try:
            from meetandread.playback.bookmark import BookmarkManager
            if self._bookmark_manager is None:
                self._bookmark_manager = BookmarkManager(md_path)
            bookmarks = self._bookmark_manager.list_bookmarks()
        except Exception as exc:
            logger.warning(
                "bookmark_load_failed: stem=%s error=%s",
                md_path.stem, exc,
            )
            self._bookmark_combo.clear()
            self._bookmark_combo.addItem("(Bookmark error)")
            self._bookmark_combo.setEnabled(False)
            self._bookmark_items = []
            return

        # Block signals while rebuilding to avoid triggering navigation
        self._bookmark_combo.blockSignals(True)
        self._bookmark_combo.clear()
        self._bookmark_items = []

        if not bookmarks:
            self._bookmark_combo.addItem("No bookmarks")
        else:
            for bm in bookmarks:
                from meetandread.playback.bookmark import _format_position
                label = f"{bm.name} ({_format_position(bm.position_ms)})"
                self._bookmark_combo.addItem(label)
                self._bookmark_items.append((bm.created_at, bm.position_ms))

        self._bookmark_combo.blockSignals(False)

        # Update enabled state
        has_bookmarks = len(self._bookmark_items) > 0
        helper = self._playback_helper
        has_audio = helper is not None and helper.is_audio_available
        self._bookmark_combo.setEnabled(has_audio and has_bookmarks)
        self._bookmark_delete_btn.setEnabled(has_bookmarks)

    def _on_bookmark_delete_clicked(self) -> None:
        """Delete the currently selected bookmark.

        Looks up ``created_at`` from ``_bookmark_items`` at the current combo
        index, calls ``BookmarkManager.delete()``, and refreshes UI state.
        Safe no-op when no bookmark is selected, no transcript is active, or
        the bookmark has already been deleted.
        """
        idx = self._bookmark_combo.currentIndex()
        if idx < 0 or idx >= len(self._bookmark_items):
            return

        created_at, _position_ms = self._bookmark_items[idx]
        md_path = self._current_history_md_path
        if md_path is None:
            return

        try:
            from meetandread.playback.bookmark import BookmarkManager
            if self._bookmark_manager is None:
                self._bookmark_manager = BookmarkManager(md_path)
            deleted = self._bookmark_manager.delete(created_at)
            if deleted:
                logger.info(
                    "bookmark_deleted_ui: stem=%s",
                    md_path.stem,
                )
            # If not found (already deleted), silently refresh
        except Exception as exc:
            logger.warning("bookmark_delete_failed: stem=%s error=%s", md_path.stem, exc)
            self._playback_status_label.setText(f"Bookmark delete error: {exc}")
            return

        self._refresh_bookmark_combo()

    def _load_bookmarks_for_transcript(self, md_path: Path) -> None:
        """Initialize bookmark manager and combo for a newly selected transcript.

        Called from _on_history_item_clicked after setting _current_history_md_path.
        """
        self._bookmark_manager = None  # Reset — will lazy-init on first use
        self._refresh_bookmark_combo()

    def _on_progress_slider_pressed(self) -> None:
        """Mark slider as being dragged to suppress position updates."""
        self._is_dragging_progress_slider = True

    def _on_progress_slider_released(self) -> None:
        """Seek to the slider position and resume position tracking."""
        helper = self._playback_helper
        if helper is None or not helper.is_audio_available:
            self._is_dragging_progress_slider = False
            return
        slider_value = self._playback_progress_slider.value()
        duration = helper.duration_ms
        if duration > 0:
            percent = slider_value / 1000.0
            target_ms = int(percent * duration)
            logger.info(
                "slider_seek_triggered position_ms=%d percent=%.3f",
                target_ms, percent,
            )
            helper.seek_to(target_ms)
        self._is_dragging_progress_slider = False

    def _on_progress_slider_value_changed(self, value: int) -> None:
        """Handle slider value changes (only seek on drag release, not live)."""
        # Live value changes during drag are tracked but not applied;
        # actual seek happens in _on_progress_slider_released.
        pass

    def _stop_playback(self) -> None:
        """Stop playback and reset helper. Called on panel hide/close."""
        if self._playback_helper is not None:
            self._playback_helper.stop()
        self._reset_highlight_state()

    # -- Keyboard shortcuts for History playback -----------------------------

    def keyPressEvent(self, event) -> None:  # noqa: N802
        """Handle keyboard shortcuts for History playback.

        Shortcuts only fire when the History nav page is active and audio is
        available:

        - Space: toggle play/pause
        - Arrow Left: skip backward 5 seconds
        - Arrow Right: skip forward 5 seconds
        - ``+`` / ``=``: increase playback speed
        - ``-`` / ``_``: decrease playback speed
        """
        key = event.key()
        modifier = event.modifiers()

        # Only handle shortcuts when on the History page
        if self._content_stack.currentIndex() == self._NAV_HISTORY:
            helper = self._playback_helper
            if helper is not None and helper.is_audio_available:
                action = None

                # Space — play/pause (no modifier)
                if key == Qt.Key.Key_Space and modifier == Qt.KeyboardModifier.NoModifier:
                    action = "play_pause"
                    self._on_playback_play_clicked()

                # Arrow Left — skip backward (no modifier)
                elif key == Qt.Key.Key_Left and modifier == Qt.KeyboardModifier.NoModifier:
                    action = "skip_backward"
                    self._on_playback_skip_back_clicked()

                # Arrow Right — skip forward (no modifier)
                elif key == Qt.Key.Key_Right and modifier == Qt.KeyboardModifier.NoModifier:
                    action = "skip_forward"
                    self._on_playback_skip_fwd_clicked()

                # Plus / Equal — increase speed (no modifier)
                elif key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal) and modifier == Qt.KeyboardModifier.NoModifier:
                    action = "speed_up"
                    self._step_playback_speed(1)

                # Minus — decrease speed (no modifier)
                elif key == Qt.Key.Key_Minus and modifier == Qt.KeyboardModifier.NoModifier:
                    action = "speed_down"
                    self._step_playback_speed(-1)

                # M — add bookmark (no modifier)
                elif key == Qt.Key.Key_M and modifier == Qt.KeyboardModifier.NoModifier:
                    action = "bookmark_add"
                    self._on_bookmark_add_clicked()

                if action is not None:
                    logger.info(
                        "keyboard_shortcut_triggered key=%s action=%s",
                        event.text() or str(key), action,
                    )
                    event.accept()
                    return

        # Not a shortcut we handle — pass to base class
        super().keyPressEvent(event)

    def _step_playback_speed(self, delta: int) -> None:
        """Step the playback speed combo up (``+1``) or down (``-1``).

        Clamps at combo bounds so the index never goes out of range.
        """
        combo = self._playback_speed_combo
        new_index = combo.currentIndex() + delta
        new_index = max(0, min(new_index, combo.count() - 1))
        if new_index != combo.currentIndex():
            combo.setCurrentIndex(new_index)

    @staticmethod
    def _extract_transcript_body(md_path: Optional[Path]) -> str:
        """Extract the markdown body (before METADATA footer) from a transcript.

        Args:
            md_path: Path to the transcript .md file.

        Returns:
            The markdown body text, or an error message string.
        """
        if md_path is None or not md_path.exists():
            return "(file not found)"
        try:
            content = md_path.read_text(encoding="utf-8")
        except OSError as exc:
            return f"(error reading file: {exc})"

        footer_marker = "\n---\n\n<!-- METADATA:"
        marker_idx = content.find(footer_marker)
        if marker_idx != -1:
            content = content[:marker_idx]
        return content.strip()

    @staticmethod
    def _validate_timed_word(word: dict, index: int) -> Optional[int]:
        """Validate a metadata word entry and return its start_ms, or None.

        A valid timed word must have ``start_time`` as a non-negative number.
        Returns the start time in milliseconds (int) for seek anchors.

        Args:
            word: A word dict from the metadata ``words`` array.
            index: Zero-based index of the word in the array.

        Returns:
            Start time in milliseconds, or None if the word lacks valid timing.
        """
        try:
            start = word.get("start_time")
            if start is None:
                return None
            start_ms = int(float(start) * 1000)
            if start_ms < 0:
                return None
            return start_ms
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _escape_html_text(text: str) -> str:
        """Escape text for safe inclusion in HTML content.

        Uses :func:`html.escape` with ``quote=True`` to handle
        ampersands, angle brackets, and quotes.

        Args:
            text: Raw text to escape.

        Returns:
            HTML-safe string.
        """
        return _html_module.escape(str(text), quote=True)

    def _extract_timed_words(self, md_path: Path) -> List[tuple]:
        """Extract and cache (start_ms, end_ms) pairs from transcript metadata.

        Reads the metadata ``words`` array and builds a list of
        ``(start_ms, end_ms)`` tuples for words that have valid timing.
        Words with missing or negative timing are represented as ``(None, None)``
        to keep indices aligned with the metadata array.

        The result is cached in ``_cached_timed_words`` and reused across
        highlight lookups until the transcript selection changes.

        Args:
            md_path: Path to the transcript .md file.

        Returns:
            List of ``(start_ms, end_ms)`` tuples aligned with the words array.
        """
        try:
            content = md_path.read_text(encoding="utf-8")
        except OSError:
            self._cached_timed_words = []
            return self._cached_timed_words

        footer_marker = "\n---\n\n<!-- METADATA:"
        marker_idx = content.find(footer_marker)
        if marker_idx == -1:
            self._cached_timed_words = []
            return self._cached_timed_words

        metadata_text = content[marker_idx + len(footer_marker):]
        if metadata_text.strip().endswith(" -->"):
            metadata_text = metadata_text.strip()[:-len(" -->")]

        try:
            data = json.loads(metadata_text)
        except json.JSONDecodeError:
            self._cached_timed_words = []
            return self._cached_timed_words

        words = data.get("words", [])
        result: List[tuple] = []
        for w in words:
            start = self._validate_timed_word(w, len(result))
            if start is not None:
                # Parse end_time
                try:
                    end_val = w.get("end_time")
                    if end_val is not None:
                        end_ms = int(float(end_val) * 1000)
                        if end_ms < start:
                            end_ms = start + 1  # fallback: 1ms minimum span
                    else:
                        end_ms = start + 1  # fallback for missing end_time
                except (TypeError, ValueError):
                    end_ms = start + 1
                result.append((start, end_ms))
            else:
                result.append((None, None))

        self._cached_timed_words = result
        return result

    def _find_active_word_index(self, position_ms: int) -> int:
        """Map a playback position to the active word index.

        Uses ``[start_ms, end_ms)`` semantics: a word is active when
        ``start_ms <= position_ms < end_ms``.

        For position gaps (silence between words), returns -1.
        For positions before the first word, returns -1.
        For positions at or beyond the last word's end, returns -1.

        Args:
            position_ms: Current playback position in milliseconds.

        Returns:
            Zero-based index of the active word, or -1 if no word is active.
        """
        if position_ms < 0:
            return -1

        cached = self._cached_timed_words
        if not cached:
            return -1

        # Binary search for efficiency on large transcripts
        # Find the last word whose start_ms <= position_ms
        lo, hi = 0, len(cached) - 1
        candidate = -1
        while lo <= hi:
            mid = (lo + hi) // 2
            start, end = cached[mid]
            if start is None:
                # Untimed word — skip; search both sides
                # We can't binary search through None entries efficiently,
                # fall back to linear for correctness.
                # In practice, mixed transcripts are rare.
                return self._find_active_word_index_linear(position_ms)
            if start <= position_ms:
                candidate = mid
                lo = mid + 1
            else:
                hi = mid - 1

        if candidate >= 0:
            start, end = cached[candidate]
            if start is not None and end is not None and start <= position_ms < end:
                return candidate

        return -1

    def _find_active_word_index_linear(self, position_ms: int) -> int:
        """Linear fallback for _find_active_word_index when None entries exist.

        Used when binary search encounters untimed (None) entries that
        prevent correct halving.
        """
        for i, (start, end) in enumerate(self._cached_timed_words):
            if start is not None and end is not None:
                if start <= position_ms < end:
                    return i
        return -1

    def _reset_highlight_state(self) -> None:
        """Reset all highlight state (on transcript change, stop, etc.).

        Clears the cached timed words, resets the current highlight index,
        and resets the highlight throttle timestamp.
        """
        self._cached_timed_words = []
        self._current_highlight_word_idx = -1
        self._last_highlight_update_ms = 0

    def _render_highlighted_transcript(self, md_path: Path, highlight_idx: int) -> None:
        """Re-render the transcript with the word at *highlight_idx* highlighted.

        Preserves vertical scroll position around the re-render.  Skips
        the re-render entirely if the HTML content would be identical
        (e.g., same highlight index as current).

        Args:
            md_path: Path to the transcript .md file.
            highlight_idx: Zero-based word index to highlight, or -1 for none.
        """
        if highlight_idx == self._current_highlight_word_idx:
            # No change — skip expensive setHtml() call
            return

        # Save scroll position
        scrollbar = self._history_viewer.verticalScrollBar()
        scroll_pos = scrollbar.value()

        self._current_highlight_word_idx = highlight_idx
        html = self._render_history_transcript_highlighted(md_path, highlight_idx)
        if html is not None:
            self._history_viewer.setHtml(html)

        # Restore scroll position
        scrollbar.setValue(scroll_pos)

    def _render_history_transcript_highlighted(
        self, md_path: Path, highlight_idx: int
    ) -> Optional[str]:
        """Render transcript HTML with a highlighted word.

        Produces the same output as ``_render_history_transcript`` but wraps
        the word at *highlight_idx* in a ``<span>`` with a highlight style.

        Args:
            md_path: Path to the transcript .md file.
            highlight_idx: Zero-based word index to highlight, or -1 for none.

        Returns:
            HTML string, or None if the transcript cannot be rendered.
        """
        try:
            content = md_path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.error("Failed to read transcript for highlighting: %s: %s", md_path, exc)
            return None

        footer_marker = "\n---\n\n<!-- METADATA:"
        marker_idx = content.find(footer_marker)
        if marker_idx == -1:
            return None

        metadata_text = content[marker_idx + len(footer_marker):]
        if metadata_text.strip().endswith(" -->"):
            metadata_text = metadata_text.strip()[:-len(" -->")]

        try:
            data = json.loads(metadata_text)
        except json.JSONDecodeError:
            return None

        words = data.get("words", [])
        has_timed_words = any(self._validate_timed_word(w, i) is not None
                             for i, w in enumerate(words))

        if not has_timed_words:
            # No timing data — nothing to highlight; return plain render
            return self._render_history_transcript(md_path)

        # Collect unique speakers
        speakers = []
        seen = set()
        has_unknown = False
        for word in words:
            sid = word.get("speaker_id")
            if sid is not None and sid not in seen:
                seen.add(sid)
                speakers.append(sid)
            elif sid is None:
                has_unknown = True

        # Build highlighted HTML
        _SENTINEL = object()
        current_speaker = _SENTINEL
        html_lines = []
        highlight_style = (
            "background-color: rgba(79, 195, 247, 0.25); "
            "border-radius: 2px; padding: 0 1px;"
        )

        for idx, w in enumerate(words):
            sid = w.get("speaker_id")
            if sid != current_speaker:
                current_speaker = sid
                if sid is not None and sid in seen:
                    color = speaker_color(sid)
                    html_lines.append(
                        f'<p><a href="speaker:{sid}" '
                        f'style="color:{color}; font-weight:bold; text-decoration:none;">'
                        f'[{sid}]</a></p>'
                    )
                elif sid is None and has_unknown:
                    html_lines.append(
                        f'<p><a href="speaker:__unknown__" '
                        f'style="color:#888888; font-weight:bold; text-decoration:none;">'
                        f'[Unknown Speaker]</a></p>'
                    )
                else:
                    label = sid if sid is not None else "Unknown Speaker"
                    html_lines.append(f"<p><b>{self._escape_html_text(label)}</b></p>")

            word_text = self._escape_html_text(w.get("text", ""))
            start_ms = self._validate_timed_word(w, idx)

            if start_ms is not None and word_text:
                if idx == highlight_idx:
                    html_lines.append(
                        f'<a href="word:{idx}:{start_ms}" '
                        f'style="color:inherit; text-decoration:none; {highlight_style}">'
                        f'{word_text}</a>'
                    )
                else:
                    html_lines.append(
                        f'<a href="word:{idx}:{start_ms}" '
                        f'style="color:inherit; text-decoration:none;">'
                        f'{word_text}</a>'
                    )
            else:
                if idx == highlight_idx and word_text:
                    html_lines.append(
                        f'<span style="{highlight_style}">{word_text}</span>'
                    )
                else:
                    html_lines.append(word_text)
            html_lines.append(" ")

        return "".join(html_lines)

    def _render_history_transcript(self, md_path: Path) -> Optional[str]:
        """Render a transcript .md file as HTML with clickable speaker anchors.

        Reads the .md file, parses the JSON metadata footer to get speakers,
        and returns HTML where each speaker label is an anchor tag with
        format ``speaker:{speaker_label}``.

        Args:
            md_path: Path to the transcript .md file.

        Returns:
            HTML string for the viewer, or None if no metadata is found.
        """
        try:
            content = md_path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.error("Failed to read transcript for rendering: %s: %s", md_path, exc)
            return None

        footer_marker = "\n---\n\n<!-- METADATA:"
        marker_idx = content.find(footer_marker)
        if marker_idx == -1:
            return None

        md_body = _strip_confidence_percentages(content[:marker_idx])

        metadata_text = content[marker_idx + len(footer_marker):]
        if metadata_text.strip().endswith(" -->"):
            metadata_text = metadata_text.strip()[:-len(" -->")]

        try:
            data = json.loads(metadata_text)
        except json.JSONDecodeError as exc:
            logger.warning("Malformed metadata in %s: %s", md_path, exc)
            return None

        # Collect unique speaker IDs from words (None counts as "Unknown Speaker")
        speakers = []
        seen = set()
        has_unknown = False
        for word in data.get("words", []):
            sid = word.get("speaker_id")
            if sid is not None and sid not in seen:
                seen.add(sid)
                speakers.append(sid)
            elif sid is None:
                has_unknown = True

        # Check if words have timing metadata for clickable word anchors
        words = data.get("words", [])
        has_timed_words = any(self._validate_timed_word(w, i) is not None
                             for i, w in enumerate(words))

        # Build HTML with clickable speaker anchors
        # The markdown body has lines like "**SPK_0**" — make them anchors
        html_lines = []

        if has_timed_words:
            # Render words from metadata as clickable timed anchors,
            # grouped by speaker with speaker heading anchors preserved.
            _SENTINEL = object()
            current_speaker = _SENTINEL
            for idx, w in enumerate(words):
                sid = w.get("speaker_id")
                if sid != current_speaker:
                    current_speaker = sid
                    if sid is not None and sid in seen:
                        color = speaker_color(sid)
                        html_lines.append(
                            f'<p><a href="speaker:{sid}" '
                            f'style="color:{color}; font-weight:bold; text-decoration:none;">'
                            f'[{sid}]</a></p>'
                        )
                    elif sid is None and has_unknown:
                        html_lines.append(
                            f'<p><a href="speaker:__unknown__" '
                            f'style="color:#888888; font-weight:bold; text-decoration:none;">'
                            f'[Unknown Speaker]</a></p>'
                        )
                    else:
                        label = sid if sid is not None else "Unknown Speaker"
                        html_lines.append(f"<p><b>{self._escape_html_text(label)}</b></p>")

                word_text = self._escape_html_text(w.get("text", ""))
                start_ms = self._validate_timed_word(w, idx)
                if start_ms is not None and word_text:
                    html_lines.append(
                        f'<a href="word:{idx}:{start_ms}" '
                        f'style="color:inherit; text-decoration:none;">'
                        f'{word_text}</a>'
                    )
                else:
                    html_lines.append(word_text)
                html_lines.append(" ")

            return "".join(html_lines)

        # Fallback: render markdown body (legacy / no-timing transcripts)
        for line in md_body.splitlines():
            match = re.match(r"^\*\*(.+?)\*\*\s*$", line)
            if match:
                speaker_label = match.group(1)
                if speaker_label in seen:
                    color = speaker_color(speaker_label)
                    html_lines.append(
                        f'<p><a href="speaker:{speaker_label}" '
                        f'style="color:{color}; font-weight:bold; text-decoration:none;">'
                        f'[{speaker_label}]</a></p>'
                    )
                elif speaker_label == "Unknown Speaker" and has_unknown:
                    # Make "Unknown Speaker" clickable so user can assign an identity
                    color = "#888888"
                    html_lines.append(
                        f'<p><a href="speaker:__unknown__" '
                        f'style="color:{color}; font-weight:bold; text-decoration:none;">'
                        f'[{speaker_label}]</a></p>'
                    )
                else:
                    html_lines.append(f"<p><b>{speaker_label}</b></p>")
            else:
                escaped = (
                    line.replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                )
                if escaped.strip():
                    escaped = re.sub(r"\*(.+?)\*", r"<i>\1</i>", escaped)
                    html_lines.append(f"<p>{escaped}</p>")
                elif not escaped:
                    html_lines.append("<br>")

        return "\n".join(html_lines)

    def _reselect_history_item(self, md_path: Path) -> None:
        """Re-select a history list item by its transcript path.

        Args:
            md_path: Path to the transcript .md file to re-select.
        """
        md_str = str(md_path)
        for i in range(self._history_list.count()):
            item = self._history_list.item(i)
            if item and item.data(Qt.ItemDataRole.UserRole) == md_str:
                self._history_list.setCurrentItem(item)
                return
        logger.debug("Could not re-select history item for %s", md_path)

    def _on_history_context_menu(self, pos) -> None:
        """Show context menu on history list items."""
        item = self._history_list.itemAt(pos)
        if item is None:
            return

        menu = QMenu(self._history_list)
        p = current_palette()
        menu.setStyleSheet(context_menu_css(p, accent_color=p.danger))

        scrub_action = menu.addAction("🔄  Scrub Recording")
        delete_action = menu.addAction("🗑  Delete Recording")
        scrub_action.triggered.connect(lambda: self._on_scrub_clicked())
        delete_action.triggered.connect(lambda: self._delete_recording(item))
        menu.exec(self._history_list.mapToGlobal(pos))

    def _on_delete_btn_clicked(self) -> None:
        """Handle Delete button click in the detail header."""
        current = self._history_list.currentItem()
        if current is None:
            return
        self._delete_recording(current)

    def _delete_recording(self, item: QListWidgetItem) -> None:
        """Delete a recording after user confirmation."""
        md_path_str = item.data(Qt.ItemDataRole.UserRole)
        if not md_path_str:
            return

        md_path = Path(md_path_str)
        stem = md_path.stem
        recording_name = item.text().split("|")[0].strip()

        try:
            from meetandread.recording.management import enumerate_recording_files, delete_recording
            files = enumerate_recording_files(stem)
        except Exception as exc:
            logger.error("Failed to enumerate recording files: %s", exc)
            files = []

        file_count = len(files)

        parent = self.parent() if self.parent() else self
        reply = QMessageBox.question(
            parent,
            "Delete Recording",
            f"Delete '{recording_name}'?\n\n"
            f"This will permanently remove {file_count} file{'s' if file_count != 1 else ''}.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            count, deleted = delete_recording(stem)
            logger.info(
                "Deleted recording '%s': %d files removed",
                recording_name, count,
            )
        except Exception as exc:
            logger.error("Failed to delete recording '%s': %s", recording_name, exc)
            QMessageBox.warning(
                parent,
                "Delete Failed",
                f"Could not delete recording '{recording_name}'.\n\n{exc}",
            )
            return

        self._current_history_md_path = None
        self._history_viewer.clear()
        self._history_viewer.setPlaceholderText("Select a recording to view its transcript")
        self._history_detail_header.hide()
        self._stop_playback()
        self._sync_playback_controls()

        self._refresh_history()

    def _on_history_anchor_clicked(self, url: QUrl) -> None:
        """Handle clicks on speaker and word anchors in the history viewer.

        ``speaker:`` anchors trigger the existing identity-link dialog.
        ``word:{index}:{start_ms}`` anchors seek playback to the word's
        start time and resume playing.
        """
        link = url.toString()

        # Handle word anchors: word:{index}:{start_ms}
        word_prefix = "word:"
        if link.startswith(word_prefix):
            self._on_word_anchor_clicked(link)
            return

        # Handle speaker anchors: speaker:{label}
        prefix = "speaker:"
        if not link.startswith(prefix):
            return

        raw_label = link[len(prefix):]
        if not raw_label:
            return

        parent = self.parent() if self.parent() else self
        md_path = self._current_history_md_path

        linked = _open_identity_link_dialog(md_path, raw_label, parent)
        if not linked:
            return

        html = self._render_history_transcript(md_path)
        if html is not None:
            self._history_viewer.setHtml(html)
            # Re-extract timed words after file change and re-apply highlight
            self._extract_timed_words(md_path)
            self._current_highlight_word_idx = -1
        else:
            self._history_viewer.setPlainText("(Error refreshing after link)")

        # Refresh the identities list so the newly linked identity appears
        # immediately when the user switches to the Identities tab.
        self._refresh_identities()

        # Refresh the history list so speaker counts update immediately,
        # and re-select the current item so the user stays on the same recording.
        if md_path is not None:
            self._refresh_history()
            self._reselect_history_item(md_path)

    def _on_word_anchor_clicked(self, link: str) -> None:
        """Handle a ``word:{index}:{start_ms}`` anchor click.

        Validates the payload, checks helper/audio availability, then
        calls ``seek_to(start_ms)`` and ``play()``.  Logs structured
        diagnostics for success, skipped seeks, and malformed payloads.
        """
        payload = link[len("word:"):]
        parts = payload.split(":")

        if len(parts) != 2:
            logger.warning("word_anchor_malformed: link=%s", link)
            return

        try:
            word_index = int(parts[0])
            start_ms = int(parts[1])
        except (ValueError, TypeError):
            logger.warning("word_anchor_malformed: link=%s", link)
            return

        if word_index < 0 or start_ms < 0:
            logger.warning(
                "word_anchor_malformed: index=%d start_ms=%d",
                word_index, start_ms,
            )
            return

        # Check helper availability
        helper = self._playback_helper
        if helper is None:
            logger.info(
                "word_seek_skipped: index=%d start_ms=%d reason=no_helper",
                word_index, start_ms,
            )
            return

        if not helper.is_audio_available:
            logger.info(
                "word_seek_skipped: index=%d start_ms=%d reason=audio_unavailable",
                word_index, start_ms,
            )
            return

        helper.seek_to(start_ms)
        helper.play()
        logger.info(
            "word_seek_success: index=%d start_ms=%d",
            word_index, start_ms,
        )

    def _rename_speaker_in_file(
        self, md_path: Path, old_name: str, new_name: str
    ) -> None:
        """Rename a speaker in a transcript .md file.

        Updates both the JSON metadata (words and segments arrays) and the
        markdown body speaker labels.
        """
        content = md_path.read_text(encoding="utf-8")

        footer_marker = "\n---\n\n<!-- METADATA:"
        marker_idx = content.find(footer_marker)
        if marker_idx == -1:
            logger.warning("No metadata footer in %s — cannot rename speaker", md_path)
            return

        md_body = content[:marker_idx]
        metadata_text = content[marker_idx + len(footer_marker):]
        if metadata_text.strip().endswith(" -->"):
            metadata_text = metadata_text.strip()[:-len(" -->")]

        try:
            data = json.loads(metadata_text)
        except json.JSONDecodeError as exc:
            logger.warning("Malformed metadata in %s: %s", md_path, exc)
            return

        # Update speaker names in words
        for word in data.get("words", []):
            if word.get("speaker_id") == old_name:
                word["speaker_id"] = new_name

        # Update speaker names in segments
        for seg in data.get("segments", []):
            if seg.get("speaker") == old_name:
                seg["speaker"] = new_name

        # Update speaker names in markdown body
        md_body = md_body.replace(f"**{old_name}**", f"**{new_name}**")

        # Write back
        new_content = md_body + footer_marker + json.dumps(data, indent=2) + " -->\n"
        md_path.write_text(new_content, encoding="utf-8")
        logger.info("Renamed speaker '%s' -> '%s' in %s", old_name, new_name, md_path)

    def _propagate_rename_to_signatures(
        self, md_path: Path, old_name: str, new_name: str
    ) -> None:
        """Propagate a speaker rename to the VoiceSignatureStore (best-effort).

        If the old speaker name has a saved embedding in the signature
        database (located in the same directory as the transcript file),
        saves the embedding under the new name and deletes the old entry.
        """
        try:
            from meetandread.speaker.signatures import VoiceSignatureStore
        except ImportError:
            logger.warning(
                "VoiceSignatureStore not available — skipping rename propagation"
            )
            return

        db_path = md_path.parent / "speaker_signatures.db"
        if not db_path.exists():
            # Try the default data directory
            try:
                from meetandread.audio.storage.paths import get_recordings_dir
                default_db = get_recordings_dir() / "speaker_signatures.db"
                if default_db.exists():
                    db_path = default_db
                else:
                    logger.info(
                        "No signature database found — speaker '%s' not in store",
                        old_name,
                    )
                    return
            except Exception:
                logger.info(
                    "No signature database found — speaker '%s' not in store",
                    old_name,
                )
                return

        try:
            with VoiceSignatureStore(db_path=str(db_path)) as store:
                profiles = store.load_signatures()
                old_profile = None
                for profile in profiles:
                    if profile.name == old_name:
                        old_profile = profile
                        break

                if old_profile is None:
                    logger.info(
                        "Speaker '%s' not found in signature store — no propagation needed",
                        old_name,
                    )
                    return

                store.save_signature(
                    new_name,
                    old_profile.embedding,
                    averaged_from_segments=old_profile.num_samples,
                )
                store.delete_signature(old_name)

                logger.info(
                    "Propagated rename '%s' -> '%s' to signature store at %s",
                    old_name, new_name, db_path,
                )
        except Exception as exc:
            logger.warning(
                "Failed to propagate rename to signature store: %s", exc,
            )

    def _on_scrub_clicked(self) -> None:
        """Handle Scrub button click — placeholder for S03 full scrub."""
        if self._is_scrubbing:
            return

        current = self._history_list.currentItem()
        if current is None:
            return

        md_path_str = current.data(Qt.ItemDataRole.UserRole)
        if not md_path_str:
            return
        md_path = Path(md_path_str)
        stem = md_path.stem

        # Check for WAV file
        try:
            from meetandread.audio.storage.paths import get_recordings_dir
            wav_path = get_recordings_dir() / f"{stem}.wav"
        except Exception:
            wav_path = md_path.parent.parent / "recordings" / f"{stem}.wav"

        if not wav_path.exists():
            parent = self.parent() if self.parent() else self
            QMessageBox.information(
                parent,
                "Cannot Scrub",
                "Cannot scrub — audio file missing.\n\n"
                "The original .wav recording file is required for re-transcription.",
            )
            return

        # Show model picker dialog
        dialog = self._create_scrub_dialog()
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        model_size = dialog._model_combo.currentData()
        if not model_size:
            return

        # Start the scrub
        self._start_scrub(wav_path, md_path, model_size)

    def _create_scrub_dialog(self) -> QDialog:
        """Create the model picker dialog for scrub."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Scrub Recording")
        dialog.setFixedSize(340, 180)
        p = current_palette()
        dialog.setStyleSheet(dialog_css(p))

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        title_label = QLabel("Re-transcribe with a different model:")
        title_label.setStyleSheet(f"font-weight: bold; color: {p.info}; font-size: 13px;")
        layout.addWidget(title_label)

        combo = QComboBox()
        combo.setStyleSheet(combo_box_css(p, accent_color=p.info))

        try:
            from meetandread.config import get_config
            _cfg = get_config()
            _bench_history = _cfg.transcription.benchmark_history
            _default_model = _cfg.transcription.postprocess_model_size
        except Exception:
            _bench_history = {}
            _default_model = "base"

        _model_order = ["tiny", "base", "small", "medium", "large"]
        _select_idx = 0
        for _i, _mn in enumerate(_model_order):
            _entry = _bench_history.get(_mn)
            if _entry and "wer" in _entry:
                _wer_pct = _entry["wer"] * 100
                _item_text = f"{_mn} — WER: {_wer_pct:.1f}%"
            else:
                _item_text = f"{_mn} (not benchmarked)"
            combo.addItem(_item_text, _mn)
            if _mn == _default_model:
                _select_idx = _i
        combo.setCurrentIndex(_select_idx)

        layout.addWidget(combo)
        dialog._model_combo = combo

        layout.addStretch()

        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
        )
        btn_box.setStyleSheet(action_button_css(p, "dialog"))
        btn_box.accepted.connect(dialog.accept)
        btn_box.rejected.connect(dialog.reject)
        layout.addWidget(btn_box)

        return dialog

    def _get_app_settings(self):
        """Get the current AppSettings from config."""
        try:
            from meetandread.config import get_config
            return get_config()
        except Exception:
            from meetandread.config.models import AppSettings
            return AppSettings()

    def _start_scrub(self, wav_path: Path, md_path: Path, model_size: str) -> None:
        """Start a ScrubRunner background re-transcription."""
        from meetandread.transcription.scrub import ScrubRunner

        self._scrub_model_size = model_size
        self._is_scrubbing = True
        self._is_comparison_mode = False

        self._scrub_original_html = self._history_viewer.toHtml()

        self._scrub_btn.setEnabled(False)
        self._scrub_btn.setText("Scrubbing... 0%")

        self._scrub_runner = ScrubRunner(
            settings=self._get_app_settings(),
            on_progress=self._on_scrub_progress,
            on_complete=self._on_scrub_complete,
        )
        self._scrub_sidecar_path = self._scrub_runner.scrub_recording(
            wav_path, md_path, model_size,
        )

    def _on_scrub_progress(self, pct: int) -> None:
        """Update scrub button text with progress percentage.

        Called from the ScrubRunner background thread — uses
        QTimer.singleShot to marshal the update to the GUI thread.
        """
        QTimer.singleShot(0, lambda: self._scrub_btn.setText(f"Scrubbing... {pct}%"))

    def _on_scrub_complete(self, sidecar_path: str, error: Optional[str]) -> None:
        """Handle scrub completion."""
        QTimer.singleShot(0, lambda: self._handle_scrub_complete(sidecar_path, error))

    def _handle_scrub_complete(self, sidecar_path: str, error: Optional[str]) -> None:
        """Process scrub completion on the GUI thread."""
        self._is_scrubbing = False
        self._scrub_btn.setEnabled(True)
        self._scrub_btn.setText("🔄 Scrub")

        if error:
            parent = self.parent() if self.parent() else self
            QMessageBox.warning(
                parent,
                "Scrub Failed",
                f"Re-transcription failed:\n\n{error}",
            )
            logger.error("Scrub failed: %s", error)
            return

        self._show_scrub_comparison(sidecar_path)

    def _show_scrub_comparison(self, sidecar_path: str) -> None:
        """Show side-by-side comparison of original vs scrubbed transcript."""
        sidecar = Path(sidecar_path)
        if not sidecar.exists():
            logger.warning("Sidecar not found for comparison: %s", sidecar_path)
            return

        self._is_comparison_mode = True
        self._scrub_sidecar_path = sidecar_path

        original_text = self._extract_transcript_body(
            self._current_history_md_path
        )
        scrubbed_text = self._extract_transcript_body(sidecar)

        html = f"""
        <html>
        <head><style>
            body {{ margin: 0; padding: 4px; background-color: #2a2a2a; color: #fff; font-size: 12px; }}
            .comparison {{ display: flex; gap: 8px; }}
            .column {{ flex: 1; }}
            .column-header {{
                font-weight: bold;
                padding: 4px 8px;
                border-radius: 4px 4px 0 0;
                font-size: 11px;
                text-align: center;
            }}
            .original .column-header {{ background-color: #37474F; color: #B0BEC5; }}
            .scrubbed .column-header {{ background-color: #1B5E20; color: #A5D6A7; }}
            .content {{
                padding: 6px 8px;
                background-color: #333;
                border-radius: 0 0 4px 4px;
                min-height: 50px;
                line-height: 1.4;
                white-space: pre-wrap;
                word-wrap: break-word;
            }}
        </style></head>
        <body>
        <div class="comparison">
            <div class="column original">
                <div class="column-header">Original</div>
                <div class="content">{_escape_html(original_text)}</div>
            </div>
            <div class="column scrubbed">
                <div class="column-header">Scrubbed ({_escape_html(self._scrub_model_size or "?")})</div>
                <div class="content">{_escape_html(scrubbed_text)}</div>
            </div>
        </div>
        </body></html>
        """

        self._history_viewer.setHtml(html)
        self._show_scrub_accept_reject()

    def _show_scrub_accept_reject(self) -> None:
        """Replace the scrub button with Accept/Reject during comparison mode."""
        self._scrub_btn.hide()

        if not hasattr(self, '_scrub_accept_btn'):
            self._scrub_accept_btn = QPushButton("✓ Accept")
            self._scrub_accept_btn.setObjectName("AethericHistoryActionButton")
            self._scrub_accept_btn.setProperty("action", "accept")
            self._scrub_accept_btn.setFixedHeight(26)
            p = current_palette()
            self._scrub_accept_btn.setStyleSheet(aetheric_history_action_button_css(p))
            self._scrub_accept_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self._scrub_accept_btn.clicked.connect(self._on_scrub_accept)

            self._scrub_reject_btn = QPushButton("✗ Reject")
            self._scrub_reject_btn.setObjectName("AethericHistoryActionButton")
            self._scrub_reject_btn.setProperty("action", "reject")
            self._scrub_reject_btn.setFixedHeight(26)
            self._scrub_reject_btn.setStyleSheet(aetheric_history_action_button_css(p))
            self._scrub_reject_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self._scrub_reject_btn.clicked.connect(self._on_scrub_reject)

            header_layout = self._history_detail_header.layout()
            delete_idx = header_layout.indexOf(self._delete_btn)
            header_layout.insertWidget(delete_idx, self._scrub_accept_btn)
            header_layout.insertWidget(delete_idx + 1, self._scrub_reject_btn)
        else:
            self._scrub_accept_btn.show()
            self._scrub_reject_btn.show()

    def _hide_scrub_accept_reject(self) -> None:
        """Hide Accept/Reject buttons and show the scrub button again."""
        if hasattr(self, '_scrub_accept_btn'):
            self._scrub_accept_btn.hide()
        if hasattr(self, '_scrub_reject_btn'):
            self._scrub_reject_btn.hide()
        self._scrub_btn.show()
        self._is_comparison_mode = False

    def _on_scrub_accept(self) -> None:
        """Accept the scrub result — promote sidecar to canonical transcript."""
        if self._current_history_md_path is None or self._scrub_model_size is None:
            return

        try:
            from meetandread.transcription.scrub import ScrubRunner
            ScrubRunner.accept_scrub(
                self._current_history_md_path, self._scrub_model_size,
            )
            logger.info(
                "Accepted scrub: %s model %s",
                self._current_history_md_path, self._scrub_model_size,
            )
        except FileNotFoundError:
            parent = self.parent() if self.parent() else self
            QMessageBox.warning(
                parent, "Accept Failed",
                "Sidecar file not found. It may have been deleted.",
            )
            self._hide_scrub_accept_reject()
            return
        except Exception as exc:
            parent = self.parent() if self.parent() else self
            QMessageBox.warning(
                parent, "Accept Failed", f"Could not accept scrub result:\n\n{exc}",
            )
            self._hide_scrub_accept_reject()
            return

        self._hide_scrub_accept_reject()
        self._refresh_after_scrub()

    def _on_scrub_reject(self) -> None:
        """Reject the scrub result — delete the sidecar file."""
        if self._current_history_md_path is None or self._scrub_model_size is None:
            return

        try:
            from meetandread.transcription.scrub import ScrubRunner
            ScrubRunner.reject_scrub(
                self._current_history_md_path, self._scrub_model_size,
            )
            logger.info(
                "Rejected scrub: %s model %s",
                self._current_history_md_path, self._scrub_model_size,
            )
        except Exception as exc:
            logger.warning("Error rejecting scrub: %s", exc)

        self._hide_scrub_accept_reject()
        self._refresh_after_scrub()

    def _refresh_after_scrub(self) -> None:
        """Refresh the history list and viewer after accept/reject."""
        md_path = self._current_history_md_path

        self._refresh_history()

        if md_path is not None:
            self._reselect_history_item(md_path)

        if md_path is not None and md_path.exists():
            html = self._render_history_transcript(md_path)
            if html is not None:
                self._history_viewer.setHtml(html)
            else:
                try:
                    content = md_path.read_text(encoding="utf-8")
                except OSError:
                    content = ""
                footer_marker = "\n---\n\n<!-- METADATA:"
                marker_idx = content.find(footer_marker)
                if marker_idx != -1:
                    content = content[:marker_idx]
                self._history_viewer.setMarkdown(_strip_confidence_percentages(content))
        else:
            self._history_viewer.clear()
            self._history_viewer.setPlaceholderText(
                "Select a recording to view its transcript",
            )

    def closeEvent(self, event):
        """Handle close event."""
        self.closed.emit()
        event.accept()
    
    def mousePressEvent(self, event):
        """Start dragging."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
    
    def mouseMoveEvent(self, event):
        """Handle dragging."""
        if self._dragging and self._drag_pos is not None:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()
    
    def mouseReleaseEvent(self, event):
        """Stop dragging."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            event.accept()


if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    
    panel = FloatingTranscriptPanel()
    panel.show_panel()
    
    # Test adding some content
    panel.update_segment("Hello", 85, 0, is_final=False)
    panel.update_segment("world", 90, 1, is_final=False)
    panel.update_segment("this is", 75, 2, is_final=True)
    
    # New phrase
    panel.update_segment("New phrase", 80, 0, phrase_start=True, is_final=False)
    panel.update_segment("here", 85, 1, is_final=True)
    
    sys.exit(app.exec())
