"""Speaker identity linking — domain logic for linking and renaming identities in transcripts.

Provides pure-file-mutation functions that update transcript metadata (words,
segments, speaker_matches) and markdown body headings, plus best-effort
propagation to the VoiceSignatureStore.  No Qt imports.

Public functions:
    - link_identity(md_path, raw_label, identity_name)
    - rename_identity(md_path, old_name, new_name)

The module-level functions in floating_panels.py (the UI adapters) call into
these and handle dialog lifecycle.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from meetandread.transcription import transcript_footer
from meetandread.utils.file_utils import atomic_write

logger = logging.getLogger(__name__)


def _norm_label(s: str) -> str:
    s = s.lower()
    m = re.match(r"^(spk)(\d+)$", s)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    return s


def _resolve_unknown_speaker_labels(md_path: Path) -> list[str]:
    try:
        content = md_path.read_text(encoding="utf-8")
        data = transcript_footer.parse(content)
        if data is None:
            return []

        labels: set[str] = set()
        for w in data.get("words", []):
            sid = w.get("speaker_id")
            if sid is not None:
                labels.add(sid)
        return sorted(labels)
    except Exception:
        return []


def _replace_speaker_heading(md_body: str, old_label: str, new_label: str) -> str:
    """Replace a ``**label**`` speaker heading line with ``**new_label**``.

    The replacement is anchored to a line that consists solely of the bold
    label (plus optional trailing whitespace), so bold mentions of the same
    name inside segment body text are left untouched (issue #42).
    """
    pattern = rf"(?m)^\*\*{re.escape(old_label)}\*\*\s*$"
    return re.sub(pattern, f"**{new_label}**", md_body)


def _resolve_speaker_matches_key(
    sm: Dict[str, Any], raw_label: str
) -> Optional[str]:
    """Find the actual key in speaker_matches for *raw_label*, handling
    case/format variants (e.g. ``SPK_0`` vs ``spk0``).

    Returns the matching key, or None.
    """
    if raw_label in sm:
        return raw_label
    norm_target = _norm_label(raw_label)
    for existing_key in sm:
        if _norm_label(existing_key) == norm_target:
            return existing_key
    return None


def _dedup_speaker_matches_keys(
    sm: Dict[str, Any], canonical_key: str
) -> None:
    """Remove stale duplicate keys that differ only by casing or raw-label format."""
    norm_canonical = _norm_label(canonical_key)
    for dup_key in list(sm.keys()):
        if dup_key != canonical_key and _norm_label(dup_key) == norm_canonical:
            del sm[dup_key]


def _set_speaker_match(
    sm: Dict[str, Any],
    key: str,
    identity_name: str,
    existing_info: Optional[Dict[str, Any]],
) -> None:
    """Update or create a speaker_matches entry for *key*."""
    if existing_info is not None and "score" in existing_info and "confidence" in existing_info:
        sm[key] = {
            "identity_name": identity_name,
            "score": existing_info["score"],
            "confidence": existing_info["confidence"],
        }
    else:
        sm[key] = {
            "identity_name": identity_name,
            "score": 1.0,
            "confidence": "manual",
        }


def _resolve_signature_db(md_path: Path) -> Optional[Path]:
    """Resolve the signature DB path, preferring one next to the transcript
    and falling back to the default recordings dir.

    Returns None when no database exists.
    """
    db_path = md_path.parent / "speaker_signatures.db"
    if db_path.exists():
        return db_path
    try:
        from meetandread.audio.storage.paths import get_recordings_dir
        default_db = get_recordings_dir() / "speaker_signatures.db"
        if default_db.exists():
            return default_db
    except Exception:
        pass
    return None


def _propagate_signatures(
    md_path: Path,
    replacements: list[tuple[str, str]],
    *,
    verb: Literal["link", "rename"],
) -> None:
    """Shared db-resolution + store interaction for signature propagation.

    For each ``(old_name, new_name)`` pair, if a profile named ``old_name``
    exists in the resolved store, re-save its embedding under ``new_name`` and
    delete the old entry.  Best-effort: logs a warning and stops on store error.

    PII-safe: only counts and the operation *verb* appear in log records —
    never a Speaker name, a raw label, or a filesystem path.
    """
    try:
        from meetandread.speaker.signatures import VoiceSignatureStore
    except ImportError:
        logger.info("VoiceSignatureStore unavailable — skipping %s propagation", verb)
        return

    db_path = _resolve_signature_db(md_path)
    if db_path is None:
        logger.info("No signature database found — skipping %s propagation", verb)
        return

    try:
        with VoiceSignatureStore(db_path=str(db_path)) as store:
            profiles = store.load_signatures()
            profile_map = {p.name: p for p in profiles}
            propagated = 0
            for old_name, new_name in replacements:
                old_profile = profile_map.get(old_name)
                if old_profile is None:
                    continue
                store.save_signature(
                    new_name,
                    old_profile.embedding,
                    averaged_from_segments=old_profile.num_samples,
                )
                store.delete_signature(old_name)
                propagated += 1
    except Exception as exc:
        # Log only the exception class — the message could embed a path or name.
        logger.warning(
            "Failed to propagate %s signatures: %s", verb, type(exc).__name__
        )
        return

    if propagated:
        logger.info(
            "Propagated %s to signature store (%d profile(s))", verb, propagated
        )
    else:
        logger.info("No profiles matched for %s propagation", verb)


def _propagate_link_to_signature_store(
    md_path: Path, raw_label: str, identity_name: str
) -> None:
    """Link-path propagation: remap the raw label's embedding to *identity_name*.

    For ``__unknown__``, every SPK label found in the transcript is remapped.
    Delegates db-resolution + store interaction to ``_propagate_signatures``.
    """
    if raw_label == "__unknown__":
        resolved_labels = _resolve_unknown_speaker_labels(md_path)
        if not resolved_labels:
            logger.info(
                "No SPK labels found in transcript for __unknown__ — skipping propagation"
            )
            return
        replacements = [(label, identity_name) for label in resolved_labels]
    else:
        replacements = [(raw_label, identity_name)]

    _propagate_signatures(md_path, replacements, verb="link")


def link_identity(md_path: Path, raw_label: str, identity_name: str) -> None:
    """Link a raw speaker label to an identity name in a transcript.

    Updates the transcript .md file so that *raw_label* (e.g. ``SPK_0``) is
    replaced with *identity_name* (e.g. ``Alice``) in the JSON metadata
    (words, segments), the markdown body headings, and the ``speaker_matches``
    map.  Propagates to the ``VoiceSignatureStore`` best-effort.

    PII-safe: identity names are never logged.

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
    split_result = transcript_footer.split(content)
    if split_result is None:
        logger.warning("No metadata footer found — cannot link identity")
        return

    md_body, data = split_result

    matching_label = None if raw_label == "__unknown__" else raw_label

    words_updated = 0
    for word in data.get("words", []):
        if word.get("speaker_id") == matching_label:
            word["speaker_id"] = identity_name
            words_updated += 1

    segments_updated = 0
    for seg in data.get("segments", []):
        if seg.get("speaker_id") == matching_label:
            seg["speaker_id"] = identity_name
            segments_updated += 1
        if seg.get("speaker") == matching_label:
            seg["speaker"] = identity_name
            if seg.get("speaker_id") != matching_label:
                segments_updated += 1

    display_label = "Unknown Speaker" if raw_label == "__unknown__" else raw_label
    updated_body = _replace_speaker_heading(md_body, display_label, identity_name)

    if "speaker_matches" not in data:
        data["speaker_matches"] = {}

    match_key = "__unknown__" if raw_label == "__unknown__" else raw_label
    sm = data["speaker_matches"]

    actual_key = _resolve_speaker_matches_key(sm, match_key)
    if actual_key is None:
        actual_key = match_key

    existing = sm.get(actual_key)
    _set_speaker_match(sm, actual_key, identity_name, existing)

    _dedup_speaker_matches_keys(sm, actual_key)

    atomic_write(md_path, transcript_footer.join(updated_body, data))

    if raw_label != "__unknown__":
        _propagate_link_to_signature_store(md_path, raw_label, identity_name)


def rename_identity(md_path: Path, old_name: str, new_name: str) -> None:
    """Rename a speaker identity in a transcript.

    Updates both the JSON metadata (words and segments arrays) and the
    markdown body speaker labels from *old_name* to *new_name*.
    """
    content = md_path.read_text(encoding="utf-8")
    split_result = transcript_footer.split(content)
    if split_result is None:
        raise ValueError(f"No metadata footer found in {md_path}")

    md_body, data = split_result

    words_updated = 0
    for word in data.get("words", []):
        if word.get("speaker_id") == old_name:
            word["speaker_id"] = new_name
            words_updated += 1

    segments_updated = 0
    for seg in data.get("segments", []):
        if seg.get("speaker_id") == old_name:
            seg["speaker_id"] = new_name
            segments_updated += 1

    updated_body = _replace_speaker_heading(md_body, old_name, new_name)

    atomic_write(md_path, transcript_footer.join(updated_body, data))


def propagate_rename_to_signature_store(
    md_path: Path, old_name: str, new_name: str
) -> None:
    """Rename-path propagation: re-save the old name's embedding under the new
    name and delete the old entry.

    Delegates db-resolution + store interaction to ``_propagate_signatures``.
    """
    _propagate_signatures(md_path, [(old_name, new_name)], verb="rename")
