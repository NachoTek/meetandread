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

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

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
        marker = "\n---\n\n<!-- METADATA: "
        idx = content.find(marker)
        if idx < 0:
            return []
        data = json.loads(content[idx + len(marker):].rstrip(" -->\n"))

        labels: set[str] = set()
        for w in data.get("words", []):
            sid = w.get("speaker_id")
            if sid is not None:
                labels.add(sid)
        return sorted(labels)
    except Exception:
        return []


def _parse_metadata_footer(content: str) -> Optional[tuple[str, dict]]:
    """Split a transcript file into (md_body, metadata_dict).

    Returns None when no metadata footer is found or JSON is malformed.
    """
    footer_marker = "\n---\n\n<!-- METADATA: "
    marker_idx = content.find(footer_marker)
    if marker_idx == -1:
        return None

    md_body = content[:marker_idx]
    after_marker = content[marker_idx + len(footer_marker):]
    space_before_json = ""
    if after_marker.startswith(" "):
        space_before_json = " "
        after_marker = after_marker[1:]

    metadata_text = after_marker
    if metadata_text.strip().endswith(" -->"):
        metadata_text = metadata_text.strip()[:-len(" -->")]

    try:
        data = json.loads(metadata_text)
    except json.JSONDecodeError:
        logger.warning("Malformed metadata — leaving file unchanged")
        return None

    return md_body, data, footer_marker, space_before_json


def _rebuild_file(
    md_body: str,
    data: dict,
    footer_marker: str,
    space_before_json: str,
) -> str:
    updated_json = json.dumps(data, indent=2)
    return (
        md_body + footer_marker + space_before_json + updated_json + " -->\n"
    )


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


def _propagate_to_signature_store(
    md_path: Path, raw_label: str, identity_name: str
) -> None:
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

    resolved_labels: list[str] = []
    if raw_label == "__unknown__":
        resolved_labels = _resolve_unknown_speaker_labels(md_path)
        if not resolved_labels:
            logger.info(
                "No SPK labels found in transcript for __unknown__ — skipping propagation"
            )
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
                    logger.info(
                        "Raw speaker '%s' not found in signature store — skipping", label
                    )
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
    parsed = _parse_metadata_footer(content)
    if parsed is None:
        logger.warning("No metadata footer found — cannot link identity")
        return

    md_body, data, footer_marker, space_before_json = parsed

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
    updated_body = re.sub(
        re.escape(f"**{display_label}**"),
        f"**{identity_name}**",
        md_body,
    )

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

    atomic_write(md_path, _rebuild_file(updated_body, data, footer_marker, space_before_json))

    if raw_label != "__unknown__":
        _propagate_to_signature_store(md_path, raw_label, identity_name)


def rename_identity(md_path: Path, old_name: str, new_name: str) -> None:
    """Rename a speaker identity in a transcript.

    Updates both the JSON metadata (words and segments arrays) and the
    markdown body speaker labels from *old_name* to *new_name*.
    """
    content = md_path.read_text(encoding="utf-8")
    parsed = _parse_metadata_footer(content)
    if parsed is None:
        raise ValueError(f"No metadata footer found in {md_path}")

    md_body, data, footer_marker, space_before_json = parsed

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

    updated_body = re.sub(
        re.escape(f"**{old_name}**"),
        f"**{new_name}**",
        md_body,
    )

    atomic_write(md_path, _rebuild_file(updated_body, data, footer_marker, space_before_json))


def propagate_rename_to_signature_store(
    md_path: Path, old_name: str, new_name: str
) -> None:
    """Propagate a speaker rename to the VoiceSignatureStore (best-effort).

    If the old speaker name has a saved embedding, saves it under the new
    name and deletes the old entry.
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
            return

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
