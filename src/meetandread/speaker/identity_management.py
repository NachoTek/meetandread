"""Identity management service for speaker profiles.

Non-UI operations for the Settings → Identities tab: scan transcript
metadata for identity usage, rename/merge/delete identities while
keeping VoiceSignatureStore and transcript files consistent.

PII constraint: identity names are local PII.  This module's logger
messages use counts, operation labels, and short hashes — never raw
identity names.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Collection, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metadata footer markers (must match TranscriptStore.save_to_file)
# ---------------------------------------------------------------------------
_FOOTER_MARKER = "\n---\n\n<!-- METADATA: "
_FOOTER_END = " -->\n"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class IdentityManagementError(Exception):
    """Base exception for identity management operations."""


class RenameError(IdentityManagementError):
    """Raised when a rename operation fails."""


class MergeError(IdentityManagementError):
    """Raised when a merge operation fails."""


class DeleteError(IdentityManagementError):
    """Raised when a delete operation fails."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IdentityRecordingRef:
    """Reference to a single transcript that mentions an identity.

    Attributes:
        path: Absolute path to the transcript .md file.
        recording_count: Number of word-level mentions of this identity
            in the transcript (typically 1 per speaker segment group).
        last_modified: File mtime as a UNIX timestamp, or None if
            unavailable.
    """

    path: Path
    recording_count: int
    last_modified: Optional[float] = None


@dataclass
class IdentityUsage:
    """Aggregated usage information for a single identity.

    Attributes:
        identity_name: The speaker identity name.
        recordings: Transcript files that reference this identity.
        total_mentions: Sum of word-level speaker_id matches across
            all transcripts.
        last_activity: Latest file mtime among all recordings, or None.
    """

    identity_name: str
    recordings: List[IdentityRecordingRef] = field(default_factory=list)
    total_mentions: int = 0
    last_activity: Optional[float] = None

    @property
    def recording_count(self) -> int:
        """Number of distinct transcript files referencing this identity."""
        return len(self.recordings)


# ---------------------------------------------------------------------------
# Metadata parsing helpers
# ---------------------------------------------------------------------------


def parse_metadata_footer(content: str) -> Optional[Dict[str, Any]]:
    """Extract the JSON metadata dict from a transcript file's content.

    Returns None if the footer is missing or contains malformed JSON.
    """
    marker_idx = content.find(_FOOTER_MARKER)
    if marker_idx == -1:
        return None

    after_marker = content[marker_idx + len(_FOOTER_MARKER) :]
    metadata_text = after_marker
    if metadata_text.strip().endswith(" -->"):
        metadata_text = metadata_text.strip()[: -len(" -->")]
    else:
        # Try to find the closing --> even without trailing newline
        end_idx = metadata_text.find(" -->")
        if end_idx != -1:
            metadata_text = metadata_text[:end_idx]

    try:
        return json.loads(metadata_text)  # type: ignore[no-any-return]
    except (json.JSONDecodeError, ValueError):
        return None


def _rebuild_transcript(
    md_body: str, metadata: Dict[str, Any]
) -> str:
    """Rebuild a transcript file from markdown body and metadata dict."""
    return md_body + _FOOTER_MARKER + json.dumps(metadata, indent=2) + " -->\n"


# ---------------------------------------------------------------------------
# Scan identity usage
# ---------------------------------------------------------------------------


def scan_identity_usage(
    transcripts_dir: Path,
    identity_names: Collection[str],
) -> Dict[str, IdentityUsage]:
    """Scan transcript .md files for identity usage metadata.

    Reads all ``.md`` files in *transcripts_dir*, parses their metadata
    footers, and counts how many times each identity name appears as a
    ``speaker_id`` in words.  Malformed files are skipped with a warning.

    Args:
        transcripts_dir: Directory containing transcript .md files.
        identity_names: Identity names to look for (from VoiceSignatureStore).

    Returns:
        Mapping from identity name to IdentityUsage.  Names with no
        recordings still appear with empty recording lists.
    """
    # Initialise result for every requested identity (including those with
    # zero recordings) so callers can rely on the key existing.
    usage: Dict[str, IdentityUsage] = {
        name: IdentityUsage(identity_name=name) for name in identity_names
    }

    if not transcripts_dir.is_dir():
        logger.warning("Transcripts directory does not exist: %s", transcripts_dir)
        return usage

    md_files = sorted(transcripts_dir.glob("*.md"))
    if not md_files:
        return usage

    name_set = set(identity_names)
    skipped_count = 0

    for md_path in md_files:
        try:
            content = md_path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning(
                "Cannot read transcript (skipping): error=%s", exc
            )
            skipped_count += 1
            continue

        data = parse_metadata_footer(content)
        if data is None:
            skipped_count += 1
            continue

        # Count word-level mentions per identity in this transcript
        per_identity_counts: Dict[str, int] = {}
        for word in data.get("words", []):
            sid = word.get("speaker_id")
            if sid and sid in name_set:
                per_identity_counts[sid] = per_identity_counts.get(sid, 0) + 1

        if not per_identity_counts:
            continue

        # Get file mtime
        try:
            mtime = md_path.stat().st_mtime
        except OSError:
            mtime = None

        for name, count in per_identity_counts.items():
            ref = IdentityRecordingRef(
                path=md_path,
                recording_count=count,
                last_modified=mtime,
            )
            usage[name].recordings.append(ref)
            usage[name].total_mentions += count
            if mtime is not None:
                if (
                    usage[name].last_activity is None
                    or mtime > usage[name].last_activity
                ):
                    usage[name].last_activity = mtime

    if skipped_count:
        logger.info(
            "Identity scan: %d file(s) skipped (malformed/unreadable)",
            skipped_count,
        )

    logger.info(
        "Identity scan complete: %d identities, %d transcript(s)",
        len(identity_names),
        len(md_files),
    )
    return usage


# ---------------------------------------------------------------------------
# Transcript rewrite helpers
# ---------------------------------------------------------------------------


def replace_speaker_label_in_file(
    md_path: Path,
    old_label: str,
    new_label: str,
) -> int:
    """Replace an exact speaker label in a transcript file.

    Updates ``words[].speaker_id``, ``segments[].speaker_id``,
    ``segments[].speaker``, and markdown ``**label**`` headings.
    Only replaces when the value exactly equals *old_label* — no
    substring matching (so ``SPK_0`` won't touch ``SPK_01``).

    Returns:
        Number of word-level replacements made.

    Raises:
        OSError: If the file cannot be read or written.
        IdentityManagementError: If the metadata footer is missing or
            malformed and the file would need rewriting.
    """
    content = md_path.read_text(encoding="utf-8")

    marker_idx = content.find(_FOOTER_MARKER)
    if marker_idx == -1:
        raise IdentityManagementError(
            f"No metadata footer in {md_path} — cannot rewrite"
        )

    md_body = content[:marker_idx]
    after_marker = content[marker_idx + len(_FOOTER_MARKER) :]
    metadata_text = after_marker
    if metadata_text.strip().endswith(" -->"):
        metadata_text = metadata_text.strip()[: -len(" -->")]
    else:
        end_idx = metadata_text.find(" -->")
        if end_idx != -1:
            metadata_text = metadata_text[:end_idx]

    try:
        data = json.loads(metadata_text)
    except (json.JSONDecodeError, ValueError) as exc:
        raise IdentityManagementError(
            f"Malformed metadata in {md_path} — cannot rewrite"
        ) from exc

    # Replace in words — exact match only
    words_replaced = 0
    for word in data.get("words", []):
        if word.get("speaker_id") == old_label:
            word["speaker_id"] = new_label
            words_replaced += 1

    # Replace in segments — both speaker_id and speaker keys
    for seg in data.get("segments", []):
        if seg.get("speaker_id") == old_label:
            seg["speaker_id"] = new_label
        if seg.get("speaker") == old_label:
            seg["speaker"] = new_label

    # Replace in speaker_matches values
    for _raw_label, match_info in data.get("speaker_matches", {}).items():
        if isinstance(match_info, dict) and match_info.get("identity_name") == old_label:
            match_info["identity_name"] = new_label

    # Replace in markdown body — exact **label** only
    updated_body = re.sub(
        re.escape(f"**{old_label}**"),
        f"**{new_label}**",
        md_body,
    )

    new_content = _rebuild_transcript(updated_body, data)
    md_path.write_text(new_content, encoding="utf-8")
    return words_replaced


def _find_transcripts_with_label(
    transcripts_dir: Path, label: str
) -> List[Path]:
    """Find transcript .md files that contain *label* as a speaker_id.

    Skips files with malformed metadata.
    """
    results: List[Path] = []
    if not transcripts_dir.is_dir():
        return results

    for md_path in sorted(transcripts_dir.glob("*.md")):
        try:
            content = md_path.read_text(encoding="utf-8")
        except OSError:
            continue
        data = parse_metadata_footer(content)
        if data is None:
            continue
        # Check words for exact match
        for word in data.get("words", []):
            if word.get("speaker_id") == label:
                results.append(md_path)
                break
    return results


# ---------------------------------------------------------------------------
# Rename identity
# ---------------------------------------------------------------------------


def rename_identity(
    store: Any,
    transcripts_dir: Path,
    old_name: str,
    new_name: str,
) -> None:
    """Rename a speaker identity across store and transcripts.

    Steps (rollback-minded):
    1. Validate inputs.
    2. Load old profile from store.
    3. Save old embedding under new name with old ``num_samples``.
    4. Delete old name from store.
    5. Rewrite matching transcript files.

    Raises:
        RenameError: On validation failure or if any step fails.
    """
    _validate_rename_inputs(old_name, new_name, store)

    # Load old profile
    profiles = store.load_signatures()
    old_profile = _find_profile(profiles, old_name)
    if old_profile is None:
        raise RenameError(f"Source identity not found in store")

    # Check target doesn't already exist
    existing_names = {p.name for p in profiles}
    if new_name in existing_names:
        raise RenameError("Target name already exists in store")

    # Save under new name (preserves embedding and num_samples)
    try:
        store.save_signature(
            new_name,
            old_profile.embedding,
            averaged_from_segments=old_profile.num_samples,
        )
    except Exception as exc:
        raise RenameError(f"Failed to save renamed profile: {exc}") from exc

    # Delete old
    try:
        store.delete_signature(old_name)
    except Exception as exc:
        # Attempt rollback: remove the new entry we just created
        try:
            store.delete_signature(new_name)
        except Exception:
            pass
        raise RenameError(f"Failed to delete old profile: {exc}") from exc

    # Rewrite transcripts
    _rewrite_transcripts_safely(
        transcripts_dir, old_name, new_name, operation="rename"
    )

    logger.info(
        "Renamed identity: %d transcript(s) rewritten",  # PII-safe: no names
        len(_find_transcripts_with_label(transcripts_dir, new_name)),
    )


def _validate_rename_inputs(
    old_name: str, new_name: str, store: Any
) -> None:
    """Validate rename inputs before any mutations."""
    if not old_name or not old_name.strip():
        raise RenameError("Source identity name must not be empty")
    if not new_name or not new_name.strip():
        raise RenameError("New identity name must not be empty")
    if new_name.strip() == old_name.strip():
        raise RenameError("New name must differ from old name")


# ---------------------------------------------------------------------------
# Merge identities
# ---------------------------------------------------------------------------


def merge_identities(
    store: Any,
    transcripts_dir: Path,
    source_name: str,
    target_name: str,
) -> None:
    """Merge source identity into target identity.

    Steps (rollback-minded):
    1. Validate inputs (including no self-merge).
    2. Load both profiles from store.
    3. Compute weighted average embedding.
    4. Save merged embedding under target with summed samples.
    5. Rewrite source transcript labels to target.
    6. Delete source from store.

    Raises:
        MergeError: On validation failure or if any step fails.
    """
    _validate_merge_inputs(source_name, target_name)

    profiles = store.load_signatures()
    source_profile = _find_profile(profiles, source_name)
    if source_profile is None:
        raise MergeError("Source identity not found in store")

    target_profile = _find_profile(profiles, target_name)
    if target_profile is None:
        raise MergeError("Target identity not found in store")

    # Weighted average embedding
    total_samples = source_profile.num_samples + target_profile.num_samples
    merged_embedding = (
        source_profile.embedding * source_profile.num_samples
        + target_profile.embedding * target_profile.num_samples
    ) / total_samples

    # Save merged embedding under target
    try:
        store.save_signature(
            target_name,
            merged_embedding.astype(np.float32),
            averaged_from_segments=total_samples,
        )
    except Exception as exc:
        raise MergeError(f"Failed to save merged profile: {exc}") from exc

    # Rewrite transcripts (source → target) before deleting source
    rewrite_errors = _rewrite_transcripts_safely(
        transcripts_dir, source_name, target_name, operation="merge"
    )

    # Delete source
    try:
        store.delete_signature(source_name)
    except Exception as exc:
        logger.warning(
            "Merge: failed to delete source from store after rewrite "
            "(target is already updated): error=%s",
            exc,
        )
        # Target was saved successfully, transcripts were rewritten —
        # not a fatal error.  Source will be orphaned but not corrupting.

    logger.info(
        "Merged identity into target: %d file(s) rewritten",
        len(rewrite_errors) if rewrite_errors else 0,
    )


def _validate_merge_inputs(source_name: str, target_name: str) -> None:
    """Validate merge inputs before any mutations."""
    if not source_name or not source_name.strip():
        raise MergeError("Source identity name must not be empty")
    if not target_name or not target_name.strip():
        raise MergeError("Target identity name must not be empty")
    if source_name.strip() == target_name.strip():
        raise MergeError("Cannot merge identity into itself")


# ---------------------------------------------------------------------------
# Delete identity
# ---------------------------------------------------------------------------


def delete_identity(
    store: Any,
    transcripts_dir: Path,
    name: str,
) -> None:
    """Delete a speaker identity from the store.

    Removes the voice signature from the SQLite database.  Transcript
    files that reference this identity are left unchanged — the identity
    name remains in their metadata as a historical record.

    Args:
        store: VoiceSignatureStore instance.
        transcripts_dir: Directory containing transcripts (used for
            reporting usage before delete).
        name: Identity name to delete.

    Raises:
        DeleteError: If the identity is not found or deletion fails.
    """
    if not name or not name.strip():
        raise DeleteError("Identity name must not be empty")

    # Check existence first
    profiles = store.load_signatures()
    if not any(p.name == name for p in profiles):
        raise DeleteError("Identity not found in store")

    try:
        deleted = store.delete_signature(name)
    except Exception as exc:
        raise DeleteError(f"Failed to delete identity: {exc}") from exc

    if not deleted:
        raise DeleteError("Identity not found in store (concurrent delete?)")

    logger.info("Deleted identity from store")  # PII-safe


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_profile(profiles: Sequence[Any], name: str) -> Any:
    """Find a profile by name in a list of SpeakerProfile objects."""
    for profile in profiles:
        if profile.name == name:
            return profile
    return None


def _rewrite_transcripts_safely(
    transcripts_dir: Path,
    old_label: str,
    new_label: str,
    *,
    operation: str,
) -> List[Path]:
    """Rewrite transcript files, collecting errors instead of aborting.

    Returns list of files that were successfully rewritten.

    Raises MergeError/RenameError only if ALL files that needed rewriting
    failed (i.e., no partial success was possible).
    """
    matching = _find_transcripts_with_label(transcripts_dir, old_label)
    if not matching:
        return []

    rewritten: List[Path] = []
    failed: List[str] = []

    for md_path in matching:
        try:
            replace_speaker_label_in_file(md_path, old_label, new_label)
            rewritten.append(md_path)
        except (OSError, IdentityManagementError) as exc:
            failed.append(f"{md_path}: {exc}")
            logger.warning(
                "%s: failed to rewrite transcript: error=%s",
                operation.capitalize(),
                exc,
            )

    if failed and not rewritten:
        error_cls = RenameError if operation == "rename" else MergeError
        raise error_cls(
            f"All transcript rewrites failed during {operation}: "
            f"{len(failed)} file(s)"
        )

    return rewritten
