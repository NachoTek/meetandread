"""Public-interface tests for the Transcript Footer module.

The Transcript Footer is the JSON metadata block appended to a Transcript's
Markdown body.  This module owns every read and write of that format through
four operations: ``parse``, ``split``, ``join``, and ``strip``.

These tests exercise ONLY the four public operations.  They never reach for
private marker constants, internal helpers, or framing literals belonging to
the implementation.  Inputs that need a specific framing are constructed
inline so the tests remain stable when the private details change.
"""

import json

import pytest

from meetandread.transcription.transcript_footer import (
    join,
    parse,
    split,
    strip,
)


# ---------------------------------------------------------------------------
# Representative footer data
# ---------------------------------------------------------------------------

META = {
    "recording_start_time": "2026-07-01T10:00:00",
    "word_count": 2,
    "words": [
        {"text": "Hello", "start_time": 0.0, "end_time": 0.5, "speaker_id": "SPK_0"},
        {"text": "world", "start_time": 0.5, "end_time": 1.0, "speaker_id": "SPK_0"},
    ],
    "speaker_matches": {"SPK_0": {"identity_name": "David"}},
}


def _no_space_footer(body: str, metadata: dict) -> str:
    """Build a footer using the tolerated no-space form (no space after the
    metadata label).  Used only to construct parse/split inputs."""
    return (
        body
        + "\n\n---\n\n<!-- METADATA:"
        + json.dumps(metadata, indent=2)
        + " -->\n"
    )


def _crlf(content: str) -> str:
    """Rewrite a transcript with Windows CRLF line endings.

    Editors on Windows save transcripts with ``\\r\\n`` line endings; footer
    operations must recognise that framing without rewriting the file first.
    """
    return content.replace("\r\n", "\n").replace("\n", "\r\n")


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


class TestPublicInterface:
    """The module exposes the four core operations plus the Outcome interface."""

    _CORE = {"parse", "split", "join", "strip"}
    _OUTCOME = {
        "PostProcessOutcome",
        "outcome_from_block",
        "read_post_process_outcome",
        "write_post_process_outcome",
        "clear_post_process_outcome",
    }

    def test_all_exports_are_core_plus_outcome_operations(self):
        import meetandread.transcription.transcript_footer as mod

        assert set(mod.__all__) == self._CORE | self._OUTCOME

    def test_no_public_marker_constants(self):
        """Marker literals and framing strings stay private.

        A leaked marker would be a public string constant; the public
        operations are callables, so the only public bare strings are the
        documented Outcome vocabulary constants (issue #62).
        """
        import meetandread.transcription.transcript_footer as mod

        outcome_vocabulary = (
            mod.OUTCOME_STATUSES | mod.OUTCOME_STAGES | {mod.OUTCOME_KEY}
        )
        leaked_strings = {
            name
            for name, value in vars(mod).items()
            if not name.startswith("_")
            and isinstance(value, str)
            and value not in outcome_vocabulary
        }
        assert not leaked_strings, f"Public string constants leaked: {leaked_strings}"

    def test_all_four_are_callable(self):
        assert callable(parse)
        assert callable(split)
        assert callable(join)
        assert callable(strip)


# ---------------------------------------------------------------------------
# parse
# ---------------------------------------------------------------------------


class TestParse:
    def test_parses_canonical_footer(self):
        assert parse(join("# Body", META)) == META

    def test_parses_single_line_json(self):
        content = "# Body\n\n---\n\n<!-- METADATA: {\"key\": \"value\"} -->\n"
        assert parse(content) == {"key": "value"}

    def test_parses_empty_json_object(self):
        content = "# Body\n\n---\n\n<!-- METADATA: {} -->\n"
        assert parse(content) == {}

    def test_accepts_no_space_form(self):
        assert parse(_no_space_footer("# Body", META)) == META

    def test_tolerates_extra_leading_newlines(self):
        # Legacy files had a body ending in "\n" plus the "\n\n---\n\n"
        # separator, yielding several newlines before the horizontal rule.
        content = "# Body\n" + "\n\n---\n\n<!-- METADATA: " + json.dumps(META) + " -->\n"
        assert parse(content) == META

    def test_trailing_content_after_closer_is_ignored(self):
        content = join("# Body", META).removesuffix("\n") + "\ntrailing line\n"
        assert parse(content) == META

    def test_missing_marker_returns_none(self):
        assert parse("# Just markdown\n\nNo footer here.") is None

    def test_empty_content_returns_none(self):
        assert parse("") is None

    def test_whitespace_only_returns_none(self):
        assert parse("   \n\n  \n") is None

    def test_malformed_json_returns_none(self):
        content = "# Body\n\n---\n\n<!-- METADATA: {not valid json} -->\n"
        assert parse(content) is None

    def test_missing_closer_returns_none(self):
        content = "# Body\n\n---\n\n<!-- METADATA: {\"key\": \"value\"}\n"
        assert parse(content) is None

    def test_json_containing_closer_text(self):
        meta = {"note": "ends with --> here", "count": 42}
        assert parse(join("# Body", meta)) == meta

    def test_parses_crlf_footer(self):
        assert parse(_crlf(join("# Body", META))) == META


class TestParseSelectsLastFooter:
    """Body text resembling the footer marker must not shadow the real footer."""

    def test_earlier_marker_like_text_in_body(self):
        body = (
            "# Transcript\n\n"
            "The format looks like:\n"
            "\n\n---\n\n<!-- METADATA: fake earlier marker\n"
            "But that was just discussion.\n\n"
        )
        content = join(body, {"real": True, "word_count": 3})
        assert parse(content) == {"real": True, "word_count": 3}

    def test_complete_earlier_footer_is_ignored(self):
        body_with_fake = (
            "# Transcript\n\n"
            "Earlier footer:\n"
            "\n\n---\n\n<!-- METADATA: "
            + json.dumps({"fake": True})
            + " -->\n"
            + "More content after the fake.\n\n"
        )
        content = join(body_with_fake, {"real": True, "word_count": 10})
        result = parse(content)
        assert result == {"real": True, "word_count": 10}
        assert "fake" not in result

    def test_multiple_earlier_footers(self):
        parts = ["# Transcript\n\n"]
        for i in range(3):
            parts.append(f"Section {i}\n")
            parts.append(
                "\n\n---\n\n<!-- METADATA: "
                + json.dumps({"fake_index": i})
                + " -->\n"
            )
        parts.append("\n")
        body = "".join(parts)
        content = join(body, {"real": True})
        result = parse(content)
        assert result == {"real": True}
        assert "fake_index" not in result


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------


class TestSplit:
    def test_returns_body_and_metadata(self):
        assert split(join("# Body\n\nText.", META)) == ("# Body\n\nText.", META)

    def test_body_with_no_trailing_newline(self):
        assert split(join("Just one line", META)) == ("Just one line", META)

    def test_missing_footer_returns_none(self):
        assert split("# No footer") is None

    def test_missing_closer_returns_none(self):
        content = "# Body\n\n---\n\n<!-- METADATA: {\"k\": \"v\"}\n"
        assert split(content) is None

    def test_malformed_json_returns_none(self):
        content = "# Body\n\n---\n\n<!-- METADATA: {bad} -->\n"
        assert split(content) is None

    def test_accepts_no_space_form(self):
        assert split(_no_space_footer("# Body", META)) == ("# Body", META)

    def test_splits_crlf_footer(self):
        content = _crlf(join("# Body\n\nText.", META))
        assert split(content) == (_crlf("# Body\n\nText."), META)


# ---------------------------------------------------------------------------
# join + split round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "trailing",
    ["", "\n", "\n\n", "\n\n\n"],
    ids=["no_newline", "one_newline", "two_newlines", "many_newlines"],
)
class TestRoundTrip:
    def test_split_inverts_join(self, trailing):
        body = "# Transcript\n\nHello world." + trailing
        assert split(join(body, META)) == (body, META)

    def test_metadata_round_trips_through_join_and_parse(self, trailing):
        body = "# Transcript\n\nHello world." + trailing
        assert parse(join(body, META)) == META


class TestJoinFraming:
    """Canonical output keeps valid Markdown framing: a blank line before the
    horizontal rule, regardless of the body's trailing-newline state."""

    @pytest.mark.parametrize(
        "trailing",
        ["", "\n", "\n\n"],
        ids=["no_newline", "one_newline", "two_newlines"],
    )
    def test_blank_line_before_horizontal_rule(self, trailing):
        body = "# Heading\n\nA paragraph." + trailing
        joined = join(body, {"k": "v"})
        # The rule sits on its own line, preceded by a blank line, so it is a
        # thematic break rather than a setext heading underline.
        assert "\n\n---\n" in joined

    def test_body_is_not_mutated_by_join(self):
        body = "# Body"
        joined = join(body, {"k": "v"})
        assert joined.startswith(body)


# ---------------------------------------------------------------------------
# strip
# ---------------------------------------------------------------------------


class TestStrip:
    def test_returns_body_without_decoding_json(self):
        assert strip(join("# Body\n\nReadable text.", META)) == "# Body\n\nReadable text."

    def test_returns_body_when_json_is_malformed(self):
        content = "# Body\n\nReadable text.\n\n---\n\n<!-- METADATA: {bad json} -->\n"
        assert strip(content) == "# Body\n\nReadable text."

    def test_returns_body_when_closer_is_missing(self):
        content = "# Body\n\nReadable text.\n\n---\n\n<!-- METADATA: {\"k\": \"v\"}\n"
        assert strip(content) == "# Body\n\nReadable text."

    def test_no_footer_returns_content_unchanged(self):
        content = "# Just markdown\n\nNo footer here."
        assert strip(content) == content

    def test_empty_content_returns_empty(self):
        assert strip("") == ""

    def test_strips_crlf_footer(self):
        content = _crlf(join("# Body\n\nReadable text.", META))
        assert strip(content) == _crlf("# Body\n\nReadable text.")

    def test_selects_last_footer(self):
        body = (
            "# Transcript\n\n"
            "Intro mentions:\n"
            "\n\n---\n\n<!-- METADATA: {\"fake\": true} -->\n"
            "More body.\n\n"
        )
        content = join(body, {"real": True})
        assert strip(content) == body
