"""Transcript canary registry tests (issue #108, ADR 0004).

The capture boundary's leak DETECTOR: during a capture run, the app
samples fragments of transcript text and recording titles into a
registry of hashed n-grams (never plaintext). At assembly, the
Diagnostics Bundle runs every component through the same n-grammer â€”
a match means transcript/title content leaked into a diagnostics
artifact, and assembly FAILS CLOSED.

Pure-logic, fast-lane (ADR 0001): writer and reader are stdlib-only.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

import meetandread.transcript_canary as tcan
from meetandread.transcript_canary import (
    CANARY_FILE_NAME,
    CANARY_NGRAM_SIZE,
    canary_ngrams,
    install_canary_registry,
    canary_installed,
    record_canary_text,
    read_canary_ngrams,
    scan_text_for_canary_hits,
)


@pytest.fixture(autouse=True)
def _reset_canary_state():
    """Isolate the process-global writer between tests (the #105/#106
    test pattern)."""
    tcan._writer = None
    yield
    if tcan._writer is not None:
        tcan._writer.close()
        tcan._writer = None


def _plain_ngrams(text: str, size: int = CANARY_NGRAM_SIZE):
    """Plaintext windows, mirroring the hasher's tokenization."""
    words = text.split()
    return {
        " ".join(words[i : i + size]).lower()
        for i in range(len(words) - size + 1)
    }


def _digest(plain_window: str) -> str:
    import hashlib

    return hashlib.sha256(plain_window.encode("utf-8")).hexdigest()[:16]


class TestCanaryNgrams:
    def test_ngrams_of_plain_sentence(self):
        grams = canary_ngrams("the quick brown fox jumped over", size=3)
        expected = {
            _digest(w)
            for w in _plain_ngrams(
                "the quick brown fox jumped over", size=3
            )
        }
        assert grams == expected
        assert len(grams) == 4

    def test_ngrams_are_hashed_not_plaintext(self):
        grams = canary_ngrams("sebastopol secret words here now", size=3)
        for gram in grams:
            assert "sebastopol" not in gram
            assert len(gram) == 16  # sha256 hexdigest, truncated

    def test_ngrams_deterministic(self):
        assert canary_ngrams("same text twice here now", size=3) == (
            canary_ngrams("same text twice here now", size=3)
        )

    def test_short_text_yields_no_ngrams(self):
        assert canary_ngrams("two words", size=3) == set()
        assert canary_ngrams("", size=3) == set()

    def test_whitespace_and_case_normalization(self):
        assert canary_ngrams("A  b   C d E f", size=3) == canary_ngrams(
            "a b c d e f", size=3
        )


class TestRegistryWriteRead:
    def test_file_name_is_contract(self):
        assert CANARY_FILE_NAME == "transcript_canary.jsonl"

    def test_install_creates_file_exclusively(self, tmp_path):
        path = install_canary_registry(tmp_path)
        assert path == tmp_path / CANARY_FILE_NAME
        assert path.exists()
        assert canary_installed() is True

    def test_second_install_same_dir_is_idempotent(self, tmp_path):
        p1 = install_canary_registry(tmp_path)
        p2 = install_canary_registry(tmp_path)
        assert p1 == p2

    def test_record_writes_hashed_grams_only(self, tmp_path):
        install_canary_registry(tmp_path)
        assert record_canary_text("Sebastopol canary spoken words recorded") > 0
        raw = (tmp_path / CANARY_FILE_NAME).read_text(encoding="utf-8")
        assert "Sebastopol" not in raw
        assert "sebastopol" not in raw
        for line in raw.splitlines():
            payload = json.loads(line)
            assert set(payload) == {"gram"}
            assert len(payload["gram"]) == 16

    def test_read_roundtrip(self, tmp_path):
        install_canary_registry(tmp_path)
        record_canary_text("alpha beta gamma delta epsilon")
        record_canary_text("epsilon zeta eta theta iota")
        grams = read_canary_ngrams(tmp_path)
        expected = canary_ngrams(
            "alpha beta gamma delta epsilon"
        ) | canary_ngrams("epsilon zeta eta theta iota")
        assert grams == expected

    def test_read_absent_dir_returns_empty(self, tmp_path):
        assert read_canary_ngrams(tmp_path) == set()

    def test_read_drops_corrupt_lines(self, tmp_path):
        d = tmp_path / "capture"
        d.mkdir()
        (d / CANARY_FILE_NAME).write_text(
            '{"gram": "aaaaaaaaaaaaaaaa"}\n'
            "not json at all\n"
            '{"gram": "bbbbbbbbbbbbbbbb"}\n',
            encoding="utf-8",
        )
        grams = read_canary_ngrams(d)
        assert grams == {"aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb"}

    def test_record_without_install_is_noop(self):
        assert record_canary_text("never recorded anywhere") == 0


class TestScanForHits:
    def test_clean_text_has_no_hits(self):
        hits = scan_text_for_canary_hits(
            "routine DEBUG record with no secrets",
            known={"0000000000000000"},
        )
        assert hits == set()

    def test_leaked_fragment_is_a_hit(self):
        secret = "Sebastopol canary spoken words recorded"
        known = canary_ngrams(secret)
        hits = scan_text_for_canary_hits(
            f"transcribe failed on {secret} mid-run", known=known
        )
        assert hits, "canary fragment in log must scan as a hit"
        assert hits <= known

    def test_scan_matches_across_line_and_record_boundaries(self):
        # A leak wrapped by log formatting (timestamps, extra
        # whitespace between words) still matches: tokenization is
        # over whitespace runs, not single spaces.
        secret = "quantum hedgehog prairie vocal chorus"
        known = canary_ngrams(secret)
        hits = scan_text_for_canary_hits(
            "2026-09-10 10:00:00 - DEBUG - segment text: "
            "quantum  hedgehog   prairie vocal chorus (chunk 3)",
            known=known,
        )
        assert hits

    def test_dedup_scan(self):
        secret = "wobbly llama sermon tape tonight"
        known = canary_ngrams(secret)
        text = f"{secret} and again {secret}"
        assert len(scan_text_for_canary_hits(text, known)) == len(known)


class TestDefensiveBoundary:
    def test_importing_transcript_canary_pulls_no_heavy_stack(self):
        import os as _os

        code = (
            "import sys, meetandread.transcript_canary; "
            "heavy = sorted(m for m in sys.modules "
            "if m.split('.')[0] in {'PyQt6','sounddevice','numpy',"
            "'psutil','sherpa_onnx','pyaudiowpatch','comtypes',"
            "'webrtcvad','scipy','pywhispercpp'}); "
            "print(heavy)"
        )
        env = dict(_os.environ)
        env["PYTHONPATH"] = str(
            Path(
                __import__(
                    "meetandread.transcript_canary", fromlist=["x"]
                ).__file__
            )
            .resolve()
            .parent.parent
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
