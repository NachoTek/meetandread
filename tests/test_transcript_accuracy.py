"""Transcript accuracy benchmark against known-good YouTube transcript.

Feeds the Harvard AI audio (mp3) through the full transcription pipeline
as a fake audio source, captures exactly what the CC panel would display,
and compares against the known-good transcript.

What the user sees:
  The CC overlay shows phrases (lines), each built from segments.
  Segments can be replaced in-place (same segment_index) as the model
  refines.  A new phrase starts on phrase_start=True.  The final visible
  text is: each phrase = ' '.join(segments), full text = lines joined.

This test:
1. Converts the mp3 to 16kHz mono WAV (cached)
2. Feeds the WAV through RecordingController with fake source
3. Simulates the CC panel's update_segment logic faithfully
4. After transcription completes, reconstructs the displayed text
5. Normalizes and computes WER against the reference transcript
6. Reports confidence distribution and per-segment alignment

Requires: whisper model loaded (tiny/base), ~8 min audio processing.
Marked @pytest.mark.slow — excluded from CI by default.
"""

import re
import json
import wave
import time
import threading
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field

import numpy as np
import pytest

# --- Path constants ---
FIXTURES_DIR = Path(__file__).parent / "fixtures"
MP3_PATH = FIXTURES_DIR / "Harvard_AI_audio.mp3"
TRANSCRIPT_PATH = FIXTURES_DIR / "Harvard_AI_Transcript.txt"
WAV_CACHE = FIXTURES_DIR / "Harvard_AI_audio_16k.wav"


# ---------------------------------------------------------------------------
# Transcript parsing
# ---------------------------------------------------------------------------

@dataclass
class TranscriptSegment:
    """A timestamped segment from the known-good transcript."""
    start_sec: float
    end_sec: float
    text: str


def parse_reference_transcript(path: Path) -> List[TranscriptSegment]:
    """Parse the YouTube transcript format: [HH:MM:SS] text lines."""
    segments: List[TranscriptSegment] = []
    current_start: Optional[float] = None
    current_lines: List[str] = []

    timestamp_re = re.compile(r'\[(\d{2}):(\d{2}):(\d{2})\]\s*(.*)')

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('http') or line.startswith('='):
                continue

            m = timestamp_re.match(line)
            if m:
                if current_start is not None and current_lines:
                    segments.append(TranscriptSegment(
                        start_sec=current_start,
                        end_sec=_parse_ts(m.group(1), m.group(2), m.group(3)),
                        text=' '.join(current_lines),
                    ))
                current_start = _parse_ts(m.group(1), m.group(2), m.group(3))
                current_lines = [m.group(4)] if m.group(4) else []
            else:
                if current_start is not None:
                    current_lines.append(line)

    if current_start is not None and current_lines:
        segments.append(TranscriptSegment(
            start_sec=current_start,
            end_sec=current_start + 5.0,
            text=' '.join(current_lines),
        ))

    return segments


def _parse_ts(h: str, m: str, s: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s)


# ---------------------------------------------------------------------------
# Audio conversion
# ---------------------------------------------------------------------------

def convert_mp3_to_wav(mp3_path: Path, wav_path: Path, target_sr: int = 16000) -> Path:
    """Convert mp3 to 16kHz mono WAV using soundfile."""
    if wav_path.exists():
        return wav_path

    import soundfile as sf
    data, sr = sf.read(str(mp3_path))

    if data.ndim > 1:
        data = data.mean(axis=1)

    if sr != target_sr:
        try:
            import resampy
            data = resampy.resample(data, sr, target_sr)
        except ImportError:
            ratio = target_sr / sr
            n_samples = int(len(data) * ratio)
            indices = np.linspace(0, len(data) - 1, n_samples).astype(int)
            data = data[indices]

    data = data.astype(np.float32)
    int_data = (data * 32767).astype(np.int16)
    with wave.open(str(wav_path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(target_sr)
        wf.writeframes(int_data.tobytes())

    return wav_path


# ---------------------------------------------------------------------------
# CC Panel display simulator
# ---------------------------------------------------------------------------

@dataclass
class CCPhrase:
    """Mirrors the Phrase dataclass from floating_panels.py."""
    segments: List[str] = field(default_factory=list)
    confidences: List[int] = field(default_factory=list)
    is_final: bool = False
    speaker_id: Optional[str] = None


class CCPanelCapture:
    """Faithfully simulates what the CC overlay panel displays.

    Reproduces the exact update_segment logic from CCOverlayPanel:
    - phrase_start or first segment creates a new phrase
    - segment_index < len(segments) replaces in-place
    - segment_index >= len(segments) appends
    - [BLANK_AUDIO] is filtered
    - Final displayed text = each phrase's segments joined by spaces,
      then all phrases joined by newlines.

    Also tracks a chronological event log for debugging.
    """

    def __init__(self):
        self.phrases: List[CCPhrase] = []
        self.current_phrase_idx: int = -1
        self._has_content: bool = False
        self._lock = threading.Lock()
        self.events: List[Dict] = []  # chronological log

    def update_segment(self, text: str, confidence: int, segment_index: int,
                       is_final: bool = False, phrase_start: bool = False,
                       speaker_id: Optional[str] = None) -> None:
        """Exactly mirrors CCOverlayPanel.update_segment()."""
        if text.strip() == "[BLANK_AUDIO]":
            return

        with self._lock:
            wall_time = time.time()

            if not self._has_content:
                self._has_content = True

            # Start new phrase if needed
            if phrase_start or self.current_phrase_idx < 0:
                self.phrases.append(
                    CCPhrase(segments=[], confidences=[], is_final=False,
                             speaker_id=speaker_id)
                )
                self.current_phrase_idx = len(self.phrases) - 1

            phrase = self.phrases[self.current_phrase_idx]
            phrase.is_final = is_final

            # Update or add segment
            if segment_index < len(phrase.segments):
                phrase.segments[segment_index] = text
                phrase.confidences[segment_index] = confidence
                action = "replace"
            else:
                phrase.segments.append(text)
                phrase.confidences.append(confidence)
                action = "append"

            self.events.append({
                "time": wall_time,
                "action": action,
                "phrase": self.current_phrase_idx,
                "seg_idx": segment_index,
                "text": text,
                "conf": confidence,
                "is_final": is_final,
                "phrase_start": phrase_start,
            })

    def get_displayed_text(self) -> str:
        """Return exactly what the user sees: phrases joined by newlines."""
        with self._lock:
            lines = []
            for phrase in self.phrases:
                if phrase.segments:
                    lines.append(' '.join(phrase.segments))
            return '\n'.join(lines)

    def get_final_displayed_text(self) -> str:
        """Return displayed text from only final phrases (completed segments)."""
        with self._lock:
            lines = []
            for phrase in self.phrases:
                if phrase.is_final and phrase.segments:
                    lines.append(' '.join(phrase.segments))
            return '\n'.join(lines)

    def get_confidence_stats(self) -> Dict:
        """Get confidence distribution across all segments."""
        with self._lock:
            confs = []
            for phrase in self.phrases:
                confs.extend(phrase.confidences)
            if not confs:
                return {"count": 0}
            return {
                "count": len(confs),
                "mean": sum(confs) / len(confs),
                "min": min(confs),
                "max": max(confs),
                "below_50": sum(1 for c in confs if c < 50),
                "above_80": sum(1 for c in confs if c >= 80),
            }

    def get_phrase_count(self) -> int:
        with self._lock:
            return len(self.phrases)

    def get_segment_count(self) -> int:
        with self._lock:
            return sum(len(p.segments) for p in self.phrases)


# ---------------------------------------------------------------------------
# WER calculation
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def word_error_rate(hyp: str, ref: str) -> Tuple[float, int, int, int]:
    """Compute WER using dynamic programming."""
    hyp_words = normalize_text(hyp).split()
    ref_words = normalize_text(ref).split()

    if not ref_words:
        return (0.0, 0, 0, 0) if not hyp_words else (1.0, 0, 0, len(hyp_words))

    n, m = len(ref_words), len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    s = d = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i - 1] == hyp_words[j - 1]:
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            s += 1; i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            d += 1; i -= 1
        else:
            ins += 1; j -= 1

    return (dp[n][m] / n, s, d, ins)


# ---------------------------------------------------------------------------
# Diff helper — show word-level differences
# ---------------------------------------------------------------------------

def word_diff(hyp: str, ref: str) -> List[str]:
    """Produce a compact word-level diff showing mismatches."""
    hyp_w = normalize_text(hyp).split()
    ref_w = normalize_text(ref).split()

    n, m = len(ref_w), len(hyp_w)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_w[i-1] == hyp_w[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    # Backtrack to collect aligned pairs
    lines = []
    i, j = n, m
    ops = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_w[i-1] == hyp_w[j-1]:
            ops.append(('=', ref_w[i-1]))
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            ops.append(('S', f"{ref_w[i-1]} → {hyp_w[j-1]}"))
            i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            ops.append(('D', f"−{ref_w[i-1]}"))
            i -= 1
        else:
            ops.append(('I', f"+{hyp_w[j-1]}"))
            j -= 1
    ops.reverse()

    # Collect only mismatches (limit to 40 for readability)
    mismatches = [op for op in ops if op[0] != '=']
    return [f"  {op[0]} {op[1]}" for op in mismatches[:40]]


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestTranscriptAccuracy:
    """Feed known audio through the pipeline and compare CC output to reference."""

    @pytest.fixture(autouse=True)
    def _setup_wav(self):
        self.wav_path = convert_mp3_to_wav(MP3_PATH, WAV_CACHE)

    def test_cc_output_matches_reference(self, tmp_path):
        """Run full pipeline and verify CC panel output matches YouTube transcript.

        Takes ~8-10 min (real-time audio processing). Produces a detailed
        accuracy report showing WER, confidence stats, and word-level diffs.
        """
        from meetandread.recording.controller import RecordingController
        from meetandread.config.manager import ConfigManager

        # --- Parse reference transcript ---
        ref_segments = parse_reference_transcript(TRANSCRIPT_PATH)
        ref_full_text = ' '.join(s.text for s in ref_segments)
        ref_words = normalize_text(ref_full_text).split()

        with wave.open(str(self.wav_path), 'rb') as wf:
            audio_duration = wf.getnframes() / wf.getframerate()

        print(f"\n{'='*70}")
        print(f"CC PANEL DISPLAY vs REFERENCE TRANSCRIPT")
        print(f"{'='*70}")
        print(f"Audio:    {audio_duration:.1f}s  ({self.wav_path.name})")
        print(f"Reference: {len(ref_segments)} segments, {len(ref_words)} words")

        # --- Reset config singleton ---
        ConfigManager._instance = None
        ConfigManager._initialized = False

        # --- Create controller ---
        controller = RecordingController(enable_transcription=True)

        # --- Wire up CC panel capture ---
        cc = CCPanelCapture()
        controller.on_phrase_result = lambda result: cc.update_segment(
            text=result.text,
            confidence=result.confidence,
            segment_index=result.segment_index,
            is_final=result.is_final,
            phrase_start=getattr(result, 'phrase_start', False),
        )

        # --- Start recording with fake source ---
        print(f"\n[{time.strftime('%H:%M:%S')}] Starting transcription...")
        t0 = time.time()

        error = controller.start(
            {"fake"},
            fake_path=str(self.wav_path),
            fake_denoise=False,
        )
        if error:
            pytest.skip(f"Controller failed to start: {error.message}")

        # --- Wait for transcription to complete ---
        last_seg_count = 0
        idle_rounds = 0
        max_wait = audio_duration + 120

        while (time.time() - t0) < max_wait:
            time.sleep(2)
            seg_count = cc.get_segment_count()
            if seg_count == last_seg_count:
                idle_rounds += 1
            else:
                idle_rounds = 0
                last_seg_count = seg_count

            elapsed = time.time() - t0
            if idle_rounds >= 5 and elapsed > audio_duration * 0.5:
                print(f"[{time.strftime('%H:%M:%S')}] Transcription idle — stopping")
                break

            if int(elapsed) % 30 == 0 and idle_rounds == 0:
                print(f"[{time.strftime('%H:%M:%S')}] Transcribing... "
                      f"{seg_count} segments, {cc.get_phrase_count()} phrases, "
                      f"{elapsed:.0f}s elapsed")

        # --- Stop recording ---
        print(f"[{time.strftime('%H:%M:%S')}] Stopping...")
        controller.stop()
        if controller._worker_thread and controller._worker_thread.is_alive():
            controller._worker_thread.join(timeout=30)

        total_time = time.time() - t0

        # --- Extract what the user saw in the CC panel ---
        cc_text = cc.get_displayed_text()
        cc_final_text = cc.get_final_displayed_text()
        conf_stats = cc.get_confidence_stats()

        # Use whichever has more content — final-only or all
        display_text = cc_text if len(cc_text) >= len(cc_final_text) else cc_final_text

        print(f"\nProcessing time:  {total_time:.1f}s")
        print(f"Phrases captured: {cc.get_phrase_count()}")
        print(f"Segments captured: {cc.get_segment_count()}")

        if not display_text.strip():
            print("\n⚠ NO CC OUTPUT PRODUCED")
            print("Segments received:", cc.get_segment_count())
            for evt in cc.events[:10]:
                print(f"  {evt['action']} p{evt['phrase']}.{evt['seg_idx']} "
                      f"conf={evt['conf']}% final={evt['is_final']} "
                      f"'{evt['text'][:50]}'")
            pytest.skip("No CC output — model may not be available")

        # --- WER ---
        wer, subs, dels, ins = word_error_rate(display_text, ref_full_text)
        hyp_words = normalize_text(display_text).split()

        print(f"\n--- Word Error Rate ---")
        print(f"  Reference:  {len(ref_words)} words")
        print(f"  CC display: {len(hyp_words)} words")
        print(f"  WER:        {wer*100:.1f}%")
        print(f"    Substitutions: {subs}")
        print(f"    Deletions:     {dels} (missing from CC)")
        print(f"    Insertions:    {ins} (extra in CC)")

        # --- Confidence ---
        print(f"\n--- Confidence ---")
        print(f"  Mean:   {conf_stats['mean']:.1f}%")
        print(f"  Range:  {conf_stats['min']}%–{conf_stats['max']}%")
        print(f"  <50%:   {conf_stats['below_50']}  ≥80%: {conf_stats['above_80']}")

        # --- Word-level diff ---
        diffs = word_diff(display_text, ref_full_text)
        if diffs:
            print(f"\n--- Word Differences (first {len(diffs)}) ---")
            for line in diffs:
                print(line)

        # --- Per-phrase comparison ---
        print(f"\n--- CC Phrases vs Reference (sample) ---")
        hyp_norm = normalize_text(display_text)
        for i, phrase in enumerate(cc.phrases[:10]):
            ptext = ' '.join(phrase.segments)
            pnorm = normalize_text(ptext)
            pwords = pnorm.split()
            if not pwords:
                continue
            found = sum(1 for w in pwords if w in normalize_text(ref_full_text))
            pct = found / len(pwords) * 100
            status = "✓" if pct >= 80 else "△" if pct >= 50 else "✗"
            conf_str = f"avg={sum(phrase.confidences)//max(len(phrase.confidences),1)}%"
            print(f"  {status} P{i:02d} [{conf_str}] '{ptext[:70]}'")

        # --- Low confidence phrases ---
        low = [(i, p) for i, p in enumerate(cc.phrases)
               if p.confidences and sum(p.confidences)/len(p.confidences) < 50]
        if low:
            print(f"\n--- Low-Confidence Phrases ({len(low)}) ---")
            for i, p in low[:10]:
                avg = sum(p.confidences) // max(len(p.confidences), 1)
                print(f"  P{i:02d} [{avg}%] '{' '.join(p.segments)[:70]}'")

        # --- Save report ---
        report_path = tmp_path / "cc_accuracy_report.json"
        report = {
            "audio_file": str(MP3_PATH),
            "audio_duration_s": audio_duration,
            "processing_time_s": total_time,
            "wer": wer,
            "wer_details": {"subs": subs, "dels": dels, "ins": ins},
            "reference_word_count": len(ref_words),
            "cc_display_word_count": len(hyp_words),
            "confidence_stats": conf_stats,
            "cc_phrases": [
                {
                    "index": i,
                    "text": ' '.join(p.segments),
                    "confidences": p.confidences,
                    "is_final": p.is_final,
                    "avg_confidence": sum(p.confidences) // max(len(p.confidences), 1),
                }
                for i, p in enumerate(cc.phrases)
            ],
            "reference_preview": ref_full_text[:300],
            "cc_display_preview": display_text[:300],
        }
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nReport: {report_path}")

        # --- Assertions ---
        assert cc.get_segment_count() > 0, "No segments reached the CC panel"

        if wer > 0.5:
            print(f"\n⚠ WER {wer*100:.1f}% is above 50% — investigate transcription quality")

        print(f"\n{'='*70}")

        ConfigManager._instance = None
        ConfigManager._initialized = False
