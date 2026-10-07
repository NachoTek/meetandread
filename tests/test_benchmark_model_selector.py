"""Tests for Performance tab benchmark model selector (T04).

Validates:
- Benchmark model dropdown shows all 5 models with WER annotations
- _on_benchmark_clicked uses the selected model, not config default
- _on_benchmark_complete persists per-model WER to config.benchmark_history
- _refresh_benchmark_model_combo updates dropdown text after benchmark
- Benchmark history display shows per-model format
"""

import pytest
from unittest.mock import patch, MagicMock, call
from PyQt6.QtWidgets import QApplication, QComboBox

from meetandread.config.models import AppSettings, TranscriptionSettings
from meetandread.performance.benchmark import BenchmarkResult
from meetandread.widgets.floating_panels import FloatingSettingsPanel


# ---------------------------------------------------------------------------
# Qt application fixture (session-scoped)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def qapp():
    """Provide a QApplication for the test session."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def settings_with_history():
    """Create AppSettings with benchmark history for some models."""
    return AppSettings(
        transcription=TranscriptionSettings(
            realtime_model_size="base",
            postprocess_model_size="small",
            benchmark_history={
                "tiny": {"wer": 0.25, "timestamp": "2026-04-26T12:00:00"},
                "base": {"wer": 0.1735, "timestamp": "2026-04-26T12:01:00"},
            },
        )
    )


@pytest.fixture
def settings_empty():
    """Create AppSettings with no benchmark history."""
    return AppSettings(
        transcription=TranscriptionSettings(
            realtime_model_size="tiny",
            postprocess_model_size="base",
        )
    )


def _make_panel():
    """Create a lightweight FloatingSettingsPanel for testing (no full __init__)."""
    panel = FloatingSettingsPanel.__new__(FloatingSettingsPanel)
    panel._controller = None
    panel._tray_manager = None
    panel._main_widget = None
    panel._resource_monitor = MagicMock()
    panel._metrics_timer = MagicMock()
    panel._benchmark_runner = None
    panel._benchmark_history = []
    panel._perf_tab_active = False
    # Set up combo boxes that _refresh methods expect
    panel._live_model_combo = QComboBox()
    panel._postprocess_model_combo = QComboBox()
    panel._benchmark_model_combo = QComboBox()
    panel._benchmark_btn = MagicMock()
    panel._benchmark_history_label = MagicMock()
    panel._wer_label = MagicMock()
    return panel


class TestBenchmarkModelCombo:
    """Test benchmark model dropdown in Performance tab."""

    @patch("meetandread.config.get_config")
    def test_combo_has_five_models(self, mock_get_config, settings_empty, qapp):
        """Benchmark model combo lists all 5 model sizes."""
        mock_get_config.return_value = settings_empty

        panel = _make_panel()
        panel._refresh_benchmark_model_combo()

        assert panel._benchmark_model_combo.count() == 5
        model_ids = [panel._benchmark_model_combo.itemData(i) for i in range(5)]
        assert model_ids == ["tiny", "base", "small", "medium", "large"]

    @patch("meetandread.config.get_config")
    def test_combo_shows_wer_annotations(self, mock_get_config, settings_with_history, qapp):
        """Benchmarked models show WER in the combo text."""
        mock_get_config.return_value = settings_with_history

        panel = _make_panel()
        panel._refresh_benchmark_model_combo()

        # tiny is benchmarked: WER 25%
        assert "WER: 25.0%" in panel._benchmark_model_combo.itemText(0)
        # base is benchmarked: WER 17.35%
        assert "WER: 17.3%" in panel._benchmark_model_combo.itemText(1)
        # small is not benchmarked
        assert "not benchmarked" in panel._benchmark_model_combo.itemText(2)

    @patch("meetandread.config.get_config")
    def test_combo_defaults_to_live_model(self, mock_get_config, settings_with_history, qapp):
        """Benchmark model combo defaults to current live model from config."""
        mock_get_config.return_value = settings_with_history

        panel = _make_panel()
        panel._refresh_benchmark_model_combo()

        # realtime_model_size = "base" -> index 1
        assert panel._benchmark_model_combo.currentIndex() == 1
        assert panel._benchmark_model_combo.currentData() == "base"

    @patch("meetandread.config.get_config")
    def test_combo_preserves_selection_on_refresh(self, mock_get_config, settings_with_history, qapp):
        """After refresh, the combo keeps its current model selection."""
        mock_get_config.return_value = settings_with_history

        panel = _make_panel()
        panel._refresh_benchmark_model_combo()

        # Select "medium" (index 3)
        panel._benchmark_model_combo.setCurrentIndex(3)
        assert panel._benchmark_model_combo.currentData() == "medium"

        # Refresh — should keep "medium" selected
        panel._refresh_benchmark_model_combo()
        assert panel._benchmark_model_combo.currentData() == "medium"

    @patch("meetandread.config.get_config")
    def test_combo_updates_wer_after_benchmark(self, mock_get_config, settings_empty, qapp):
        """After a model is benchmarked, refresh shows its WER."""
        mock_get_config.return_value = settings_empty

        panel = _make_panel()
        panel._refresh_benchmark_model_combo()

        # Initially: all show "not benchmarked"
        assert "not benchmarked" in panel._benchmark_model_combo.itemText(2)

        # Simulate benchmark data now in config
        updated_settings = AppSettings(
            transcription=TranscriptionSettings(
                benchmark_history={
                    "small": {"wer": 0.14, "timestamp": "2026-04-26T14:00:00"},
                }
            )
        )
        mock_get_config.return_value = updated_settings
        panel._refresh_benchmark_model_combo()

        # Now "small" shows WER
        assert "WER: 14.0%" in panel._benchmark_model_combo.itemText(2)


class TestBenchmarkClickedModelSelection:
    """Test _on_benchmark_clicked uses selected model from combo."""

    @patch("meetandread.widgets.floating_panels.BenchmarkRunner")
    @patch("meetandread.transcription.engine.WhisperTranscriptionEngine")
    def test_uses_combo_model_not_config(
        self, mock_engine_cls, mock_runner_cls, qapp
    ):
        """Benchmark uses the model selected in the combo, not config default."""
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_runner = MagicMock()
        mock_runner.is_running = False
        mock_runner_cls.return_value = mock_runner

        panel = _make_panel()
        # Populate combo and select "small"
        panel._benchmark_model_combo.clear()
        for m in ["tiny", "base", "small", "medium", "large"]:
            panel._benchmark_model_combo.addItem(m, m)
        panel._benchmark_model_combo.setCurrentIndex(2)  # small

        panel._on_benchmark_clicked()

        # Engine created with "small"
        mock_engine_cls.assert_called_once_with(model_size="small")
        mock_engine.load_model.assert_called_once()

    @patch("meetandread.widgets.floating_panels.BenchmarkRunner")
    @patch("meetandread.transcription.engine.WhisperTranscriptionEngine")
    def test_uses_large_model_when_selected(
        self, mock_engine_cls, mock_runner_cls, qapp
    ):
        """Benchmark uses 'large' when selected in combo."""
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_runner = MagicMock()
        mock_runner.is_running = False
        mock_runner_cls.return_value = mock_runner

        panel = _make_panel()
        panel._benchmark_model_combo.clear()
        for m in ["tiny", "base", "small", "medium", "large"]:
            panel._benchmark_model_combo.addItem(m, m)
        panel._benchmark_model_combo.setCurrentIndex(4)  # large

        panel._on_benchmark_clicked()

        mock_engine_cls.assert_called_once_with(model_size="large")


class TestBenchmarkCompletePersistence:
    """Test _on_benchmark_complete persists per-model results to config."""

    @patch("meetandread.config.save_config")
    @patch("meetandread.config.set_config")
    @patch("meetandread.config.get_config")
    def test_persists_per_model_wer(
        self, mock_get_config, mock_set, mock_save, settings_empty, qapp
    ):
        """Benchmark result for 'base' model persists to config.benchmark_history."""
        mock_get_config.return_value = settings_empty

        panel = _make_panel()
        panel._refresh_benchmark_model_combo()

        result = BenchmarkResult(
            wer=0.173,
            total_audio_s=10.0,
            total_latency_s=5.0,
            throughput_ratio=2.0,
            model_info={"model_size": "base", "device": "cpu"},
        )

        panel._on_benchmark_complete(result)

        # set_config called with benchmark_history containing "base"
        history_call = mock_set.call_args_list
        history_set = [c for c in history_call if c[0][0] == "transcription.benchmark_history"]
        assert len(history_set) == 1
        updated_history = history_set[0][0][1]
        assert "base" in updated_history
        assert abs(updated_history["base"]["wer"] - 0.173) < 0.001
        assert "timestamp" in updated_history["base"]
        mock_save.assert_called_once()

    @patch("meetandread.config.save_config")
    @patch("meetandread.config.set_config")
    @patch("meetandread.config.get_config")
    def test_persists_small_model_wer(
        self, mock_get_config, mock_set, mock_save, settings_empty, qapp
    ):
        """Benchmark result for 'small' model persists separately."""
        mock_get_config.return_value = settings_empty

        panel = _make_panel()

        result = BenchmarkResult(
            wer=0.14,
            total_audio_s=10.0,
            total_latency_s=8.0,
            throughput_ratio=1.25,
            model_info={"model_size": "small", "device": "cpu"},
        )

        panel._on_benchmark_complete(result)

        history_set = [c for c in mock_set.call_args_list if c[0][0] == "transcription.benchmark_history"]
        updated_history = history_set[0][0][1]
        assert "small" in updated_history
        assert abs(updated_history["small"]["wer"] - 0.14) < 0.001

    @patch("meetandread.config.save_config")
    @patch("meetandread.config.set_config")
    @patch("meetandread.config.get_config")
    def test_preserves_existing_history_on_new_benchmark(
        self, mock_get_config, mock_set, mock_save, settings_with_history, qapp
    ):
        """Benchmarking a new model preserves existing entries in history."""
        mock_get_config.return_value = settings_with_history

        panel = _make_panel()

        # Benchmark "small" model — "tiny" and "base" should be preserved
        result = BenchmarkResult(
            wer=0.14,
            total_audio_s=10.0,
            total_latency_s=8.0,
            throughput_ratio=1.25,
            model_info={"model_size": "small", "device": "cpu"},
        )

        panel._on_benchmark_complete(result)

        history_set = [c for c in mock_set.call_args_list if c[0][0] == "transcription.benchmark_history"]
        updated_history = history_set[0][0][1]
        # "tiny" and "base" from fixture preserved
        assert "tiny" in updated_history
        assert "base" in updated_history
        # "small" added
        assert "small" in updated_history

    @patch("meetandread.config.save_config")
    @patch("meetandread.config.set_config")
    @patch("meetandread.config.get_config")
    def test_history_label_shows_per_model_format(
        self, mock_get_config, mock_set, mock_save, settings_empty, qapp
    ):
        """Benchmark history display shows model name per entry."""
        mock_get_config.return_value = settings_empty

        panel = _make_panel()

        result = BenchmarkResult(
            wer=0.173,
            total_audio_s=10.0,
            total_latency_s=5.0,
            throughput_ratio=2.0,
            model_info={"model_size": "base", "device": "cpu"},
        )

        panel._on_benchmark_complete(result)

        # Check the label was updated with model name
        label_text = panel._benchmark_history_label.setText.call_args[0][0]  # pyright: ignore[reportAttributeAccessIssue]  # intentional mock seam
        assert "base" in label_text
        assert "WER 17.3%" in label_text
        assert "Speed 2.0x" in label_text

    @patch("meetandread.config.save_config")
    @patch("meetandread.config.set_config")
    @patch("meetandread.config.get_config")
    def test_error_result_does_not_persist(
        self, mock_get_config, mock_set, mock_save, settings_empty, qapp
    ):
        """Failed benchmark does not write to config."""
        mock_get_config.return_value = settings_empty

        panel = _make_panel()

        result = BenchmarkResult(
            error="Model not loaded",
        )

        panel._on_benchmark_complete(result)

        # set_config should not be called for benchmark_history
        history_set = [c for c in mock_set.call_args_list if c[0][0] == "transcription.benchmark_history"]
        assert len(history_set) == 0
