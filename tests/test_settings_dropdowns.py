"""Tests for Settings tab model dropdowns (Live Model + Post Process Model).

Validates that the two QComboBox dropdowns replace the old radio buttons,
show all 5 models with WER annotations, and persist selections to config.
"""

import pytest
from unittest.mock import patch, MagicMock
from PyQt6.QtWidgets import QApplication, QComboBox

from meetandread.config.models import AppSettings, TranscriptionSettings
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
    return panel


class TestPopulateModelDropdown:
    """Test _populate_model_dropdown helper method."""

    @patch("meetandread.config.get_config")
    def test_five_models_listed(self, mock_get_config, settings_empty, qapp):
        """Dropdown should list all 5 model sizes."""
        mock_get_config.return_value = settings_empty

        panel = _make_panel()
        combo = QComboBox()
        panel._populate_model_dropdown(combo, "realtime_model_size")

        assert combo.count() == 5
        model_ids = [combo.itemData(i) for i in range(5)]
        assert model_ids == ["tiny", "base", "small", "medium", "large"]

    @patch("meetandread.config.get_config")
    def test_wer_annotation_displayed(self, mock_get_config, settings_with_history, qapp):
        """Benchmarked models show 'WER: X.X%' in dropdown text."""
        mock_get_config.return_value = settings_with_history

        panel = _make_panel()
        combo = QComboBox()
        panel._populate_model_dropdown(combo, "realtime_model_size")

        # tiny is benchmarked: WER 25%
        assert "WER: 25.0%" in combo.itemText(0)
        # base is benchmarked: WER 17.35%
        assert "WER: 17.3%" in combo.itemText(1)
        # small is not benchmarked
        assert "not benchmarked" in combo.itemText(2)
        # medium is not benchmarked
        assert "not benchmarked" in combo.itemText(3)

    @patch("meetandread.config.get_config")
    def test_current_selection_set_from_config(self, mock_get_config, settings_with_history, qapp):
        """Dropdown current index matches config's model size."""
        mock_get_config.return_value = settings_with_history

        panel = _make_panel()

        # realtime_model_size = "base" -> index 1
        live_combo = QComboBox()
        panel._populate_model_dropdown(live_combo, "realtime_model_size")
        assert live_combo.currentIndex() == 1
        assert live_combo.currentData() == "base"

        # postprocess_model_size = "small" -> index 2
        pp_combo = QComboBox()
        panel._populate_model_dropdown(pp_combo, "postprocess_model_size")
        assert pp_combo.currentIndex() == 2
        assert pp_combo.currentData() == "small"


class TestDropdownConfigPersistence:
    """Test that dropdown selection changes persist to config."""

    @patch("meetandread.config.save_config")
    @patch("meetandread.config.set_config")
    @patch("meetandread.config.get_config")
    def test_live_model_change_saves_config(self, mock_get_config, mock_set, mock_save, settings_empty, qapp):
        """Selecting a live model calls set_config and save_config."""
        mock_get_config.return_value = settings_empty
        mock_save.return_value = True

        panel = _make_panel()
        panel._live_model_combo = QComboBox()
        panel._populate_model_dropdown(panel._live_model_combo, "realtime_model_size")

        # Mock model_changed.emit to avoid QWidget init requirement
        panel.model_changed = MagicMock()

        # Simulate selecting "small" (index 2)
        panel._live_model_combo.setCurrentIndex(2)
        panel._on_live_model_changed(2)

        mock_set.assert_called_with("transcription.realtime_model_size", "small")
        mock_save.assert_called_once()

    @patch("meetandread.config.save_config")
    @patch("meetandread.config.set_config")
    @patch("meetandread.config.get_config")
    def test_postprocess_model_change_saves_config(self, mock_get_config, mock_set, mock_save, settings_empty, qapp):
        """Selecting a post-process model calls set_config and save_config."""
        mock_get_config.return_value = settings_empty
        mock_save.return_value = True

        panel = _make_panel()
        panel._postprocess_model_combo = QComboBox()
        panel._populate_model_dropdown(panel._postprocess_model_combo, "postprocess_model_size")

        # Simulate selecting "medium" (index 3)
        panel._postprocess_model_combo.setCurrentIndex(3)
        panel._on_postprocess_model_changed(3)

        mock_set.assert_called_with("transcription.postprocess_model_size", "medium")
        mock_save.assert_called_once()

    @patch("meetandread.config.save_config")
    @patch("meetandread.config.set_config")
    @patch("meetandread.config.get_config")
    def test_live_model_emits_model_changed_signal(self, mock_get_config, mock_set, mock_save, settings_empty, qapp):
        """Live model change emits model_changed signal."""
        mock_get_config.return_value = settings_empty
        mock_save.return_value = True

        panel = _make_panel()
        panel._live_model_combo = QComboBox()
        panel._populate_model_dropdown(panel._live_model_combo, "realtime_model_size")

        panel.model_changed = MagicMock()
        panel._live_model_combo.setCurrentIndex(2)
        panel._on_live_model_changed(2)

        panel.model_changed.emit.assert_called_with("small")


class TestUpdateBenchmarkDisplay:
    """Test update_benchmark_display refreshes dropdowns after benchmark."""

    @patch("meetandread.config.save_config")
    @patch("meetandread.config.set_config")
    @patch("meetandread.config.get_config")
    def test_benchmark_display_updates_config_and_refreshes(
        self, mock_get_config, mock_set, mock_save, settings_empty, qapp
    ):
        """update_benchmark_display writes WER to config and refreshes dropdowns."""
        mock_get_config.return_value = settings_empty
        mock_save.return_value = True

        panel = _make_panel()
        panel._live_model_combo = QComboBox()
        panel._postprocess_model_combo = QComboBox()
        panel._populate_model_dropdown(panel._live_model_combo, "realtime_model_size")
        panel._populate_model_dropdown(panel._postprocess_model_combo, "postprocess_model_size")

        # Before: small shows "not benchmarked"
        assert "not benchmarked" in panel._live_model_combo.itemText(2)

        # Update with benchmark results
        panel.update_benchmark_display({"base": 0.173, "small": 0.142})

        # set_config called with updated benchmark_history
        set_calls = mock_set.call_args_list
        history_call = [c for c in set_calls if c[0][0] == "transcription.benchmark_history"]
        assert len(history_call) == 1
        updated_history = history_call[0][0][1]
        assert "base" in updated_history
        assert abs(updated_history["base"]["wer"] - 0.173) < 0.001
        assert "small" in updated_history
        assert abs(updated_history["small"]["wer"] - 0.142) < 0.001

        # After refresh: get_config returns updated settings with history
        updated_settings = AppSettings(
            transcription=TranscriptionSettings(
                benchmark_history={
                    "base": {"wer": 0.173, "timestamp": "2026-04-26T12:00:00"},
                    "small": {"wer": 0.142, "timestamp": "2026-04-26T12:00:00"},
                }
            )
        )
        mock_get_config.return_value = updated_settings
        panel._refresh_dropdown_wer()

        # Verify WER now appears in text
        assert "WER: 14.2%" in panel._live_model_combo.itemText(2)

    @patch("meetandread.config.get_config")
    def test_all_models_show_not_benchmarked_initially(self, mock_get_config, settings_empty, qapp):
        """When no benchmark_history exists, all items show 'not benchmarked'."""
        mock_get_config.return_value = settings_empty

        panel = _make_panel()
        combo = QComboBox()
        panel._populate_model_dropdown(combo, "realtime_model_size")

        for i in range(5):
            assert "not benchmarked" in combo.itemText(i)
