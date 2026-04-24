"""Tests for startup hardware warning dialog in main.py.

Covers: check_hardware_requirements with below-minimum, above-minimum, and error cases.
"""

import pytest
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, 'src')

from metamemory.hardware.detector import HardwareDetector, SystemSpecs


# --- Fixtures ---

def make_low_specs():
    """SystemSpecs below single-mode minimum (4GB RAM, 2 cores)."""
    return SystemSpecs(
        total_ram_gb=2.0,
        available_ram_gb=1.0,
        cpu_count_logical=1,
        cpu_count_physical=1,
        cpu_freq_mhz=1000.0,
        is_64bit=True,
        platform='Windows',
    )


def make_good_specs():
    """SystemSpecs meeting single-mode minimum (4GB RAM, 2 cores)."""
    return SystemSpecs(
        total_ram_gb=8.0,
        available_ram_gb=4.0,
        cpu_count_logical=4,
        cpu_count_physical=2,
        cpu_freq_mhz=2400.0,
        is_64bit=True,
        platform='Windows',
    )


def make_settings_mock(auto_detect=True):
    """Create a mock settings object with hardware.auto_detect_on_startup."""
    settings = MagicMock()
    settings.hardware.auto_detect_on_startup = auto_detect
    return settings


# --- Tests ---

class TestCheckHardwareRequirements:
    """Test check_hardware_requirements function."""

    @patch('metamemory.main.QMessageBox')
    @patch('metamemory.hardware.detector.HardwareDetector')
    @patch('metamemory.main.get_config')
    def test_below_minimum_shows_dialog(self, mock_get_config, MockDetector, MockQMessageBox):
        """Below-minimum specs should show a QMessageBox warning."""
        from metamemory.main import check_hardware_requirements

        detector = MockDetector.return_value
        detector.detect.return_value = make_low_specs()
        detector.has_minimum_requirements.return_value = False
        detector.get_warning_message.return_value = "RAM: 2.0GB (need 4GB+), CPU cores: 1 (need 2+)"

        mock_get_config.return_value = make_settings_mock(auto_detect=True)

        mock_msg_box = MagicMock()
        MockQMessageBox.return_value = mock_msg_box

        check_hardware_requirements()

        # Verify dialog was shown
        MockQMessageBox.assert_called_once()
        mock_msg_box.setWindowTitle.assert_called_once_with("Hardware Notice")
        mock_msg_box.exec.assert_called_once()

    @patch('metamemory.main.QMessageBox')
    @patch('metamemory.hardware.detector.HardwareDetector')
    @patch('metamemory.main.get_config')
    def test_above_minimum_no_dialog(self, mock_get_config, MockDetector, MockQMessageBox):
        """Above-minimum specs should NOT show a dialog."""
        from metamemory.main import check_hardware_requirements

        detector = MockDetector.return_value
        detector.detect.return_value = make_good_specs()
        detector.has_minimum_requirements.return_value = True

        mock_get_config.return_value = make_settings_mock(auto_detect=True)

        check_hardware_requirements()

        # QMessageBox should not be instantiated
        MockQMessageBox.assert_not_called()

    @patch('metamemory.main.QMessageBox')
    @patch('metamemory.main.get_config')
    def test_auto_detect_disabled_no_dialog(self, mock_get_config, MockQMessageBox):
        """When auto_detect_on_startup is False, no dialog should appear."""
        from metamemory.main import check_hardware_requirements

        mock_get_config.return_value = make_settings_mock(auto_detect=False)

        check_hardware_requirements()

        MockQMessageBox.assert_not_called()

    @patch('metamemory.main.get_config')
    def test_exception_does_not_crash(self, mock_get_config):
        """Exceptions during detection should be caught and not crash the app."""
        from metamemory.main import check_hardware_requirements

        mock_get_config.side_effect = RuntimeError("Config broken")

        # Should not raise — function catches all exceptions
        check_hardware_requirements()

    @patch('metamemory.main.QMessageBox')
    @patch('metamemory.hardware.detector.HardwareDetector')
    @patch('metamemory.main.get_config')
    def test_warning_message_used_as_informative_text(self, mock_get_config, MockDetector, MockQMessageBox):
        """The warning message from HardwareDetector should appear in InformativeText."""
        from metamemory.main import check_hardware_requirements

        warning_text = "RAM: 2.0GB (need 4GB+), CPU cores: 1 (need 2+)"
        detector = MockDetector.return_value
        detector.detect.return_value = make_low_specs()
        detector.has_minimum_requirements.return_value = False
        detector.get_warning_message.return_value = warning_text

        mock_get_config.return_value = make_settings_mock(auto_detect=True)

        mock_msg_box = MagicMock()
        MockQMessageBox.return_value = mock_msg_box

        check_hardware_requirements()

        # Verify the informative text contains the warning message
        mock_msg_box.setInformativeText.assert_called_once_with(warning_text)

    @patch('metamemory.main.QMessageBox')
    @patch('metamemory.hardware.detector.HardwareDetector')
    @patch('metamemory.main.get_config')
    def test_dialog_is_warning_icon(self, mock_get_config, MockDetector, MockQMessageBox):
        """Dialog should use Warning icon."""
        from metamemory.main import check_hardware_requirements

        detector = MockDetector.return_value
        detector.detect.return_value = make_low_specs()
        detector.has_minimum_requirements.return_value = False
        detector.get_warning_message.return_value = "Low specs"

        mock_get_config.return_value = make_settings_mock(auto_detect=True)

        mock_msg_box = MagicMock()
        MockQMessageBox.return_value = mock_msg_box

        check_hardware_requirements()

        # setIcon is called with whatever QMessageBox.Icon.Warning resolves to
        # (the mock replaces the class, so we just verify the call happened)
        mock_msg_box.setIcon.assert_called_once()
        call_args = mock_msg_box.setIcon.call_args
        # Verify the function used QMessageBox.Icon.Warning from the mock
        assert call_args[0][0] is MockQMessageBox.Icon.Warning

    @patch('metamemory.main.QMessageBox')
    @patch('metamemory.hardware.detector.HardwareDetector')
    @patch('metamemory.main.get_config')
    def test_dialog_has_ok_button_only(self, mock_get_config, MockDetector, MockQMessageBox):
        """Dialog should have only an OK button (informational, not a choice)."""
        from metamemory.main import check_hardware_requirements

        detector = MockDetector.return_value
        detector.detect.return_value = make_low_specs()
        detector.has_minimum_requirements.return_value = False
        detector.get_warning_message.return_value = "Low specs"

        mock_get_config.return_value = make_settings_mock(auto_detect=True)

        mock_msg_box = MagicMock()
        MockQMessageBox.return_value = mock_msg_box

        check_hardware_requirements()

        # setStandardButtons is called with QMessageBox.StandardButton.Ok
        mock_msg_box.setStandardButtons.assert_called_once()
        call_args = mock_msg_box.setStandardButtons.call_args
        assert call_args[0][0] is MockQMessageBox.StandardButton.Ok


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
