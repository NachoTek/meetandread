"""Post-processing starts at app startup, not first record-start.

Bug report (QA on #61/#62): recordings showed a 'Queued' pill but were
never processed automatically — no status other than the pill.  Cause:
the PostProcessingQueue (whose ``start()`` runs pending-job recovery,
dependency-repair conversion, and the Stalled requeue scan) was created
only on the first record-start.  Until the user recorded, Stalled
Recordings showed the display-time 'Queued' fallback while nothing was
scheduled and nothing ran.

Fix: ``RecordingController.initialize_post_processing()`` — called from
``main()`` at app startup via ``MeetAndReadWidget`` — creates and starts
the queue when Post-processing is enabled, so Stalled/dependency-repaired
Recordings flow through the queue while the app is idle.
"""

from unittest.mock import Mock, patch

from meetandread.config.models import AppSettings
from meetandread.recording.controller import RecordingController


def _bare_controller(settings=None, transcription_enabled=True):
    controller = RecordingController.__new__(RecordingController)
    controller.enable_transcription = transcription_enabled
    controller._post_processor = None
    controller._config_manager = Mock()
    controller._config_manager.get_settings.return_value = (
        settings or AppSettings()
    )
    return controller


class TestInitializePostProcessing:
    def test_creates_and_starts_queue_when_enabled(self):
        controller = _bare_controller()

        with patch(
            "meetandread.recording.controller.PostProcessingQueue"
        ) as queue_cls:
            controller.initialize_post_processing()

        queue_cls.assert_called_once()
        instance = queue_cls.return_value
        instance.start.assert_called_once()
        assert controller._post_processor is instance

    def test_existing_queue_is_not_recreated(self):
        controller = _bare_controller()
        existing = Mock()
        controller._post_processor = existing

        with patch(
            "meetandread.recording.controller.PostProcessingQueue"
        ) as queue_cls:
            controller.initialize_post_processing()

        queue_cls.assert_not_called()
        existing.start.assert_not_called()  # already running

    def test_disabled_post_processing_creates_no_queue(self):
        settings = AppSettings()
        settings.transcription.enable_postprocessing = False
        controller = _bare_controller(settings=settings)

        with patch(
            "meetandread.recording.controller.PostProcessingQueue"
        ) as queue_cls:
            controller.initialize_post_processing()

        queue_cls.assert_not_called()
        assert controller._post_processor is None

    def test_transcription_disabled_creates_no_queue(self):
        controller = _bare_controller(transcription_enabled=False)

        with patch(
            "meetandread.recording.controller.PostProcessingQueue"
        ) as queue_cls:
            controller.initialize_post_processing()

        queue_cls.assert_not_called()

    def test_failures_never_block_startup(self):
        controller = _bare_controller()
        controller._config_manager.get_settings = Mock(
            side_effect=RuntimeError("config store unreadable")
        )

        # Must not raise.
        controller.initialize_post_processing()
        assert controller._post_processor is None


class TestWidgetDelegation:
    def test_widget_delegates_to_controller(self):
        from types import MethodType

        from meetandread.widgets.main_widget import MeetAndReadWidget

        widget = MeetAndReadWidget.__new__(MeetAndReadWidget)
        widget._controller = Mock()
        widget.initialize_post_processing = MethodType(
            MeetAndReadWidget.initialize_post_processing, widget
        )

        widget.initialize_post_processing()

        widget._controller.initialize_post_processing.assert_called_once()
