---
status: investigating
trigger: "Model Selection Persistence - Issue #2 from Phase 2 UAT: 'It does not persist the setting.'"
created: 2026-02-10T00:00:00Z
updated: 2026-02-10T00:00:00Z
---

## Current Focus
hypothesis: Model selection setting is not being saved to disk because the FloatingSettingsPanel UI is not emitting the model_changed signal when user clicks radio buttons, and there's no code that writes the selection to ConfigManager
test: Verify that button clicks don't emit the model_changed signal and no save logic exists
expecting: Find that radio buttons need to emit model_changed.emit(model_id) and main_widget needs to connect this signal and save on close
next_action: Read the model_changed signal definition and check if it's connected anywhere

## Symptoms
expected: When user selects a model in settings panel and closes the app, the selection should persist and be loaded on restart
actual: Setting does not persist (as reported)
errors: None reported
reproduction: Select a model in settings panel, restart application, model selection is not retained
started: Phase 2 UAT - Issue #2

## Eliminated
(empty)

## Evidence
- timestamp: 2026-02-10
  checked: Settings model definition (models.py)
  found: ModelSettings dataclass has realtime_model_size and enhancement_model_size fields with defaults
  implication: Data structure exists, persistence infrastructure is ready

- timestamp: 2026-02-10
  checked: Persistence layer (persistence.py)
  found: SettingsPersistence.save_settings() method exists and writes to JSON atomically
  implication: Save infrastructure works, just needs to be called

- timestamp: 2026-02-10
  checked: ConfigManager (manager.py)
  found: ConfigManager.save() method exists and calls _persistence.save_settings()
  implication: Save method exists and works, needs to be triggered by UI

- timestamp: 2026-02-10
  checked: FloatingSettingsPanel UI (floating_panels.py lines 440-450)
  found: Radio buttons created but NO code to emit model_changed signal
  implication: User clicks change UI state but nothing happens with ConfigManager

- timestamp: 2026-02-10
  checked: Model changed signal connection (floating_panels.py line 382)
  found: Signal defined as `model_changed = pyqtSignal(str)` but NEVER EMITTED
  implication: No code listens to model changes

- timestamp: 2026-02-10
  checked: Signal usage in main_widget
  found: model_changed signal is never connected to any handler
  implication: Even if signal was emitted, nothing would happen

- timestamp: 2026-02-10
  checked: closeEvent in main_widget (main_widget.py lines 602-606)
  found: Only saves widget position, NO save_config() call
  implication: Settings not saved on application close

## Resolution
root_cause: FloatingSettingsPanel radio buttons never emit the model_changed signal, and main_widget never saves config on close

**Two critical missing pieces:**

1. **FloatingSettingsPanel.__init__ (floating_panels.py:440-450):**
   - Radio buttons created but NO code to emit model_changed.emit(model_id)
   - Missing: Toggle handlers that emit signal when selected

2. **MainWidget (main_widget.py):**
   - model_changed signal never connected to any handler
   - closeEvent only saves position, never calls save_config()

**Persistence architecture exists and works:**
- ModelSettings dataclass (models.py:12-42) - has realtime_model_size field
- SettingsPersistence.save_settings() (persistence.py:198-247) - writes to JSON atomically
- ConfigManager.save() (manager.py:206-227) - calls persistence.save_settings()
- Default config path: %APPDATA%/metamemory/config.json (persistence.py:69-92)

**Missing persistence logic:**
1. Radio button click → emit model_changed(model_id) signal
2. Signal handler → update ConfigManager via set_config('model.realtime_model_size', model_id)
3. Save → call save_config() on panel close or app exit
