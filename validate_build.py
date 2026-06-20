#!/usr/bin/env python3
"""
Validate a PyInstaller build contains all required dependencies.
Run locally before pushing tags, or in CI as a validation step.
"""
import os
import sys
import subprocess
import glob

# Fix Windows console encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# PyInstaller build directory
BUILD_DIR = "dist/meetandread"

# Critical libraries that must be present (DLLs and Python extensions)
REQUIRED_LIBRARIES = [
    # pywhispercpp (whisper.cpp bindings)
    "whisper-",
    "ggml",
    "_pywhispercpp",
    "msvcp140-",
    "vcomp140-",

    # sherpa-onnx
    "sherpa_onnx",

    # PortAudio
    "portaudio",

    # pyaudiowpatch
    "_portaudiowpatch",

    # PyQt6
    "PyQt6/Qt6/plugins/platforms/qwindows.dll",

    # Python standard
    "numpy",
    "scipy",
]

# Python modules that must be importable from the built exe
REQUIRED_IMPORTS = [
    "meetandread",
    "meetandread.main",
    "meetandread.widgets.main_widget",
    "meetandread.widgets.tray_icon",
    "meetandread.audio",
    "meetandread.config",
    "meetandread.hardware",
    "meetandread.hardware.detector",
    "meetandread.hardware.recommender",
    "meetandread.speaker",
    "meetandread.speaker.diarizer",
    "meetandread.speaker.model_downloader",
    "meetandread.speaker.signatures",
    "meetandread.speaker.models",
    "meetandread.speaker.identity_management",
    "meetandread.transcription",
    "meetandread.transcription.post_processor",
    "meetandread.transcription.transcript_store",
    "meetandread.transcription.transcript_scanner",
    "meetandread.transcription.scrub",
    "pywhispercpp",
    "pywhispercpp.model",
    "sherpa_onnx",
    "sherpa_onnx.lib",
    "sounddevice",
    "pyaudiowpatch",
    "soxr",
    "PyQt6.QtCore",
    "PyQt6.QtWidgets",
    "PyQt6.QtGui",
    "numpy",
    "scipy",
    "scipy.io.wavfile",
]


def check_build_directory():
    """Check if build directory exists."""
    if not os.path.exists(BUILD_DIR):
        print(f"❌ Build directory not found: {BUILD_DIR}")
        print("   Run: pyinstaller meetandread.spec --noconfirm")
        return False
    return True


def check_required_dlls():
    """Check that all required DLLs/extension modules are present."""
    print("\n🔍 Checking required DLLs and extension modules...")
    missing = []
    found = []

    for pattern in REQUIRED_LIBRARIES:
        # Search for files matching the pattern
        matches = glob.glob(os.path.join(BUILD_DIR, "**", f"{pattern}*"), recursive=True)
        if matches:
            found.extend(matches)
        else:
            missing.append(pattern)

    if found:
        print(f"✅ Found {len(found)} required library files:")
        for f in sorted(set(found[:10])):  # Show first 10 unique
            rel_path = os.path.relpath(f, BUILD_DIR)
            print(f"   ✓ {rel_path}")
        if len(found) > 10:
            print(f"   ... and {len(found) - 10} more")

    if missing:
        print(f"❌ Missing {len(missing)} required library patterns:")
        for pattern in missing:
            print(f"   ✗ {pattern}*")
        return False

    return True


def check_imports():
    """Check that required modules can be imported from the built exe."""
    print("\n🔍 Checking module imports from built executable...")
    exe_path = os.path.join(BUILD_DIR, "meetandread.exe")

    if not os.path.exists(exe_path):
        print(f"❌ Executable not found: {exe_path}")
        return False

    # Skip executable import checks in headless CI environments
    # PyQt6 GUI apps hang without a display, causing timeout failures
    if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
        print("   ⚠️  Skipping executable import tests in CI environment")
        print("   (PyQt6 GUI apps require a display to launch)")
        print("   ✓ DLL and file structure checks performed instead")
        return True

    # Try to use --help flag first to see if exe launches
    result = subprocess.run(
        [exe_path, "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    # If exe doesn't support --help, try import tests
    if result.returncode != 0:
        print("   Note: Executable doesn't support --help, trying import tests...")
        missing_imports = []

        for module in REQUIRED_IMPORTS:
            # Use -c flag to import module and exit
            result = subprocess.run(
                [exe_path, "-c", f"import {module}; print('OK')"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0 and "OK" in result.stdout:
                print(f"   ✓ {module}")
            else:
                print(f"   ✗ {module}")
                missing_imports.append(module)

        if missing_imports:
            print(f"\n❌ {len(missing_imports)} modules failed to import:")
            for mod in missing_imports:
                print(f"   ✗ {mod}")
            return False
    else:
        print("   ✓ Executable launches successfully")

    return True


def check_test_data():
    """Check that test data files are present."""
    print("\n🔍 Checking test data and assets...")

    # SVG icons
    svg_files = glob.glob(os.path.join(BUILD_DIR, "meetandread", "widgets", "*.svg"))
    if not svg_files:
        print("   ✗ Missing SVG icons")
        return False
    print(f"   ✓ Found {len(svg_files)} SVG icon files")

    # Performance test data
    test_data_files = glob.glob(
        os.path.join(BUILD_DIR, "meetandread", "performance", "test_data", "*")
    )
    if not test_data_files:
        print("   ✗ Missing performance test data")
        return False
    print(f"   ✓ Found {len(test_data_files)} test data files")

    return True


def main():
    print("=" * 60)
    print("PyInstaller Build Validation")
    print("=" * 60)

    checks = [
        check_build_directory,
        check_required_dlls,
        check_imports,
        check_test_data,
    ]

    all_passed = True
    for check in checks:
        if not check():
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All validation checks passed!")
        print("   The build appears complete and ready for release.")
        return 0
    else:
        print("❌ Validation failed!")
        print("   Fix the issues above before creating a release tag.")
        return 1


if __name__ == "__main__":
    sys.exit(main())