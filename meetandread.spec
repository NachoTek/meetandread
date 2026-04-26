"""PyInstaller spec for meetandread portable Windows exe.

Collects all native DLLs that PyInstaller cannot auto-detect (ctypes-loaded,
delvewheel-patched, and dynamically-discovered libraries).

Usage:
    pyinstaller meetandread.spec --noconfirm
"""
import glob
import os
import site
import sys

# --- Locate site-packages ---------------------------------------------------
site.addsitedir(site.getusersitepackages())  # user site-packages
SP = site.getsitepackages()[1]  # Lib/site-packages (index 0 is Python root)

# --- Helper -----------------------------------------------------------------

def _collect(pattern, dest):
    """Return list of (src, dest) tuples for DLLs/pyds matching glob pattern."""
    return [(f, dest) for f in glob.glob(os.path.join(SP, pattern))]


# --- Native DLL groups ------------------------------------------------------
#
# Each group collects DLLs that PyInstaller's static analysis misses because
# they are loaded via ctypes, delvewheel __init__.py patching, or dynamic
# filename hashing.

binaries = []

# 1. pywhispercpp — the hardest case.
#    Content-hash-named DLLs in site-packages root (not inside the package).
#    The .pyd extension module + ggml/whisper shared libs + MSVC runtimes.
binaries += _collect('whisper-*.dll', '.')
binaries += _collect('ggml*.dll', '.')
binaries += _collect('_pywhispercpp*.pyd', '.')
# MSVC runtime DLLs placed in SP root by delvewheel (needed by ggml DLLs)
binaries += _collect('msvcp140-*.dll', '.')
binaries += _collect('vcomp140-*.dll', '.')

# 2. sherpa-onnx — shared libs in sherpa_onnx/lib/
binaries += _collect(os.path.join('sherpa_onnx', 'lib', '*.dll'), 'sherpa_onnx/lib')
binaries += _collect(os.path.join('sherpa_onnx', 'lib', '*.pyd'), 'sherpa_onnx/lib')

# 3. sounddevice — PortAudio binaries
binaries += _collect(
    os.path.join('_sounddevice_data', 'portaudio-binaries', '*.dll'),
    '_sounddevice_data/portaudio-binaries',
)

# 4. pyaudiowpatch — .pyd extension module in site-packages root
binaries += _collect('_portaudiowpatch*.pyd', '.')

# 5. PyQt6 — platform plugins (PyInstaller usually handles this, but be safe)
binaries += _collect(
    os.path.join('PyQt6', 'Qt6', 'plugins', 'platforms', '*.dll'),
    'PyQt6/Qt6/plugins/platforms',
)

# --- Hidden imports ---------------------------------------------------------
#
# Packages that PyInstaller cannot discover through static import analysis
# (conditional imports, plugin systems, dynamic loading).

hiddenimports = [
    # Application
    'metamemory',
    'metamemory.main',
    'metamemory.widgets.main_widget',
    'metamemory.widgets.tray_icon',
    'metamemory.audio',
    'metamemory.config',
    'metamemory.hardware',
    'metamemory.hardware.detector',
    'metamemory.hardware.recommender',
    # Native packages
    'pywhispercpp',
    'pywhispercpp.model',
    'sherpa_onnx',
    'sounddevice',
    'pyaudiowpatch',
    # Qt
    'PyQt6.QtCore',
    'PyQt6.QtWidgets',
    'PyQt6.QtGui',
    # Transitive dependencies that may be missed
    'numpy',
    'scipy',
]

# --- Analysis ---------------------------------------------------------------

a = Analysis(
    [os.path.join('src', 'metamemory', '__main__.py')],
    pathex=[],
    binaries=binaries,
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtimehooks=['runtime_hook.py'],
    excludes=[],
    noarchive=False,
)

# --- Bundle -----------------------------------------------------------------

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='meetandread',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,  # Keep console for debug output; change to False for release
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='meetandread',
)
