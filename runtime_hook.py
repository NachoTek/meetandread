"""PyInstaller runtime hook — ensures native DLLs are discoverable.

When running from a PyInstaller frozen exe, the _internal/ directory contains
all bundled DLLs. Windows needs explicit path configuration to find them.
This hook runs before any package imports.
"""
import os
import sys

if getattr(sys, 'frozen', False):
    base = sys._MEIPASS
    os.add_dll_directory(base)
    os.environ['PATH'] = base + os.pathsep + os.environ.get('PATH', '')

    # sherpa-onnx: native DLLs live in sherpa_onnx/lib/ (onnxruntime.dll,
    # sherpa-onnx-c-api.dll, etc.).  The _sherpa_onnx.pyd extension needs
    # these at load time but they aren't in the root _internal/ directory.
    sherpa_lib = os.path.join(base, 'sherpa_onnx', 'lib')
    if os.path.isdir(sherpa_lib):
        os.add_dll_directory(sherpa_lib)
        os.environ['PATH'] = sherpa_lib + os.pathsep + os.environ.get('PATH', '')
