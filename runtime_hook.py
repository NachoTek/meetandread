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
