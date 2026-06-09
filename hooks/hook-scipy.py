"""PyInstaller hook for scipy.io.wavfile.

Forces collection of scipy.io.wavfile which may be missed by lazy loading.
"""

from PyInstaller.utils.hooks import collect_submodules

# Collect all scipy.io submodules including wavfile
hiddenimports = collect_submodules('scipy.io')