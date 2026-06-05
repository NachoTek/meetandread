"""PyInstaller hook for soundfile to ensure _soundfile_data is bundled."""

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Collect the _soundfile_data directory with all files
datas = collect_data_files('_soundfile_data', include_py_files=True)

# Also ensure soundfile module is collected as a hidden import
hiddenimports = ['soundfile', 'soundfile._version']