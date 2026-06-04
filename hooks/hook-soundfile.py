"""
PyInstaller hook for soundfile.

soundfile.py imports _soundfile_data to get libsndfile DLLs.
This hook collects the data directory including __init__.py and DLLs.
"""

from PyInstaller.utils.hooks import collect_data_files

datas = collect_data_files('_soundfile_data', include_py_files=True)