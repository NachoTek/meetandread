"""PyInstaller hook for soundfile to ensure _soundfile_data is bundled."""

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, is_module_satisfies

# Force this hook to always run (even if module is imported conditionally)
# by collecting the data files directly
datas = collect_data_files('_soundfile_data', include_py_files=True, deepcopy=False)

# Ensure soundfile module and its submodules are collected
hiddenimports = ['soundfile', 'soundfile._version']

# Force the module to be loaded as a package
if is_module_satisfies('soundfile'):
    pass