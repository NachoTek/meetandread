"""metamemory package.

Zero information loss during conversations — Users stay fully present
knowing every word is captured for AI agent processing.
"""

try:
    from importlib.metadata import version, PackageNotFoundError

    try:
        __version__ = version("meetandread")
    except PackageNotFoundError:
        __version__ = "0.1.0"
except Exception:
    __version__ = "0.1.0"

__author__ = "Tergi"

# Public API exports
# Config is accessible via metamemory.config submodule
# from metamemory.config import get_config, set_config, save_config