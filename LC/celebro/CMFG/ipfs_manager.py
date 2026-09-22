"""
ipfs_manager.py - SHIM de sesion (regla 400/450). Codigo en ipfs_manager_pkg/.
Generado por HRCTRC_RFCT v2026.3.1; compatible 100%.
"""
from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path as _Path
_pkgdir = _Path(__file__).resolve().parent / "ipfs_manager_pkg"
_spec = _ilu.spec_from_file_location(
    "ipfs_manager_pkg", _pkgdir / "__init__.py",
    submodule_search_locations=[str(_pkgdir)])
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)  # noqa
globals().update({k: v for k, v in vars(_mod).items() if not k.startswith('__')})
try:
    __all__ = list(getattr(_mod, '__all__', []))
except Exception:
    __all__ = []
