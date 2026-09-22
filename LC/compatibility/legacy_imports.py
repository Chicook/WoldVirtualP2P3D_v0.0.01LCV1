"""Resolucion controlada de nombres de modulo heredados.

La tabla es deliberadamente explicita: evita importar respaldos desde CHG y
permite retirar aliases de forma gradual sin tocar los consumidores actuales.
"""

from __future__ import annotations

import importlib
import warnings
from types import ModuleType


LEGACY_MODULES: dict[str, str] = {
    "BKSVCB": "LC.celebro.BKSVCB",
    "ipfs_manager": "LC.celebro.CMFG.ipfs_manager",
    "SNSBSTNPRB": "LC.celebro.CMFG.SBSTM.SNSBSTNPRB",
    "RFEN1_RN_4": "LC.celebro.red_neuronal.RF_EN.RFEN1_RN_4",
}


def resolve_module_name(name: str) -> str:
    """Devuelve el nombre canonico para un alias antiguo o actual."""
    normalized = name.strip()
    return LEGACY_MODULES.get(normalized, normalized)


def import_legacy(name: str) -> ModuleType:
    """Importa un modulo heredado y emite una advertencia de migracion."""
    canonical = resolve_module_name(name)
    if canonical != name.strip():
        warnings.warn(
            f"El import '{name}' esta obsoleto; usa '{canonical}'.",
            DeprecationWarning,
            stacklevel=2,
        )
    return importlib.import_module(canonical)


__all__ = ["LEGACY_MODULES", "import_legacy", "resolve_module_name"]
