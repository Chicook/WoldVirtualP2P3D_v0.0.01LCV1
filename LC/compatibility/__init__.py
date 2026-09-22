"""Compatibilidad y migracion de imports de LucIA.

Este paquete mantiene una frontera temporal para consumidores antiguos. Los
modulos de ``CHG`` son respaldos historicos y no forman parte del runtime.
"""

from .legacy_imports import import_legacy, resolve_module_name

__all__ = ["import_legacy", "resolve_module_name"]
