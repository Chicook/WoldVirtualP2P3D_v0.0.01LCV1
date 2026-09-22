"""Fuente factual de arquitectura y cambios visibles para LucIA.

El modelo remoto no debe inventar el historial del sistema. Esta capa le
entrega hechos procedentes del manifiesto versionado y del filesystem actual.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


LC_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = LC_ROOT.parent
MANIFEST_PATH = LC_ROOT / "ARCHITECTURE_CHANGELOG.json"


def architecture_snapshot() -> Dict[str, Any]:
    """Devuelve el estado factual y resumido de la arquitectura."""
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        manifest = {"schema_version": 1, "changes": []}

    checks = {
        "compatibility_layer": (LC_ROOT / "compatibility").is_dir(),
        "local_ai_config": (LC_ROOT / "modelosIAlocal" / "IAlocal.json").is_file(),
        "transaction_safety_fix": (LC_ROOT / "celebro" / "BKSVCB.py").is_file(),
        "architecture_tests": (LC_ROOT / "celebro" / "test" / "test_compatibility.py").is_file(),
    }
    return {
        "project": manifest.get("project", "WoldVirtualP2P3D"),
        "schema_version": manifest.get("schema_version", 1),
        "architecture_version": manifest.get("architecture_version", "unknown"),
        "changes": manifest.get("changes", []),
        "live_checks": checks,
        "runtime_root": str(PROJECT_ROOT),
        "note": "CHG contiene respaldos; el runtime activo vive bajo LC.",
    }


def architecture_context(max_changes: int = 12) -> str:
    """Genera contexto compacto, factual y apto para el prompt del modelo."""
    state = architecture_snapshot()
    lines: List[str] = [
        "ESTADO FACTUAL DE ARQUITECTURA DE LUCIA:",
        f"- Version de arquitectura: {state['architecture_version']}",
        "- Cambios aplicados en esta linea de trabajo:",
    ]
    for change in state["changes"][:max_changes]:
        lines.append(f"  - {change.get('id')}: {change.get('summary')}")
    lines.append("- Comprobaciones activas:")
    for name, value in state["live_checks"].items():
        lines.append(f"  - {name}: {'OK' if value else 'PENDIENTE'}")
    lines.append("- Restriccion: no afirmes que un cambio existe si no aparece en estos datos.")
    return "\n".join(lines)


def architecture_answer() -> str:
    """Respuesta local para evitar que una pregunta factual dependa del LLM."""
    state = architecture_snapshot()
    changes = state["changes"]
    lines = [
        "Diagnóstico factual de LucIA:",
        f"Arquitectura: {state['architecture_version']}.",
        f"Cambios registrados: {len(changes)}.",
    ]
    lines.extend(f"- {item.get('summary')}" for item in changes)
    lines.append("Estado: " + ", ".join(
        f"{key}={'OK' if value else 'PENDIENTE'}"
        for key, value in state["live_checks"].items()
    ) + ".")
    lines.append("El refactorizador automatico puede omitir modulos vivos si no superan su prueba previa.")
    return "\n".join(lines)


__all__ = ["architecture_answer", "architecture_context", "architecture_snapshot"]
