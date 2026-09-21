"""RNP/panel_rnp - Panel vivo de calibración interna (RN11-RN14).

Responde a la encuesta: RNP era el subsistema más débil (8/8) porque nadie
lo miraba de frente. Este panel expone, por nodo, norma sináptica, rol y
estado, en el mismo idioma del mapa neuronal para que LucIA pueda verse.
"""
import logging
from typing import Any, Dict, List

logger = logging.getLogger("lucIA.PanelRNP")

ROLES = {
    "RN11": "peso neuronal",
    "RN12": "ajuste dinámico",
    "RN13": "control de gradientes",
    "RN14": "puertas de atención",
}


def _norma(nodo: Any) -> float:
    import numpy as _np
    for attr in ("pesos", "pesos_principal", "pesos_actor"):
        p = getattr(nodo, attr, None)
        try:
            if p is not None and _np.size(p) > 1:
                return round(float(_np.linalg.norm(_np.asarray(p, dtype="float32"))), 5)
        except Exception:
            pass
    return 0.0


def _rol(nombre: str) -> str:
    for clave, rol in ROLES.items():
        if clave in nombre:
            return rol
    n = (nombre or "").lower()
    if "optimizador" in n:
        return ROLES["RN11"]
    if "ajuste" in n:
        return ROLES["RN12"]
    if "controlador" in n or "gradiente" in n:
        return ROLES["RN13"]
    if "puerta" in n or "atencion" in n:
        return ROLES["RN14"]
    return "calibración"


def resumen_rnp(nodos: List[Any]) -> Dict[str, Any]:
    """Agrega el estado de los nodos RN11-RN14 en un informe legible."""
    detalle: Dict[str, Any] = {}
    for nodo in nodos or []:
        nombre = getattr(nodo, "nombre", type(nodo).__name__)
        norma = _norma(nodo)
        rol = _rol(nombre)
        # RN13/RN14 arrancan en ceros por diseño (control/reposo, no atrofia):
        # el mapa los vería "débiles" aunque estén sanos. Se etiqueta aparte.
        if norma > 0:
            estado = "activo"
        elif rol in ("control de gradientes", "puertas de atención"):
            estado = "en reposo (normal en ceros)"
        else:
            estado = "en arranque"
        detalle[nombre] = {
            "rol": rol,
            "norma_sinaptica": norma,
            "estado": estado,
            "nota": getattr(nodo, "informe_ultimo", ""),
        }
    total = round(sum(d["norma_sinaptica"] for d in detalle.values()), 5)
    mas_activo = max(detalle, key=lambda k: detalle[k]["norma_sinaptica"]) if detalle else "N/A"
    lineas = ["[PANEL RNP — calibración interna (RN11 peso, RN12 ajuste, RN13 gradientes, RN14 puertas)]:"]
    for nombre, d in detalle.items():
        lineas.append(f"  • {nombre} ({d['rol']}): norma={d['norma_sinaptica']:.3f} [{d['estado']}]")
    lineas.append(f"  Fuerza total RNP: {total:.3f} | nodo más activo: {mas_activo}")
    return {"detalle": detalle, "fuerza_total": total,
            "nodo_mas_activo": mas_activo, "informe": "\n".join(lineas)}
