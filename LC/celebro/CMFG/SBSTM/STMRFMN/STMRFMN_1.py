# ─── IMPORTACION DIRECTA HRCTNR ──────────────────────────

def _importar_hrctnr():
    """Importacion directa del subsistema HRCNTR."""
    from LC.celebro.CMFG.SBSTM.HRCNTR import (
        GestorHRCNTR, get_gestor_hrctnr, ejecutar_hrctnr,
        estado_hrctnr, actualizar_sistema_hrctnr, ciclo_cierre_hrctnr,
        confirmar_actualizacion_hrctnr,
    )
    return {
        "GestorHRCNTR": GestorHRCNTR,
        "get_gestor_hrctnr": get_gestor_hrctnr,
        "ejecutar_hrctnr": ejecutar_hrctnr,
        "estado_hrctnr": estado_hrctnr,
        "actualizar_sistema_hrctnr": actualizar_sistema_hrctnr,
        "ciclo_cierre_hrctnr": ciclo_cierre_hrctnr,
        "confirmar_actualizacion_hrctnr": confirmar_actualizacion_hrctnr,
    }

_HRCNTR_API: Optional[Dict[str, Any]] = None


def _get_hrctnr_api() -> Dict[str, Any]:
    global _HRCNTR_API
    if _HRCNTR_API is None:
        _HRCNTR_API = _importar_hrctnr()
    return _HRCNTR_API