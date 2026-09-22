# ─── VERSION DE SESION HRCTRC_RFCT ───────────────────────────────────────

def version_sesion(overlay: str = "") -> Dict[str, Any]:
    """Divide oversized a regla 400/450 con modelo local y barra de progreso."""
    from pathlib import Path as _P
    from LC.celebro.CMFG.SBSTM.HRCTRC import get_gestor_hrctrc
    from LC.celebro.CMFG.SBSTM.HRCTRC_RFCT import RefactorizadorSesion
    ov = _P(overlay) if overlay else get_gestor_hrctrc().overlay
    return RefactorizadorSesion(ov).ejecutar(mostrar_barra=True)


def estado_version_sesion(overlay: str = "") -> Dict[str, Any]:
    """Oversized restantes, paquetes y cumplimiento de la regla."""
    from pathlib import Path as _P
    from LC.celebro.CMFG.SBSTM.HRCTRC import get_gestor_hrctrc
    from LC.celebro.CMFG.SBSTM.HRCTRC_RFCT import RefactorizadorSesion
    ov = _P(overlay) if overlay else get_gestor_hrctrc().overlay
    return RefactorizadorSesion(ov).estado_version()