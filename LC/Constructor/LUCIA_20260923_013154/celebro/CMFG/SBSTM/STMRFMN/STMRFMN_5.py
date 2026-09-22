# ─── INTEGRACION Y CIERRE (INTEGRACIONRF) ────────────────────────────────

def bitacora() -> Dict[str, Any]:
    """Genera el .md de actividad en CHG/ con monitoreo del sistema."""
    from LC.celebro.CMFG.SBSTM.INTEGRACIONRF import get_integrador
    return get_integrador().generar_md()


def documentar_respaldos() -> Dict[str, Any]:
    """Convierte los .bak_sesion de CHG/ en .md explicativo y luego a pesos."""
    from LC.celebro.CMFG.SBSTM.INTEGRACIONRF import get_integrador
    return get_integrador().documentar_respaldos_chg()


def cierre_integracion() -> Dict[str, Any]:
    """Pipeline de cierre: refactor -> .md -> pesos -> IPFS -> devopencode."""
    from LC.celebro.CMFG.SBSTM.INTEGRACIONRF import cierre_integracion as _c
    return _c()


def informe_cierre(reporte: Dict[str, Any]) -> str:
    """Resumen de una linea del pipeline de cierre para consola y voz."""
    from LC.celebro.CMFG.SBSTM.INTEGRACIONRF import get_integrador
    return get_integrador().resumen_cierre_txt(reporte)