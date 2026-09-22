# ─── CONSTRUCTOR HRCTRC ──────────────────────────────────────────────────

def abrir_constructor(sesion_id: str = "") -> Dict[str, Any]:
    """Abre la copia de trabajo en LC/Constructor para la sesion."""
    from LC.celebro.CMFG.SBSTM.HRCTRC import get_gestor_hrctrc
    return get_gestor_hrctrc(sesion_id).iniciar_sesion()


def estado_constructor() -> Dict[str, Any]:
    """Overlay activo, pendientes y si Constructor esta vacia."""
    from LC.celebro.CMFG.SBSTM.HRCTRC import get_gestor_hrctrc
    return get_gestor_hrctrc().estado()


def unificar_constructor() -> Dict[str, Any]:
    """Unifica el overlay en rutas reales y vacia Constructor."""
    from LC.celebro.CMFG.SBSTM.HRCTRC import get_gestor_hrctrc
    return get_gestor_hrctrc().finalizar_sesion(aplicar=True)


def crear_carpeta_lc(ruta_rel: str, tambien_en_lc: bool = False) -> Dict[str, Any]:
    """Crea carpetas con permisos via codigo (overlay y opcionalmente LC)."""
    from LC.celebro.CMFG.SBSTM.HRCTRC import get_gestor_hrctrc
    return get_gestor_hrctrc().crear_carpeta(ruta_rel, tambien_en_lc=tambien_en_lc)