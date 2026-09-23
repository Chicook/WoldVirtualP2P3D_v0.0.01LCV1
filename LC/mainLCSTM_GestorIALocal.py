class GestorIALocal:
    @staticmethod
    def perfilar_hardware() -> Dict[str, Any]:
        from LC.celebro.CMFG.SBSTM.IALOCAL import perfilar_hardware_ialocal
        return perfilar_hardware_ialocal(guardar=True)
    @staticmethod
    def verificar_capacidades(perfil: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        from LC.celebro.CMFG.SBSTM.IALOCAL import verificar_capacidades
        return verificar_capacidades(perfil=perfil)
    @staticmethod
    def listar_modelos_hf() -> List[Dict[str, Any]]:
        from LC.celebro.CMFG.SBSTM.IALOCAL import listar_modelos_hf
        return listar_modelos_hf()
    @staticmethod
    def modelos_compatibles(perfil: Optional[Dict[str, Any]] = None,
                               limite: Optional[int] = None) -> List[Dict[str, Any]]:
        from LC.celebro.CMFG.SBSTM.IALOCAL import recomendar_modelos_hf
        return recomendar_modelos_hf(perfil=perfil, limite=limite)
    @staticmethod
    def descargar_modelo_hf(modelo_id: str) -> Dict[str, Any]:
        from LC.celebro.CMFG.SBSTM.IALOCAL import descargar_modelo_hf
        return descargar_modelo_hf(modelo_id)
    @staticmethod
    def recomendar_y_descargar(limite: int = 2, mostrar_banner: bool = True) -> List[Dict[str, Any]]:
        from LC.celebro.CMFG.SBSTM.IALOCAL import recomendar_y_descargar
        return recomendar_y_descargar(limite=limite, mostrar_banner=mostrar_banner)
    @staticmethod
    def info_completa() -> Dict[str, Any]:
        from LC.celebro.CMFG.SBSTM.IALOCAL import info_completa
        return info_completa()
    @staticmethod
    def pre_sesion(modelo_id: str = None) -> Dict[str, Any]:
        from LC.celebro.CMFG.SBSTM.IALOCAL import orquestar_pre_sesion
        return orquestar_pre_sesion(modelo_id=modelo_id)
    @staticmethod
    def limpiar_sesion() -> Dict[str, Any]:
        from LC.celebro.CMFG.SBSTM.IALOCAL import limpiar_ia_temporal
        return limpiar_ia_temporal()
    @staticmethod
    def listar_py_constructor() -> List[Any]:
        from LC.celebro.CMFG.SBSTM.IALOCAL import listar_py_constructor
        return listar_py_constructor()
    @staticmethod
    def refactorizar_constructor(modelo_hf: str = "google/gemma-2-2b-it:free",
                                     usar_local: bool = True) -> Dict[str, Any]:
        from LC.celebro.CMFG.SBSTM.IALOCAL import refactorizar_constructor
        return refactorizar_constructor(modelo_hf=modelo_hf, usar_local=usar_local)


# CLASE 3: GestorConstructor — sesiones del constructor HRCTRC: carpetas, versionado y refactor de sesión (HRCTRC + HRCTRC_RFCT).
