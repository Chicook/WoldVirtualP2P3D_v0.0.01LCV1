"""mainLCSTM.py - Orquestador Central del Sistema Cognitivo LucIA (2026)

Refactor 2026.4.0: la lógica se agrupa en clases-gestor cohesivas (una
responsabilidad por clase); las funciones de módulo históricas quedan
como wrappers de una línea para no romper importaciones existentes.
Regla de arquitectura: 400/450 líneas por módulo.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
from typing import Any, Dict, Final, List, Optional

if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from LC.compatibility import import_legacy
from LC.celebro.CMFG.SBSTM.BASELUC import EstiloTerminalLucIA, get_cliente_iafree
from LC.celebro.CMFG.SBSTM.CMDLUC import CMDLUCMixin
from LC.celebro.CMFG.SBSTM.PRTLUC import PRTLUCMixin
from LC.celebro.CMFG.SBSTM.TRNLUC import TRNLUCMixin
from LC.celebro.CMFG.SBSTM.STMRFMN.STMRFMN_1 import GestorSTMRFMN, get_gestor_stmrfmn
from LC.celebro.CMFG.SBSTM.STMRFMN.STMRFMN_2 import (
    ConfiguracionLucia, ServidorBKS, ConversorPSN, GestorHRCTRC,
    ClienteIAFree, ClienteIPFS, TelemetriaLucia, GestorMemoria, DemoLucia,
    iniciar_lucia, detener_lucia, turno, estado,
)

__version__: Final[str] = "2026.4.0"

# CLASE 1: STMRFMNFacade — fachada compacta sobre GestorSTMRFMN para las operaciones HRCNTR (monitor/refactor neural).
class STMRFMNFacade:
    def __init__(self, gestor: Optional[GestorSTMRFMN] = None) -> None:
        self._gestor = gestor or get_gestor_stmrfmn()

    @property
    def gestor(self) -> GestorSTMRFMN: return self._gestor
    def refactorizar(self, overlay="", mostrar_barra=True, usar_cache=False) -> Dict[str, Any]:
        return self._gestor.refactorizar(overlay=overlay, mostrar_barra=mostrar_barra, usar_cache=usar_cache)
    def actualizar(self, confirmar=False) -> Dict[str, Any]: return self._gestor.actualizar(confirmar=confirmar)
    def ciclo_cierre(self) -> Dict[str, Any]: return self._gestor.ciclo_cierre()
    def estado(self, forzar=False, usar_cache=True) -> Dict[str, Any]:
        return self._gestor.estado(forzar=forzar, usar_cache=usar_cache)
    def confirmar_actualizacion(self) -> str: return self._gestor.confirmar_actualizacion()
    def monitor_sistema(self, forzar=False, usar_cache=True) -> Dict[str, Any]:
        return self._gestor.monitor_sistema(forzar=forzar, usar_cache=usar_cache)
    def listar_modulos(self) -> List[Dict[str, Any]]: return self._gestor.listar_modulos()
    def listar_neuronas(self) -> List[Dict[str, Any]]: return self._gestor.listar_neuronas()
    def seleccionar_clases(self) -> List[Dict[str, Any]]: return self._gestor.seleccionar_clases()
    def diagnostico(self) -> Dict[str, Any]: return self._gestor.diagnostico()
    def limpiar_cache(self) -> None: self._gestor.limpiar_cache()

# CLASE 2: GestorDescargasIA — perfilado de hardware, catálogo local y consulta offline (DSIALCLGRG + MDSTM), respaldo de IAFREE.
class GestorDescargasIA:
    @staticmethod
    def perfilar_pc() -> Dict[str, Any]:
        from LC.celebro.CMFG.SBSTM.DSIALCLGRG import perfilar_hardware
        return perfilar_hardware(guardar=True)
    @staticmethod
    def listar_modelos_locales() -> List[Dict[str, Any]]:
        from LC.celebro.CMFG.SBSTM.DSIALCLGRG import listar_modelos_locales
        return listar_modelos_locales()
    @staticmethod
    def descargar(modelo_id: str) -> Dict[str, Any]:
        from LC.modelosIAlocal.MDSTM import get_gestor_mdstm
        return get_gestor_mdstm().descargar(modelo_id)
    @staticmethod
    def preguntar(prompt: str) -> Dict[str, Any]:
        from LC.celebro.CMFG.SBSTM.DSIALCLGRG import consultar_lucia_local
        texto, mid, lat = consultar_lucia_local(prompt, None)
        return {"texto": texto, "modelo": mid, "latencia_ms": lat}

# CLASE 3: GestorConstructor — sesiones del constructor HRCTRC: carpetas, versionado y refactor de sesión (HRCTRC + HRCTRC_RFCT).
class GestorConstructor:
    @staticmethod
    def abrir(sesion_id: str = "") -> Dict[str, Any]:
        from LC.celebro.CMFG.SBSTM.HRCTRC import get_gestor_hrctrc
        return get_gestor_hrctrc(sesion_id).iniciar_sesion()
    @staticmethod
    def estado() -> Dict[str, Any]:
        from LC.celebro.CMFG.SBSTM.HRCTRC import get_gestor_hrctrc
        return get_gestor_hrctrc().estado()
    @staticmethod
    def unificar() -> Dict[str, Any]:
        from LC.celebro.CMFG.SBSTM.HRCTRC import get_gestor_hrctrc
        return get_gestor_hrctrc().finalizar_sesion(aplicar=True)
    @staticmethod
    def crear_carpeta(ruta_rel: str, tambien_en_lc: bool = False) -> Dict[str, Any]:
        from LC.celebro.CMFG.SBSTM.HRCTRC import get_gestor_hrctrc
        return get_gestor_hrctrc().crear_carpeta(ruta_rel, tambien_en_lc=tambien_en_lc)
    @staticmethod
    def version_sesion(overlay: str = "") -> Dict[str, Any]:
        from LC.celebro.CMFG.SBSTM.HRCTRC import get_gestor_hrctrc
        from LC.celebro.CMFG.SBSTM.HRCTRC_RFCT import RefactorizadorSesion
        ruta = Path(overlay) if overlay else get_gestor_hrctrc().overlay
        return RefactorizadorSesion(ruta).ejecutar(mostrar_barra=True)
    @staticmethod
    def estado_version_sesion(overlay: str = "") -> Dict[str, Any]:
        from LC.celebro.CMFG.SBSTM.HRCTRC import get_gestor_hrctrc
        from LC.celebro.CMFG.SBSTM.HRCTRC_RFCT import RefactorizadorSesion
        ruta = Path(overlay) if overlay else get_gestor_hrctrc().overlay
        return RefactorizadorSesion(ruta).estado_version()

# CLASE 4: GestorIntegracion — bitácora .md, respaldos CHG y el ciclo pesos -> IPFS -> devopencode (INTEGRACIONRF).
class GestorIntegracion:
    @staticmethod
    def bitacora() -> Dict[str, Any]:
        from LC.celebro.CMFG.SBSTM.INTEGRACIONRF import get_integrador
        return get_integrador().generar_md()
    @staticmethod
    def documentar_respaldos() -> Dict[str, Any]:
        from LC.celebro.CMFG.SBSTM.INTEGRACIONRF import get_integrador
        return get_integrador().documentar_respaldos_chg()
    @staticmethod
    def cierre() -> Dict[str, Any]:
        from LC.celebro.CMFG.SBSTM.INTEGRACIONRF import cierre_integracion
        return cierre_integracion()
    @staticmethod
    def informe_cierre(reporte: Dict[str, Any]) -> str:
        from LC.celebro.CMFG.SBSTM.INTEGRACIONRF import get_integrador
        return get_integrador().resumen_cierre_txt(reporte)

# CLASE 5: GestorBlockchain — minado y validación de la cadena BKSVCB donde se registran los turnos.
class GestorBlockchain:
    @staticmethod
    def minar(orq: "OrquestadorSistemaLucIA") -> Dict[str, Any]:
        blk = orq.servidor_bks.minar_transacciones_pendientes()
        if not blk:
            return {"exito": False, "mensaje": "Sin transacciones pendientes."}
        return {"exito": True, "indice": blk.indice, "hash": blk.hash_bloque}
    @staticmethod
    def validar_cadena(orq: "OrquestadorSistemaLucIA") -> Dict[str, Any]:
        valida, err = orq.servidor_bks.validar_cadena()
        return {"valida": bool(valida), "error": err, "bloques": len(orq.servidor_bks.cadena)}

# CLASE 6: GestorVoz — voz neuronal es-ES: habla asíncrona y cancelación inmediata.
class GestorVoz:
    @staticmethod
    def decir(texto: str, emocion: Optional[str] = None) -> None:
        from LC.celebro.CMFG.SBSTM.voice_engine import speak
        speak(texto, esperar=False, emocion=emocion)
    @staticmethod
    def silenciar() -> None:
        from LC.celebro.CMFG.SBSTM.voice_engine import cancel_speech
        cancel_speech()
    @staticmethod
    def motor() -> Any:
        from LC.celebro.CMFG.SBSTM.voice_engine import get_voice_engine
        return get_voice_engine()

# CLASE 7: GestorIAFree — inferencia gratuita vía OpenRouter: consulta, catálogo, selección de modelo y benchmark.
class GestorIAFree:
    @staticmethod
    def consulta(prompt: str) -> str:
        from LC.celebro.CMFG.SBSTM.IAFREE import consultar_lucia_gratis
        return consultar_lucia_gratis(prompt)
    @staticmethod
    def catalogo() -> List[Dict[str, Any]]:
        from LC.celebro.CMFG.SBSTM.IAFREE import listar_catalogo_gratis
        return listar_catalogo_gratis()
    @staticmethod
    def seleccionar(modelo_id: str) -> bool:
        from LC.celebro.CMFG.SBSTM.IAFREE import get_cliente_iafree
        return get_cliente_iafree().gestor.seleccionar_por_id(modelo_id)
    @staticmethod
    def benchmark(max_modelos: int = 3) -> Dict[str, float]:
        from LC.celebro.CMFG.SBSTM.IAFREE import get_cliente_iafree
        return get_cliente_iafree().benchmark_rapido_modelos(max_modelos=max_modelos)

# CLASE 8: GestorPesos — persistencia de pesos neuronales (PSNRL) y su ciclo de vida en IPFS.
class GestorPesos:
    @staticmethod
    def exportar(orq: "OrquestadorSistemaLucIA", etiqueta: str = "") -> Dict[str, str]:
        npz, js = orq.conversor_psn.persistir_pesos_en_psnrl(etiqueta=etiqueta or f"export_{orq.sesion_id}")
        return {"npz": str(npz), "json": str(js)}
    @staticmethod
    def pin_ipfs() -> Dict[str, Any]:
        from LC.celebro.CMFG.ipfs_manager import get_ipfs_manager
        return get_ipfs_manager().subir_y_limpiar_psnrl(forzar_borrado_sin_daemon=False)
    @staticmethod
    def restaurar(cid: str) -> Dict[str, Any]:
        from LC.celebro.CMFG.ipfs_manager import get_ipfs_manager
        data = get_ipfs_manager().recuperar_pesos(cid)
        n = len(data) if isinstance(data, (bytes, list)) else len(str(data))
        return {"cid": cid, "bytes": n}

# CLASE 9: GestorSistema — metadatos del orquestador: versión, ayuda de la API, capacidades y diagnóstico agregado.
class GestorSistema:
    CAPACIDADES: Final[Dict[str, str]] = {
        "descargas_ia": "MDSTM descarga modelos ligeros reales en LC/modelosIAlocal.",
        "ia_local": "DSIALCLGRG responde offline con Ollama/GGUF si OpenRouter falla.",
        "constructor": "HRCTRC crea carpetas y versiona el sistema en LC/Constructor.",
        "refactor": "HRCTRC_RFCT divide oversized a 400/450 con modelo local.",
        "hrctnr": "HRCNTR monitorea, refactoriza y actualiza todo el sistema.",
        "integracion": "INTEGRACIONRF: .md en CHG -> pesos -> IPFS -> devopencode.",
        "voz": "voice_engine habla con voz neuronal es-ES y cancela al instante.",
        "blockchain": "BKSVCB registra cada turno y mina cada 3 turnos.",
    }
    @staticmethod
    def version() -> Dict[str, str]:
        return {"orquestador": __version__, "arquitectura": "WoldVirtualP2P3D-2026",
                "regla": "400/450 lineas por modulo", "coste": "$0.00"}
    @staticmethod
    def ayuda_api() -> str:
        return ("LucIA API: iniciar_lucia/detener_lucia/turno/estado | "
                "consulta_gratis/listar_catalogo_gratis/seleccionar_modelo_gratis/benchmark_gratis | "
                "perfilar_pc/listar_ia_local/descargar_modelo_local/preguntar_ia_local | "
                "abrir_constructor/estado_constructor/unificar_constructor/crear_carpeta_lc | "
                "version_sesion/estado_version_sesion | "
                "refactorizar_sistema/actualizar_sistema_hrctnr/ciclo_cierre_hrctnr/gestor_hrctnr/estado_hrctnr | "
                "bitacora/documentar_respaldos/cierre_integracion/informe_cierre | "
                "minar/validar_cadena | decir/silenciar | exportar_pesos/pin_ipfs/restaurar_pesos | texto_capacidades.")
    @classmethod
    def texto_capacidades(cls) -> str:
        return "Soy LucIA y SI puedo: " + " ".join(cls.CAPACIDADES.values())
    @staticmethod
    def diagnostico_orquestador() -> Dict[str, Any]:
        from LC.celebro.CMFG.SBSTM.IAFREE import get_cliente_iafree
        from LC.compatibility.architecture import architecture_snapshot
        cliente_free = get_cliente_iafree()
        return {"orquestador_version": __version__, "iafree_activo": cliente_free.esta_autenticado(),
                "modelos_gratuitos_total": len(cliente_free.gestor.listar_modelos()),
                "modelo_predeterminado": cliente_free.gestor.obtener_modelo_activo()["id"],
                "costo_acumulado": cliente_free.obtener_metricas_consumo()["costo_acumulado_usd"],
                "arquitectura": architecture_snapshot(), "timestamp": time.time()}

# CLASE 10: LucIACicloVida — ciclo de vida del orquestador: iniciar, detener, turno de diálogo y estado global.
class LucIACicloVida:
    def iniciar(self) -> "OrquestadorSistemaLucIA":
        if not self.inicializar_subsistemas():
            raise RuntimeError("[mainLCSTM] Fallo inicializando subsistemas.")
        self.activa = True
        return self

    def detener(self) -> None: self.cerrar_sistema()

    def turno(self, prompt: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        self.procesar_turno_dialogo(prompt)
        return {"turno": self.turno_actual, "segundos": round(time.perf_counter() - t0, 2)}

    def estado_sistema(self) -> Dict[str, Any]:
        try:
            valida, _ = self.servidor_bks.validar_cadena()
        except Exception:
            valida = False
        try:
            mod = (self.cliente_iafree.gestor.obtener_modelo_activo()["id"]
                   if self.cliente_iafree else "N/A")
        except Exception:
            mod = "N/A"
        return {"sesion": self.sesion_id, "activa": self.activa,
                "turnos": self.turno_actual, "cadena_valida": bool(valida),
                "bloques": len(self.servidor_bks.cadena) if self.servidor_bks else 0,
                "neuronas": len(self.conversor_psn.neuronas) if self.conversor_psn else 0,
                "modelo_activo": mod, "ia_local_lista": self.ia_local_lista,
                "rotacion_ia": self.rotador_ia.estado() if self.rotador_ia else {},
                "sintesis_neuronal": bool(self.sintetizador_lucia),
                "constructor_activo": bool(self.gestor_hrctrc and self.gestor_hrctrc.esta_activa())}

# CLASE 11: STMRFMNMixin — expone STMRFMNFacade como métodos de instancia y añade los comandos interactivos hrctnr.
class STMRFMNMixin:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.stmrfmn: STMRFMNFacade = STMRFMNFacade()
        super().__init__(*args, **kwargs)

    @property
    def gestor_stmrfmn(self) -> GestorSTMRFMN: return self.stmrfmn.gestor
    def refactorizar_sistema(self, overlay="", mostrar_barra=True, usar_cache=False) -> Dict[str, Any]:
        return self.stmrfmn.refactorizar(overlay=overlay, mostrar_barra=mostrar_barra, usar_cache=usar_cache)
    def actualizar_sistema_hrctnr(self, confirmar=False) -> Dict[str, Any]: return self.stmrfmn.actualizar(confirmar=confirmar)
    def ciclo_cierre_hrctnr(self) -> Dict[str, Any]: return self.stmrfmn.ciclo_cierre()
    def gestor_hrctnr(self) -> Any: return self.stmrfmn.gestor
    def estado_hrctnr(self, forzar=False, usar_cache=True) -> Dict[str, Any]:
        return self.stmrfmn.estado(forzar=forzar, usar_cache=usar_cache)
    def confirmar_actualizacion_hrctnr(self) -> str: return self.stmrfmn.confirmar_actualizacion()
    def monitor_sistema(self, forzar=False, usar_cache=True) -> Dict[str, Any]:
        return self.stmrfmn.monitor_sistema(forzar=forzar, usar_cache=usar_cache)
    def listar_modulos(self) -> List[Dict[str, Any]]: return self.stmrfmn.listar_modulos()
    def listar_neuronas(self) -> List[Dict[str, Any]]: return self.stmrfmn.listar_neuronas()
    def seleccionar_clases(self) -> List[Dict[str, Any]]: return self.stmrfmn.seleccionar_clases()
    def diagnostico_stmrfmn(self) -> Dict[str, Any]: return self.stmrfmn.diagnostico()
    def limpiar_cache_stmrfmn(self) -> None: self.stmrfmn.limpiar_cache()

    def _hrctnr_refactor_inicial(self) -> None:
        try:
            print("  [9/9] Refactor Neural HRCNTR:")
            reporte = self.stmrfmn.refactorizar(mostrar_barra=True)
            print(f"        \033[38;5;48m{reporte.get('mensaje')}\033[0m")
        except Exception as exc:
            print(f"  [9/9] Refactor Neural HRCNTR: \033[38;5;214mAVISO ({exc})\033[0m")

    def _cmd_hrctnr(self) -> None:
        try:
            print("  HRCNTR - Monitor y Refactorizador Neural")
            estado = self.estado_hrctnr()
            print(f"  Archivos: {estado['monitoreo']['archivos']}"
                  f" | Oversized: {estado['monitoreo']['oversized']}"
                  f" | Generados: {estado['generados']}")
            print(self.confirmar_actualizacion_hrctnr())
            entrada = input("  Confirmar [S/N]? ").strip().lower()
            if entrada in ("s", "si", "sí", "y", "yes"):
                reporte = self.actualizar_sistema_hrctnr(confirmar=True)
                print(f"  {reporte.get('mensaje')}")
            else:
                print("  Actualizacion cancelada.")
        except Exception as exc:
            print(f"  [hrctnr] {exc}")

# CLASE 12: OrquestadorSistemaLucIA — orquestador maestro: integra BKSVCB, IAFREE, STYLOS, las 50 neuronas y cada gestor.
class OrquestadorSistemaLucIA(STMRFMNMixin, LucIACicloVida, PRTLUCMixin, TRNLUCMixin, CMDLUCMixin):
    """Punto de entrada único al sistema cognitivo LucIA."""
    descargas: Final = GestorDescargasIA
    constructor: Final = GestorConstructor
    integracion: Final = GestorIntegracion
    blockchain: Final = GestorBlockchain
    voz: Final = GestorVoz
    iafree: Final = GestorIAFree
    pesos: Final = GestorPesos
    sistema: Final = GestorSistema

# CLASE 13: ContextoOrquestadorLucIA — gestor de contexto ("with ...") que abre y cierra el orquestador de forma segura.
class ContextoOrquestadorLucIA:
    def __init__(self) -> None:
        self.orquestador = OrquestadorSistemaLucIA()

    def __enter__(self) -> OrquestadorSistemaLucIA: return self.orquestador.iniciar()

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.orquestador.detener()
        return False

# WRAPPERS DE COMPATIBILIDAD — funciones de módulo que delegan en su clase-gestor; conservan todos los nombres de __all__.
def _reexportar(nombre: str) -> Any:
    mapa = {
        "IAFREE": "LC.celebro.CMFG.SBSTM.IAFREE", "STYLOS": "LC.celebro.CMFG.SBSTM.STYLOS",
        "RPLC": "LC.celebro.CMFG.SBSTM.RPLC", "VOZ": "LC.celebro.CMFG.SBSTM.voice_engine",
        "DSIALCLGRG": "LC.celebro.CMFG.SBSTM.DSIALCLGRG", "MDSTM": "LC.modelosIAlocal.MDSTM",
        "HRCTRC": "LC.celebro.CMFG.SBSTM.HRCTRC", "HRCTRC_RFCT": "LC.celebro.CMFG.SBSTM.HRCTRC_RFCT",
        "HRCNTR": "LC.celebro.CMFG.SBSTM.HRCNTR", "INTEGRACIONRF": "LC.celebro.CMFG.SBSTM.INTEGRACIONRF",
        "MIXINS": "LC.celebro.CMFG.SBSTM",
    }
    return import_legacy(mapa[nombre])

def subsistema(nombre: str) -> Any: return _reexportar(nombre.upper())

# Ciclo de vida programático
def iniciar_lucia() -> OrquestadorSistemaLucIA: return OrquestadorSistemaLucIA().iniciar()
def detener_lucia(orq: OrquestadorSistemaLucIA) -> None: orq.detener()
def turno(orq: OrquestadorSistemaLucIA, prompt: str) -> Dict[str, Any]: return orq.turno(prompt)
def estado(orq: OrquestadorSistemaLucIA) -> Dict[str, Any]: return orq.estado_sistema()

# IA local y descargas -> GestorDescargasIA
def perfilar_pc() -> Dict[str, Any]: return GestorDescargasIA.perfilar_pc()
def listar_ia_local() -> List[Dict[str, Any]]: return GestorDescargasIA.listar_modelos_locales()
def descargar_modelo_local(modelo_id: str) -> Dict[str, Any]: return GestorDescargasIA.descargar(modelo_id)
def preguntar_ia_local(prompt: str) -> Dict[str, Any]: return GestorDescargasIA.preguntar(prompt)

# Constructor HRCTRC -> GestorConstructor
def abrir_constructor(sesion_id: str = "") -> Dict[str, Any]: return GestorConstructor.abrir(sesion_id)
def estado_constructor() -> Dict[str, Any]: return GestorConstructor.estado()
def unificar_constructor() -> Dict[str, Any]: return GestorConstructor.unificar()
def crear_carpeta_lc(ruta_rel, tambien_en_lc=False) -> Dict[str, Any]:
    return GestorConstructor.crear_carpeta(ruta_rel, tambien_en_lc=tambien_en_lc)
def version_sesion(overlay: str = "") -> Dict[str, Any]: return GestorConstructor.version_sesion(overlay)
def estado_version_sesion(overlay: str = "") -> Dict[str, Any]: return GestorConstructor.estado_version_sesion(overlay)

# HRCNTR / STMRFMN -> STMRFMNFacade (instancia compartida del gestor global)
def _stmrfmn() -> GestorSTMRFMN: return get_gestor_stmrfmn()
def refactorizar_sistema(overlay="", mostrar_barra=True, usar_cache=False) -> Dict[str, Any]:
    return _stmrfmn().refactorizar(overlay=overlay, mostrar_barra=mostrar_barra, usar_cache=usar_cache)
def actualizar_sistema_hrctnr(confirmar=False) -> Dict[str, Any]: return _stmrfmn().actualizar(confirmar=confirmar)
def ciclo_cierre_hrctnr() -> Dict[str, Any]: return _stmrfmn().ciclo_cierre()
def gestor_hrctnr() -> Any: return _stmrfmn()
def estado_hrctnr(forzar=False, usar_cache=True) -> Dict[str, Any]: return _stmrfmn().estado(forzar=forzar, usar_cache=usar_cache)
def confirmar_actualizacion_hrctnr() -> str: return _stmrfmn().confirmar_actualizacion()
def monitor_sistema(forzar=False, usar_cache=True) -> Dict[str, Any]: return _stmrfmn().monitor_sistema(forzar=forzar, usar_cache=usar_cache)
def listar_modulos() -> List[Dict[str, Any]]: return _stmrfmn().listar_modulos()
def listar_neuronas() -> List[Dict[str, Any]]: return _stmrfmn().listar_neuronas()
def seleccionar_clases() -> List[Dict[str, Any]]: return _stmrfmn().seleccionar_clases()
def diagnostico_stmrfmn() -> Dict[str, Any]: return _stmrfmn().diagnostico()
def limpiar_cache_stmrfmn() -> None: _stmrfmn().limpiar_cache()

# Integración y cierre -> GestorIntegracion
def bitacora() -> Dict[str, Any]: return GestorIntegracion.bitacora()
def documentar_respaldos() -> Dict[str, Any]: return GestorIntegracion.documentar_respaldos()
def cierre_integracion() -> Dict[str, Any]: return GestorIntegracion.cierre()
def informe_cierre(reporte: Dict[str, Any]) -> str: return GestorIntegracion.informe_cierre(reporte)

# Blockchain BKSVCB -> GestorBlockchain
def minar(orq: OrquestadorSistemaLucIA) -> Dict[str, Any]: return GestorBlockchain.minar(orq)
def validar_cadena(orq: OrquestadorSistemaLucIA) -> Dict[str, Any]: return GestorBlockchain.validar_cadena(orq)

# Voz -> GestorVoz
def decir(texto: str, emocion=None) -> None: GestorVoz.decir(texto, emocion)
def silenciar() -> None: GestorVoz.silenciar()
def motor_voz() -> Any: return GestorVoz.motor()

# IAFREE -> GestorIAFree
def consulta_gratis(prompt: str) -> str: return GestorIAFree.consulta(prompt)
def listar_catalogo_gratis() -> List[Dict[str, Any]]: return GestorIAFree.catalogo()
def seleccionar_modelo_gratis(modelo_id: str) -> bool: return GestorIAFree.seleccionar(modelo_id)
def benchmark_gratis(max_modelos=3) -> Dict[str, float]: return GestorIAFree.benchmark(max_modelos)

# Pesos y PSNRL -> GestorPesos
def exportar_pesos(orq: OrquestadorSistemaLucIA, etiqueta="") -> Dict[str, str]: return GestorPesos.exportar(orq, etiqueta)
def pin_ipfs() -> Dict[str, Any]: return GestorPesos.pin_ipfs()
def restaurar_pesos(cid: str) -> Dict[str, Any]: return GestorPesos.restaurar(cid)

# Acceso directo a gestores externos
def gestor_mdstm() -> Any:
    from LC.modelosIAlocal.MDSTM import get_gestor_mdstm
    return get_gestor_mdstm()
def gestor_hrctrc() -> Any:
    from LC.celebro.CMFG.SBSTM.HRCTRC import get_gestor_hrctrc
    return get_gestor_hrctrc()
def integrador() -> Any:
    from LC.celebro.CMFG.SBSTM.INTEGRACIONRF import get_integrador
    return get_integrador()

# Metadatos del sistema -> GestorSistema
def version_sistema() -> Dict[str, str]: return GestorSistema.version()
def ayuda_api() -> str: return GestorSistema.ayuda_api()
def texto_capacidades() -> str: return GestorSistema.texto_capacidades()
def obtener_diagnostico_orquestador() -> Dict[str, Any]: return GestorSistema.diagnostico_orquestador()

CAPACIDADES: Final[Dict[str, str]] = GestorSistema.CAPACIDADES

# EXPORTACIÓN PÚBLICA Y PUNTO DE ENTRADA CLI
__all__: Final[List[str]] = [
    "OrquestadorSistemaLucIA", "STMRFMNFacade", "STMRFMNMixin", "ContextoOrquestadorLucIA", "GestorDescargasIA",
    "GestorConstructor", "GestorIntegracion", "GestorBlockchain", "GestorVoz", "GestorIAFree", "GestorPesos",
    "GestorSistema", "subsistema", "iniciar_lucia", "detener_lucia", "turno", "estado", "perfilar_pc",
    "listar_ia_local", "descargar_modelo_local", "preguntar_ia_local", "abrir_constructor", "estado_constructor",
    "unificar_constructor", "crear_carpeta_lc", "version_sesion", "estado_version_sesion", "refactorizar_sistema",
    "actualizar_sistema_hrctnr", "ciclo_cierre_hrctnr", "gestor_hrctnr", "estado_hrctnr",
    "confirmar_actualizacion_hrctnr", "monitor_sistema", "listar_modulos", "listar_neuronas", "seleccionar_clases",
    "diagnostico_stmrfmn", "limpiar_cache_stmrfmn", "bitacora", "documentar_respaldos", "cierre_integracion",
    "informe_cierre", "minar", "validar_cadena", "decir", "silenciar", "consulta_gratis", "listar_catalogo_gratis",
    "seleccionar_modelo_gratis", "benchmark_gratis", "exportar_pesos", "pin_ipfs", "restaurar_pesos",
    "gestor_mdstm", "gestor_hrctrc", "integrador", "motor_voz", "version_sistema", "ayuda_api",
    "texto_capacidades", "obtener_diagnostico_orquestador", "main"
]

def main() -> int:
    orquestador = OrquestadorSistemaLucIA()
    if not orquestador.inicializar_subsistemas():
        sys.stderr.write("[mainLCSTM] Fallo durante la inicializacion de subsistemas.\n")
        return 1
    try:
        orquestador.ejecutar_bucle_interactivo()
        return 0
    except Exception as err:
        sys.stderr.write(f"[mainLCSTM] Error no controlado: {err}\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
