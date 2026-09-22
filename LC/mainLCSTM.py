"""
mainLCSTM.py - Orquestador Central del Sistema Cognitivo LucIA (2026)
=====================================================================
Clase delgada (mixins PRTLUC/TRNLUC/CMDLUC) + fachada API programatica:
re-exporta cada subsistema y expone funciones de conveniencia para usar
a LucIA desde codigo: turnos, IA local, constructor, refactor, integracion,
blockchain, voz y diagnostico. Todo importado correctamente desde SBSTM/.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, Final, List, Optional

if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from LC.compatibility import import_legacy

from LC.celebro.CMFG.SBSTM.BASELUC import (
    EstiloTerminalLucIA,
    get_cliente_iafree,
)
from LC.celebro.CMFG.SBSTM.CMDLUC import CMDLUCMixin
from LC.celebro.CMFG.SBSTM.PRTLUC import PRTLUCMixin
from LC.celebro.CMFG.SBSTM.TRNLUC import TRNLUCMixin

__version__: Final[str] = "2026.3.1"


class OrquestadorSistemaLucIA(PRTLUCMixin, TRNLUCMixin, CMDLUCMixin):
    """
    Orquestador maestro que integra BKSVCB, SNSBSTNPRB, IAFREE, STYLOS y las 50 neuronas.
    """


# ─── RE-EXPORT OFICIAL DE SUBSISTEMAS ─────────────────────────────────────
def _reexportar(nombre: str) -> Any:
    """Importa perezosamente un simbolo de SBSTM sin romper el arranque."""
    mapa = {
        "IAFREE": "LC.celebro.CMFG.SBSTM.IAFREE",
        "STYLOS": "LC.celebro.CMFG.SBSTM.STYLOS",
        "RPLC": "LC.celebro.CMFG.SBSTM.RPLC",
        "VOZ": "LC.celebro.CMFG.SBSTM.voice_engine",
        "DSIALCLGRG": "LC.celebro.CMFG.SBSTM.DSIALCLGRG",
        "MDSTM": "LC.modelosIAlocal.MDSTM",
        "HRCTRC": "LC.celebro.CMFG.SBSTM.HRCTRC",
        "HRCTRC_RFCT": "LC.celebro.CMFG.SBSTM.HRCTRC_RFCT",
        "INTEGRACIONRF": "LC.celebro.CMFG.SBSTM.INTEGRACIONRF",
        "MIXINS": "LC.celebro.CMFG.SBSTM",
    }
    return import_legacy(mapa[nombre])


def subsistema(nombre: str) -> Any:
    """Devuelve el modulo de un subsistema: IAFREE, STYLOS, RPLC, VOZ,
    DSIALCLGRG, MDSTM, HRCTRC, HRCTRC_RFCT o INTEGRACIONRF."""
    return _reexportar(nombre.upper())


# ─── CICLO DE VIDA PROGRAMATICO ──────────────────────────────────────────
def iniciar_lucia() -> OrquestadorSistemaLucIA:
    """Crea el orquestador e inicializa todos los subsistemas (pasos 0-8)."""
    orq = OrquestadorSistemaLucIA()
    if not orq.inicializar_subsistemas():
        raise RuntimeError("[mainLCSTM] Fallo inicializando subsistemas.")
    return orq


def detener_lucia(orq: OrquestadorSistemaLucIA) -> None:
    """Cierre ordenado: refactor -> .md -> pesos -> IPFS -> devopencode."""
    orq.cerrar_sistema()


def turno(orq: OrquestadorSistemaLucIA, prompt: str) -> Dict[str, Any]:
    """Ejecuta un turno de dialogo y devuelve su telemetria basica."""
    t0 = time.perf_counter()
    orq.procesar_turno_dialogo(prompt)
    return {"turno": orq.turno_actual, "segundos": round(time.perf_counter() - t0, 2)}


def estado(orq: OrquestadorSistemaLucIA) -> Dict[str, Any]:
    """Foto del sistema: cadena, neuronas, deriva, turnos y modelos."""
    try:
        valida, _ = orq.servidor_bks.validar_cadena()
    except Exception:
        valida = False
    try:
        mod = orq.cliente_iafree.gestor.obtener_modelo_activo()["id"] if orq.cliente_iafree else "N/A"
    except Exception:
        mod = "N/A"
    return {"sesion": orq.sesion_id, "activa": orq.activa, "turnos": orq.turno_actual,
            "cadena_valida": bool(valida),
            "bloques": len(orq.servidor_bks.cadena) if orq.servidor_bks else 0,
            "neuronas": len(orq.conversor_psn.neuronas) if orq.conversor_psn else 0,
            "modelo_activo": mod,
            "ia_local_lista": orq.ia_local_lista,
            "constructor_activo": bool(orq.gestor_hrctrc and orq.gestor_hrctrc.esta_activa())}


# ─── IA LOCAL Y DESCARGAS (DSIALCLGRG + MDSTM) ────────────────────────────
def perfilar_pc() -> Dict[str, Any]:
    """Registra RAM/VRAM del PC en LC/modelosIAlocal/perfil_hardware.json."""
    from LC.celebro.CMFG.SBSTM.DSIALCLGRG import perfilar_hardware
    return perfilar_hardware(guardar=True)


def listar_ia_local() -> List[Dict[str, Any]]:
    """Modelos ligeros fisicamente presentes en LC/modelosIAlocal."""
    from LC.celebro.CMFG.SBSTM.DSIALCLGRG import listar_modelos_locales
    return listar_modelos_locales()


def descargar_modelo_local(modelo_id: str) -> Dict[str, Any]:
    """Descarga autonoma real de un modelo ligero (Ollama o GGUF)."""
    from LC.modelosIAlocal.MDSTM import get_gestor_mdstm
    return get_gestor_mdstm().descargar(modelo_id)


def preguntar_ia_local(prompt: str) -> Dict[str, Any]:
    """Consulta al modelo local (fallback cuando OpenRouter falla)."""
    from LC.celebro.CMFG.SBSTM.DSIALCLGRG import consultar_lucia_local
    texto, mid, lat = consultar_lucia_local(prompt, None)
    return {"texto": texto, "modelo": mid, "latencia_ms": lat}


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


# ─── BLOCKCHAIN BKSVCB ───────────────────────────────────────────────────
def minar(orq: OrquestadorSistemaLucIA) -> Dict[str, Any]:
    """Mina transacciones pendientes y devuelve el bloque nuevo."""
    blk = orq.servidor_bks.minar_transacciones_pendientes()
    if not blk:
        return {"exito": False, "mensaje": "Sin transacciones pendientes."}
    return {"exito": True, "indice": blk.indice, "hash": blk.hash_bloque}


def validar_cadena(orq: OrquestadorSistemaLucIA) -> Dict[str, Any]:
    """Valida la cadena de bloques (inmutabilidad)."""
    valida, err = orq.servidor_bks.validar_cadena()
    return {"valida": bool(valida), "error": err,
            "bloques": len(orq.servidor_bks.cadena)}


# ─── VOZ ─────────────────────────────────────────────────────────────────
def decir(texto: str, emocion: Optional[float] = None) -> None:
    """Emite texto con la voz neuronal de LucIA (no bloquea)."""
    from LC.celebro.CMFG.SBSTM.voice_engine import speak
    speak(texto, esperar=False, emocion=emocion)


def silenciar() -> None:
    """Interrumpe inmediatamente la voz de LucIA."""
    from LC.celebro.CMFG.SBSTM.voice_engine import cancel_speech
    cancel_speech()


# ─── IAFREE (INFERENCIA GRATUITA) ────────────────────────────────────────
def consulta_gratis(prompt: str) -> str:
    """Respuesta de LucIA a coste $0.00 con rotacion automatica de modelos."""
    from LC.celebro.CMFG.SBSTM.IAFREE import consultar_lucia_gratis
    return consultar_lucia_gratis(prompt)


def listar_catalogo_gratis() -> List[Dict[str, Any]]:
    """Catalogo de modelos :free disponibles en OpenRouter."""
    from LC.celebro.CMFG.SBSTM.IAFREE import listar_catalogo_gratis as _l
    return _l()


def seleccionar_modelo_gratis(modelo_id: str) -> bool:
    """Fija el modelo gratuito activo por su identificador."""
    from LC.celebro.CMFG.SBSTM.IAFREE import get_cliente_iafree as _g
    return _g().gestor.seleccionar_por_id(modelo_id)


def benchmark_gratis(max_modelos: int = 3) -> Dict[str, float]:
    """1 llamada corta por modelo para medir latencia sin quemar cuota."""
    from LC.celebro.CMFG.SBSTM.IAFREE import get_cliente_iafree as _g
    return _g().benchmark_rapido_modelos(max_modelos=max_modelos)


# ─── PESOS Y PSNRL ───────────────────────────────────────────────────────
def exportar_pesos(orq: OrquestadorSistemaLucIA, etiqueta: str = "") -> Dict[str, str]:
    """Persiste los pesos de las 50 neuronas en PSNRL (npz + json)."""
    npz, js = orq.conversor_psn.persistir_pesos_en_psnrl(
        etiqueta=etiqueta or f"export_{orq.sesion_id}")
    return {"npz": str(npz), "json": str(js)}


def pin_ipfs() -> Dict[str, Any]:
    """Sube PSNRL a IPFS sin borrar sin confirmacion."""
    from LC.celebro.CMFG.ipfs_manager import get_ipfs_manager
    return get_ipfs_manager().subir_y_limpiar_psnrl(forzar_borrado_sin_daemon=False)


def restaurar_pesos(cid: str) -> Dict[str, Any]:
    """Recupera pesos neuronales desde IPFS por su CID."""
    from LC.celebro.CMFG.ipfs_manager import get_ipfs_manager
    data = get_ipfs_manager().recuperar_pesos(cid)
    n = len(data) if isinstance(data, (bytes, list)) else len(str(data))
    return {"cid": cid, "bytes": n}


# ─── ACCESO DIRECTO A GESTORES ───────────────────────────────────────────
def gestor_mdstm() -> Any:
    """Singleton de descarga autonoma de modelos ligeros."""
    from LC.modelosIAlocal.MDSTM import get_gestor_mdstm as _g
    return _g()


def gestor_hrctrc() -> Any:
    """Herrero constructor: overlay en LC/Constructor + unificacion."""
    from LC.celebro.CMFG.SBSTM.HRCTRC import get_gestor_hrctrc as _g
    return _g()


def integrador() -> Any:
    """Integrador de cierre: .md -> pesos -> IPFS -> devopencode."""
    from LC.celebro.CMFG.SBSTM.INTEGRACIONRF import get_integrador as _g
    return _g()


def motor_voz() -> Any:
    """Motor de voz neuronal de LucIA (es-ES-ElviraNeural)."""
    from LC.celebro.CMFG.SBSTM.voice_engine import get_voice_engine
    return get_voice_engine()

def version_sistema() -> Dict[str, str]:
    """Versiones de cada subsistema del stack LucIA."""
    return {"orquestador": __version__, "arquitectura": "WoldVirtualP2P3D-2026",
            "regla": "400/450 lineas por modulo", "coste": "$0.00"}


def ayuda_api() -> str:
    """Lista las funciones publicas de esta fachada y que hace cada una."""
    return ("LucIA API: iniciar_lucia/detener_lucia/turno/estado | consulta_gratis/"
            "listar_catalogo_gratis/seleccionar_modelo_gratis/benchmark_gratis | "
            "perfilar_pc/listar_ia_local/descargar_modelo_local/preguntar_ia_local | "
            "abrir_constructor/estado_constructor/unificar_constructor/crear_carpeta_lc | "
            "version_sesion/estado_version_sesion | bitacora/documentar_respaldos/"
            "cierre_integracion/informe_cierre | minar/validar_cadena | decir/silenciar | "
            "exportar_pesos/pin_ipfs/restaurar_pesos | texto_capacidades.")


# ─── CAPACIDADES FACTUALES DE LUCIA ──────────────────────────────────────
CAPACIDADES: Final[Dict[str, str]] = {
    "descargas_ia": "MDSTM descarga modelos ligeros reales en LC/modelosIAlocal.",
    "ia_local": "DSIALCLGRG responde offline con Ollama/GGUF si OpenRouter falla.",
    "constructor": "HRCTRC crea carpetas y versiona el sistema en LC/Constructor.",
    "refactor": "HRCTRC_RFCT divide oversized a 400/450 con modelo local.",
    "integracion": "INTEGRACIONRF: .md en CHG -> pesos -> IPFS -> devopencode.",
    "voz": "voice_engine habla con voz neuronal es-ES y cancela al instante.",
    "blockchain": "BKSVCB registra cada turno y mina cada 3 turnos.",
}


def texto_capacidades() -> str:
    """Frase factual de capacidades para inyectar en el contexto de LucIA."""
    return "Soy LucIA y SI puedo: " + " ".join(CAPACIDADES.values())


# ─── GESTOR DE CONTEXTO PARA INTEGRACIONES EXTERNAS ─────────────────────────
class ContextoOrquestadorLucIA:
    """Gestor de contexto para pruebas, evaluacion o invocacion programatica."""

    def __init__(self) -> None:
        self.orquestador = OrquestadorSistemaLucIA()

    def __enter__(self) -> OrquestadorSistemaLucIA:
        if not self.orquestador.inicializar_subsistemas():
            raise RuntimeError("Fallo durante la inicializacion de subsistemas en LucIA")
        return self.orquestador

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        self.orquestador.cerrar_sistema()
        return False


# ─── SERVICIOS DE CONSULTA Y DIAGNOSTICO PROGRAMATICO ───────────────────────
def obtener_diagnostico_orquestador() -> Dict[str, Any]:
    """Genera un reporte estructural para validar la salud de todo el stack."""
    from LC.celebro.CMFG.SBSTM.IAFREE import get_cliente_iafree
    from LC.compatibility.architecture import architecture_snapshot
    cliente_free = get_cliente_iafree()
    return {
        "orquestador_version": "2026.3.1",
        "iafree_activo": cliente_free.esta_autenticado(),
        "modelos_gratuitos_total": len(cliente_free.gestor.listar_modelos()),
        "modelo_predeterminado": cliente_free.gestor.obtener_modelo_activo()["id"],
        "costo_acumulado": cliente_free.obtener_metricas_consumo()["costo_acumulado_usd"],
        "arquitectura": architecture_snapshot(),
        "timestamp": time.time(),
    }

__all__: Final[List[str]] = [
    "OrquestadorSistemaLucIA",
    "ContextoOrquestadorLucIA",
    "subsistema",
    "iniciar_lucia",
    "detener_lucia",
    "turno",
    "estado",
    "perfilar_pc",
    "listar_ia_local",
    "descargar_modelo_local",
    "preguntar_ia_local",
    "abrir_constructor",
    "estado_constructor",
    "unificar_constructor",
    "crear_carpeta_lc",
    "version_sesion",
    "estado_version_sesion",
    "bitacora",
    "documentar_respaldos",
    "cierre_integracion",
    "informe_cierre",
    "minar",
    "validar_cadena",
    "decir",
    "silenciar",
    "consulta_gratis",
    "listar_catalogo_gratis",
    "seleccionar_modelo_gratis",
    "benchmark_gratis",
    "exportar_pesos",
    "pin_ipfs",
    "restaurar_pesos",
    "gestor_mdstm",
    "gestor_hrctrc",
    "integrador",
    "motor_voz",
    "version_sistema",
    "ayuda_api",
    "texto_capacidades",
    "obtener_diagnostico_orquestador",
    "main",
]

# ─── FUNCION PRINCIPAL DE ENTRADA AL SISTEMA ─────────────────────────────────
def main() -> int:
    """Punto de arranque del orquestador unificado mainLCSTM."""
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
