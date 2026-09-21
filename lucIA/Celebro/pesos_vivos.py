"""lucIA.Celebro.pesos_vivos - Pesos en vivo + subida a IPFS al cerrar sesion.

Contrato:
1. En tiempo de ejecucion, cada turno genera deltas que van a las neuronas
   del conversor (ENRN/RF_SL/RF_EN/RNP/SLRN) via actualizar_memoria_salida.
   Cada uno de los 15 modulos anatomicos aporta su linea de estado.
2. Al cerrar sesion, se guarda un checkpoint .npz y se registra en la sesion
   para que session_manager lo suba a IPFS y borre el local.
"""
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger("lucIA.PesosVivos")

CELEBRO_DIR = Path(__file__).parent.resolve()


def inyectar_pesos_turno(cerebro, conversor, pregunta: str, respuesta: str,
                         info_pesos: Dict[str, Any]) -> int:
    """Inyecta el turno en vivo a las neuronas. Retorna nº de inyecciones."""
    hechos = 0
    emo = float(info_pesos.get("estado_emocional", 0.0))
    delta = float(info_pesos.get("norma_delta_aplicada",
                                 info_pesos.get("norma_delta_pesos", 0.0)))
    # 1. Estado de cada modulo anatomico -> neuronas (via conversor).
    modulos = list(getattr(cerebro, "modulos", []) or [])
    if getattr(cerebro, "hipocampo", None) is not None:
        modulos.append(cerebro.hipocampo)
    for m in modulos:
        try:
            nombre = getattr(m, "nombre", type(m).__name__)
            try:
                linea = m.resumen() if hasattr(m, "resumen") else m.resumen_estado()
            except Exception:
                linea = "sin resumen"
            conversor.actualizar_memoria_salida(
                f"[{nombre} turno emo={emo:+.2f} delta={delta:.4f}] {pregunta}",
                str(linea)[:400], nombre)
            hechos += 1
        except Exception as e:
            logger.debug(f"pesos_vivos turno {m}: {e}")
    # 2. Respuesta completa -> memoria de salida (refuerzo global).
    try:
        conversor.actualizar_memoria_salida(pregunta, respuesta, "turno_vivo")
        hechos += 1
    except Exception as e:
        logger.debug(f"pesos_vivos refuerzo: {e}")
    return hechos


def checkpoint_y_registrar(conversor, sesion, etiqueta: str = "pesos_activos") -> List[Path]:
    """Guarda checkpoint .npz con timestamp y lo registra para IPFS.

    El borrado local lo hace session_manager.cerrar() tras confirmar el CID.
    Retorna lista de archivos registrados.
    """
    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    destino = CELEBRO_DIR / f"{etiqueta}_{ts}.npz"
    archivo = conversor.guardar_pesos_en_celebro(archivo=destino)
    try:
        sesion.registrar_archivo_pesos(archivo)
        logger.info(f"pesos_vivos: {archivo.name} registrado para IPFS al cerrar sesion")
    except Exception as e:
        logger.warning(f"pesos_vivos no pudo registrar {archivo}: {e}")
    return [archivo]
