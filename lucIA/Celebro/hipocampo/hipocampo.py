"""
lucIA.Celebro.hipocampo.hipocampo - Buffer de corto plazo y consolidacion.
==========================================================================
Solo stdlib + numpy. No toca disco: el buffer vive en RAM; el largo plazo
sigue siendo pesos de Celebro + IPFS (lo gestiona session_manager).
"""

import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("lucIA.Hipocampo")

MAX_BUFFER = 30          # turnos en corto plazo (igual ventana que memory_manager)
MAX_CHARS = 800          # tope por texto (igual que memory_manager)
UMBRAL_EMOCION_DIRECTA = 0.3   # |emocion| sobre esto => consolidacion inmediata


def _importancia(emocion: float, delta: float, novedad: float) -> float:
    """0.5*|emocion| + 0.3*delta_norm + 0.2*novedad, en [0, 1]."""
    e = min(1.0, abs(float(emocion)))
    d = min(1.0, float(delta) / 5.0) if delta > 0 else 0.0
    n = min(1.0, max(0.0, float(novedad)))
    return round(0.5 * e + 0.3 * d + 0.2 * n, 4)


class Hipocampo:
    """Consolidador de memoria a corto -> largo plazo."""

    def __init__(self, max_buffer: int = MAX_BUFFER):
        self.max_buffer = max_buffer
        self.buffer: List[Dict[str, Any]] = []   # turnos pendientes
        self.consolidados = 0
        self._media_vector: Optional[list] = None

    # -- registro ----------------------------------------------------
    def registrar(self, pregunta: str, respuesta: str, modelo: str = "LucIA_Core",
                  emocion: float = 0.0, delta: float = 0.0,
                  vector: Optional[list] = None) -> Dict[str, Any]:
        """Guarda un turno en el buffer con su importancia. Lo mas antiguo se
        comprime en resumen si se supera el tope (igual que memory_manager)."""
        novedad = self._novedad(vector)
        imp = _importancia(emocion, delta, novedad)
        turno = {
            "ts": time.time(),
            "user": (pregunta or "")[:MAX_CHARS],
            "lucia": (respuesta or "")[:MAX_CHARS],
            "modelo": modelo,
            "emocion": round(float(emocion), 3),
            "delta": round(float(delta), 5),
            "importancia": imp,
            "consolidado": False,
        }
        self.buffer.append(turno)
        if vector:
            self._actualizar_media(vector)
        if len(self.buffer) > self.max_buffer:
            self.buffer.pop(0)  # sale lo mas antiguo (ya resumido por memory_manager)
        if abs(float(emocion)) >= UMBRAL_EMOCION_DIRECTA:
            turno["consolidacion_directa"] = True
        return turno

    def _novedad(self, vector: Optional[list]) -> float:
        if not vector or not self._media_vector:
            return 0.5
        try:
            dist = math.dist(list(vector)[:4], self._media_vector[:4])
            return min(1.0, dist / 2.0)
        except Exception:
            return 0.5

    def _actualizar_media(self, vector: list) -> None:
        try:
            v = [float(x) for x in list(vector)[:4]]
            if self._media_vector is None:
                self._media_vector = v
            else:
                self._media_vector = [
                    0.9 * m + 0.1 * x for m, x in zip(self._media_vector, v)
                ]
        except Exception:
            pass

    # -- consolidacion (replay) ---------------------------------------
    def pendientes(self) -> List[Dict[str, Any]]:
        return [t for t in self.buffer if not t.get("consolidado")]

    def consolidar(self, conversor, n: int = 5) -> int:
        """Pasa los n pendientes mas importantes por el conversor
        (actualizar_memoria_salida, factor x importancia). Devuelve cuantos."""
        pend = sorted(self.pendientes(), key=lambda t: t["importancia"], reverse=True)[:n]
        hechos = 0
        for t in pend:
            try:
                factor = 0.5 + t["importancia"]  # 0.5 .. 1.5
                v = conversor.encoder.encode_pair(t["user"], t["lucia"])
                act, _ = conversor._propagar_todas_las_neuronas(v)
                conversor._actualizar_pesos_en_todas_las_neuronas(
                    v, act, factor=factor, fase="hipocampo")
                t["consolidado"] = True
                self.consolidados += 1
                hechos += 1
            except Exception as e:
                logger.debug(f"Hipocampo: no se pudo consolidar turno: {e}")
        if hechos:
            logger.info(f"Hipocampo: {hechos} turnos consolidados en pesos.")
        return hechos

    # -- cierre de sesion ----------------------------------------------
    def volcar_cierre(self, conversor, sesion=None) -> Dict[str, Any]:
        """Consolida TODO lo pendiente, guarda .npz y lo registra en la sesion
        para subida a IPFS. El buffer queda a 0 (nada generado queda en RAM).
        No borra nada de disco (eso lo hace session_manager.cerrar)."""
        total = len(self.pendientes())
        hechos = 0
        while self.pendientes():
            hechos += self.consolidar(conversor, n=10)
        archivo = None
        if hechos:
            try:
                celebro_dir = Path(__file__).parent.parent.resolve()
                archivo = conversor.guardar_pesos_en_celebro(
                    archivo=celebro_dir / "pesos_hipocampo.npz")
                if sesion is not None:
                    sesion.registrar_archivo_pesos(archivo)
            except Exception as e:
                logger.error(f"Hipocampo: error guardando pesos de cierre: {e}")
        self.buffer.clear()
        self._media_vector = None
        logger.info(f"Hipocampo: cierre con {hechos}/{total} turnos -> pesos.")
        return {"pendientes": total, "consolidados": hechos,
                "archivo": str(archivo) if archivo else None,
                "buffer_restante": 0}

    def resumen_estado(self) -> str:
        imp_media = (round(sum(t["importancia"] for t in self.buffer)
                           / len(self.buffer), 3) if self.buffer else 0.0)
        return (f"Buffer corto plazo: {len(self.buffer)}/{self.max_buffer} | "
                f"Consolidados: {self.consolidados} | "
                f"Importancia media: {imp_media}")


_hipocampo_global: Optional[Hipocampo] = None


def get_hipocampo() -> Hipocampo:
    """Singleton del hipocampo (igual patron que get_memory_manager)."""
    global _hipocampo_global
    if _hipocampo_global is None:
        _hipocampo_global = Hipocampo()
    return _hipocampo_global
