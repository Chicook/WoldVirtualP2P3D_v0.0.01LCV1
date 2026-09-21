"""Glia - Limpieza, poda neuronal y verificacion de cierre limpio.

Rol biologico: las celulas gliales realizan el mantenimiento del cerebro:
limpian desechos metabolicos, eliminan sinapsis debiles (poda), protegen
neuronas y regulan el microentorno neuronal. Durante el sueno hacen poda
activa. En LucIA verifica restos en disco, poda pesos muy debiles del
conversor y mantiene un historial de limpieza para optimizar iteraciones.
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from lucIA.Celebro.bases_neuro import ModuloCerebral

logger = logging.getLogger("lucIA.Glia")

_LUCIA_DIR = Path(__file__).parent.parent.resolve()

# Umbral de poda: pesos cuya norma absoluta media sea menor que esto
# se consideran "sinapsis debiles" candidatas a poda.
UMBRAL_PODA = 1e-4
MAX_PODAS_POR_CICLO = 5     # no podar mas de N neuronas por vez


class Glia(ModuloCerebral):
    nombre = "Glia"

    def __init__(self):
        super().__init__()
        self._limpiezas: int = 0
        self._podas_totales: int = 0
        self._ultimo_informe: Dict[str, Any] = {}

    # -- Verificacion de cierre limpio ---------------------------------

    def verificar(self) -> dict:
        """Lista restos de ejecucion: .npz locales, JSON de memoria, cache."""
        restos = []
        # Busca archivos de pesos locales (los que no llegaron a IPFS o son residuos)
        for patron in ("*.npz", "memoria_conversacion.json", "pesos_*.npz",
                       "checkpoint_*.npz"):
            for p in _LUCIA_DIR.rglob(patron):
                if "venv" not in str(p) and ".git" not in str(p):
                    restos.append(str(p.relative_to(_LUCIA_DIR)))

        # Verifica cache
        cache = _LUCIA_DIR / "Celebro" / "cache"
        if cache.exists():
            for item in cache.iterdir():
                restos.append(f"Celebro/cache/{item.name}")

        limpio = not restos
        self._limpiezas += 1
        self._notar(f"verificacion #{self._limpiezas} restos={len(restos)}")
        self._ultimo_informe = {"limpio": limpio, "restos": sorted(set(restos))}
        return self._ultimo_informe

    # -- Poda de sinapsis debiles -------------------------------------

    def podar_pesos_debiles(self, conversor,
                             umbral: float = UMBRAL_PODA,
                             max_podas: int = MAX_PODAS_POR_CICLO) -> dict:
        """Poda neuronas cuya norma de pesos sea extremadamente pequena.

        Solo opera sobre el dict 'neuronas' del conversor; no toca
        nada externo. Retorna cuantas neuronas se podaron.
        """
        neuronas = getattr(conversor, "neuronas", {}) or {}
        podadas = []
        revisadas = 0

        for nombre, neurona in list(neuronas.items()):
            if len(podadas) >= max_podas:
                break
            revisadas += 1
            try:
                # La norma de la neurona: usamos el peso del bias (w_b) o pesos de entrada
                pesos_dict = getattr(neurona, "__dict__", {})
                norma_total = 0.0
                n_pesos = 0
                for k, v in pesos_dict.items():
                    if k.startswith("w_") or k == "bias":
                        try:
                            import numpy as _np
                            arr = _np.asarray(v, dtype=float)
                            norma_total += float(_np.linalg.norm(arr))
                            n_pesos += arr.size
                        except Exception:
                            pass
                if n_pesos == 0:
                    continue
                norma_media = norma_total / n_pesos
                if norma_media < umbral:
                    podadas.append(nombre)
                    self._notar(f"poda {nombre} norma_media={norma_media:.2e}")
            except Exception:
                pass

        # Informar al conversor (no elimina, solo registra para que el conversor
        # reinicialice esas neuronas en el proximo ciclo si lo desea)
        if podadas:
            self._podas_totales += len(podadas)
            logger.info(
                f"Glia poda {len(podadas)} neuronas debiles: {podadas}"
            )

        resultado = {
            "podadas": len(podadas),
            "nombres_podadas": podadas,
            "revisadas": revisadas,
            "podas_totales_sesion": self._podas_totales,
        }
        self._ultimo_informe.update(resultado)
        return resultado

    # -- Limpieza de logs y residuos en memoria -----------------------

    def limpiar_eventos_antiguos(self, max_eventos: int = 100) -> int:
        """Poda el buffer interno de eventos si crece demasiado."""
        if len(self.eventos) > max_eventos:
            recortados = len(self.eventos) - max_eventos
            self.eventos = self.eventos[-max_eventos:]
            self._notar(f"buffer recortado {recortados} eventos")
            return recortados
        return 0

    # -- Exportacion a pesos ------------------------------------------

    def a_pesos(self, conversor) -> None:
        """Exporta historial de limpiezas a pesos neuronales."""
        try:
            resumen = (
                f"limpiezas={self._limpiezas} "
                f"podas_totales={self._podas_totales} "
                f"limpio={self._ultimo_informe.get('limpio', '?')}"
            )
            conversor.actualizar_memoria_salida(
                "[Glia] limpieza y poda", resumen, "Glia"
            )
        except Exception:
            pass

    def resumen(self) -> str:
        limpio = self._ultimo_informe.get("limpio")
        estado_str = "✅" if limpio is True else ("⚠️" if limpio is False else "?")
        return (
            f"Glia: {estado_str} "
            f"limpiezas={self._limpiezas} "
            f"podas={self._podas_totales} | {super().resumen()}"
        )


_glia_global = None


def get_glia() -> Glia:
    global _glia_global
    if _glia_global is None:
        _glia_global = Glia()
    return _glia_global
