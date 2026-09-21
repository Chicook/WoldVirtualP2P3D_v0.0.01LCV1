"""bases_neuro - Micro-base compartida de los modulos cerebrales de LucIA."""
import logging
import time

logger = logging.getLogger("lucIA.ModuloCerebral")


class ModuloCerebral:
    """Base: estado en RAM, sin disco. Al cerrar, a_pesos() lo vuelca."""
    nombre = "Modulo"

    def __init__(self):
        self.eventos = []
        self._t0 = time.time()

    def _notar(self, texto: str) -> None:
        self.eventos.append({"ts": time.time(), "e": str(texto)[:200]})
        if len(self.eventos) > 50:
            self.eventos.pop(0)

    def a_pesos(self, conversor) -> int:
        """Vuelca el estado al conversor. Retorna nº de consolidaciones."""
        hechos = 0
        for ev in self.eventos:
            try:
                conversor.actualizar_memoria_salida(f"[{self.nombre}]", ev["e"], self.nombre)
                hechos += 1
            except Exception as e:
                logger.debug(f"{self.nombre}: sin consolidar ({e})")
        self.eventos.clear()
        return hechos

    def resumen(self) -> str:
        return f"{self.nombre}: {len(self.eventos)} eventos en RAM"
