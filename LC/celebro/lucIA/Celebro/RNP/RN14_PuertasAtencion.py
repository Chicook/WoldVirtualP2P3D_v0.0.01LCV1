"""
RN14_PuertasAtencion.py - Puertas de Atención por neurona (plan de la propia Lucía)
====================================================================================
Cada neurona de Celebro tiene una "puerta" (gate 0.05-1.0) que modula cuánto
aprende en cada turno según el contexto: carga emocional, carga cognitiva,
inestabilidad reciente y tendencia de la neurona (datos del mapa neuronal).

La atención es un recurso limitado: las puertas se normalizan a media 0.5
(abrir una implica entornar otra), con suelo anti-atrofia para que ninguna
neurona se cierre del todo. Media móvil lenta (EMA) para no oscilar.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger("lucIA.RN14_PuertasAtencion")


class PuertasAtencion:
    def __init__(self, suelo: float = 0.05, alfa_ema: float = 0.3):
        self.nombre = "RN14_PuertasAtencion"
        self.input_size = 4
        self.output_size = 8
        self.suelo = suelo
        self.alfa_ema = alfa_ema
        self.puertas: Dict[str, float] = {}
        self.informe_ultimo: str = "Puertas sin calibrar aún."
        # Matriz propia: RN14 tambien aprende como las demas (pesos por igual)
        self.pesos = np.zeros((4, 8), dtype=np.float32)
        self.sesgo = np.zeros((1, 8), dtype=np.float32)

    def _grupo(self, nombre: str) -> str:
        if nombre.startswith("EN"):
            return "ENRN"
        if nombre.startswith("RF_SL"):
            return "RF_SL"
        if nombre.startswith("RF_EN"):
            return "RF_EN"
        if nombre.startswith("RNP") or "Optimizador" in nombre or "Ajuste" in nombre or "Controlador" in nombre:
            return "RNP"
        return "SLRN"

    def _base_por_grupo(self, grupo: str, emo: float, cog: float, inestable: bool) -> float:
        if grupo == "RF_SL":
            return 0.5 + 0.4 * emo          # memoria se abre con emoción
        if grupo == "RF_EN":
            return 0.5 + 0.4 * cog          # decisión se abre con carga cognitiva
        if grupo == "RNP":
            return 0.8 if inestable else 0.4  # calibración solo cuando hay lío
        if grupo == "SLRN":
            return 0.6
        return 0.5                           # ENRN estable

    def calcular(self, nombres: List[str], vector_4d: np.ndarray,
                 emocion: float, tendencias: Optional[Dict[str, str]] = None,
                 inestable: bool = False) -> Dict[str, float]:
        emo = min(1.0, abs(float(emocion)))
        try:
            cog = min(1.0, abs(float(vector_4d[0, 0])))
        except Exception:
            cog = 0.5
        tendencias = tendencias or {}

        objetivos: Dict[str, float] = {}
        for n in nombres:
            g = self._grupo(n)
            v = self._base_por_grupo(g, emo, cog, inestable)
            t = tendencias.get(n, "")
            if t == "debilitándose":
                v += 0.15                    # estimular la débil
            elif t in ("creciendo",):
                v -= 0.10                    # entornar la dominante
            objetivos[n] = v

        # Recurso limitado: normalizar a media 0.5
        media = sum(objetivos.values()) / max(1, len(objetivos))
        if media > 1e-8:
            for n in objetivos:
                objetivos[n] = objetivos[n] * (0.5 / media)

        # EMA lenta + suelo anti-atrofia (sin oscilar de golpe)
        for n in nombres:
            previo = self.puertas.get(n, 0.5)
            nuevo = self.alfa_ema * float(np.clip(objetivos[n], self.suelo, 1.0)) \
                + (1.0 - self.alfa_ema) * previo
            self.puertas[n] = round(float(np.clip(nuevo, self.suelo, 1.0)), 3)

        self._generar_informe()
        return dict(self.puertas)

    def puerta(self, nombre: str) -> float:
        return float(self.puertas.get(nombre, 0.5))

    def media_grupo(self, nombres: List[str]) -> float:
        vals = [self.puertas.get(n, 0.5) for n in nombres]
        return float(sum(vals) / max(1, len(vals)))

    def _generar_informe(self) -> None:
        grupos: Dict[str, list] = {}
        for n, v in self.puertas.items():
            grupos.setdefault(self._grupo(n), []).append(v)
        medias = {g: sum(v) / len(v) for g, v in grupos.items()}
        if not medias:
            return
        abierto = max(medias, key=medias.get)
        cerrado = min(medias, key=medias.get)
        self.informe_ultimo = (
            f"Puertas: abro {abierto} ({medias[abierto]:.2f}), "
            f"entorno {cerrado} ({medias[cerrado]:.2f})."
        )

    def exportar(self) -> Dict[str, Any]:
        return {"nombres": list(self.puertas.keys()),
                "valores": [float(v) for v in self.puertas.values()]}

    def importar(self, datos: Dict[str, Any]) -> None:
        try:
            for n, v in zip(datos.get("nombres", []), datos.get("valores", [])):
                self.puertas[str(n)] = float(np.clip(float(v), self.suelo, 1.0))
        except Exception as e:
            logger.debug(f"RN14: no se pudieron importar puertas: {e}")
