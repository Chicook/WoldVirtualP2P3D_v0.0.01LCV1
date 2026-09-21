"""Hipotalamo - Homeostasis de LucIA: energia, ritmo circadiano y alertas.

Rol biologico: regula el hambre, la sed, la temperatura, el sueno y los
ritmos circadianos. En LucIA controla el nivel de energia cognitiva,
detecta la sobrecarga acumulada, sugiere pausas y mantiene un ritmo
de sesion saludable con umbrales adaptativos.
"""
import time
import math
from lucIA.Celebro.bases_neuro import ModuloCerebral

# Umbrales de energia y alerta
ENERGIA_INICIAL = 1.0
COSTE_POR_DELTA = 0.015          # gasto base por unidad de delta (suavizado)
COSTE_MAX_POR_TURNO = 0.08       # tope de gasto por turno
RECUPERACION_REPOSO = 0.30       # energia que repone un /descansa
UMBRAL_ALERTA_BAJA = 0.35        # por debajo -> sugiere descanso
UMBRAL_ALERTA_MUY_BAJA = 0.15   # por debajo -> estado critico
RECUPERACION_TURNO = 0.005       # mini recuperacion pasiva cada turno


class Hipotalamo(ModuloCerebral):
    nombre = "Hipotalamo"

    def __init__(self):
        super().__init__()
        self.energia: float = ENERGIA_INICIAL
        self._turnos: int = 0
        self._suma_deltas: float = 0.0
        self._t_inicio: float = time.time()
        self._alertas_enviadas: int = 0
        self._estado: str = "activo"          # activo | alerta | critico | reposo

    # -- Ciclo turno a turno -------------------------------------------

    def latido(self, delta: float) -> dict:
        """Procesa un turno: gasta energia segun esfuerzo y recupera un poco.

        Retorna dict con energia, estado y sugiere_descanso.
        """
        d = float(delta)
        self._turnos += 1
        self._suma_deltas += d

        # Gasto proporcional al delta (logaritmico para no castigar picos puntales)
        gasto = min(COSTE_MAX_POR_TURNO,
                    COSTE_POR_DELTA * math.log1p(d))
        # Mini recuperacion pasiva (el cuerpo siempre se recupera un poco)
        self.energia = round(
            min(ENERGIA_INICIAL, max(0.0,
                self.energia - gasto + RECUPERACION_TURNO)), 4
        )

        # Actualizar estado
        if self.energia <= UMBRAL_ALERTA_MUY_BAJA:
            self._estado = "critico"
        elif self.energia <= UMBRAL_ALERTA_BAJA:
            self._estado = "alerta"
        else:
            self._estado = "activo"

        sugiere = self.energia <= UMBRAL_ALERTA_BAJA
        if sugiere:
            self._alertas_enviadas += 1

        self._notar(
            f"energia={self.energia} estado={self._estado} "
            f"delta={d:.4f} turno={self._turnos}"
        )
        return {
            "energia": self.energia,
            "estado": self._estado,
            "sugiere_descanso": sugiere,
            "turnos_sesion": self._turnos,
        }

    # -- Descanso / recarga --------------------------------------------

    def reponer(self, parcial: bool = False) -> float:
        """Repone energia. Si parcial=True, solo recupera RECUPERACION_REPOSO."""
        if parcial:
            self.energia = round(
                min(ENERGIA_INICIAL, self.energia + RECUPERACION_REPOSO), 4
            )
            self._estado = "activo" if self.energia > UMBRAL_ALERTA_BAJA else "alerta"
            self._notar(f"descanso parcial -> energia={self.energia}")
        else:
            self.energia = ENERGIA_INICIAL
            self._estado = "activo"
            self._notar("reposo completo -> energia=1.0")
        return self.energia

    # -- Estadisticas de sesion ----------------------------------------

    def ritmo_sesion(self) -> dict:
        """Tiempo transcurrido, delta medio y cadencia de alertas."""
        elapsed = round(time.time() - self._t_inicio, 1)
        delta_medio = round(self._suma_deltas / max(1, self._turnos), 4)
        return {
            "segundos_sesion": elapsed,
            "turnos": self._turnos,
            "delta_medio": delta_medio,
            "alertas": self._alertas_enviadas,
            "energia": self.energia,
            "estado": self._estado,
        }

    def a_pesos(self, conversor) -> None:
        """Exporta telemetria energetica a pesos neuronales."""
        try:
            r = self.ritmo_sesion()
            resumen = (
                f"energia_final={r['energia']} estado={r['estado']} "
                f"turnos={r['turnos']} delta_medio={r['delta_medio']} "
                f"alertas={r['alertas']} sesion={r['segundos_sesion']}s"
            )
            conversor.actualizar_memoria_salida(
                "[Hipotalamo] homeostasis sesion", resumen, "Hipotalamo"
            )
        except Exception:
            pass

    def resumen(self) -> str:
        return (
            f"Hipotalamo: energia={self.energia} "
            f"estado={self._estado} "
            f"turnos={self._turnos} | {super().resumen()}"
        )


_hipotalamo_global = None


def get_hipotalamo() -> Hipotalamo:
    global _hipotalamo_global
    if _hipotalamo_global is None:
        _hipotalamo_global = Hipotalamo()
    return _hipotalamo_global
