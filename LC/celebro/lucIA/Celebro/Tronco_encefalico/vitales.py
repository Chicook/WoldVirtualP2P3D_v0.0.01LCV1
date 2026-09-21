"""Tronco_encefalico/vitales.py - Constantes vitales del sistema."""
import time


class ConstantesVitales:
    """Pulso de sesion: turnos, tiempo activo, presupuesto y alertas."""

    LIMITE_MB = 125.0

    def __init__(self):
        self.inicio = time.time()
        self.turnos = 0
        self.alertas = []

    def pulso(self, tamano_mb: float = -1.0) -> dict:
        self.turnos += 1
        vivo_min = round((time.time() - self.inicio) / 60.0, 1)
        estado = {"turnos": self.turnos, "min_activo": vivo_min,
                  "tamano_mb": tamano_mb, "alerta": ""}
        if tamano_mb > 0 and tamano_mb > self.LIMITE_MB * 0.9:
            estado["alerta"] = f"cerca del limite ({tamano_mb:.1f} MB)"
            self.alertas.append(estado["alerta"])
        return estado

    def descanso(self) -> dict:
        return {"turnos": self.turnos, "alertas": list(self.alertas)}
