"""Corteza_sensorial/vias.py - Registro y estado de vias sensoriales."""
import time


class RegistroVias:
    """Tabla de sentidos: texto (activo), vision/audio (reservados)."""

    def __init__(self):
        self.vias = {"texto": {"activa": True, "usos": 0, "ultimo": 0.0},
                     "vision": {"activa": False, "usos": 0, "ultimo": 0.0},
                     "audio": {"activa": False, "usos": 0, "ultimo": 0.0}}

    def usar(self, via: str, detalle: str = "") -> bool:
        if via not in self.vias or not self.vias[via]["activa"]:
            return False
        self.vias[via]["usos"] += 1
        self.vias[via]["ultimo"] = round(time.time(), 0)
        return True

    def activar(self, via: str) -> bool:
        if via not in self.vias:
            return False
        self.vias[via]["activa"] = True
        return True

    def estado(self) -> dict:
        return {k: {"activa": v["activa"], "usos": v["usos"]} for k, v in self.vias.items()}
