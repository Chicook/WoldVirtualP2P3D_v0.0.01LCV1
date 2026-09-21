"""Talamo/relevo.py - Relevo sensorial con compuerta adaptativa."""
import numpy as np


class RelevoSensorial:
    """Compuerta: deja pasar lo coherente, atenua lo ruidoso, registra el motivo."""

    def __init__(self, umbral: float = 8.0):
        self.umbral = umbral
        self._fondo = None
        self.pasan = 0
        self.atenuados = 0

    def relevar(self, vector) -> tuple:
        v = np.asarray(vector, dtype=np.float32)
        if self._fondo is None:
            self._fondo = v.copy()
        self._fondo = 0.9 * v + 0.1 * self._fondo
        distancia = float(np.linalg.norm(v - self._fondo))
        if distancia > self.umbral:
            self.atenuados += 1
            return np.clip(self._fondo, -5.0, 5.0), {"pasa": False, "motivo": f"pico {distancia:.2f}"}
        self.pasan += 1
        return np.clip(v, -5.0, 5.0), {"pasa": True, "motivo": "coherente"}

    def balance(self) -> dict:
        total = self.pasan + self.atenuados
        return {"pasan": self.pasan, "atenuados": self.atenuados,
                "tasa_paso": round(self.pasan / max(1, total), 3)}
