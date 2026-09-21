"""Cuerpo_calloso/puente.py - Integracion de hemisferios local/cloud."""
import re


class PuenteHemisferios:
    """Fusiona la via rapida local (L) y la creativa cloud (C).
    Estrategias: alternancia (la usada por el router), sintesis y contraste."""

    ESTRATEGIAS = ("alternancia", "sintesis", "contraste")

    def __init__(self):
        self.estrategia = "alternancia"
        self.integraciones = 0

    def _frases(self, texto: str, n: int = 3):
        partes = [p.strip() for p in re.split(r"[.!?]+", texto or "") if p.strip()]
        return partes[:n]

    def integrar(self, via_local: str, via_cloud: str, estrategia: str = "") -> dict:
        est = estrategia or self.estrategia
        fl, fc = self._frases(via_local), self._frases(via_cloud)
        if est == "contraste":
            texto = (f"Mi parte rapida dice: {fl[0] if fl else '-'}. "
                     f"Mi parte reflexiva matiza: {fc[0] if fc else '-'}.")
        elif est == "sintesis":
            texto = " ".join((fl[:2] + fc[:1])[:3])
        else:
            texto = " ".join(fl[:1] + fc[:1])
        self.integraciones += 1
        return {"texto": texto.strip(), "estrategia": est,
                "n": self.integraciones}

    def acuerdo(self, via_local: str, via_cloud: str) -> float:
        a = set(re.findall(r"[a-záéíóúñ]{4,}", (via_local or "").lower()))
        b = set(re.findall(r"[a-záéíóúñ]{4,}", (via_cloud or "").lower()))
        if not a or not b:
            return 0.0
        return round(len(a & b) / max(1, len(a | b)), 3)
