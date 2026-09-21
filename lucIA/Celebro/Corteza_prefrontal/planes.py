"""Corteza_prefrontal/planes.py - Pila de objetivos y su avance."""
import re


class PilaObjetivos:
    """Guarda objetivos detectados y mide avance por solape de vocabulario."""

    def __init__(self, max_obj: int = 5):
        self.max_obj = max_obj
        self.objetivos = []

    def detectar(self, texto: str) -> str:
        t = (texto or "").strip()
        if re.search(r"(?i)(plan|objetivo|meta|quiero|necesito|ayudame|elabora|crea|disena|explica|dime)", t):
            claves = [w.lower() for w in re.findall(r"[a-zA-Záéíóúñ]{4,}", t)][:10]
            self.objetivos.append({"texto": t[:200], "claves": claves, "avance": 0.0})
            if len(self.objetivos) > self.max_obj:
                self.objetivos.pop(0)
            return t[:200]
        return ""

    def medir(self, respuesta: str) -> float:
        if not self.objetivos or not respuesta:
            return 0.0
        obj = self.objetivos[-1]
        resp = respuesta.lower()
        hit = sum(1 for w in obj["claves"] if w in resp)
        obj["avance"] = round(hit / max(1, len(obj["claves"])), 2)
        return obj["avance"]

    def actual(self) -> dict:
        return dict(self.objetivos[-1]) if self.objetivos else {}
