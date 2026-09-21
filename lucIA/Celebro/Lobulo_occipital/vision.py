"""Lobulo_occipital/vision.py - Via visual reservada (stub funcional)."""
import hashlib


class ViaVisual:
    """Interfaz lista para vision futura. Hoy: huella estructural del texto
    (longitud, lineas, bloques) como 'esbozo' sin sensor real."""

    def __init__(self):
        self.esbozos = 0
        self.ultima_huella = ""

    def esbozar(self, texto: str) -> dict:
        t = texto or ""
        lineas = t.count("\n") + 1 if t else 0
        self.ultima_huella = hashlib.md5(t.encode("utf-8", "ignore")).hexdigest()[:12]
        self.esbozos += 1
        return {"huella": self.ultima_huella, "chars": len(t),
                "lineas": lineas, "sensor": False, "n": self.esbozos}

    def describir(self) -> str:
        if not self.ultima_huella:
            return "aun no he mirado nada"
        return f"he esbozado {self.esbozos} textos (huella {self.ultima_huella})"
