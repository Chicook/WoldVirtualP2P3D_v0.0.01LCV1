"""Lobulo_temporal/lenguaje.py - Oido y lenguaje basico de LucIA."""
import re

PALABRAS_CALIDAS = {"gracias", "genial", "perfecto", "bien", "encanta", "bravo",
                    "estupendo", "maravilla", "contenta", "feliz", "alegria",
                    "bueno", "buena", "excelente", "fantastico", "increible",
                    "hermoso", "precioso", "alegre", "divertido", "cariño",
                    "abrazo", "enhorabuena", "felicidades", "suerte", "animo",
                    "fuerza", "adelante", "logrado", "conseguido", "brava"}
PALABRAS_TENSAS = {"mal", "error", "falla", "triste", "miedo", "rabia",
                   "preocupa", "cansada", "presion", "fatal", "horrible",
                   "tristeza", "ansiedad", "nerviosa", "estres", "mala",
                   "terrible", "dolor", "llorar", "sola", "solo", "aburrida",
                   "enfado", "odio", "culpa", "verguenza", "duelo", "muerte",
                   "urgente", "socorro", "ayuda", "peligro", "fallo"}


class OidoLinguistico:
    """Escucha cada frase: cuenta, mide calidez/tension y extrae temas."""

    def __init__(self):
        self.frases = 0
        self.calidez_total = 0.0
        self.tension_total = 0.0

    def oir(self, texto: str) -> dict:
        t = (texto or "").lower()
        toks = set(re.findall(r"[a-záéíóúñ]+", t))
        cal = len(toks & PALABRAS_CALIDAS)
        ten = len(toks & PALABRAS_TENSAS)
        self.frases += 1
        self.calidez_total += cal
        self.tension_total += ten
        temas = [w for w in toks if len(w) > 5][:5]
        return {"n": self.frases, "calidez": cal, "tension": ten,
                "temas": temas,
                "clima": "calido" if cal > ten else "tenso" if ten > cal else "neutro"}

    def clima_global(self) -> dict:
        if not self.frases:
            return {"clima": "neutro", "frases": 0}
        c = self.calidez_total / self.frases
        t = self.tension_total / self.frases
        clima = "calido" if c > t else "tenso" if t > c else "neutro"
        return {"clima": clima, "frases": self.frases,
                "calidez_media": round(c, 3), "tension_media": round(t, 3)}
