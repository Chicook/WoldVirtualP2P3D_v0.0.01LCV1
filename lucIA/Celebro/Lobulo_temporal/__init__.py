"""Lobulo_temporal - Lenguaje, prosodia y memoria auditiva de LucIA.

Rol biologico: comprension del habla, reconocimiento de palabras,
memoria semántica y musical. En LucIA gestiona el vocabulario activo,
detecta el tono conversacional (pregunta, orden, emocion) y mantiene
un lexicon de conceptos mas frecuentes de la sesion.
"""
import re
import math
from collections import Counter
from lucIA.Celebro.bases_neuro import ModuloCerebral


class LobuloTemporal(ModuloCerebral):
    nombre = "Temporal"

    # Patrones de tono conversacional
    _PATRON_PREGUNTA = re.compile(
        r"(?i)^(que|como|quien|donde|cuando|por que|cuanto|cual|"
        r"dime|explica|sabes|tienes|puedes|podr[íi]as|es|son|hay)", re.UNICODE
    )
    _PATRON_ORDEN = re.compile(
        r"(?i)^(crea|genera|escribe|haz|dame|busca|analiza|muestra|"
        r"calcula|lista|compara|traduce|resume|explica|refactoriza)", re.UNICODE
    )
    _PATRON_EMOCION = re.compile(
        r"(?i)(amor|odio|miedo|alegr[ía]a|triste|nervios|eufori|enojo|"
        r"frustrac|emocion|siento|me gusta|me encanta|me molesta|me preocupa)", re.UNICODE
    )
    _STOP = frozenset({
        "que", "como", "con", "para", "este", "esta", "esto", "eso",
        "esos", "esas", "del", "las", "los", "una", "unos", "unas",
        "donde", "cuando", "porque", "pues", "alla", "muy", "mas",
        "pero", "sin", "sobre", "hay", "son", "por", "tiene", "tengo",
    })

    def __init__(self):
        super().__init__()
        self.frases_oidas: int = 0
        self.lexicon: Counter = Counter()   # vocabulario activo de sesion
        self._ultimo_tono: str = "neutral"
        self._chars_totales: int = 0

    # -- Escucha / procesamiento ----------------------------------------

    def oir(self, texto: str) -> dict:
        """Registra y analiza una frase (usuario o LucIA).

        Retorna dict con tono, tokens_nuevos y tamanio_lexicon.
        """
        if not texto or len(texto.strip()) < 3:
            return {"tono": self._ultimo_tono, "tokens_nuevos": 0,
                    "lexicon": len(self.lexicon)}

        t = texto.strip()
        self.frases_oidas += 1
        self._chars_totales += len(t)

        # -- Tokenizacion limpia -----------------------------------------
        tokens = [
            w.lower() for w in re.findall(r"[a-zA-Záéíóúñü]{3,}", t)
            if w.lower() not in self._STOP
        ]

        # -- Clasificar tono principal -----------------------------------
        tono = self._clasificar_tono(t)
        self._ultimo_tono = tono

        # -- Actualizar lexico -------------------------------------------
        nuevos_antes = len(self.lexicon)
        self.lexicon.update(tokens)
        tokens_nuevos = len(self.lexicon) - nuevos_antes

        self._notar(
            f"#{self.frases_oidas} tono={tono} "
            f"tokens={len(tokens)} lexico={len(self.lexicon)}"
        )
        return {
            "tono": tono,
            "tokens_nuevos": tokens_nuevos,
            "lexicon": len(self.lexicon),
            "chars_sesion": self._chars_totales,
        }

    def _clasificar_tono(self, texto: str) -> str:
        """Detecta: pregunta | orden | emocion | reflexion | neutral."""
        if self._PATRON_EMOCION.search(texto):
            return "emocion"
        primera = texto.split()[0] if texto.split() else ""
        if texto.strip().endswith("?") or self._PATRON_PREGUNTA.match(primera):
            return "pregunta"
        if self._PATRON_ORDEN.match(primera):
            return "orden"
        # Reflexion: frases largas sin patron claro
        if len(texto) > 120:
            return "reflexion"
        return "neutral"

    # -- Lexicon y estadisticas ----------------------------------------

    def conceptos_top(self, n: int = 10) -> list:
        """Devuelve los n conceptos mas frecuentes del lexico de sesion."""
        return [w for w, _ in self.lexicon.most_common(n)]

    def riqueza_lexica(self) -> float:
        """Tipo/token ratio estimado (0-1): diversidad del vocabulario."""
        total_tokens = sum(self.lexicon.values())
        if total_tokens == 0:
            return 0.0
        # Clamp logaritmico: ratio crece con diversidad, baja con repeticion
        return round(min(1.0, math.log1p(len(self.lexicon)) /
                         math.log1p(total_tokens)), 3)

    def a_pesos(self, conversor) -> None:
        """Exporta resumen del lexico a pesos neuronales via conversor."""
        try:
            top = " ".join(self.conceptos_top(20))
            resumen = (
                f"lexico sesion: {len(self.lexicon)} conceptos | "
                f"riqueza={self.riqueza_lexica()} | tono={self._ultimo_tono} | "
                f"top: {top}"
            )
            conversor.actualizar_memoria_salida(
                "[Temporal] lexico de sesion", resumen, "Temporal"
            )
        except Exception:
            pass

    def resumen(self) -> str:
        return (
            f"Temporal: frases={self.frases_oidas} "
            f"lexico={len(self.lexicon)} "
            f"riqueza={self.riqueza_lexica()} "
            f"tono={self._ultimo_tono} | {super().resumen()}"
        )


_temporal_global = None


def get_temporal() -> LobuloTemporal:
    global _temporal_global
    if _temporal_global is None:
        _temporal_global = LobuloTemporal()
    return _temporal_global
