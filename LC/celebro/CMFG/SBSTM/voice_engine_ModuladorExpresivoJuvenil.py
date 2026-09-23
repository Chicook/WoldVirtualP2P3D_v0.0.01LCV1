from typing import Final

class ModuladorExpresivoJuvenil:
    """

    MULETILLAS_APERTURA: Final[List[str]] = [
        "Oye, ",
        "A ver, te cuento: ",
        "Mira, ",
        "Bueno, pues a ver: ",
        "Fijate, ",
        "Pues mira, ",
        "Osea, escucha: ",
    ]

    CONECTORES_ESPONTANEOS: Final[List[Tuple[str, str]]] = [
        (r"\bpor lo tanto\b", "asi que"),
        (r"\bpor consiguiente\b", "con lo cual"),
        (r"\bsin embargo\b", "pero bueno,"),
        (r"\bno obstante\b", "aunque la verdad,"),
        (r"\bes decir\b", "en plan,"),
        (r"\ben efecto\b", "totalmente,"),
        (r"\bcabe destacar que\b", "lo que mola es que"),
        (r"\bes menester\b", "toca"),
        (r"\bciertamente\b", "de verdad,"),
    ]

    CIERRES_FRESCOS: Final[List[str]] = [
        " ¿Tiene sentido, verdad?",
        " ¿Que te parece?",
        " Ya me diras que opinas.",
        " Esta genial, a que si?",
        " Cualquier duda me dices!",
    ]

    def __init__(self) -> None:
        self._contador_turnos: int = 0
        self._lock = threading.Lock()

    def humanizar_texto(self, texto: str, aplicar_muletillas: bool = True) -> str:
        """Transforma formulas formales y rigidas en un estilo oral natural."""
        if not texto:
            return ""
        t = texto
        for pat, rep in self.CONECTORES_ESPONTANEOS:
            t = re.sub(pat, rep, t, flags=re.IGNORECASE)

        if aplicar_muletillas and len(t) > 30 and not t.startswith(("¡", "¿", "Oye", "Mira")):
            with self._lock:
                self._contador_turnos += 1
                if self._contador_turnos % 2 == 1:
                    prefijo = random.choice(self.MULETILLAS_APERTURA)
                    t = prefijo + t[0].lower() + t[1:]

        t = re.sub(r"\s+", " ", t).strip()
        return t

    def calcular_prosodia(self, texto: str, emocion: float = 0.5) -> Tuple[str, str]:
        """Genera rate y pitch adaptativos (+pitch para vivacidad juvenil)."""
        if emocion > 0.6:
            pitch_val = min(12, int(6 + (emocion * 8)))
            rate_val = min(12, int(4 + (emocion * 6)))
        elif emocion < 0.3:
            pitch_val = max(2, int(3 + (emocion * 4)))
            rate_val = max(0, int(emocion * 5))
        else:
            pitch_val = 6
            rate_val = 5

        rate_str = f"+{rate_val}%" if rate_val >= 0 else f"{rate_val}%"
        pitch_str = f"+{pitch_val}Hz" if pitch_val >= 0 else f"{pitch_val}Hz"
        return rate_str, pitch_str


