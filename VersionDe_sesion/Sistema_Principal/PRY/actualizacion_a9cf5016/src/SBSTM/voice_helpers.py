"""
voice_helpers.py - Clases auxiliares para el motor de voz neuronal LucIA.
========================================================================
Extraído de voice_engine.py para mantener el límite estricto de <= 450 líneas.
Incluye:
  - ModuladorExpresivoJuvenil: Modulación de estilo oral juvenil y prosodia.
  - LimpiadorFoneticoLucIA: Sanitización y segmentación de texto para TTS.
  - ReproductorMCIWindows: Reproducción y control en tiempo real mediante winmm.dll.
"""

from __future__ import annotations

import ctypes
import logging
import os
import random
import re
import threading
import time
from typing import Final, List, Tuple

logger = logging.getLogger("lucIA.Voice.Helpers")


class ModuladorExpresivoJuvenil:
    """
    Modula la cadencia, tono emocional y muletillas orales para lograr una
    expresion fresca, viva y espontanea caracteristica de una joven de 18 anos.
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


class LimpiadorFoneticoLucIA:
    """Sanitiza y normaliza textos para una fonetica fluida y sin tropiezos."""

    @staticmethod
    def sanitizar(texto: str) -> str:
        if not texto:
            return ""
        t = texto.strip()
        t = re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL)
        t = re.sub(r"```.*?```", " codigo omitido ", t, flags=re.DOTALL)
        t = re.sub(r"`[^`]*`", "", t)
        t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)
        t = re.sub(r"^\s*#+\s*", "", t, flags=re.MULTILINE)
        t = re.sub(r"^\s*\d+\.\s*", "", t, flags=re.MULTILINE)
        t = re.sub(r"^\s*[-*+•◆]\s*", "", t, flags=re.MULTILINE)
        t = re.sub(r"\*+", "", t)
        t = re.sub(r"[_~>|\\(){}\[\]]", " ", t)
        t = "".join(c for c in t if ord(c) < 0x10000)
        t = re.sub(r"[\u2600-\u27BF]", "", t)
        t = re.sub(r"[ \t]+", " ", t)
        t = re.sub(r"\n\s*\n+", ". ", t)
        t = re.sub(r"\n", ". ", t)
        t = re.sub(r"\.{2,}", ".", t)
        t = t.strip()
        if t and t[-1] not in ".!?":
            t += "."
        return t

    @staticmethod
    def segmentar_frases(texto: str, max_chars: int = 420) -> List[str]:
        if len(texto) <= max_chars:
            return [texto]
        bloques: List[str] = []
        inicio = 0
        while inicio < len(texto):
            resto = len(texto) - inicio
            if resto <= max_chars:
                bloques.append(texto[inicio:].strip())
                break
            fin = inicio + max_chars
            corte = -1
            for sep in (". ", "! ", "? ", "; "):
                pos = texto.rfind(sep, inicio, fin)
                if pos > corte:
                    corte = pos + len(sep)
            if corte > inicio:
                bloques.append(texto[inicio:corte].strip())
                inicio = corte
            else:
                espacio = texto.rfind(" ", inicio, fin)
                if espacio > inicio:
                    bloques.append(texto[inicio:espacio].strip())
                    inicio = espacio + 1
                else:
                    bloques.append(texto[inicio:fin].strip())
                    inicio = fin
        return [b for b in bloques if b]


class ReproductorMCIWindows:
    """Controlador directo de audio nativo mediante winmm.dll de Windows."""

    def __init__(self, alias: str = "lucia_audio_active") -> None:
        self.alias: str = alias
        self._lock = threading.Lock()

    def detener_inmediato(self) -> None:
        with self._lock:
            try:
                mci = ctypes.windll.winmm.mciSendStringW
                mci(f"stop {self.alias}", None, 0, None)
                mci(f"close {self.alias}", None, 0, None)
            except Exception:
                pass

    def reproducir_mp3(self, ruta_mp3: str, cancel_ev: threading.Event, stop_ev: threading.Event) -> bool:
        if cancel_ev.is_set():
            return False
        with self._lock:
            try:
                mci = ctypes.windll.winmm.mciSendStringW
                mci(f"close {self.alias}", None, 0, None)
                ruta_norm = os.path.abspath(ruta_mp3).replace("\\", "/")
                if mci(f'open "{ruta_norm}" type mpegvideo alias {self.alias}', None, 0, None) != 0:
                    return False
                mci(f"play {self.alias}", None, 0, None)
            except Exception as ex:
                logger.warning(f"Error al abrir audio MCI: {ex}")
                return False

        buf = ctypes.create_unicode_buffer(128)
        time.sleep(0.08)
        while not cancel_ev.is_set() and not stop_ev.is_set():
            time.sleep(0.09)
            with self._lock:
                res = mci(f"status {self.alias} mode", buf, 128, None)
                if res != 0 or buf.value not in ("playing", ""):
                    break

        self.detener_inmediato()
        return True

    def reproducir_wav(self, ruta_wav: str, cancel_ev: threading.Event, stop_ev: threading.Event) -> bool:
        """Reproduce un archivo WAV via MCI waveaudio (para sintesis OneCore COM)."""
        if cancel_ev.is_set():
            return False
        mci = ctypes.windll.winmm.mciSendStringW
        alias_wav = self.alias + "_wav"
        with self._lock:
            try:
                mci(f"close {alias_wav}", None, 0, None)
                ruta_norm = os.path.abspath(ruta_wav).replace("\\", "/")
                if mci(f'open "{ruta_norm}" type waveaudio alias {alias_wav}', None, 0, None) != 0:
                    return False
                mci(f"play {alias_wav}", None, 0, None)
            except Exception as ex:
                logger.warning(f"Error al abrir WAV MCI: {ex}")
                return False

        buf = ctypes.create_unicode_buffer(128)
        time.sleep(0.05)
        while not cancel_ev.is_set() and not stop_ev.is_set():
            time.sleep(0.09)
            with self._lock:
                res = mci(f"status {alias_wav} mode", buf, 128, None)
                if res != 0 or buf.value not in ("playing", ""):
                    break

        with self._lock:
            try:
                mci(f"stop {alias_wav}", None, 0, None)
                mci(f"close {alias_wav}", None, 0, None)
            except Exception:
                pass
        return True
