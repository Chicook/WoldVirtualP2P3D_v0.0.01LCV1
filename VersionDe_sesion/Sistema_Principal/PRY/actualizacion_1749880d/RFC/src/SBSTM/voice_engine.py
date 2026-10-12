"""

voice_engine.py - Motor de Sintesis de Voz Neuronal para LucIA (WoldVirtualP2P3D 2026)
=====================================================================================
Modo de expresion: Voz juvenil, espontanea, dinamica y fresca (estilo chica 18 anos).
Voz oficial: Voz Neuronal LucIA 18 anos (es-ES / Natural / DSP 1.16x / Tono Humano).
- Sintesis primaria: gTTS es-ES (peninsular nativo) + DSP vocal (1.16x + calidez intima de estudio).
- Fallbacks integrados: SAPI OneCore (Laura/Helena) y System.Speech.
- Control en tiempo real: cancel_tts con winmm.dll (MCI) y soporte asincrono en worker.

"""

# ===========================================================================
# voice_engine.py - Motor de Sintesis de Voz Neuronal para LucIA (v0.0.01)
# ===========================================================================

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import queue
import subprocess
import tempfile
import threading
import time
import winreg
from typing import Any, Dict, Final, List, Optional, Tuple

from SBSTM.voice_helpers import (
    LimpiadorFoneticoLucIA,
    ModuladorExpresivoJuvenil,
    ReproductorMCIWindows,
)

logger = logging.getLogger("lucIA.Voice")

VOZ_OFICIAL: Final[str] = "LucIA-18anos-esES-vocal"
VERSION_VOICE: Final[str] = "2026.4.3"


class VoiceEngine:
    """
    Sintetizador neuronal de voz para LucIA con voz oficial es-ES natural
    y modulacion prosodica adaptativa de estilo juvenil espontaneo (18 anos).
    """

    def __init__(self, voz: str = VOZ_OFICIAL) -> None:
        self.voz: str = voz
        self._cola_voz: queue.Queue[str] = queue.Queue()
        self._detener_evento = threading.Event()
        self._cancelar_actual = threading.Event()
        self._hilo_worker: Optional[threading.Thread] = None
        self.mci = ReproductorMCIWindows()
        self.modulador = ModuladorExpresivoJuvenil()
        self.limpiador = LimpiadorFoneticoLucIA()
        self._gtts_disponible: bool = False
        self._nivel_emocion: float = 0.65
        self._habilitado: bool = True
        # OneCore COM voice (fallback cuando no haya conexion externa)
        self._onecore_token: Optional[object] = self._descubrir_voz_onecore()
        self._onecore_rate: int = 2
        self._procesador_rn = None
        self._inicializar_procesador_rn()
        self._verificar_gtts()
        self._iniciar_worker()

    def _inicializar_procesador_rn(self) -> None:
        try:
            try:
                from LC_STM.red_neuronal.SLRN.procesador_voz_neuronal import get_procesador_voz_neuronal
            except ImportError:
                import sys
                src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if src_dir not in sys.path:
                    sys.path.insert(0, src_dir)
                from LC_STM.red_neuronal.SLRN.procesador_voz_neuronal import get_procesador_voz_neuronal
            self._procesador_rn = get_procesador_voz_neuronal()
            logger.info("Motor de voz conectado con la Red Neuronal de LucIA (SLRN)")
        except Exception as e:
            logger.warning(f"No se pudo enlazar ProcesadorVozNeuronal: {e}")

    def _verificar_gtts(self) -> None:
        try:
            import gtts
            self._gtts_disponible = True
            logger.info(f"Voz oficial configurada: LucIA 18 anos (es-ES / DSP 1.16x)")
        except ImportError:
            self._gtts_disponible = False
            logger.warning("gTTS no detectado, usando fallback local")

    def _iniciar_worker(self) -> None:
        self._hilo_worker = threading.Thread(
            target=self._procesar_cola, daemon=True, name="LucIA-Voice-Worker"
        )
        self._hilo_worker.start()

    def set_emocion(self, nivel: float) -> None:
        """Ajusta la valencia emocional de la prosodia (0.0 a 1.0)."""
        self._nivel_emocion = max(0.0, min(1.0, float(nivel)))

    def limpiar_texto_para_habla(self, texto: str) -> str:
        """Sanitiza markdown, listas y caracteres extraños."""
        return self.limpiador.sanitizar(texto)

    def _sintetizar_y_reproducir_chunk(self, chunk: str) -> bool:
        """Sintetiza mediante la voz oficial de LucIA (es-ES + DSP 1.16x)."""
        if self._cancelar_actual.is_set():
            return False

        temp_fd_mp3, temp_path_mp3 = tempfile.mkstemp(suffix=".mp3")
        os.close(temp_fd_mp3)
        temp_fd_wav, temp_path_wav = tempfile.mkstemp(suffix=".wav")
        os.close(temp_fd_wav)

        try:
            from gtts import gTTS
            import soundfile as sf
            import numpy as np
            from scipy import signal
            from pedalboard import Pedalboard, LowShelfFilter, PeakFilter, Compressor

            # 1. Sintetizar base peninsular nativa
            tts = gTTS(text=chunk, lang="es", tld="es", slow=False)
            tts.save(temp_path_mp3)

            if not os.path.exists(temp_path_mp3) or os.path.getsize(temp_path_mp3) == 0:
                return False

            if self._cancelar_actual.is_set():
                return False

            # 2. Leer audio base
            audio, sr = sf.read(temp_path_mp3)
            if audio.ndim > 1:
                audio = audio[:, 0]
            audio = audio.astype(np.float32)

            # 3. Aplicar cadencia dinámica elegida (1.16x de la voz 1)
            factor_velocidad = 1.16
            if self._nivel_emocion > 0.7:
                factor_velocidad = 1.19
            elif self._nivel_emocion < 0.4:
                factor_velocidad = 1.14

            nueva_longitud = int(len(audio) / factor_velocidad)
            audio_resample = signal.resample(audio, nueva_longitud)

            # 4. Masterización de estudio (timbre cálido, cuerpo de 18 años y sedosidad)
            board = Pedalboard([
                LowShelfFilter(cutoff_frequency_hz=280.0, gain_db=1.0),
                PeakFilter(cutoff_frequency_hz=1000.0, gain_db=0.9, q=0.7),
                PeakFilter(cutoff_frequency_hz=3400.0, gain_db=-1.5, q=1.0),
                Compressor(threshold_db=-18.0, ratio=1.7)
            ])
            audio_final = board(audio_resample.astype(np.float32), sr)

            sf.write(temp_path_wav, audio_final, sr)

            # 5. Reproducir con el reproductor nativo MCI
            if os.path.exists(temp_path_wav) and os.path.getsize(temp_path_wav) > 0:
                if not self._cancelar_actual.is_set():
                    self.mci.reproducir_wav(temp_path_wav, self._cancelar_actual, self._detener_evento)
                    return True
        except Exception as err:
            logger.warning(f"Fallo en sintesis oficial LucIA: {err}")
        finally:
            for p in (temp_path_mp3, temp_path_wav):
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass
        return False

    @staticmethod
    def _descubrir_voz_onecore() -> Optional[object]:
        """
        Busca la mejor voz OneCore disponible en el sistema.
        Preferencia: ElviraNeural > Laura > Helena > cualquier es-ES femenina.
        Retorna el token COM de la voz seleccionada o None si falla.
        """
        PREFERENCIAS = [
            "elvira",   # ideal: Neural (instalar via Configuracion > Voz)
            "laura",    # OneCore juvenil femenina - segunda opcion
            "helena",   # OneCore femenina - tercera opcion
        ]
        try:
            import comtypes.client  # type: ignore
            cat = comtypes.client.CreateObject("SAPI.SpObjectTokenCategory")
            cat.SetId(
                r"HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech_OneCore\Voices", False
            )
            tokens = cat.EnumerateTokens()
            # Recopilar todas las voces primero para busqueda priorizada
            mapa_voces: List[Tuple[str, object]] = []
            for i in range(tokens.Count):
                tok = tokens.Item(i)
                desc = tok.GetDescription(1033).lower()
                mapa_voces.append((desc, tok))
            # Seleccionar por preferencia en orden estricto
            for pref in PREFERENCIAS:
                for desc, tok in mapa_voces:
                    if pref in desc:
                        logger.info(
                            f"Voz OneCore seleccionada: {tok.GetDescription(1033)}"
                        )
                        return tok
            # Ultimo recurso: cualquier voz en espanol
            for desc, tok in mapa_voces:
                if "spanish" in desc or "espa" in desc:
                    logger.info(
                        f"Voz OneCore fallback es-ES: {tok.GetDescription(1033)}"
                    )
                    return tok
            return None
        except Exception as err:
            logger.warning(f"No se pudo acceder a voces OneCore: {err}")
            return None

    def _sintetizar_fallback_windows(self, texto_limpio: str) -> None:
        """Sintetiza con SAPI OneCore (Laura/Helena) via COM y reproduce con MCI."""
        if self._cancelar_actual.is_set():
            return

        # Intentar sintesis COM OneCore
        if self._onecore_token is not None and self._sintetizar_onecore_com(texto_limpio):
            return

        # Ultimo recurso: PowerShell System.Speech (sin seleccion de voz especifica)
        logger.warning("Usando System.Speech como ultimo recurso (calidad basica)")
        t_esc = texto_limpio.replace("'", "").replace('"', "")
        cmd = [
            "powershell", "-NoProfile", "-NonInteractive", "-Command",
            f"Add-Type -AssemblyName System.Speech; "
            f"$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$synth.SelectVoiceByHints([System.Speech.Synthesis.VoiceGender]::Female, "
            f"[System.Speech.Synthesis.VoiceAge]::Teen); "
            f"$synth.Rate = 2; "
            f"$synth.Speak([string]'{t_esc}')"
        ]
        try:
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=25,
            )
        except Exception as err:
            logger.warning(f"Error System.Speech final: {err}")

    def _sintetizar_onecore_com(self, texto: str) -> bool:
        """Genera WAV con SAPI OneCore via COM y lo reproduce con MCI."""
        if self._cancelar_actual.is_set():
            return False
        temp_fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(temp_fd)
        try:
            import comtypes.client  # type: ignore
            sapi = comtypes.client.CreateObject("SAPI.SpVoice")
            sapi.Voice = self._onecore_token
            sapi.Rate = self._onecore_rate

            stream = comtypes.client.CreateObject("SAPI.SpFileStream")
            stream.Open(temp_path, 3, False)  # SSFMCreateForWrite
            sapi.AudioOutputStream = stream
            sapi.Speak(texto, 0)  # SPF_DEFAULT (sincrono)
            stream.Close()

            if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                if not self._cancelar_actual.is_set():
                    self.mci.reproducir_wav(temp_path, self._cancelar_actual, self._detener_evento)
                return True
        except Exception as err:
            logger.warning(f"Error sintesis OneCore COM: {err}")
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
        return False

    def _sintetizar_red_neuronal_local(self, chunk: str) -> bool:
        """Sintetiza mediante la red neuronal de LucIA (SLRN) de forma autonoma."""
        if self._cancelar_actual.is_set() or self._procesador_rn is None:
            return False
        try:
            import soundfile as sf
            res = self._procesador_rn.generar_audio_frase(chunk, self._nivel_emocion)
            if res is None:
                return False
            audio, sr = res
            temp_fd, temp_path = tempfile.mkstemp(suffix=".wav")
            os.close(temp_fd)
            try:
                sf.write(temp_path, audio, sr)
                if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                    if not self._cancelar_actual.is_set():
                        self.mci.reproducir_wav(temp_path, self._cancelar_actual, self._detener_evento)
                        return True
            finally:
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass
        except Exception as ex:
            logger.warning(f"Error en sintesis con red neuronal local: {ex}")
        return False

    def _procesar_mensaje(self, texto_completo: str) -> None:
        if not self._habilitado:
            return
        limpio = self.limpiador.sanitizar(texto_completo)
        if not limpio:
            return
        humanizado = self.modulador.humanizar_texto(limpio, aplicar_muletillas=True)
        chunks = self.limpiador.segmentar_frases(humanizado, max_chars=420)

        for fragmento in chunks:
            if self._cancelar_actual.is_set() or self._detener_evento.is_set():
                break
            # Prioridad 1: Voz oficial de LucIA (gTTS es-ES + DSP 1.16x de 18 anos)
            exito = False
            if self._gtts_disponible and not self._cancelar_actual.is_set():
                exito = self._sintetizar_y_reproducir_chunk(fragmento)
            # Prioridad 2: Red Neuronal autonoma de LucIA (SLRN)
            if not exito and not self._cancelar_actual.is_set():
                exito = self._sintetizar_red_neuronal_local(fragmento)
            # Prioridad 3: Fallback Windows SAPI OneCore
            if not exito and not self._cancelar_actual.is_set():
                self._sintetizar_fallback_windows(fragmento)

    def _procesar_cola(self) -> None:
        while not self._detener_evento.is_set():
            try:
                mensaje = self._cola_voz.get(timeout=0.3)
            except queue.Empty:
                continue

            if mensaje:
                self._cancelar_actual.clear()
                self._procesar_mensaje(mensaje)
            self._cola_voz.task_done()

    def cancel_tts(self) -> None:
        """Detiene inmediatamente el audio en curso y purga la cola."""
        self._cancelar_actual.set()
        self.mci.detener_inmediato()
        while not self._cola_voz.empty():
            try:
                self._cola_voz.get_nowait()
            except Exception:
                pass

    def speak(self, texto: str, esperar: bool = False, emocion: Optional[float] = None) -> None:
        """Emite un enunciado en la voz natural juvenil de LucIA."""
        if not texto or not texto.strip():
            return
        if emocion is not None:
            self.set_emocion(emocion)
        if esperar:
            self._cancelar_actual.clear()
            self._procesar_mensaje(texto)
        else:
            self._cola_voz.put(texto)

    def activar(self, estado: bool = True) -> None:
        """Habilita o desactiva la sintesis de voz global."""
        self._habilitado = bool(estado)
        if not self._habilitado:
            self.cancel_tts()

    def esta_activa(self) -> bool:
        return self._habilitado

    def detener(self) -> None:
        """Cierre definitivo del motor de audio y sus hilos asociados."""
        self._detener_evento.set()
        self.cancel_tts()


_voice_instance: Optional[VoiceEngine] = None
_global_lock_voice = threading.Lock()


def get_voice_engine() -> VoiceEngine:
    """Singleton thread-safe del VoiceEngine juvenil de LucIA."""
    global _voice_instance
    with _global_lock_voice:
        if _voice_instance is None:
            _voice_instance = VoiceEngine(voz=VOZ_OFICIAL)
    return _voice_instance


def speak(texto: str, esperar: bool = False, emocion: Optional[float] = None) -> None:
    """Conveniencia directa: emite voz con es-ES-ElviraNeural y tono fresco."""
    motor = get_voice_engine()
    motor.speak(texto, esperar=esperar, emocion=emocion)


def cancel_speech() -> None:
    """Interrumpe cualquier reproduccion sonora activa al instante."""
    motor = get_voice_engine()
    motor.cancel_tts()


def configurar_prosodia_juvenil(tono: str = "positivo") -> None:
    """Configura la expresion de la voz segun el tono neuronal detectado."""
    motor = get_voice_engine()
    mapeo_tonos = {"positivo": 0.85, "neutro": 0.60, "tecnico": 0.45, "negativo": 0.30}
    val = mapeo_tonos.get(tono.lower(), 0.60)
    motor.set_emocion(val)


if __name__ == "__main__":
    print("\033[38;5;51m" + "=" * 72 + "\033[0m")
    print("  Motor de Voz LucIA (18 Anos / es-ES / DSP 1.16x) - Autotest")
    print("\033[38;5;51m" + "=" * 72 + "\033[0m")
    v = get_voice_engine()
    demo_txt = "Hola! Soy Lucía. Estoy encantada de saludarte y mi voz suena completamente natural."
    print(f"  Texto original : {demo_txt}")
    h = v.modulador.humanizar_texto(demo_txt)
    r, p = v.modulador.calcular_prosodia(h, 0.75)
    print(f"  Humanizado     : {h}")
    print(f"  Prosodia       : Rate={r} | Pitch={p}")
    print("\033[38;5;48m  [OK] Subsistema de voz listo para integracion en mainLCSTM\033[0m")
    print("\033[38;5;51m" + "=" * 72 + "\033[0m")
