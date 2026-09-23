"""
voice_engine.py - Motor de Sintesis de Voz Neuronal para LucIA (WoldVirtualP2P3D 2026)
=====================================================================================
Modo de expresion: Voz juvenil, espontanea, dinamica y fresca (estilo chica 18 anos).
Voz neuronal oficial: es-ES-ElviraNeural (es-ES / Microsoft Speech Services).
- Modulador prosodico: variaciones naturales de pitch y cadencia segun estado cognitivo.
- Insercion organica de muletillas de expresion conversacional dinamica.
- Diccion fluida: sanitizacion profunda de markdown, simbolos y elementos tecnicos.
- Interrupcion instantanea (cancel_tts) con winmm.dll (MCI) y soporte asincrono en hilo worker.
- Fallback automatico a System.Speech.Synthesis en entornos sin conexion.
"""
from __future__ import annotations

import asyncio
import ctypes
import hashlib
import logging
import os
import queue
import random
import re
import subprocess
import tempfile
import threading
import time
from typing import Any, Dict, Final, List, Optional, Tuple

logger = logging.getLogger("lucIA.Voice")

VOZ_OFICIAL: Final[str] = "es-ES-ElviraNeural"
VERSION_VOICE: Final[str] = "2026.4.2"


class VoiceEngine:
    """
    Sintetizador neuronal de voz para LucIA con voz oficial es-ES-ElviraNeural
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
        self._edge_tts_disponible: bool = False
        self._nivel_emocion: float = 0.65
        self._habilitado: bool = True
        self._verificar_edge_tts()
        self._iniciar_worker()

    def _verificar_edge_tts(self) -> None:
        try:
            import edge_tts
            self._edge_tts_disponible = True
            logger.info(f"Voz neuronal configurada: {self.voz} (18 anos / ElviraNeural)")
        except ImportError:
            self._edge_tts_disponible = False
            logger.warning("edge_tts no detectado, usando fallback Windows")

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
        if self._cancelar_actual.is_set():
            return False
        if not self._edge_tts_disponible:
            return False
        temp_fd, temp_path = tempfile.mkstemp(suffix=".mp3")
        os.close(temp_fd)

        rate_str, pitch_str = self.modulador.calcular_prosodia(chunk, self._nivel_emocion)

        async def _gen() -> None:
            import edge_tts  # import diferido: nunca rompe el worker si falta
            comm = edge_tts.Communicate(chunk, self.voz, rate=rate_str, pitch=pitch_str)
            await comm.save(temp_path)

        try:
            try:
                asyncio.run(asyncio.wait_for(_gen(), timeout=18.0))
            except RuntimeError:  # ya hay un loop corriendo en este hilo
                loop = asyncio.new_event_loop()
                try:
                    loop.run_until_complete(asyncio.wait_for(_gen(), timeout=18.0))
                finally:
                    loop.close()
            if os.path.exists(temp_path) and os.path.getsize(temp_path) > 1024:
                if not self._cancelar_actual.is_set():
                    self.mci.reproducir_mp3(temp_path, self._cancelar_actual, self._detener_evento)
                    return True
        except Exception as err:
            logger.warning(f"Fallo en sintesis edge_tts: {err}")
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
        return False

    def _sintetizar_fallback_windows(self, texto_limpio: str) -> None:
        """Fallback offline con voz espanola (Sabina/Helena), nunca robotica inglesa."""
        if self._cancelar_actual.is_set():
            return
        t_esc = texto_limpio.replace("'", " ").replace('"', " ").replace("`", " ")[:900]
        ps = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "$voz = $s.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Culture.Name -like 'es*' } "
            "| Select-Object -First 1; "
            "if ($voz) { $s.SelectVoice($voz.VoiceInfo.Name) }; "
            "$s.Rate = 0; $s.Volume = 100; "
            f"$s.Speak('{t_esc}')"
        )
        try:
            subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                capture_output=True,
                text=True,
                timeout=40,
            )
        except Exception as err:
            logger.warning(f"Error fallback System.Speech: {err}")

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
            exito = False
            if self._edge_tts_disponible:
                exito = self._sintetizar_y_reproducir_chunk(fragmento)
            if not exito and not self._cancelar_actual.is_set():
                self._sintetizar_fallback_windows(fragmento)

    def _procesar_cola(self) -> None:
        while not self._detener_evento.is_set():
            try:
                mensaje = self._cola_voz.get(timeout=0.3)
            except queue.Empty:
                continue

            try:
                if mensaje:
                    self._cancelar_actual.clear()
                    try:
                        self._procesar_mensaje(mensaje)
                    except Exception as err:
                        logger.warning(f"Error en worker de voz (recuperado): {err}")
            finally:
                self._cola_voz.task_done()

    def cancel_tts(self) -> None:
        """Detiene inmediatamente el audio en curso y purga la cola sin bloquearla."""
        self._cancelar_actual.set()
        self.mci.detener_inmediato()
        while not self._cola_voz.empty():
            try:
                self._cola_voz.get_nowait()
                self._cola_voz.task_done()
            except Exception:
                break

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
    print("  Motor de Voz LucIA (18 Anos / es-ES-ElviraNeural) - Autotest")
    print("\033[38;5;51m" + "=" * 72 + "\033[0m")
    v = get_voice_engine()
    demo_txt = "Hola! Por lo tanto estoy procesando todo genial y mi voz suena super natural."
    print(f"  Texto original : {demo_txt}")
    h = v.modulador.humanizar_texto(demo_txt)
    r, p = v.modulador.calcular_prosodia(h, 0.75)
    print(f"  Humanizado     : {h}")
    print(f"  Prosodia       : Rate={r} | Pitch={p}")
    print("\033[38;5;48m  [OK] Subsistema de voz listo para integracion en mainLCSTM\033[0m")
    print("\033[38;5;51m" + "=" * 72 + "\033[0m")
from voice_engine_ModuladorExpresivoJuvenil import ModuladorExpresivoJuvenil  # CLASSPACK
from voice_engine_LimpiadorFoneticoLucIA import LimpiadorFoneticoLucIA  # CLASSPACK
from voice_engine_ReproductorMCIWindows import ReproductorMCIWindows  # CLASSPACK
