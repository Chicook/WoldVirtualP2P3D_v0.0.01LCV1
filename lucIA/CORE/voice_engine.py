"""
lucIA.CORE.voice_engine - Motor de síntesis de voz neuronal (TTS) para LucIA
=============================================================================

Voz oficial asignada: es-ES-ElviraNeural (Voz neuronal femenina en español).
- Sintetiza texto completo sin truncamiento mediante partición inteligente de frases.
- Limpieza profunda de markdown, listas y símbolos para dicción fluida en español.
- Capacidad de interrupción instantánea (cancel_tts) cuando el usuario formula una nueva consulta.
- Reproducción directa con MCI nativo de Windows (winmm.dll) con control de estado y parada inmediata.
- Fallback automático a System.Speech.Synthesis en caso offline.
"""

import os
import re
import queue
import ctypes
import asyncio
import logging
import tempfile
import threading
import subprocess
from typing import Optional, List

logger = logging.getLogger("lucIA.Voice")

VOZ_OFICIAL = "es-ES-ElviraNeural"


class VoiceEngine:
    """
    Sintetizador de voz neuronal para LucIA con voz oficial es-ES-ElviraNeural.
    """

    def __init__(self, voz: str = VOZ_OFICIAL):
        self.voz = voz
        self._cola_voz: queue.Queue[str] = queue.Queue()
        self._detener_evento = threading.Event()
        self._cancelar_actual = threading.Event()
        self._hilo_worker: Optional[threading.Thread] = None
        self._alias_actual = "lucia_audio_active"
        self._mci_lock = threading.Lock()
        self._edge_tts_disponible = False
        self._verificar_edge_tts()
        self._iniciar_worker()

    def _verificar_edge_tts(self) -> None:
        try:
            import edge_tts
            self._edge_tts_disponible = True
            logger.info(f"Voz neuronal configurada con éxito: {self.voz}")
        except ImportError:
            self._edge_tts_disponible = False
            logger.warning(f"edge_tts no detectado, usando fallback del sistema")

    def _iniciar_worker(self) -> None:
        """Inicia el hilo trabajador en segundo plano."""
        self._hilo_worker = threading.Thread(target=self._procesar_cola, daemon=True, name="LucIA-Voice-Worker")
        self._hilo_worker.start()

    def limpiar_texto_para_habla(self, texto: str) -> str:
        """
        Limpia markdown, viñetas, tablas, símbolos y etiquetas
        para que la dicción con es-ES-ElviraNeural sea 100% natural, fluida y sin tropiezos.
        """
        if not texto:
            return ""
        texto = texto.strip()
        # Eliminar bloques de pensamiento o razonamiento residuales <think>...</think>
        texto = re.sub(r'<think>.*?</think>', '', texto, flags=re.DOTALL)
        # Eliminar bloques de código ```...```
        texto = re.sub(r'```.*?```', ' bloque de código omitido ', texto, flags=re.DOTALL)
        # Eliminar código en línea `...`
        texto = re.sub(r'`[^`]*`', '', texto)
        # Eliminar enlaces markdown [texto](url) -> texto
        texto = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', texto)
        # Limpiar encabezados markdown (##, ###)
        texto = re.sub(r'^\s*#+\s*', '', texto, flags=re.MULTILINE)
        # Limpiar viñetas numéricas "1. ", "2. "
        texto = re.sub(r'^\s*\d+\.\s*', '', texto, flags=re.MULTILINE)
        # Limpiar viñetas de guion o asterisco "* ", "- ", "+ "
        texto = re.sub(r'^\s*[-*+]\s+', '', texto, flags=re.MULTILINE)
        # Eliminar asteriscos de negrita/cursiva
        texto = re.sub(r'\*+', '', texto)
        # Eliminar caracteres especiales de markdown o tablas
        texto = re.sub(r'[_~>|\\]', ' ', texto)
        # Preservar caracteres estándar, tildes y eñes
        texto = "".join(c for c in texto if ord(c) < 0x10000)
        texto = re.sub(r'[\u2600-\u27BF]', '', texto)
        # Limpiar espacios múltiples y normalizar saltos de línea a pausas
        texto = re.sub(r'[ \t]+', ' ', texto)
        texto = re.sub(r'\n\s*\n+', '. ', texto)
        texto = re.sub(r'\n', '. ', texto)
        texto = re.sub(r'\.{2,}', '.', texto)
        texto = texto.strip()
        if texto and texto[-1] not in '.!?':
            texto += '.'
        return texto

    @staticmethod
    def _chunk_text(texto: str, max_chars: int = 1500) -> List[str]:
        """
        Divide textos en bloques de hasta 1500 caracteres (conforme a la arquitectura original de lucIA)
        para que respuestas extensas se sinteticen en un único flujo de audio continuo sin pausas.
        """
        if len(texto) <= max_chars:
            return [texto]
        chunks = []
        start = 0
        while start < len(texto):
            remaining = len(texto) - start
            if remaining <= max_chars:
                chunks.append(texto[start:].strip())
                break
            end = start + max_chars
            split_at = -1
            for sep in ('. ', '! ', '? ', '; '):
                idx = texto.rfind(sep, start, end)
                if idx > split_at:
                    split_at = idx + len(sep)
            if split_at > start:
                chunks.append(texto[start:split_at].strip())
                start = split_at
            else:
                idx = texto.rfind(' ', start, end)
                if idx > start:
                    chunks.append(texto[start:idx].strip())
                    start = idx + 1
                else:
                    chunks.append(texto[start:end].strip())
                    start = end
        return [c for c in chunks if c]

    def _detener_mci_inmediato(self) -> None:
        """Detiene cualquier audio reproduciéndose en el dispositivo MCI."""
        with self._mci_lock:
            try:
                mci = ctypes.windll.winmm.mciSendStringW
                mci(f"stop {self._alias_actual}", None, 0, None)
                mci(f"close {self._alias_actual}", None, 0, None)
            except Exception:
                pass

    def _reproducir_con_mci(self, ruta_mp3: str) -> bool:
        """Reproduce un archivo MP3 con soporte de interrupción instantánea."""
        with self._mci_lock:
            if self._cancelar_actual.is_set():
                return False
            try:
                mci = ctypes.windll.winmm.mciSendStringW
                mci(f"close {self._alias_actual}", None, 0, None)
                # Usar ruta absoluta normalizada
                ruta_limpia = os.path.abspath(ruta_mp3).replace("\\", "/")
                res_open = mci(f'open "{ruta_limpia}" type mpegvideo alias {self._alias_actual}', None, 0, None)
                if res_open != 0:
                    return False
                mci(f"play {self._alias_actual}", None, 0, None)
            except Exception as e:
                logger.warning(f"Error al iniciar audio MCI: {e}")
                return False

        # Monitorear reproducción comprobando si se solicita cancelar
        buf = ctypes.create_unicode_buffer(128)
        import time
        # Dar margen inicial para que el buffer de MCI entre en reproducción
        time.sleep(0.08)
        while not self._cancelar_actual.is_set() and not self._detener_evento.is_set():
            time.sleep(0.1)
            with self._mci_lock:
                res = mci(f"status {self._alias_actual} mode", buf, 128, None)
                if res != 0 or buf.value not in ("playing", ""):
                    break

        with self._mci_lock:
            mci(f"stop {self._alias_actual}", None, 0, None)
            mci(f"close {self._alias_actual}", None, 0, None)

        return True

    def _sintetizar_y_reproducir_chunk(self, chunk: str) -> bool:
        """Sintetiza un fragmento con es-ES-ElviraNeural y lo reproduce."""
        if self._cancelar_actual.is_set():
            return False

        import edge_tts
        temp_fd, temp_path = tempfile.mkstemp(suffix=".mp3")
        os.close(temp_fd)

        async def _gen():
            comm = edge_tts.Communicate(chunk, self.voz, rate="+0%", pitch="+0Hz")
            await comm.save(temp_path)

        try:
            # Timeout generoso de 15 segundos para síntesis fiable
            asyncio.run(asyncio.wait_for(_gen(), timeout=15.0))
            if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                if not self._cancelar_actual.is_set():
                    self._reproducir_con_mci(temp_path)
                    return True
        except Exception as e:
            logger.warning(f"Fallo al sintetizar chunk con {self.voz}: {e}")
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
        return False

    def _sintetizar_fallback_windows(self, texto_limpio: str) -> None:
        """Fallback local rápido en caso offline."""
        if self._cancelar_actual.is_set():
            return
        texto_escapado = texto_limpio.replace("'", "''").replace('"', '`"')
        comando_ps = (
            f"Add-Type -AssemblyName System.Speech; "
            f"$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$synth.Rate = 0; "
            f"$synth.Speak('{texto_escapado}')"
        )
        try:
            subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", comando_ps],
                capture_output=True,
                text=True,
                timeout=30
            )
        except Exception as e:
            logger.warning(f"Error en fallback de voz: {e}")

    def _procesar_mensaje(self, texto_completo: str) -> None:
        """Procesa y habla el texto completo dividiéndolo en fragmentos sin truncar."""
        texto_limpio = self.limpiar_texto_para_habla(texto_completo)
        if not texto_limpio:
            return

        chunks = self._chunk_text(texto_limpio, max_chars=450)
        for chunk in chunks:
            if self._cancelar_actual.is_set() or self._detener_evento.is_set():
                break
            exito = False
            if self._edge_tts_disponible:
                exito = self._sintetizar_y_reproducir_chunk(chunk)
            if not exito and not self._cancelar_actual.is_set():
                self._sintetizar_fallback_windows(chunk)

    def _procesar_cola(self) -> None:
        """Bucle consumidor de la cola de mensajes de voz."""
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
        """
        Interrumpe inmediatamente cualquier habla en curso y vacía la cola.
        Llamado automáticamente cuando el usuario formula una nueva pregunta.
        """
        self._cancelar_actual.set()
        self._detener_mci_inmediato()
        while not self._cola_voz.empty():
            try:
                self._cola_voz.get_nowait()
            except Exception:
                pass

    def speak(self, texto: str, esperar: bool = False) -> None:
        """
        Encola el texto para que LucIA lo hable completo con es-ES-ElviraNeural.
        """
        if not texto or not texto.strip():
            return
        if esperar:
            self._cancelar_actual.clear()
            self._procesar_mensaje(texto)
        else:
            self._cola_voz.put(texto)

    def detener(self) -> None:
        """Detiene completamente el sintetizador."""
        self._detener_evento.set()
        self.cancel_tts()


# Instancia global única
_voice_instance: Optional[VoiceEngine] = None


def get_voice_engine() -> VoiceEngine:
    global _voice_instance
    if _voice_instance is None:
        _voice_instance = VoiceEngine(voz=VOZ_OFICIAL)
    return _voice_instance


def speak(texto: str, esperar: bool = False) -> None:
    """Función de alto nivel para hacer hablar a LucIA con es-ES-ElviraNeural."""
    motor = get_voice_engine()
    motor.speak(texto, esperar=esperar)


def cancel_speech() -> None:
    """Detiene cualquier habla en reproducción."""
    motor = get_voice_engine()
    motor.cancel_tts()
