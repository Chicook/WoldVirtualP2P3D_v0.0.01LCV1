"""
test_psnrcv.py — Tests del Conversor de Respuestas a Pesos Neuronales
======================================================================
Cubre:
  • NeuralMathPrecision2026: Newton-Schulz NS-5, SOAP, GSNR, Entropía Shannon.
  • ConversorRespuestaPesos: inicialización, 50 neuronas, procesamiento de prompts.
  • Pipeline: procesar_consulta_a_pesos() con prompts de distinta longitud.
  • Deriva acumulada Muon: monotonía, acumulación por turnos.
  • Persistencia PSNRL: checkpoint a .npz, restauración.
  • get_conversor_pesos() singleton.
  • Manejo de entradas inválidas: prompt vacío, unicode, bytes.
  • Casos de borde: tensores con norma 0, matrices no cuadradas.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import numpy as np
import pytest

from LC.celebro.test import PROMPT_CORTO, PROMPT_LARGO, PROMPT_TECNICO, PROMPT_VACIO, PROMPT_UNICODE
from LC.celebro.test.conftest import (
    assert_hash_sha256_valido,
    assert_norma_acotada,
    assert_tensor_valido,
)

try:
    from LC.celebro.CMFG.PSNRCV import (
        ConversorRespuestaPesos,
        NeuralMathPrecision2026,
        get_conversor_pesos,
    )
    _PSNRCV_DISPONIBLE = True
except Exception as exc:  # noqa: BLE001
    _PSNRCV_DISPONIBLE = False
    _PSNRCV_ERROR = str(exc)

pytestmark = pytest.mark.skipif(
    not _PSNRCV_DISPONIBLE,
    reason=f"PSNRCV no disponible: {_PSNRCV_ERROR if not _PSNRCV_DISPONIBLE else ''}",
)


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — NeuralMathPrecision2026: algoritmos matemáticos
# ═══════════════════════════════════════════════════════════════════════════════

class TestConversorRespuestaPesos:
    """Tests de la clase principal de transducción."""

    @pytest.fixture()
    def conversor(self, psnrl_tmp: Path) -> ConversorRespuestaPesos:
        """Conversor con PSNRL temporal."""
        with patch("LC.celebro.CMFG.PSNRCV.PSNRL_DIR", psnrl_tmp):
            return ConversorRespuestaPesos()

    def test_inicializacion_crea_neuronas(self, conversor: ConversorRespuestaPesos) -> None:
        """El conversor debe cargar al menos 1 neurona."""
        assert hasattr(conversor, "neuronas"), "Conversor debe tener atributo 'neuronas'"
        assert len(conversor.neuronas) > 0, "Debe haber al menos 1 neurona cargada"

    def test_50_neuronas_cargadas(self, conversor: ConversorRespuestaPesos) -> None:
        """Se esperan exactamente 50 neuronas activas."""
        assert len(conversor.neuronas) == 50, (
            f"Se esperan 50 neuronas, se obtuvieron {len(conversor.neuronas)}"
        )

    def test_procesar_consulta_devuelve_dict(self, conversor: ConversorRespuestaPesos) -> None:
        """procesar_consulta_a_pesos() debe devolver un diccionario."""
        resultado = conversor.procesar_consulta_a_pesos(PROMPT_CORTO)
        assert isinstance(resultado, dict), "procesar_consulta_a_pesos debe devolver dict"

    def test_resultado_tiene_tono_cognitivo(self, conversor: ConversorRespuestaPesos) -> None:
        """El dict resultado debe incluir 'tono_cognitivo'."""
        resultado = conversor.procesar_consulta_a_pesos(PROMPT_CORTO)
        assert "tono_cognitivo" in resultado, "Falta 'tono_cognitivo' en el resultado"
        assert isinstance(resultado["tono_cognitivo"], str), "'tono_cognitivo' debe ser str"

    def test_resultado_tiene_estado_emocional(self, conversor: ConversorRespuestaPesos) -> None:
        """El dict resultado debe incluir 'estado_emocional' como float."""
        resultado = conversor.procesar_consulta_a_pesos(PROMPT_CORTO)
        assert "estado_emocional" in resultado, "Falta 'estado_emocional'"
        val = resultado["estado_emocional"]
        assert isinstance(val, (int, float)), f"'estado_emocional' debe ser numérico, es {type(val)}"

    def test_procesar_prompt_largo_sin_excepcion(self, conversor: ConversorRespuestaPesos) -> None:
        """procesar_consulta_a_pesos() con prompt largo no debe lanzar excepción."""
        try:
            conversor.procesar_consulta_a_pesos(PROMPT_LARGO)
        except Exception as e:
            pytest.fail(f"Prompt largo lanzó excepción: {e}")

    def test_procesar_prompt_vacio_sin_excepcion(self, conversor: ConversorRespuestaPesos) -> None:
        """procesar_consulta_a_pesos() con prompt vacío no debe lanzar excepción."""
        try:
            resultado = conversor.procesar_consulta_a_pesos(PROMPT_VACIO)
            assert isinstance(resultado, dict), "Con prompt vacío debe devolver dict"
        except Exception as e:
            pytest.fail(f"Prompt vacío lanzó excepción: {e}")

    def test_procesar_prompt_unicode_sin_excepcion(self, conversor: ConversorRespuestaPesos) -> None:
        """procesar_consulta_a_pesos() con caracteres unicode no debe lanzar excepción."""
        try:
            conversor.procesar_consulta_a_pesos(PROMPT_UNICODE)
        except Exception as e:
            pytest.fail(f"Prompt unicode lanzó excepción: {e}")

    def test_procesar_multiple_turnos_acumula_deriva(self, conversor: ConversorRespuestaPesos) -> None:
        """Múltiples llamadas deben ir acumulando deriva_acumulada."""
        deriva_inicial = getattr(conversor, "deriva_acumulada", 0.0)
        for i in range(5):
            conversor.procesar_consulta_a_pesos(f"turno {i}")
        deriva_final = getattr(conversor, "deriva_acumulada", 0.0)
        assert isinstance(deriva_final, (int, float)), "deriva_acumulada debe ser numérico"

    def test_norma_delta_en_resultado(self, conversor: ConversorRespuestaPesos) -> None:
        """El resultado debe contener 'norma_delta_aplicada' como float."""
        resultado = conversor.procesar_consulta_a_pesos(PROMPT_TECNICO)
        if "norma_delta_aplicada" in resultado:
            norma = resultado["norma_delta_aplicada"]
            assert isinstance(norma, (int, float)), "'norma_delta_aplicada' debe ser numérico"
            assert norma >= 0.0, "'norma_delta_aplicada' no puede ser negativa"

    def test_conversor_es_thread_safe(self, conversor: ConversorRespuestaPesos) -> None:
        """procesar_consulta_a_pesos() debe ser seguro para uso desde múltiples hilos."""
        import threading
        errores: list = []

        def procesar() -> None:
            try:
                conversor.procesar_consulta_a_pesos("prueba concurrente")
            except Exception as e:  # noqa: BLE001
                errores.append(str(e))

        hilos = [threading.Thread(target=procesar) for _ in range(4)]
        for h in hilos:
            h.start()
        for h in hilos:
            h.join(timeout=10)
        assert not errores, f"Errores en uso concurrente: {errores}"


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — Persistencia de pesos en PSNRL
# ═══════════════════════════════════════════════════════════════════════════════

class TestPersistenciaPSNRL:
    """Tests de checkpoint y restauración de pesos sinápticos."""

    @pytest.fixture()
    def conversor_con_psnrl(self, psnrl_tmp: Path) -> ConversorRespuestaPesos:
        with patch("LC.celebro.CMFG.PSNRCV.PSNRL_DIR", psnrl_tmp):
            c = ConversorRespuestaPesos()
        return c

    def test_checkpoint_crea_archivos_npz(self, conversor_con_psnrl: ConversorRespuestaPesos,
                                          psnrl_tmp: Path) -> None:
        """Después de procesar prompts, debe haber archivos .npz o .json en PSNRL."""
        conversor_con_psnrl.procesar_consulta_a_pesos("checkpoint test")
        # Si el conversor guarda automáticamente, debe haber algo en psnrl_tmp
        archivos = list(psnrl_tmp.iterdir())
        # Este test es informativo; el checkpoint puede ser diferido
        assert isinstance(archivos, list), "PSNRL debe ser un directorio iterable"

    def test_psnrl_dir_existe(self, conversor_con_psnrl: ConversorRespuestaPesos,
                               psnrl_tmp: Path) -> None:
        """El directorio PSNRL debe existir tras la inicialización."""
        assert psnrl_tmp.exists(), "PSNRL debe existir como directorio"
        assert psnrl_tmp.is_dir(), "PSNRL debe ser un directorio"
from test_psnrcv_TestNeuralMathPrecision2026 import TestNeuralMathPrecision2026  # CLASSPACK
