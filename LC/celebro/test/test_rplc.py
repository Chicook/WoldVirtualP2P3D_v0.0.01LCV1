"""
test_rplc.py — Suite exhaustiva de pruebas para el subsistema RPLC de LucIA
=============================================================================
Cubre el procesador lingüístico cognitivo RPLC (Reestructurador de Prosa LucIA):
  1. Capa 1: AnalizadorTextualNeuronal (tokenización, vectorización 50D, tonos, densidad semántica).
  2. Detección de marcadores de voz ajena ("como modelo de lenguaje", "soy gpt", "por supuesto").
  3. Capa 2: TransformadorCognitivoRPLC (modulación Hebbiana, cálculo de valencia, deriva sináptica).
  4. Capa 3: ReformuladorLinguisticoLucia (limpieza regex, conectores, estructura de voz propia).
  5. CacheMemoriaCognitiva: registro de turnos, hashes sha256 y detección de respuestas repetitivas.
  6. Pipeline integral ProcesadorRPLC: procesar, procesar_simple, diagnostico y métricas globales.
  7. Casos límite de texto y estrés (textos largos, caracteres especiales, números y listas).
  8. Funciones de conveniencia del módulo: get_procesador_rplc, reprocesar_con_metricas, etc.
  9. Concurrencia y estabilidad de estados con múltiples hilos en paralelo.
"""
from __future__ import annotations

import math
import threading
from typing import Any, Dict, List
import pytest

from LC.celebro.CMFG.SBSTM.RPLC import (
    NUM_NEURONAS,
    UMBRAL_ACTIVACION,
    AnalizadorTextualNeuronal,
    CacheMemoriaCognitiva,
    ProcesadorRPLC,
    ReformuladorLinguisticoLucia,
    TransformadorCognitivoRPLC,
    get_procesador_rplc,
    obtener_estadisticas_rplc,
    reprocesar_con_metricas,
    reprocesar_para_lucia,
)


# ============================================================================
# 1. CAPA 1: ANALIZADOR TEXTUAL NEURONAL
# ============================================================================

class TestAnalizadorTextualNeuronal:
    """Pruebas unitarias para tokenización, vectorización y análisis semántico."""

    def test_tokenizar_texto_estandar(self) -> None:
        """Verifica que el analizador limpie signos y devuelva tokens en minúsculas."""
        analizador = AnalizadorTextualNeuronal()
        texto = "¡Hola! Este es un análisis semántico de 50 neuronas, ¿correcto?"
        tokens = analizador.tokenizar(texto)
        assert isinstance(tokens, list)
        assert "hola" in tokens
        assert "análisis" in tokens or "analisis" in tokens
        assert "semántico" in tokens or "semantico" in tokens
        assert "50" in tokens
        assert "y" not in tokens

    def test_tokenizar_texto_vacio(self) -> None:
        """Comprueba que una entrada vacía o con solo espacios devuelva lista vacía."""
        analizador = AnalizadorTextualNeuronal()
        assert analizador.tokenizar("") == []
        assert [t for t in analizador.tokenizar("   !?.   ") if t] == []

    def test_vectorizar_dimension_exacta_50d(self) -> None:
        """Verifica que la salida vectorial sea siempre de dimensión NUM_NEURONAS (50)."""
        analizador = AnalizadorTextualNeuronal(num_neuronas=50)
        texto = "Inferencia neuronal distribuida con optimización de pesos sinápticos"
        vector = analizador.vectorizar(texto)
        assert isinstance(vector, list)
        assert len(vector) == 50
        assert all(isinstance(v, float) for v in vector)
        assert all(0.0 <= v <= 1.0 for v in vector)

    def test_vectorizar_texto_vacio_retorna_ceros(self) -> None:
        """Comprueba que texto vacío genere un vector de 50 ceros."""
        analizador = AnalizadorTextualNeuronal(num_neuronas=50)
        vector_ceros = analizador.vectorizar("")
        assert len(vector_ceros) == 50
        assert all(v == 0.0 for v in vector_ceros)

    def test_vectorizar_estabilidad_textos_largos(self) -> None:
        """Comprueba que párrafos extensos no saturen el vector por encima de 1.0."""
        analizador = AnalizadorTextualNeuronal()
        parrafo = "Convergencia sináptica recursiva y ajuste continuo de gradientes. " * 50
        vec = analizador.vectorizar(parrafo)
        assert len(vec) == 50
        assert max(vec) <= 1.0
        assert min(vec) >= 0.0

    @pytest.mark.parametrize("texto,tono_esperado", [
        ("Excelente resultado, el algoritmo es perfecto y un éxito rotundo", "positivo"),
        ("Error crítico, fallo grave, problema imposible de resolver", "negativo"),
        ("El tensor semántico y la matriz de la red neuronal modulan el código", "tecnico"),
        ("Hoy es martes por la tarde y la temperatura es moderada", "neutro"),
    ])
    def test_analizar_tono(self, texto: str, tono_esperado: str) -> None:
        """Verifica la clasificación de tono semántico."""
        analizador = AnalizadorTextualNeuronal()
        tono_detectado = analizador.analizar_tono(texto)
        assert tono_detectado == tono_esperado

    def test_calcular_densidad_semantica_rango(self) -> None:
        """Comprueba que la densidad semántica se encuentre acotada entre 0.0 y 1.0."""
        analizador = AnalizadorTextualNeuronal()
        densidad = analizador.calcular_densidad_semantica("Texto representativo de prueba cognitiva")
        assert 0.0 <= densidad <= 1.0
        assert analizador.calcular_densidad_semantica("") == 0.0


# ============================================================================
# 2. DETECCIÓN Y FILTRADO DE VOZ AJENA
# ============================================================================

class TestDeteccionVozAjena:
    """Verifica la identificación de clichés de LLMs comerciales externos."""

    @pytest.mark.parametrize("frase_ajena", [
        "Por supuesto, puedo ayudarte con eso.",
        "Como modelo de lenguaje, no tengo sentimientos.",
        "Soy un asistente virtual creado para responder preguntas.",
        "Hola, soy Claude de Anthropic.",
        "Recuerda que como inteligencia artificial debo recordarte esto.",
    ])
    def test_detectar_voz_ajena_positivo(self, frase_ajena: str) -> None:
        """Comprueba que marcadores ajenos conocidos sean detectados."""
        analizador = AnalizadorTextualNeuronal()
        assert analizador.detectar_voz_ajena(frase_ajena) is True

    def test_detectar_voz_ajena_negativo(self) -> None:
        """Verifica que textos formulados con voz cognitiva propia no se marquen como ajenos."""
        analizador = AnalizadorTextualNeuronal()
        texto_propio = "Desde mi análisis sináptico, identifico el vector de solución apropiado."
        assert analizador.detectar_voz_ajena(texto_propio) is False


# ============================================================================
# 3. CAPA 2: TRANSFORMADOR COGNITIVO HEBBIANO
# ============================================================================

from LC.celebro.test.test_rplc_TestTransformadorCognitivoRPLC import TestTransformadorCognitivoRPLC  # CLASSPACK
from LC.celebro.test.test_rplc_TestReformuladorLinguisticoLucia import TestReformuladorLinguisticoLucia  # CLASSPACK
from LC.celebro.test.test_rplc_TestCacheMemoriaCognitiva import TestCacheMemoriaCognitiva  # CLASSPACK
from LC.celebro.test.test_rplc_TestProcesadorRPLC import TestProcesadorRPLC  # CLASSPACK
from LC.celebro.test.test_rplc_TestRobustezLingüística import TestRobustezLingüística  # CLASSPACK
from LC.celebro.test.test_rplc_TestRPLCConcurrenciaYGlobales import TestRPLCConcurrenciaYGlobales  # CLASSPACK
