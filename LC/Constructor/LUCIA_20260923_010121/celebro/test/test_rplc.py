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

class TestTransformadorCognitivoRPLC:
    """Pruebas de adaptación Hebbiana, cálculo de valencia y deriva."""

    def test_transformar_vector_activa_neuronas(self) -> None:
        """Comprueba que el transformador aplique sigmoide y devuelva vector acotado."""
        transf = TransformadorCognitivoRPLC()
        v_in = [0.5] * NUM_NEURONAS
        v_out = transf.transformar(v_in)
        assert len(v_out) == NUM_NEURONAS
        assert all(0.0 <= x <= 1.0 for x in v_out)

    def test_valencia_sinaptica_rango(self) -> None:
        """Verifica que la valencia compare los hemisferios del vector [-1.0, 1.0]."""
        transf = TransformadorCognitivoRPLC()
        # Primeros 25 en 1.0, segundos 25 en 0.0 -> valencia positiva
        v_pos = [1.0] * 25 + [0.0] * 25
        val_pos = transf.valencia(v_pos)
        assert val_pos > 0.0

        # Primeros 25 en 0.0, segundos 25 en 1.0 -> valencia negativa
        v_neg = [0.0] * 25 + [1.0] * 25
        val_neg = transf.valencia(v_neg)
        assert val_neg < 0.0

        assert transf.valencia([]) == 0.0

    def test_neuronas_activas_umbral(self) -> None:
        """Comprueba que se identifiquen las neuronas por encima del umbral de activación."""
        transf = TransformadorCognitivoRPLC()
        v = [0.1] * NUM_NEURONAS
        v[5] = 0.85
        v[12] = 0.95
        activos = transf.activos(v)
        assert 5 in activos
        assert 12 in activos
        assert len(activos) == 2

    def test_deriva_media_evolucion(self) -> None:
        """Verifica que transformaciones sucesivas acumulen deriva en el historial."""
        transf = TransformadorCognitivoRPLC()
        v = [0.8] * NUM_NEURONAS
        transf.transformar(v)
        transf.transformar(v)
        deriva = transf.deriva_media()
        assert isinstance(deriva, float)
        assert deriva >= 0.0


# ============================================================================
# 4. CAPA 3: REFORMULADOR LINGÜÍSTICO (VOZ PROPIA LUCIA)
# ============================================================================

class TestReformuladorLinguisticoLucia:
    """Verifica la eliminación de muletillas y la inserción de aperturas y conectores."""

    def test_limpiar_muletillas_comerciales(self) -> None:
        """Comprueba que la regex interna remueva frases genéricas."""
        ref = ReformuladorLinguisticoLucia()
        sucio = "¡Hola! Claro que si, como modelo de lenguaje puedo ayudarte con esto."
        limpio = ref._limpiar(sucio)
        assert "hola" not in limpio.lower()
        assert "claro que si" not in limpio.lower()
        assert "como modelo de lenguaje" not in limpio.lower()
        assert "puedo ayudarte" in limpio

    def test_reformular_cuerpo_perspectiva(self) -> None:
        """Verifica que se adapten expresiones impersonales o asistenciales."""
        ref = ReformuladorLinguisticoLucia()
        original = "Te sugiero que revises este parámetro. Debes asegurarte de compilar."
        adaptado = ref._reformular_cuerpo(original)
        assert "mi analisis apunta a" in adaptado.lower()
        assert "considero que" in adaptado.lower()

    def test_reformular_completo_estructura(self) -> None:
        """Comprueba que la salida final contenga apertura, conector, cuerpo y cierre."""
        ref = ReformuladorLinguisticoLucia()
        texto_in = "El procesamiento paralelo incrementa el rendimiento del cluster."
        salida = ref.reformular(
            texto=texto_in, tono="tecnico", valencia=0.42, vector=[0.5] * 50, n_activas=15
        )
        lineas = salida.splitlines()
        assert len(lineas) >= 4
        assert "La arquitectura" in lineas[0] or "Sintetizando" in lineas[0] or "tensor" in lineas[0]
        assert "Valencia sinaptica" in salida or "neuronas activas" in salida or "Tono: tecnico" in salida

    def test_reformular_texto_vacio_retorna_fallback(self) -> None:
        """Comprueba que una entrada vacía después de limpiar retorne mensaje de fallback."""
        ref = ReformuladorLinguisticoLucia()
        salida = ref.reformular("", "neutro", 0.0, [], 0)
        assert "Mi red no ha capturado contenido suficiente" in salida


# ============================================================================
# 5. CACHÉ DE MEMORIA COGNITIVA Y REPETICIONES
# ============================================================================

class TestCacheMemoriaCognitiva:
    """Verifica el buffer de corto plazo y el control de redundancia lingüística."""

    def test_registro_en_cache_y_tamano_acotado(self) -> None:
        """Comprueba que la caché respete la capacidad máxima configurada."""
        cache = CacheMemoriaCognitiva(capacidad=10)
        for i in range(15):
            cache.registrar(f"Respuesta #{i}", valencia=0.1 * i, tono="positivo")

        assert cache.total_registros() == 10
        assert len(cache._entradas) == 10

    def test_deteccion_repeticion_por_hash(self) -> None:
        """Verifica que cadenas idénticas sean detectadas como repetitivas."""
        cache = CacheMemoriaCognitiva(capacidad=10)
        texto = "Este es un enunciado de prueba exacto."
        cache.registrar(texto, valencia=0.5, tono="tecnico")
        assert cache.es_repetitivo(texto) is True
        assert cache.es_repetitivo("Texto completamente diferente") is False

    def test_limpiar_cache(self) -> None:
        """Comprueba el vaciado total de la memoria de corto plazo."""
        cache = CacheMemoriaCognitiva(capacidad=5)
        cache.registrar("Prueba", 0.0, "neutro")
        assert cache.total_registros() == 1
        cache.vaciar()
        assert cache.total_registros() == 0


# ============================================================================
# 6. PIPELINE INTEGRAL: PROCESADOR RPLC
# ============================================================================

class TestProcesadorRPLC:
    """Pruebas sobre el orquestador maestro ProcesadorRPLC."""

    def test_procesar_pipeline_completo(self) -> None:
        """Verifica el flujo a través de las 3 capas y la emisión de métricas completas."""
        proc = ProcesadorRPLC()
        prompt = "Hola, por supuesto. Las redes neuronales profundas aprenden representaciones jerárquicas."
        ctx = {"norma_delta_aplicada": 5.0, "estado_emocional": 0.3}

        texto_out, metricas = proc.procesar(prompt, ctx=ctx)
        assert isinstance(texto_out, str)
        assert len(texto_out) > len(prompt) * 0.5

        # Verificación de métricas obligatorias
        assert "tono" in metricas
        assert "valencia_sinaptica" in metricas
        assert "densidad_semantica" in metricas
        assert "neuronas_activas" in metricas
        assert "deriva_media" in metricas
        assert "voz_ajena_detectada" in metricas
        assert "tiempo_rplc_ms" in metricas
        assert metricas["voz_ajena_detectada"] is True

    def test_procesar_simple_retorna_solo_texto(self) -> None:
        """Comprueba el método simplificado procesar_simple()."""
        proc = ProcesadorRPLC()
        res = proc.procesar_simple("Computación cuántica y tensores")
        assert isinstance(res, str)
        assert len(res) > 0

    def test_procesar_texto_vacio(self) -> None:
        """Verifica respuesta segura ante cadena en blanco."""
        proc = ProcesadorRPLC()
        txt, met = proc.procesar("   ")
        assert "Mi canal cognitivo no recibio contenido" in txt
        assert met == {}

    def test_diagnostico_metadatos(self) -> None:
        """Comprueba la generación del informe de diagnóstico."""
        proc = ProcesadorRPLC()
        diag = proc.diagnostico("Análisis estructural del mapa vectorial")
        assert isinstance(diag, dict)
        assert diag["tokens_extraidos"] >= 4
        assert diag["tokens_unicos"] >= 4
        assert "indices_activos" in diag


# ============================================================================
# 7. CASOS LÍMITE Y ROBUSTEZ LINGÜÍSTICA
# ============================================================================

class TestRobustezLingüística:
    """Verifica el comportamiento ante entradas no convencionales."""

    def test_entrada_solo_numeros(self) -> None:
        """Comprueba que secuencias de dígitos no originen excepciones matemáticas."""
        proc = ProcesadorRPLC()
        txt, met = proc.procesar("1234567 8901234 567890")
        assert isinstance(txt, str)
        assert "valencia_sinaptica" in met

    def test_entrada_con_caracteres_unicode_especiales(self) -> None:
        """Verifica el manejo de glifos orientales, emojis o símbolos matemáticos."""
        proc = ProcesadorRPLC()
        simbolos = "✦ ◈ ● ⚡ 🧠 ∑ ∏ ∫ √ α β γ"
        txt, met = proc.procesar(simbolos)
        assert isinstance(txt, str)
        assert "neuronas_activas" in met

    def test_entrada_multiparrafos_largos(self) -> None:
        """Comprueba que textos con múltiples saltos de línea se reestructuren con fluidez."""
        proc = ProcesadorRPLC()
        texto = "Párrafo 1 con datos.\n\nPárrafo 2 con detalles.\n\nPárrafo 3 con conclusiones."
        txt, met = proc.procesar(texto)
        assert isinstance(txt, str)
        assert len(txt.splitlines()) >= 3


# ============================================================================
# 8. MÉTODOS DE CONVENIENCIA GLOBALES Y CONCURRENCIA
# ============================================================================

class TestRPLCConcurrenciaYGlobales:
    """Pruebas de concurrencia multi-hilo y atajos globales."""

    def test_singleton_get_procesador_rplc(self) -> None:
        """Verifica que get_procesador_rplc devuelva la misma instancia singleton."""
        p1 = get_procesador_rplc()
        p2 = get_procesador_rplc()
        assert p1 is p2

    def test_reprocesar_con_metricas_global(self) -> None:
        """Comprueba la función de nivel de módulo reprocesar_con_metricas."""
        salida, met = reprocesar_con_metricas("Prueba de integración directa del subsistema")
        assert isinstance(salida, str)
        assert "repeticion_detectada" in met
        assert "items_en_cache" in met

    def test_reprocesar_para_lucia_shortcut(self) -> None:
        """Comprueba el atajo reprocesar_para_lucia()."""
        txt = reprocesar_para_lucia("Texto directo sin métricas")
        assert isinstance(txt, str)
        assert len(txt) > 0

    def test_obtener_estadisticas_rplc(self) -> None:
        """Verifica el consolidado de telemetría de RPLC."""
        stats = obtener_estadisticas_rplc()
        assert isinstance(stats, dict)
        assert "total" in stats
        assert "tiempo_medio_ms" in stats
        assert "memoria_cache_items" in stats

    def test_concurrencia_multi_hilo(self) -> None:
        """Verifica la thread-safety de RPLC ejecutando 10 hilos simultáneos."""
        proc = get_procesador_rplc()
        errores: List[Exception] = []

        def worker(idx: int) -> None:
            try:
                for _ in range(5):
                    proc.procesar(f"Mensaje de hilo concurrente #{idx} con datos técnicos")
            except Exception as e:
                errores.append(e)

        hilos = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for h in hilos:
            h.start()
        for h in hilos:
            h.join()

        assert len(errores) == 0, f"Ocurrieron excepciones concurrentes: {errores}"


# Fin de suite exhaustiva test_rplc.py
