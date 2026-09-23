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

