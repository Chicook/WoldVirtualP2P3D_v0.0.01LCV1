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

