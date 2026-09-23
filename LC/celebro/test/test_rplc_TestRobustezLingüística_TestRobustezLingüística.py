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

