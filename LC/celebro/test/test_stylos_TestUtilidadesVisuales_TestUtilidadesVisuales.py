class TestUtilidadesVisuales:
    """Pruebas de filtrado ANSI y medición de ancho visible."""

    def test_limpiar_codigos_ansi_simple(self) -> None:
        """Verifica que se eliminen códigos de color y formato correctamente."""
        texto_coloreado = f"{ColoresLucIA.BOLD}{ColoresLucIA.CIAN_ELECTRICO}Hola Mundo{ColoresLucIA.RESET}"
        limpio = limpiar_codigos_ansi(texto_coloreado)
        assert limpio == "Hola Mundo"

    def test_limpiar_codigos_ansi_texto_sin_formato(self) -> None:
        """Comprueba que el texto sin formato se preserve intacto."""
        texto_plano = "Texto ordinario sin secuencias de escape 12345"
        assert limpiar_codigos_ansi(texto_plano) == texto_plano

    def test_longitud_visual_exacta(self) -> None:
        """Comprueba que longitud_visual ignore códigos de escape al medir."""
        texto_complejo = f"{ColoresLucIA.VIOLETA_NEON}Test{ColoresLucIA.RESET} {ColoresLucIA.AMBAR_SOLAR}123{ColoresLucIA.RESET}"
        assert longitud_visual(texto_complejo) == 8

    def test_longitud_visual_vacia(self) -> None:
        """Comprueba que una cadena vacía tenga longitud cero."""
        assert longitud_visual("") == 0
        assert longitud_visual(f"{ColoresLucIA.RESET}") == 0

    def test_obtener_ancho_consola_limites(self) -> None:
        """Verifica que el ancho calculado se encuentre dentro de los límites seguros."""
        ancho = obtener_ancho_consola(max_limite=90)
        assert isinstance(ancho, int)
        assert 60 <= ancho <= 90


# ============================================================================
# 4. RENDERIZADOR MARKDOWN (SYNTAX HIGHLIGHTING Y TABLAS)
# ============================================================================

