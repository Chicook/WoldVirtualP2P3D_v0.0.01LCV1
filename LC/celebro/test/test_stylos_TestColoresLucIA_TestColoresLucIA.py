class TestColoresLucIA:
    """Verifica la integridad de las secuencias de escape ANSI."""

    def test_colores_escape_ansi_no_vacios(self) -> None:
        """Comprueba que todos los atributos de color sean cadenas ANSI no vacías."""
        atributos = [
            "RESET", "BOLD", "DIM", "ITALIC", "UNDERLINE",
            "CIAN_ELECTRICO", "CIAN_PROFUNDO", "VIOLETA_NEON", "MAGENTA_CUANTICO",
            "ESMERALDA_VIVO", "VERDE_SINAPTICO", "AMBAR_SOLAR", "DORADO_BLOCK",
            "ROJO_ALERTA", "GRIS_METALLIC", "GRIS_OSCURO", "BLANCO_LUMINOSO",
            "BG_CARD", "BG_CODE", "BG_HIGHLIGHT",
        ]
        for attr in atributos:
            assert hasattr(ColoresLucIA, attr), f"Falta el atributo {attr} en ColoresLucIA"
            val = getattr(ColoresLucIA, attr)
            assert isinstance(val, str), f"{attr} debe ser str"
            assert val.startswith("\033["), f"{attr} no inicia con escape ANSI: {repr(val)}"

    def test_reset_secuencia_valida(self) -> None:
        """Verifica que el código RESET sea el estándar ANSI para restaurar estilo."""
        assert ColoresLucIA.RESET == "\033[0m"

    def test_fondos_secuencias_validas(self) -> None:
        """Comprueba las secuencias ANSI para colores de fondo."""
        assert ColoresLucIA.BG_CARD.startswith("\033[48;")
        assert ColoresLucIA.BG_CODE.startswith("\033[48;")
        assert ColoresLucIA.BG_HIGHLIGHT.startswith("\033[48;")


# ============================================================================
# 2. GLIFOS Y CARACTERES ESTRUCTURALES UNICODE
# ============================================================================

