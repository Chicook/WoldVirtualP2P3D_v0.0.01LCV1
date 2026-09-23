class TestGlifos:
    """Verifica los caracteres tipográficos y bordes de cajas Unicode."""

    def test_glifos_existencia_y_unicidad(self) -> None:
        """Comprueba que los glifos requeridos estén definidos y sean caracteres válidos."""
        glifos_esperados = [
            "ESQ_ARR_IZQ", "ESQ_ARR_DER", "ESQ_ABA_IZQ", "ESQ_ABA_DER",
            "LINEA_H", "LINEA_V", "CRUZ_IZQ", "CRUZ_DER",
            "FLECHA", "PUNTO_VIVO", "SPARKLE",
            "BLOQUE_LLENO", "BLOQUE_MEDIO", "BLOQUE_VACIO",
            "RAYO", "CEREBRO",
        ]
        for g in glifos_esperados:
            assert hasattr(Glifos, g), f"Falta el glifo {g} en la clase Glifos"
            val = getattr(Glifos, g)
            assert isinstance(val, str)
            assert len(val) >= 1

    def test_glifos_esquinas_bordes(self) -> None:
        """Verifica que los glifos de esquinas correspondan a cajas redondeadas."""
        assert Glifos.ESQ_ARR_IZQ == "╭"
        assert Glifos.ESQ_ARR_DER == "╮"
        assert Glifos.ESQ_ABA_IZQ == "╰"
        assert Glifos.ESQ_ABA_DER == "╯"

    def test_glifos_bloques_densidad(self) -> None:
        """Verifica los caracteres de bloque para barras de progreso."""
        assert Glifos.BLOQUE_LLENO == "█"
        assert Glifos.BLOQUE_MEDIO == "▒"
        assert Glifos.BLOQUE_VACIO == "░"


# ============================================================================
# 3. UTILIDADES DE CÁLCULO Y SANEAMIENTO VISUAL
# ============================================================================

