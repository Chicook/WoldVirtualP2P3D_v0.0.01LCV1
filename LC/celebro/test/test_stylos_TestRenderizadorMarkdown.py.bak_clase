class TestRenderizadorMarkdown:
    """Pruebas de formateo de Markdown para consola de terminal."""

    def test_colorear_negrita(self) -> None:
        """Verifica que **texto** se formatee con estilo negrita y blanco luminoso."""
        entrada = "Este es un **termino clave** en la red."
        salida = RenderizadorMarkdown.colorear_sintaxis(entrada)
        assert ColoresLucIA.BOLD in salida
        assert "termino clave" in salida
        assert ColoresLucIA.RESET in salida

    def test_colorear_codigo_inline(self) -> None:
        """Verifica que `codigo` se formatee con el tono ámbar solar."""
        entrada = "Ejecuta `npm run dev` para iniciar."
        salida = RenderizadorMarkdown.colorear_sintaxis(entrada)
        assert ColoresLucIA.AMBAR_SOLAR in salida
        assert "npm run dev" in salida

    def test_colorear_cursiva(self) -> None:
        """Verifica que *texto* se formatee en cursiva con cian eléctrico."""
        entrada = "Este es un *detalle suave* importante."
        salida = RenderizadorMarkdown.colorear_sintaxis(entrada)
        assert ColoresLucIA.ITALIC in salida
        assert "detalle suave" in salida

    def test_colorear_encabezados(self) -> None:
        """Verifica que #, ## y ### generen glifos y colores jerárquicos."""
        h1 = RenderizadorMarkdown.colorear_sintaxis("# Titulo Principal")
        assert "━ Titulo Principal ━" in h1
        assert ColoresLucIA.DORADO_BLOCK in h1

        h2 = RenderizadorMarkdown.colorear_sintaxis("## Subseccion")
        assert "✦ Subseccion" in h2
        assert ColoresLucIA.CIAN_ELECTRICO in h2

        h3 = RenderizadorMarkdown.colorear_sintaxis("### Detalle Fino")
        assert "◈ Detalle Fino" in h3
        assert ColoresLucIA.VIOLETA_NEON in h3

    def test_colorear_listas_vinetas(self) -> None:
        """Verifica que listas con - o * se formateen con glifos de viñeta."""
        item = RenderizadorMarkdown.colorear_sintaxis("- Elemento de lista")
        assert "•" in item
        assert "Elemento de lista" in item

    def test_colorear_listas_numeradas(self) -> None:
        """Verifica que listas numeradas destaquen el número con color."""
        num_item = RenderizadorMarkdown.colorear_sintaxis("1. Primer paso")
        assert ColoresLucIA.CIAN_ELECTRICO in num_item
        assert "1." in num_item
        assert "Primer paso" in num_item

    def test_renderizar_tabla_unicode(self) -> None:
        """Verifica la generación de una tabla con bordes y alineación correcta."""
        lineas_tabla = [
            "| Capa | Neuronas | Activacion |",
            "|---|---|---|",
            "| Entrada | 10 | ReLU |",
            "| Salida | 4 | Sigmoid |",
        ]
        tabla_str = RenderizadorMarkdown.renderizar_tabla(lineas_tabla, ancho_max=80)
        assert isinstance(tabla_str, str)
        assert Glifos.ESQ_ARR_IZQ in tabla_str
        assert Glifos.ESQ_ABA_DER in tabla_str
        assert "Entrada" in tabla_str
        assert "Sigmoid" in tabla_str

    def test_renderizar_tabla_asimetrica(self) -> None:
        """Comprueba que una tabla con filas de distinta longitud no falle."""
        lineas_tabla = [
            "| ColA | ColB | ColC |",
            "|---|---|---|",
            "| Valor1 | Valor2 |",
            "| V1 | V2 | V3 | Extra |",
        ]
        tabla_str = RenderizadorMarkdown.renderizar_tabla(lineas_tabla, ancho_max=80)
        assert isinstance(tabla_str, str)
        assert "ColA" in tabla_str
        assert "Valor1" in tabla_str

    def test_renderizar_tabla_vacia_retorna_vacio(self) -> None:
        """Comprueba que una tabla sin filas válidas devuelva string vacío."""
        res = RenderizadorMarkdown.renderizar_tabla([], ancho_max=80)
        assert res == ""


# ============================================================================
# 5. COMPONENTES Y TARJETAS DE UI EN TERMINAL
# ============================================================================

