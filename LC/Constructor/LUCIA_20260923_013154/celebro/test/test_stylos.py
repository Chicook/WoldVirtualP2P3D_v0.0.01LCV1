"""
test_stylos.py — Suite exhaustiva de pruebas para el subsistema STYLOS de LucIA
=============================================================================
Cubre el motor de estilos visuales y renderizado de terminal (Arquitectura 2026):
  1. Constantes de color ANSI y paleta cyber-bioluminiscente (ColoresLucIA).
  2. Glifos y caracteres estructurales Unicode (Glifos).
  3. Utilidades de cálculo y saneamiento (limpiar_codigos_ansi, longitud_visual, obtener_ancho_consola).
  4. Renderizador Markdown: sintaxis inline (bold, italic, code), encabezados y tablas estructuradas.
  5. Tarjetas y componentes de UI en terminal (EstiloTerminalLucIA):
     - Encabezados de sesión y bienvenida (banner_bienvenida).
     - Tarjetas de prompt de usuario.
     - Streaming token a token y apertura/cierre de respuesta.
     - Renderizado de respuesta completa y paneles de ayuda.
     - Badges de telemetría sináptica/blockchain y notificaciones de bloque PoNL.
  6. Casos borde de formato (tablas asimétricas, markdown anidado, textos multilínea).
  7. Métodos de conveniencia y atajos globales del subsistema.
"""
from __future__ import annotations

import io
import re
import sys
from typing import Any, Dict, List
import pytest

from LC.celebro.CMFG.SBSTM.STYLOS import (
    ColoresLucIA,
    EstiloTerminalLucIA,
    Glifos,
    RenderizadorMarkdown,
    badge_turno,
    banner_bienvenida,
    formatear_respuesta_lucia,
    limpiar_codigos_ansi,
    longitud_visual,
    obtener_ancho_consola,
    panel_ayuda_comandos,
    tarjeta_diagnostico,
)


# ============================================================================
# 1. CONSTANTES DE COLOR Y PALETA CYBER-BIOLUMINISCENTE
# ============================================================================

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

class TestEstiloTerminalLucIA:
    """Verifica que las funciones de despliegue en terminal impriman sin errores."""

    def test_renderizar_encabezado_sesion_captura(self, capsys: pytest.CaptureFixture) -> None:
        """Comprueba que el encabezado principal se imprima conteniendo ID y modelo."""
        EstiloTerminalLucIA.renderizar_encabezado_sesion(
            sesion_id="SES_TEST_001", modelo_activo="openrouter/auto:free", total_neuronas=50
        )
        captured = capsys.readouterr()
        assert "LUCIA COGNITIVE CONSOLE" in captured.out
        assert "SES_TEST_001" in captured.out
        assert "openrouter/auto:free" in captured.out
        assert "50 Activas" in captured.out

    def test_renderizar_prompt_usuario_captura(self, capsys: pytest.CaptureFixture) -> None:
        """Comprueba el encuadre del prompt del usuario."""
        EstiloTerminalLucIA.renderizar_prompt_usuario("Hola LucIA, ¿cómo estás?")
        captured = capsys.readouterr()
        assert "Tu >" in captured.out
        assert "Hola LucIA, ¿cómo estás?" in captured.out

    def test_ciclo_streaming_tarjeta_respuesta(self, capsys: pytest.CaptureFixture) -> None:
        """Verifica apertura, emisión de tokens y cierre de tarjeta de streaming."""
        EstiloTerminalLucIA.iniciar_tarjeta_respuesta(modelo_id="test_model:free")
        EstiloTerminalLucIA.imprimir_token_stream("Generando ")
        EstiloTerminalLucIA.imprimir_token_stream("respuesta ")
        EstiloTerminalLucIA.imprimir_token_stream("cognitiva.")
        EstiloTerminalLucIA.cerrar_tarjeta_respuesta()

        captured = capsys.readouterr()
        limpio = limpiar_codigos_ansi(captured.out)
        assert "LucIA" in limpio
        assert "test_model:free" in limpio
        assert "Generando respuesta cognitiva." in limpio

    def test_renderizar_respuesta_completa_con_markdown(self, capsys: pytest.CaptureFixture) -> None:
        """Comprueba el renderizado íntegro de respuestas complejas con listas y negritas."""
        cuerpo = (
            "Explicación del proceso:\n"
            "- **Fase 1**: Entrada sensorial.\n"
            "- **Fase 2**: Transmisión sináptica.\n"
            "```python\nx = 42\n```"
        )
        EstiloTerminalLucIA.renderizar_respuesta_completa(cuerpo, modelo_id="free/test")
        captured = capsys.readouterr()
        assert "LucIA" in captured.out
        assert "Fase 1" in captured.out
        assert "Fase 2" in captured.out

    def test_renderizar_respuesta_con_tabla_embebida(self, capsys: pytest.CaptureFixture) -> None:
        """Comprueba que una tabla Markdown embebida en texto se procese correctamente."""
        texto = (
            "A continuación los resultados:\n"
            "| Test | Estado |\n"
            "|---|---|\n"
            "| IPFS | OK |\n"
            "| Blockchain | OK |\n"
            "Fin del reporte."
        )
        EstiloTerminalLucIA.renderizar_respuesta_completa(texto, modelo_id="free/test")
        captured = capsys.readouterr()
        assert "IPFS" in captured.out
        assert "Blockchain" in captured.out
        assert "Fin del reporte" in captured.out

    def test_renderizar_barra_telemetria_captura(self, capsys: pytest.CaptureFixture) -> None:
        """Comprueba el formato de la barra de telemetría de turno."""
        EstiloTerminalLucIA.renderizar_barra_telemetria(
            turno=5,
            duracion_ms=350.2,
            modelo_id="gemini-2.5:free",
            deriva=0.0123,
            neuronas=50,
            cid_ipfs="bafkreitestcid2026",
            txs_espera=2,
        )
        captured = capsys.readouterr()
        assert "Turno #5" in captured.out
        assert "350ms" in captured.out
        assert "gemini-2.5:free" in captured.out
        assert "0.0123" in captured.out
        assert "Sinapsis: 50" in captured.out
        assert "$0.00 USD" in captured.out

    def test_renderizar_notificacion_bloque_captura(self, capsys: pytest.CaptureFixture) -> None:
        """Verifica la notificación de minado de bloque PoNL en terminal."""
        EstiloTerminalLucIA.renderizar_notificacion_bloque(
            indice=12, hash_bloque="abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
        )
        captured = capsys.readouterr()
        assert "BLOQUE #12 MINADO PoNL" in captured.out
        assert "abcdef0123456789" in captured.out

    def test_renderizar_panel_ayuda_captura(self, capsys: pytest.CaptureFixture) -> None:
        """Comprueba que el panel de ayuda liste los comandos clave."""
        EstiloTerminalLucIA.renderizar_panel_ayuda()
        captured = capsys.readouterr()
        assert "COMANDOS DISPONIBLES EN CONSOLA" in captured.out
        assert "estado / status" in captured.out
        assert "modelos / free" in captured.out
        assert "minar / mine" in captured.out
        assert "salir / exit" in captured.out

    def test_renderizar_tarjeta_sistema_captura(self, capsys: pytest.CaptureFixture) -> None:
        """Verifica la tarjeta de diagnóstico con pares clave-valor."""
        datos = {
            "Memoria RAM": "16 GB",
            "Blockchain": "Activa (3 bloques)",
            "IPFS": "CIDv1 Autónomo",
        }
        EstiloTerminalLucIA.renderizar_tarjeta_sistema("DIAGNÓSTICO SISTEMA", datos)
        captured = capsys.readouterr()
        assert "DIAGNÓSTICO SISTEMA" in captured.out
        assert "Memoria RAM" in captured.out
        assert "16 GB" in captured.out
        assert "Blockchain" in captured.out


# ============================================================================
# 6. MÉTODOS DE CONVENIENCIA Y ACCESOS DIRECTOS GLOBALES
# ============================================================================

class TestMetodosGlobalesConveniencia:
    """Pruebas sobre las funciones exportadas directamente en el módulo."""

    def test_formatear_respuesta_lucia_shortcut(self, capsys: pytest.CaptureFixture) -> None:
        """Comprueba el atajo formatear_respuesta_lucia()."""
        formatear_respuesta_lucia("Respuesta de prueba rápida", modelo="openrouter/free")
        captured = capsys.readouterr()
        assert "Respuesta de prueba rápida" in captured.out

    def test_banner_bienvenida_shortcut(self, capsys: pytest.CaptureFixture) -> None:
        """Comprueba el atajo banner_bienvenida()."""
        banner_bienvenida("SESION_RAPIDA", "modelo_demo")
        captured = capsys.readouterr()
        assert "SESION_RAPIDA" in captured.out

    def test_badge_turno_shortcut(self, capsys: pytest.CaptureFixture) -> None:
        """Comprueba el atajo badge_turno()."""
        badge_turno(
            turno=1, dt=100.0, mod="test:free", der=0.5, neu=50, cid="--", txs=0
        )
        captured = capsys.readouterr()
        assert "Turno #1" in captured.out

    def test_panel_ayuda_comandos_shortcut(self, capsys: pytest.CaptureFixture) -> None:
        """Comprueba el atajo panel_ayuda_comandos()."""
        panel_ayuda_comandos()
        captured = capsys.readouterr()
        assert "COMANDOS DISPONIBLES EN CONSOLA" in captured.out

    def test_tarjeta_diagnostico_shortcut(self, capsys: pytest.CaptureFixture) -> None:
        """Comprueba el atajo tarjeta_diagnostico()."""
        tarjeta_diagnostico("PRUEBA_TITULO", {"Parametro": "Valor123"})
        captured = capsys.readouterr()
        assert "PRUEBA_TITULO" in captured.out
        assert "Parametro" in captured.out
        assert "Valor123" in captured.out


# Fin de suite exhaustiva test_stylos.py
