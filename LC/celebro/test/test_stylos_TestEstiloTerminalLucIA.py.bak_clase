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

