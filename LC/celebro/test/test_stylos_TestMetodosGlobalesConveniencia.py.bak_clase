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
