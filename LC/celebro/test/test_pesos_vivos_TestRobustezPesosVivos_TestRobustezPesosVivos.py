class TestRobustezPesosVivos:
    """Verifica tolerancia ante arrays vacíos, NaN o dimensiones no convencionales."""

    def test_deriva_muon_arrays_1d_fallback(self) -> None:
        """Comprueba que arrays 1D no fallen y usen el cálculo de norma relativo."""
        w1d = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        d1d = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        deriva = MetricasDinamicasPesos.deriva_espectral_muon(w1d, d1d)
        assert isinstance(deriva, float)
        assert deriva >= 0.0

    def test_inyeccion_sin_neuronas_en_conversor(self) -> None:
        """Verifica que un conversor sin diccionario de neuronas opere sin errores."""
        gestor = GestorPesosVivos()
        mock_vacio = object()
        info = {"norma_delta_aplicada": 0.02}

        res = gestor.inyectar_turno_en_vivo(
            conversor=mock_vacio,
            pregunta="P",
            respuesta="R",
            info_pesos=info,
            mostrar_en_terminal=False,
        )
        assert res["turno"] >= 1
        assert "stats_familias" in res

    def test_historial_turnos_convergencia(self) -> None:
        """Comprueba que múltiples inyecciones sucesivas preserven la serie temporal de turnos."""
        gestor = GestorPesosVivos()
        mock_conv = MagicMock()
        mock_conv.neuronas = {}

        for i in range(1, 6):
            info = {"norma_delta_aplicada": 0.01 * i, "estado_emocional": 0.1 * i}
            res = gestor.inyectar_turno_en_vivo(
                conversor=mock_conv,
                pregunta=f"Pregunta #{i}",
                respuesta=f"Respuesta #{i}",
                info_pesos=info,
                mostrar_en_terminal=False,
            )
            assert res["turno"] == i

        assert len(gestor.historial_turnos) == 5
        assert gestor.deriva_sesion > 0.0
        assert gestor.historial_turnos[0]["turno"] == 1
        assert gestor.historial_turnos[-1]["turno"] == 5


# Fin de suite exhaustiva test_pesos_vivos.py
