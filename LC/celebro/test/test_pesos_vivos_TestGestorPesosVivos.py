class TestGestorPesosVivos:
    """Pruebas del ciclo de inyección y checkpoints de GestorPesosVivos."""

    def test_inicializacion_gestor(self) -> None:
        """Comprueba estado inicial de un nuevo gestor de sesión."""
        gestor = GestorPesosVivos(session_id="SESION_TEST_2026")
        assert gestor.session_id == "SESION_TEST_2026"
        assert gestor.turno_contador == 0
        assert gestor.deriva_sesion == 0.0
        assert gestor.historial_turnos == []

    def test_inyectar_turno_con_conversor_simulado(self) -> None:
        """Verifica la inyección de un turno con mock de conversor."""
        gestor = GestorPesosVivos()

        mock_conv = MagicMock()
        mock_n_enrn = MagicMock()
        mock_n_enrn.pesos = np.random.randn(8, 4).astype(np.float32)
        mock_conv.neuronas = {
            "EN1_RN": mock_n_enrn,
        }

        info_pesos = {
            "estado_emocional": 0.5,
            "tono_cognitivo": "reflexivo",
            "norma_delta_aplicada": 0.05,
        }

        res = gestor.inyectar_turno_en_vivo(
            conversor=mock_conv,
            pregunta="¿Cómo estás?",
            respuesta="Lista para operar.",
            info_pesos=info_pesos,
            mostrar_en_terminal=False,
        )

        assert res["turno"] == 1
        assert res["delta_aplicada"] == 0.05
        assert gestor.deriva_sesion == 0.05
        assert "ENRN" in res["stats_familias"]
        assert len(gestor.historial_turnos) == 1

    def test_checkpoint_y_registrar_genera_npz(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verifica la creación de archivos comprimidos .npz en el directorio PSNRL."""
        monkeypatch.setattr("LC.celebro.CMFG.pesos_vivos.PSNRL_DIR", tmp_path)

        gestor = GestorPesosVivos()
        mock_conv = MagicMock()
        mock_n = MagicMock()
        mock_n.pesos = np.ones((4, 4), dtype=np.float32)
        mock_conv.neuronas = {"EN1_RN": mock_n}
        delattr(mock_conv, "persistir_pesos_en_psnrl")

        archivos = gestor.checkpoint_y_registrar(
            conversor=mock_conv, sesion=None, etiqueta="test_chk"
        )
        assert len(archivos) >= 1
        npz_generado = archivos[0]
        assert npz_generado.exists()
        assert npz_generado.name.endswith(".npz")

        datos_npz = np.load(npz_generado)
        assert "EN1_RN_pesos" in datos_npz
        assert datos_npz["EN1_RN_pesos"].shape == (4, 4)

    def test_obtener_telemetria_resumen(self) -> None:
        """Comprueba el reporte estadístico de la sesión."""
        gestor = GestorPesosVivos(session_id="SES_TELEMETRIA")
        gestor.turno_contador = 4
        gestor.deriva_sesion = 0.123
        gestor.historial_gsnr = [1.5, 1.8, 2.0]

        resumen = gestor.obtener_telemetria_resumen()
        assert resumen["session_id"] == "SES_TELEMETRIA"
        assert resumen["total_turnos"] == 4
        assert resumen["deriva_total"] == 0.123
        assert 1.7 <= resumen["gsnr_medio"] <= 1.8


# ============================================================================
# 5. INTEGRACIÓN CON MULTI-FAMILIAS Y CONTRATOS
# ============================================================================

