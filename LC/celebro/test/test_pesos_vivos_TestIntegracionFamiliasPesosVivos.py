class TestIntegracionFamiliasPesosVivos:
    """Pruebas de inyección contemplando las 5 familias completas."""

    def test_inyeccion_cinco_familias_completas(self) -> None:
        """Verifica que el gestor procese correctamente las 5 familias simultáneamente."""
        gestor = GestorPesosVivos()
        mock_conv = MagicMock()
        mock_conv.neuronas = {}

        prefijos = ["EN", "RFSL", "RFEN", "RNP", "SLRN"]
        for pref in prefijos:
            for i in range(1, 3):
                m = MagicMock()
                m.pesos = np.random.randn(4, 4).astype(np.float32)
                mock_conv.neuronas[f"{pref}_{i}"] = m

        info = {
            "estado_emocional": 0.2,
            "tono_cognitivo": "técnico",
            "norma_delta_aplicada": 0.03,
        }

        res = gestor.inyectar_turno_en_vivo(
            conversor=mock_conv,
            pregunta="Consulta técnica profunda",
            respuesta="Respuesta del cluster",
            info_pesos=info,
            mostrar_en_terminal=False,
        )

        assert res["turno"] == 1
        familias = res["stats_familias"]
        for f in ["ENRN", "RF_SL", "RF_EN", "RNP", "SLRN"]:
            assert f in familias

    def test_registro_en_sesion_externa(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verifica que si se pasa un objeto sesion, se invoque registrar_archivo_pesos."""
        monkeypatch.setattr("LC.celebro.CMFG.pesos_vivos.PSNRL_DIR", tmp_path)
        gestor = GestorPesosVivos()
        mock_conv = MagicMock()
        mock_conv.neuronas = {}
        delattr(mock_conv, "persistir_pesos_en_psnrl")

        mock_sesion = MagicMock()
        archs = gestor.checkpoint_y_registrar(conversor=mock_conv, sesion=mock_sesion)
        assert mock_sesion.registrar_archivo_pesos.call_count == len(archs)


# ============================================================================
# 6. CONTRATO DE INTERFAZ GLOBAL Y COMPATIBILIDAD HISTÓRICA
# ============================================================================

