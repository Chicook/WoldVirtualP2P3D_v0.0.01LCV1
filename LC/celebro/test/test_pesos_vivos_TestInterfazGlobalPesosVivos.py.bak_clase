class TestInterfazGlobalPesosVivos:
    """Verifica las funciones de módulo expuestas hacia celebro."""

    def test_singleton_gestor_pesos_vivos(self) -> None:
        """Comprueba que get_gestor_pesos_vivos retorne una única instancia singleton."""
        g1 = get_gestor_pesos_vivos()
        g2 = get_gestor_pesos_vivos()
        assert g1 is g2

    def test_inyectar_pesos_turno_wrapper(self) -> None:
        """Verifica la invocación de la función compatible inyectar_pesos_turno."""
        mock_conv = MagicMock()
        mock_conv.neuronas = {}
        info = {"norma_delta_aplicada": 0.01}

        turno_num = inyectar_pesos_turno(
            cerebro=None, conversor=mock_conv, pregunta="Pregunta", respuesta="Resp",
            info_pesos=info, mostrar_en_terminal=False
        )
        assert isinstance(turno_num, int)
        assert turno_num >= 1

    def test_checkpoint_y_registrar_wrapper(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Comprueba la invocación de la función compatible checkpoint_y_registrar."""
        monkeypatch.setattr("LC.celebro.CMFG.pesos_vivos.PSNRL_DIR", tmp_path)
        mock_conv = MagicMock()
        mock_conv.neuronas = {}
        delattr(mock_conv, "persistir_pesos_en_psnrl")

        archs = checkpoint_y_registrar(mock_conv, etiqueta="chk_global")
        assert isinstance(archs, list)
        assert len(archs) >= 1
        assert archs[0].exists()


# ============================================================================
# 7. CASOS LÍMITE Y ROBUSTEZ ANTE VALORES ATÍPICOS
# ============================================================================

