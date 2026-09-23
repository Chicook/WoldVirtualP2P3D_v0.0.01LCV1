class TestIPFSManagerBatchPSNRL:
    """Verifica la subida masiva de pesos neuronales y la higiene del directorio local."""

    def test_subir_y_limpiar_psnrl_directorio_vacio(self, tmp_path: Path) -> None:
        """Comprueba reporte estructurado cuando no hay pesos pendientes en PSNRL."""
        mgr = IPFSManager(manifest_file=str(tmp_path / "manifest_batch_vacio.json"))
        mgr._psnrl_dir = tmp_path / "psnrl_vacio"
        mgr._psnrl_dir.mkdir()

        resumen = mgr.subir_y_limpiar_psnrl()
        assert resumen["archivos_procesados"] == 0
        assert resumen["cids"] == []
        assert resumen["borrados"] == []

    def test_subir_y_limpiar_psnrl_con_archivos_forzado(self, tmp_path: Path, mock_ipfs_inactivo: None) -> None:
        """Valida que en modo offline se procesen los archivos y se generen CIDs autónomos sin borrar local."""
        dir_psnrl = tmp_path / "psnrl_con_pesos"
        dir_psnrl.mkdir()
        f1 = dir_psnrl / "neurona_en1.npz"
        f2 = dir_psnrl / "neurona_sl1.npz"
        f1.write_bytes(b"npz_pesos_en1")
        f2.write_bytes(b"npz_pesos_sl1")

        mgr = IPFSManager(manifest_file=str(tmp_path / "manifest_batch.json"))
        mgr._psnrl_dir = dir_psnrl

        resumen = mgr.subir_y_limpiar_psnrl(forzar_borrado_sin_daemon=True)
        assert resumen["archivos_procesados"] == 2
        assert len(resumen["cids"]) == 2
        # Sin confirmación de pin remoto (subida_real=False), el borrado seguro no procede
        assert resumen["borrados"] == []
        assert f1.exists()
        assert f2.exists()


# ============================================================================
# 8. MÉTODOS AUXILIARES DE CONSULTA Y TELEMETRÍA
# ============================================================================

