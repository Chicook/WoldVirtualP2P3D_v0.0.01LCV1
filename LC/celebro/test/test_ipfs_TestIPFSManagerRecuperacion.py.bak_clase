class TestIPFSManagerRecuperacion:
    """Pruebas de lectura y restauración de pesos neuronales."""

    def test_recuperar_desde_boveda(self, tmp_path: Path, mock_ipfs_inactivo: None) -> None:
        """Comprueba recuperación exacta de bytes desde el vault local."""
        mgr = IPFSManager(manifest_file=str(tmp_path / "manifest_rec.json"))
        payload_original = b"gradientes_acumulados_adamw_step_100"
        res = mgr.almacenar_pesos(payload_original, nombre_modelo="gradientes", eliminar_local=False)

        recuperado = mgr.recuperar_pesos(res["cid"])
        assert recuperado == payload_original

    def test_recuperar_hacia_archivo_destino(self, tmp_path: Path, mock_ipfs_inactivo: None) -> None:
        """Verifica la restauración física en una ruta especificada."""
        mgr = IPFSManager(manifest_file=str(tmp_path / "manifest_dest.json"))
        data = b"tensor_pesos_exportado"
        res = mgr.almacenar_pesos(data, nombre_modelo="tensor_exp", eliminar_local=False)

        destino_restauracion = tmp_path / "restaurado" / "pesos.bin"
        mgr.recuperar_pesos(res["cid"], destino=destino_restauracion)

        assert destino_restauracion.exists()
        assert destino_restauracion.read_bytes() == data

    def test_recuperar_cid_inexistente_lanza_keyerror(self, tmp_path: Path, mock_ipfs_inactivo: None) -> None:
        """Comprueba que solicitar un CID desconocido cause KeyError."""
        mgr = IPFSManager(manifest_file=str(tmp_path / "manifest_vacio.json"))
        with pytest.raises(KeyError):
            mgr.recuperar_pesos("bafy_no_existe_en_el_sistema_2026")


# ============================================================================
# 6. SIMULACIÓN DE KUBO HTTP RPC API (MOCKS DE RED)
# ============================================================================

