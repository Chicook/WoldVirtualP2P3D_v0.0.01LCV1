class TestIPFSManagerConsultas:
    """Pruebas sobre listar_pesos, ultimo_cid y descargar_bytes."""

    def test_listar_pesos_orden_temporal(self, tmp_path: Path, mock_ipfs_inactivo: None) -> None:
        """Verifica que listar_pesos ordene los registros en orden temporal descendente."""
        mgr = IPFSManager(manifest_file=str(tmp_path / "manifest_consulta.json"))
        mgr.almacenar_pesos(b"payload_uno", nombre_modelo="modelo_alfa", eliminar_local=False)
        time.sleep(0.01)
        mgr.almacenar_pesos(b"payload_dos", nombre_modelo="modelo_beta", eliminar_local=False)

        lista = mgr.listar_pesos()
        assert len(lista) == 2
        assert lista[0]["nombre_modelo"] == "modelo_beta"
        assert lista[1]["nombre_modelo"] == "modelo_alfa"

    def test_ultimo_cid_con_filtro_patron(self, tmp_path: Path, mock_ipfs_inactivo: None) -> None:
        """Comprueba la búsqueda del último CID filtrando por nombre de modelo."""
        mgr = IPFSManager(manifest_file=str(tmp_path / "manifest_patron.json"))
        mgr.almacenar_pesos(b"p1", nombre_modelo="neurona_enrn_01", eliminar_local=False)
        mgr.almacenar_pesos(b"p2", nombre_modelo="neurona_slrn_01", eliminar_local=False)

        item = mgr.ultimo_cid(patron="slrn")
        assert item is not None
        assert "slrn" in item["nombre_modelo"]

    def test_descargar_bytes_exitoso(self, tmp_path: Path, mock_ipfs_inactivo: None) -> None:
        """Valida que descargar_bytes retorne la carga binaria correctamente."""
        mgr = IPFSManager(manifest_file=str(tmp_path / "manifest_descarga.json"))
        data_raw = b"datos_bytes_crudos_2026"
        res = mgr.almacenar_pesos(data_raw, nombre_modelo="datos_crudos", eliminar_local=False)

        descargado = mgr.descargar_bytes(res["cid"])
        assert descargado == data_raw


# ============================================================================
# 9. INSTANCIA GLOBAL Y ESTADO DEL SISTEMA
# ============================================================================

