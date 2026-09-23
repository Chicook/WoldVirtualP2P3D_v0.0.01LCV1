class TestIPFSManagerOffline:
    """Pruebas de almacenamiento autónomo sin conectividad a daemon Kubo."""

    def test_almacenar_bytes_autonomo(self, tmp_path: Path, mock_ipfs_inactivo: None) -> None:
        """Valida que almacene en bóveda en memoria (_vault) ante daemon inactivo."""
        ruta_manifiesto = tmp_path / "manifest_offline.json"
        mgr = IPFSManager(manifest_file=str(ruta_manifiesto))

        data = b"pesos_capa_densa_512x64"
        res = mgr.almacenar_pesos(data, nombre_modelo="capa_densa", eliminar_local=False)

        assert res["exito"] is True
        assert res["subida_real"] is False
        assert res["nodo"] == "AUTONOMO_CIDv1"
        assert res["cid"] in mgr._vault
        assert mgr._vault[res["cid"]] == data

    def test_almacenar_desde_archivo_local(self, tmp_path: Path, mock_ipfs_inactivo: None) -> None:
        """Comprueba almacenamiento leyendo un archivo de disco real."""
        ruta_manifiesto = tmp_path / "manifest_archivo.json"
        archivo_datos = tmp_path / "pesos_origen.bin"
        contenido = b"datos_crudos_pesos_vivos"
        archivo_datos.write_bytes(contenido)

        mgr = IPFSManager(manifest_file=str(ruta_manifiesto))
        res = mgr.almacenar_pesos(
            origen=archivo_datos, nombre_modelo="origen_test", eliminar_local=False
        )

        assert res["exito"] is True
        assert res["tamano_bytes"] == len(contenido)
        assert_hash_sha256_valido(res["registro"]["sha256"])
        assert archivo_datos.exists()

    def test_almacenar_diccionario_json(self, tmp_path: Path, mock_ipfs_inactivo: None) -> None:
        """Verifica serialización automática cuando el origen es un dict de Python."""
        ruta_manifiesto = tmp_path / "manifest_dict.json"
        mgr = IPFSManager(manifest_file=str(ruta_manifiesto))

        datos_dict = {"capas": 3, "activacion": "relu", "version": "2026"}
        res = mgr.almacenar_pesos(
            origen=datos_dict, nombre_modelo="arquitectura_red", eliminar_local=False
        )

        assert res["exito"] is True
        rec = mgr.recuperar_pesos(res["cid"])
        assert isinstance(rec, dict)
        assert rec.get("activacion") == "relu"

    def test_archivo_no_existente_lanza_error(self, tmp_path: Path) -> None:
        """Valida que una ruta inexistente cause FileNotFoundError."""
        mgr = IPFSManager(manifest_file=str(tmp_path / "manifest_err.json"))
        with pytest.raises(FileNotFoundError):
            mgr.almacenar_pesos(tmp_path / "no_existe_archivo.dat")


# ============================================================================
# 5. RECUPERACIÓN Y RECONSTRUCCIÓN DE DATOS
# ============================================================================

