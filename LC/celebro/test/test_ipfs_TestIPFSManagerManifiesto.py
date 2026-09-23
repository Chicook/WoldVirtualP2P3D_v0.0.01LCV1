class TestIPFSManagerManifiesto:
    """Pruebas sobre el ciclo de vida del manifiesto local ipfs_manifest.json."""

    def test_inicializacion_con_manifiesto_nuevo(self, tmp_path: Path) -> None:
        """Valida estructura inicial cuando el archivo de manifiesto no existe."""
        ruta_manifiesto = tmp_path / "nuevo_manifest.json"
        mgr = IPFSManager(manifest_file=str(ruta_manifiesto))
        assert mgr.manifest_path == ruta_manifiesto
        assert mgr.manifest.get("version") == "2.0"
        assert mgr.manifest.get("weights") == {}
        assert mgr.manifest.get("total_stored") == 0

    def test_persistencia_manifiesto_en_disco(self, tmp_path: Path, mock_ipfs_inactivo: None) -> None:
        """Verifica que almacenar pesos guarde el manifiesto en formato JSON válido."""
        ruta_manifiesto = tmp_path / "manifiesto_disco.json"
        mgr = IPFSManager(manifest_file=str(ruta_manifiesto))

        payload = b"matriz_sinaptica_2026"
        res = mgr.almacenar_pesos(payload, nombre_modelo="test_synapse", eliminar_local=False)
        assert res["exito"] is True
        assert ruta_manifiesto.exists()

        with open(ruta_manifiesto, "r", encoding="utf-8") as f:
            contenido = json.load(f)
        assert res["cid"] in contenido["weights"]
        assert contenido["total_stored"] >= 1

    def test_recuperacion_manifiesto_corrupto(self, tmp_path: Path) -> None:
        """Verifica tolerancia y regeneración ante archivo JSON corrupto."""
        ruta_manifiesto = tmp_path / "corrupto.json"
        ruta_manifiesto.write_text("{json_invalido: 123", encoding="utf-8")

        mgr = IPFSManager(manifest_file=str(ruta_manifiesto))
        assert mgr.manifest["version"] == "2.0"
        assert mgr.manifest["weights"] == {}


# ============================================================================
# 3. ROTACIÓN CON TTL Y GESTIÓN DE MEMORIA EN MANIFIESTO
# ============================================================================

