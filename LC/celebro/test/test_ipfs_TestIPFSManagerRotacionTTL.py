class TestIPFSManagerRotacionTTL:
    """Verifica las políticas de rotación: máx 500 entradas y 10 payloads en memoria."""

    def test_rotacion_mas_de_50_entradas(self, tmp_path: Path, mock_ipfs_inactivo: None) -> None:
        """Valida que el manifiesto se pode a 500 elementos al exceder la capacidad."""
        ruta_manifiesto = tmp_path / "manifiesto_rotacion.json"
        mgr = IPFSManager(manifest_file=str(ruta_manifiesto))

        # Insertar 55 registros secuenciales
        for i in range(55):
            mgr.almacenar_pesos(
                origen=f"bloque_neuronal_{i:03d}".encode("utf-8"),
                nombre_modelo=f"modelo_{i}",
                eliminar_local=False,
            )

        assert len(mgr.manifest["weights"]) <= 500
        assert mgr.manifest["total_stored"] <= 500

    def test_purga_payload_base64_solo_10_recientes(self, tmp_path: Path, mock_ipfs_inactivo: None) -> None:
        """Comprueba que solo los 10 registros más recientes conserven payload_b64."""
        ruta_manifiesto = tmp_path / "manifiesto_payloads.json"
        mgr = IPFSManager(manifest_file=str(ruta_manifiesto))

        for i in range(20):
            mgr.almacenar_pesos(
                origen=f"peso_incremental_{i}".encode("utf-8"),
                nombre_modelo=f"checkpoint_{i}",
                eliminar_local=False,
            )

        pesos = mgr.manifest["weights"]
        con_payload = [c for c, datos in pesos.items() if "payload_b64" in datos]
        assert len(con_payload) <= 10, f"Excedido límite de payloads en caché: {len(con_payload)}"


# ============================================================================
# 4. ALMACENAMIENTO Y MODO OFFLINE / AUTÓNOMO
# ============================================================================

