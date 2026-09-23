class TestIPFSManagerKuboRPC:
    """Verifica integración con daemon IPFS Kubo mediante mocks de urllib."""

    def test_descubrir_daemon_activo(self, tmp_path: Path) -> None:
        """Valida que auto-discovery detecte un endpoint Kubo saludable."""
        mgr = IPFSManager(manifest_file=str(tmp_path / "manifest_kubo.json"))

        respuesta_id = json.dumps({"ID": "12D3KooWTestPeerIdKubo2026", "AgentVersion": "kubo/0.32.0"}).encode("utf-8")
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = respuesta_id
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            url = mgr.descubrir_daemon()
            assert url is not None
            assert "127.0.0.1" in url

    def test_subida_exitosa_daemon_kubo(self, tmp_path: Path) -> None:
        """Comprueba flujo de subida multipart con respuesta de pin OK."""
        mgr = IPFSManager(manifest_file=str(tmp_path / "manifest_upload.json"))

        cid_simulado = "bafkreifh32q22x2w6u6z5n4m3l2k1j0hgfedsba2026kuborpc"
        resp_add = json.dumps({"Name": "pesos.bin", "Hash": cid_simulado, "Size": "1024"}).encode("utf-8")
        resp_id = json.dumps({"ID": "PeerKuboTest"}).encode("utf-8")

        def fake_urlopen(req: Any, **kwargs: Any) -> MagicMock:
            m = MagicMock()
            m.status = 200
            m.__enter__.return_value = m
            url_str = req.get_full_url() if hasattr(req, "get_full_url") else str(req)
            if "api/v0/id" in url_str:
                m.read.return_value = resp_id
            else:
                m.read.return_value = resp_add
            return m

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            res = mgr.almacenar_pesos(
                b"pesos_subidos_kubo", nombre_modelo="modelo_kubo", eliminar_local=False
            )
            assert res["exito"] is True
            assert res["subida_real"] is True
            assert res["nodo"] == "KUBO_HTTP_RPC"
            assert res["cid"] == cid_simulado


# ============================================================================
# 7. PROCESAMIENTO EN LOTE (BATCH) Y LIMPIEZA DE PSNRL
# ============================================================================

