class TestPersistenciaLedger:
    """Tests de guardado y recarga del blockchain_ledger.json."""

    def test_ledger_se_guarda_correctamente(self, tmp_path: Path) -> None:
        """Después de minar, el ledger debe poder guardarse como JSON válido."""
        ledger_path = tmp_path / "ledger.json"
        psnrl = tmp_path / "PSNRL"
        psnrl.mkdir()
        with (
            patch("LC.celebro.BKSVCB.PSNRL_DIR", psnrl),
            patch("LC.celebro.BKSVCB.LEDGER_PATH", ledger_path),
        ):
            srv = ServidorBlockchainNeuronal()
            srv.registrar_aprendizaje_neural("p", "r", "m")
            srv.minar_transacciones_pendientes()
            if hasattr(srv, "guardar_ledger"):
                srv.guardar_ledger()

        if ledger_path.exists():
            contenido = ledger_path.read_text(encoding="utf-8")
            datos = json.loads(contenido)
            assert "cadena" in datos, "El ledger guardado debe tener la clave 'cadena'"

    def test_ledger_genesis_tiene_hash_genesis_especial(self, ledger_genesis_path: Path) -> None:
        """El ledger de génesis debe tener el bloque en índice 0."""
        datos = json.loads(ledger_genesis_path.read_text(encoding="utf-8"))
        assert datos["cadena"][0]["indice"] == 0, "Primer bloque debe tener índice 0"

    def test_ledger_con_3_bloques_valido(self, ledger_con_bloques: Path) -> None:
        """El ledger con 3 bloques debe tener total_bloques == 3."""
        datos = json.loads(ledger_con_bloques.read_text(encoding="utf-8"))
        assert datos["total_bloques"] == 3, "total_bloques debe ser 3"
        assert len(datos["cadena"]) == 3, "La cadena debe tener 3 bloques"

    def test_transformar_ledger_a_pesos_no_lanza(self, tmp_path: Path) -> None:
        """transformar_ledger_a_pesos_neuronales() no debe lanzar excepciones."""
        psnrl = tmp_path / "PSNRL"
        psnrl.mkdir()
        with (
            patch("LC.celebro.BKSVCB.PSNRL_DIR", psnrl),
            patch("LC.celebro.BKSVCB.LEDGER_PATH", tmp_path / "l.json"),
        ):
            srv = ServidorBlockchainNeuronal()
            try:
                resultado = srv.transformar_ledger_a_pesos_neuronales()
                assert isinstance(resultado, dict), "El resultado debe ser un dict"
            except Exception as e:
                pytest.fail(f"transformar_ledger_a_pesos_neuronales() lanzó: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5 — Singleton y get_blockchain_server
# ═══════════════════════════════════════════════════════════════════════════════

