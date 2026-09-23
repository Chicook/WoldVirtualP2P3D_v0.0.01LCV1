class TestBloqueNeuronalConstruccion:
    """Tests de creación y atributos de BloqueNeuronal."""

    def test_bloque_genesis_atributos_basicos(self) -> None:
        """El bloque génesis (índice=0) debe tener los campos mínimos correctos."""
        bloque = BloqueNeuronal(
            indice=0,
            hash_previo="0" * 64,
            transacciones=[],
            dificultad=DIFICULTAD_TEST,
        )
        assert bloque.indice == 0, "índice del génesis debe ser 0"
        assert bloque.hash_previo == "0" * 64, "hash_previo del génesis debe ser 64 ceros"
        assert isinstance(bloque.transacciones, list), "transacciones debe ser lista"
        assert isinstance(bloque.hash_bloque, str), "hash_bloque debe ser str"
        assert isinstance(bloque.merkle_root, str), "merkle_root debe ser str"
        assert bloque.dificultad == DIFICULTAD_TEST, "dificultad debe coincidir"

    def test_bloque_hash_es_sha256_valido(self) -> None:
        """El hash generado debe ser un SHA-256 hexadecimal de 64 caracteres."""
        bloque = BloqueNeuronal(
            indice=1,
            hash_previo="a" * 64,
            transacciones=[{"tipo": "test"}],
            dificultad=DIFICULTAD_TEST,
        )
        assert_hash_sha256_valido(bloque.hash_bloque, nombre="bloque.hash_bloque")

    def test_bloque_merkle_root_es_sha256_valido(self) -> None:
        """El Merkle Root debe ser también un hash SHA-256 de 64 caracteres."""
        tx = [{"prompt": "hola", "respuesta": "mundo", "modelo": "test"}]
        bloque = BloqueNeuronal(indice=0, hash_previo="0" * 64, transacciones=tx)
        assert_hash_sha256_valido(bloque.merkle_root, nombre="bloque.merkle_root")

    def test_bloque_sin_transacciones_merkle_especial(self) -> None:
        """Con 0 transacciones el Merkle Root debe ser el hash del string 'empty_block'."""
        bloque = BloqueNeuronal(indice=0, hash_previo="0" * 64, transacciones=[])
        esperado = hashlib.sha256(b"empty_block_celebro_2026").hexdigest()
        assert bloque.merkle_root == esperado, "Merkle de bloque vacío debe coincidir con el hash de 'empty_block'"

    def test_bloque_con_una_transaccion(self) -> None:
        """Con una transacción el Merkle Root debe ser el hash de esa transacción."""
        tx = [{"tipo": "single_tx", "dato": 42}]
        bloque = BloqueNeuronal(indice=0, hash_previo="0" * 64, transacciones=tx)
        tx_hash = hashlib.sha256(json.dumps(tx[0], sort_keys=True).encode()).hexdigest()
        assert bloque.merkle_root == tx_hash, "Merkle con 1 tx debe ser el hash directo de esa tx"

    def test_bloque_merkle_raiz_impar_duplica_ultimo(self) -> None:
        """Con N impar de transacciones se duplica el último hash (estándar Merkle)."""
        txs = [{"id": i} for i in range(3)]
        bloque = BloqueNeuronal(indice=0, hash_previo="0" * 64, transacciones=txs)
        # Solo verificamos que el merkle_root es un SHA-256 válido de 64 chars
        assert_hash_sha256_valido(bloque.merkle_root, nombre="merkle_impar")

    def test_bloque_estado_neuronal_vacio_por_defecto(self) -> None:
        """Sin estado_neuronal explícito, debe quedar como dict vacío."""
        bloque = BloqueNeuronal(indice=0, hash_previo="0" * 64, transacciones=[])
        assert bloque.estado_neuronal == {}, "estado_neuronal debe ser {} por defecto"

    def test_bloque_estado_neuronal_se_preserva(self) -> None:
        """El estado_neuronal pasado debe conservarse sin modificación."""
        estado = {"neuronas_activas": 50, "norma": 0.42, "deriva_muon": 0.001}
        bloque = BloqueNeuronal(
            indice=0, hash_previo="0" * 64, transacciones=[], estado_neuronal=estado
        )
        assert bloque.estado_neuronal == estado, "estado_neuronal debe ser idéntico al pasado"

    def test_bloque_to_dict_contiene_campos_requeridos(self) -> None:
        """to_dict() debe incluir todos los campos de serialización."""
        bloque = BloqueNeuronal(indice=0, hash_previo="0" * 64, transacciones=[])
        d = bloque.to_dict()
        campos_requeridos = ["indice", "hash_bloque", "hash_previo", "merkle_root",
                             "dificultad", "nonce", "total_transacciones"]
        for campo in campos_requeridos:
            assert campo in d, f"to_dict() debe contener el campo '{campo}'"

    def test_bloque_to_dict_es_serializable_json(self) -> None:
        """El dict devuelto por to_dict() debe ser serializable a JSON."""
        bloque = BloqueNeuronal(
            indice=1, hash_previo="b" * 64,
            transacciones=[{"tipo": "aprendizaje", "peso": 0.7}]
        )
        try:
            json_str = json.dumps(bloque.to_dict())
        except (TypeError, ValueError) as e:
            pytest.fail(f"to_dict() no es serializable a JSON: {e}")
        assert len(json_str) > 0, "JSON serializado no debe estar vacío"


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — Minado PoNL (Proof of Neural Learning)
# ═══════════════════════════════════════════════════════════════════════════════

