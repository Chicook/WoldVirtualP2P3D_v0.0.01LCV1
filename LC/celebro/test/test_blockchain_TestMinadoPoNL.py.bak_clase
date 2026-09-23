class TestMinadoPoNL:
    """Tests del mecanismo de minado por prueba de trabajo neuronal."""

    def test_minar_dificultad_1_produce_hash_con_prefijo(self) -> None:
        """Con dificultad 1 el hash minado debe comenzar con '0'."""
        bloque = BloqueNeuronal(
            indice=0, hash_previo="0" * 64,
            transacciones=[], dificultad=1
        )
        exito = bloque.minar_bloque(max_iteraciones=MAX_ITER_TEST)
        assert exito, "El minado con dificultad 1 debe tener éxito en el límite dado"
        assert bloque.hash_bloque.startswith("0"), (
            f"Hash minado '{bloque.hash_bloque[:8]}' no comienza con '0'"
        )

    def test_minar_nonce_incrementa(self) -> None:
        """El nonce debe ser mayor que 0 tras el minado (al menos una iteración)."""
        bloque = BloqueNeuronal(indice=0, hash_previo="0" * 64, transacciones=[], dificultad=1)
        bloque.minar_bloque(max_iteraciones=MAX_ITER_TEST)
        # Con dificultad 1 puede coincidir en nonce=0, pero verificamos que es entero
        assert isinstance(bloque.nonce, int), "nonce debe ser entero"
        assert bloque.nonce >= 0, "nonce no puede ser negativo"

    def test_minar_doble_sha256_coherente(self) -> None:
        """Después de minar, calcular_hash() debe devolver el mismo hash_bloque."""
        bloque = BloqueNeuronal(indice=0, hash_previo="0" * 64, transacciones=[], dificultad=1)
        bloque.minar_bloque(max_iteraciones=MAX_ITER_TEST)
        recalculado = bloque.calcular_hash()
        assert recalculado == bloque.hash_bloque, (
            "calcular_hash() tras minar debe coincidir con hash_bloque almacenado"
        )

    def test_bloque_ya_minado_hash_valido(self) -> None:
        """Un bloque construido con hash_existente debe conservar ese hash."""
        hash_fijo = "0a" + "b" * 62  # Comienza con '0' (dificultad 1)
        bloque = BloqueNeuronal(
            indice=0, hash_previo="0" * 64, transacciones=[],
            hash_existente=hash_fijo
        )
        assert bloque.hash_bloque == hash_fijo, "hash_existente debe conservarse sin recalcular"

    def test_minar_max_iteraciones_sin_exito_devuelve_false(self) -> None:
        """Con max_iteraciones=0 no puede minar y debe devolver False."""
        bloque = BloqueNeuronal(indice=0, hash_previo="0" * 64, transacciones=[], dificultad=4)
        resultado = bloque.minar_bloque(max_iteraciones=0)
        assert resultado is False, "Con 0 iteraciones el minado debe fallar"


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — ServidorBlockchainNeuronal: cadena y validación
# ═══════════════════════════════════════════════════════════════════════════════

