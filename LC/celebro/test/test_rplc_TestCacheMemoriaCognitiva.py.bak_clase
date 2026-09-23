class TestCacheMemoriaCognitiva:
    """Verifica el buffer de corto plazo y el control de redundancia lingüística."""

    def test_registro_en_cache_y_tamano_acotado(self) -> None:
        """Comprueba que la caché respete la capacidad máxima configurada."""
        cache = CacheMemoriaCognitiva(capacidad=10)
        for i in range(15):
            cache.registrar(f"Respuesta #{i}", valencia=0.1 * i, tono="positivo")

        assert cache.total_registros() == 10
        assert len(cache._entradas) == 10

    def test_deteccion_repeticion_por_hash(self) -> None:
        """Verifica que cadenas idénticas sean detectadas como repetitivas."""
        cache = CacheMemoriaCognitiva(capacidad=10)
        texto = "Este es un enunciado de prueba exacto."
        cache.registrar(texto, valencia=0.5, tono="tecnico")
        assert cache.es_repetitivo(texto) is True
        assert cache.es_repetitivo("Texto completamente diferente") is False

    def test_limpiar_cache(self) -> None:
        """Comprueba el vaciado total de la memoria de corto plazo."""
        cache = CacheMemoriaCognitiva(capacidad=5)
        cache.registrar("Prueba", 0.0, "neutro")
        assert cache.total_registros() == 1
        cache.vaciar()
        assert cache.total_registros() == 0


# ============================================================================
# 6. PIPELINE INTEGRAL: PROCESADOR RPLC
# ============================================================================

