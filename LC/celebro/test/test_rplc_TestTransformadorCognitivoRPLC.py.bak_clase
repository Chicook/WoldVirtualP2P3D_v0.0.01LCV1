class TestTransformadorCognitivoRPLC:
    """Pruebas de adaptación Hebbiana, cálculo de valencia y deriva."""

    def test_transformar_vector_activa_neuronas(self) -> None:
        """Comprueba que el transformador aplique sigmoide y devuelva vector acotado."""
        transf = TransformadorCognitivoRPLC()
        v_in = [0.5] * NUM_NEURONAS
        v_out = transf.transformar(v_in)
        assert len(v_out) == NUM_NEURONAS
        assert all(0.0 <= x <= 1.0 for x in v_out)

    def test_valencia_sinaptica_rango(self) -> None:
        """Verifica que la valencia compare los hemisferios del vector [-1.0, 1.0]."""
        transf = TransformadorCognitivoRPLC()
        # Primeros 25 en 1.0, segundos 25 en 0.0 -> valencia positiva
        v_pos = [1.0] * 25 + [0.0] * 25
        val_pos = transf.valencia(v_pos)
        assert val_pos > 0.0

        # Primeros 25 en 0.0, segundos 25 en 1.0 -> valencia negativa
        v_neg = [0.0] * 25 + [1.0] * 25
        val_neg = transf.valencia(v_neg)
        assert val_neg < 0.0

        assert transf.valencia([]) == 0.0

    def test_neuronas_activas_umbral(self) -> None:
        """Comprueba que se identifiquen las neuronas por encima del umbral de activación."""
        transf = TransformadorCognitivoRPLC()
        v = [0.1] * NUM_NEURONAS
        v[5] = 0.85
        v[12] = 0.95
        activos = transf.activos(v)
        assert 5 in activos
        assert 12 in activos
        assert len(activos) == 2

    def test_deriva_media_evolucion(self) -> None:
        """Verifica que transformaciones sucesivas acumulen deriva en el historial."""
        transf = TransformadorCognitivoRPLC()
        v = [0.8] * NUM_NEURONAS
        transf.transformar(v)
        transf.transformar(v)
        deriva = transf.deriva_media()
        assert isinstance(deriva, float)
        assert deriva >= 0.0


# ============================================================================
# 4. CAPA 3: REFORMULADOR LINGÜÍSTICO (VOZ PROPIA LUCIA)
# ============================================================================

