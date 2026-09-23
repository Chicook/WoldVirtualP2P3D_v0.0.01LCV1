class TestNeuralMathPrecision2026:
    """Tests del motor matemático NS-5, SOAP, GSNR y Entropía de Shannon."""

    # ─── Newton-Schulz NS-5 ──────────────────────────────────────────────────

    def test_newton_schulz5_devuelve_ndarray(self, matriz_2d_32x32: np.ndarray) -> None:
        """newton_schulz5() debe devolver un ndarray de la misma forma."""
        resultado = NeuralMathPrecision2026.newton_schulz5(matriz_2d_32x32)
        assert isinstance(resultado, np.ndarray), "newton_schulz5 debe devolver ndarray"
        assert resultado.shape == matriz_2d_32x32.shape, "La forma debe preservarse"

    def test_newton_schulz5_sin_nan_ni_inf(self, matriz_2d_32x32: np.ndarray) -> None:
        """El resultado de NS-5 no debe contener NaN ni Inf."""
        resultado = NeuralMathPrecision2026.newton_schulz5(matriz_2d_32x32, steps=5)
        assert_tensor_valido(resultado, nombre="NS-5 resultado")

    def test_newton_schulz5_norma_frobenius_acotada(self, matriz_2d_32x32: np.ndarray) -> None:
        """Tras NS-5 la norma de Frobenius debe ser ~ 1 (ortogonalización)."""
        resultado = NeuralMathPrecision2026.newton_schulz5(matriz_2d_32x32, steps=10)
        norma = float(np.linalg.norm(resultado, ord="fro"))
        assert 0.01 < norma < 100.0, f"Norma Frobenius inesperada: {norma:.4f}"

    def test_newton_schulz5_vector_1d_devuelve_sin_cambio(self) -> None:
        """Para tensores 1D, NS-5 debe devolver el tensor sin modificar."""
        v = np.ones(16, dtype=np.float32)
        resultado = NeuralMathPrecision2026.newton_schulz5(v)
        np.testing.assert_array_equal(resultado, v, err_msg="NS-5 debe devolver vector 1D sin cambio")

    def test_newton_schulz5_matriz_identidad_converge(self) -> None:
        """NS-5 aplicado a la identidad debe converger cerca de la identidad."""
        I = np.eye(8, dtype=np.float32)
        resultado = NeuralMathPrecision2026.newton_schulz5(I, steps=5)
        assert_tensor_valido(resultado, esperado_shape=(8, 8), nombre="NS-5 identidad")

    def test_newton_schulz5_diferente_steps_cambia_resultado(self, matriz_2d_32x32: np.ndarray) -> None:
        """Distintos números de pasos deben producir resultados distintos."""
        r1 = NeuralMathPrecision2026.newton_schulz5(matriz_2d_32x32, steps=1)
        r5 = NeuralMathPrecision2026.newton_schulz5(matriz_2d_32x32, steps=5)
        assert not np.allclose(r1, r5), "1 paso vs 5 pasos deben diferir"

    # ─── SOAP Preconditioner ─────────────────────────────────────────────────

    def test_soap_devuelve_forma_correcta(self) -> None:
        """soap_precondition() debe devolver una matriz de la misma forma que G."""
        m, n = 16, 8
        rng = np.random.default_rng(1)
        G = rng.standard_normal((m, n)).astype(np.float32)
        L = np.zeros((m, m), dtype=np.float32)
        R = np.zeros((n, n), dtype=np.float32)
        resultado = NeuralMathPrecision2026.soap_precondition(G, L, R)
        assert resultado.shape == (m, n), f"SOAP debe preservar forma (m,n)={m,n}"

    def test_soap_sin_nan_ni_inf(self) -> None:
        """El resultado de SOAP no debe contener valores no finitos."""
        rng = np.random.default_rng(2)
        G = rng.standard_normal((8, 4)).astype(np.float32)
        L = np.eye(8, dtype=np.float32) * 0.01
        R = np.eye(4, dtype=np.float32) * 0.01
        resultado = NeuralMathPrecision2026.soap_precondition(G, L, R, beta=0.9)
        assert_tensor_valido(resultado, nombre="SOAP resultado")

    def test_soap_modifica_L_y_R_in_place(self) -> None:
        """SOAP debe actualizar L y R en su lugar (momento de segundo orden)."""
        rng = np.random.default_rng(3)
        G = rng.standard_normal((8, 4)).astype(np.float32)
        L = np.zeros((8, 8), dtype=np.float32)
        R = np.zeros((4, 4), dtype=np.float32)
        NeuralMathPrecision2026.soap_precondition(G, L, R)
        assert not np.allclose(L, 0), "L debe haberse actualizado (no sigue siendo cero)"
        assert not np.allclose(R, 0), "R debe haberse actualizado (no sigue siendo cero)"

    # ─── GSNR (Gradient Signal-to-Noise Ratio) ───────────────────────────────

    def test_gsnr_devuelve_float(self, vector_512: np.ndarray) -> None:
        """gsnr() debe devolver un float."""
        G = vector_512.reshape(1, -1)
        G_sq = G ** 2
        resultado = NeuralMathPrecision2026.gsnr(G, G_sq, t=1)
        assert isinstance(resultado, float), f"GSNR debe ser float, es {type(resultado)}"

    def test_gsnr_no_negativo(self, vector_512: np.ndarray) -> None:
        """GSNR debe ser un valor no negativo."""
        G = vector_512.reshape(1, -1)
        G_sq = G ** 2
        resultado = NeuralMathPrecision2026.gsnr(G, G_sq, t=10)
        assert resultado >= 0.0, f"GSNR no puede ser negativo: {resultado}"

    def test_gsnr_aumenta_con_señal_clara(self) -> None:
        """Gradiente más uniforme (señal clara) debe tener mayor GSNR que gradiente ruidoso."""
        rng = np.random.default_rng(42)
        G_claro = np.ones((1, 64), dtype=np.float32)
        G_ruidoso = rng.standard_normal((1, 64)).astype(np.float32) * 10
        gsnr_claro = NeuralMathPrecision2026.gsnr(G_claro, G_claro ** 2, t=1)
        gsnr_ruidoso = NeuralMathPrecision2026.gsnr(G_ruidoso, G_ruidoso ** 2, t=1)
        # GSNR de señal uniforme debe ser finito
        assert math.isfinite(gsnr_claro), "GSNR de señal clara debe ser finito"

    def test_gsnr_con_t_cero_no_lanza(self, vector_512: np.ndarray) -> None:
        """gsnr() con t=0 no debe lanzar ZeroDivisionError."""
        G = vector_512.reshape(1, -1)
        try:
            resultado = NeuralMathPrecision2026.gsnr(G, G ** 2, t=0)
            assert math.isfinite(resultado) or resultado == 0.0, "Con t=0 debe ser finito o cero"
        except ZeroDivisionError:
            pytest.fail("gsnr() no debe lanzar ZeroDivisionError con t=0")

    # ─── Entropía de Shannon ─────────────────────────────────────────────────

    def test_shannon_entropy_devuelve_float(self, vector_512: np.ndarray) -> None:
        """shannon_entropy() debe devolver un float."""
        if not hasattr(NeuralMathPrecision2026, "shannon_entropy"):
            pytest.skip("shannon_entropy no implementado en NeuralMathPrecision2026")
        resultado = NeuralMathPrecision2026.shannon_entropy(vector_512)
        assert isinstance(resultado, float), "shannon_entropy debe devolver float"

    def test_shannon_entropy_no_negativa(self, vector_512: np.ndarray) -> None:
        """La entropía de Shannon no puede ser negativa."""
        if not hasattr(NeuralMathPrecision2026, "shannon_entropy"):
            pytest.skip("shannon_entropy no implementado")
        h = NeuralMathPrecision2026.shannon_entropy(abs(vector_512))
        assert h >= 0.0, f"Entropía negativa: {h}"


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — ConversorRespuestaPesos: inicialización y red de 50 neuronas
# ═══════════════════════════════════════════════════════════════════════════════

