class TestIntegracionInterFamilias:
    """Pruebas de paso de datos entre familias neuronales."""

    def test_pipeline_enrn_hacia_rfsl(self) -> None:
        """Verifica que la salida de ENRN pueda ser consumida directamente por RF_SL."""
        from LC.celebro.red_neuronal.ENRN import NeuronaEntradaBasica
        from LC.celebro.red_neuronal.RF_SL.RFSL1.RF_SL1_1 import GradientBoostingOptimizer

        # Capa de entrada ENRN
        enrn = NeuronaEntradaBasica(input_size=16, output_size=8)
        # Capa supervisada RF_SL
        rfsl = GradientBoostingOptimizer(input_size=8, output_size=2)

        x_raw = np.random.randn(4, 16).astype(np.float32)
        h = enrn.forward(x_raw)
        assert_tensor_valido(h, esperado_shape=(4, 8), dtype_esperado=np.float32)

        y = rfsl.forward(h)
        assert_tensor_valido(y, esperado_shape=(4, 2), dtype_esperado=np.float32)
        assert_norma_acotada(y, max_norma=100.0)

    def test_deteccion_estabilidad_numerica(self) -> None:
        """Comprueba que no se propaguen valores NaN o Inf con entradas estándar."""
        from LC.celebro.red_neuronal.ENRN import NeuronaEntradaTanh
        n = NeuronaEntradaTanh(input_size=10, output_size=10)
        x_ceros = np.zeros((1, 10), dtype=np.float32)
        res_ceros = n.forward(x_ceros)
        assert not np.any(np.isnan(res_ceros))
        assert not np.any(np.isinf(res_ceros))

    def test_conteo_total_neuronas_catalogo(self) -> None:
        """Verifica exhaustivamente que se puedan localizar las 50 neuronas en disco."""
        import os
        from LC.celebro.test.conftest import CELEBRO_DIR

        rn_path = CELEBRO_DIR / "red_neuronal"
        assert rn_path.exists()

        conteo = 0
        familias = ["ENRN", "RF_SL/RFSL1", "RF_EN", "RNP", "SLRN"]
        for fam in familias:
            dir_fam = rn_path / fam
            archivos = [f for f in os.listdir(dir_fam) if f.endswith(".py") and f != "__init__.py" and not f.startswith("_")]
            assert len(archivos) == 10, f"Familia {fam} tiene {len(archivos)} archivos, esperados 10"
            conteo += len(archivos)

        assert conteo == 50, f"Total de neuronas contabilizadas: {conteo}, esperado: 50"


# Fin de suite exhaustiva test_neuronas.py
