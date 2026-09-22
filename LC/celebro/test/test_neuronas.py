"""
test_neuronas.py — Suite de pruebas exhaustiva para las 50 neuronas de LucIA
=============================================================================
Cubre las 5 familias neuronales de la arquitectura LucIA (10 neuronas por familia):
  1. ENRN (Entrada Recurrente No Lineal): EN1..EN10
  2. RF_SL (Aprendizaje Supervisado Residual / Optimizadores de Memoria): RFSL1_1..10
  3. RF_EN (Retroalimentación y Refuerzo): RFEN1..RFEN10
  4. RNP (Plasticidad Sináptica y Optimización de Pesos 2026): RN1..RN10
  5. SLRN (Optimizadores Supervisados con Reglas de Actualización): SL1..SL10

Pruebas incluidas:
  - Verificación de metadatos del paquete y catálogo global FAMILIAS.
  - Importabilidad de módulos y clases neuronales individuales.
  - Comportamiento de forward y cálculo de tensores válidos.
  - Inicializadores de pesos (He, Xavier, LeCun, Ortogonal).
  - Manejo de excepciones esperadas (NotImplementedError en bases/stubs).
  - Verificación de tipos de datos (float32, consistencia de shapes).
  - Detección de divergencias numéricas (NaN, Inf).
"""
from __future__ import annotations

import importlib
import math
from typing import Any, Dict, List, Type
import numpy as np
import pytest

from LC.celebro.test.conftest import (
    assert_tensor_valido,
    assert_norma_acotada,
    vector_512,
    vector_64,
    matriz_2d_32x32,
)


# ============================================================================
# 1. METADATOS Y CATÁLOGO GLOBAL DE LA RED NEURONAL
# ============================================================================

class TestRedNeuronalMetadatos:
    """Valida la integridad de los metadatos globales del paquete red_neuronal."""

    def test_catalogo_familias_existencia(self) -> None:
        """Comprueba que el diccionario FAMILIAS contenga las 5 familias requeridas."""
        from LC.celebro.red_neuronal import FAMILIAS
        assert isinstance(FAMILIAS, dict)
        assert len(FAMILIAS) == 5
        familias_esperadas = {"ENRN", "RF_SL", "RF_EN", "RNP", "SLRN"}
        assert set(FAMILIAS.keys()) == familias_esperadas

    @pytest.mark.parametrize("familia,descripcion", [
        ("ENRN", "Entrada recurrente no lineal"),
        ("RF_SL", "Aprendizaje supervisado residual"),
        ("RF_EN", "Retroalimentacion y refuerzo"),
        ("RNP", "Plasticidad sinaptica"),
        ("SLRN", "Optimizadores supervisados"),
    ])
    def test_descripcion_familia(self, familia: str, descripcion: str) -> None:
        """Verifica que las descripciones de las familias contengan las claves funcionales."""
        from LC.celebro.red_neuronal import FAMILIAS
        assert familia in FAMILIAS
        assert descripcion.lower() in FAMILIAS[familia].lower()


# ============================================================================
# 2. FAMILIA ENRN (Entrada Recurrente No Lineal - 10 Neuronas)
# ============================================================================

class TestFamiliaENRN:
    """Pruebas funcionales y de tensores para las neuronas EN1..EN10."""

    @pytest.mark.parametrize("modulo,clase", [
        ("EN1_RN", "NeuronaEntradaBasica"),
        ("EN2_RN", "NeuronaEntradaXavier"),
        ("EN3_RN", "NeuronaEntradaLeCun"),
        ("EN4_RN", "NeuronaEntradaNormalizada"),
        ("EN5_RN", "NeuronaEntradaRegularizada"),
        ("EN6_RN", "NeuronaEntradaDropout"),
        ("EN7_RN", "NeuronaEntradaBatchNorm"),
        ("EN8_RN", "NeuronaEntradaReLU"),
        ("EN9_RN", "NeuronaEntradaTanh"),
        ("EN10_RN", "NeuronaEntradaSigmoid"),
    ])
    def test_import_neuronas_enrn(self, modulo: str, clase: str) -> None:
        """Verifica que cada submódulo de ENRN se pueda importar y exponga su clase."""
        mod = importlib.import_module(f"LC.celebro.red_neuronal.ENRN.{modulo}")
        assert hasattr(mod, clase), f"El módulo {modulo} no contiene la clase {clase}"
        cls = getattr(mod, clase)
        assert callable(cls)

    def test_enrn_paquete_salud(self) -> None:
        """Comprueba el reporte de salud del paquete ENRN."""
        from LC.celebro.red_neuronal.ENRN import salud
        rep = salud()
        assert isinstance(rep, dict)
        assert rep.get("ok") is True, f"Fallo en salud ENRN: {rep}"
        assert rep.get("tipos", 0) >= 10

    def test_enrn_crear_neurona_factory(self) -> None:
        """Prueba la factoría crear_neurona() de ENRN."""
        from LC.celebro.red_neuronal.ENRN import crear_neurona
        neurona = crear_neurona("basica", input_size=8, output_size=4)
        assert neurona is not None
        assert neurona.input_size == 8
        assert neurona.output_size == 4

    def test_enrn_forward_computo_basica(self) -> None:
        """Valida que NeuronaEntradaBasica ejecute forward produciendo tensor válido."""
        from LC.celebro.red_neuronal.ENRN import NeuronaEntradaBasica
        n = NeuronaEntradaBasica(input_size=16, output_size=8)
        x = np.ones((2, 16), dtype=np.float32)
        salida = n.forward(x)
        assert_tensor_valido(salida, esperado_shape=(2, 8), dtype_esperado=np.float32)
        assert np.all(np.isfinite(salida))

    def test_enrn_forward_activaciones_no_lineales(self) -> None:
        """Verifica que las neuronas ReLU, Tanh y Sigmoid acoten sus salidas."""
        from LC.celebro.red_neuronal.ENRN import (
            NeuronaEntradaReLU,
            NeuronaEntradaTanh,
            NeuronaEntradaSigmoid,
        )
        x = np.array([[-10.0, 0.0, 10.0]], dtype=np.float32)

        # ReLU
        n_relu = NeuronaEntradaReLU(input_size=3, output_size=3)
        out_relu = n_relu.forward(x)
        assert np.all(out_relu >= 0.0)

        # Tanh
        n_tanh = NeuronaEntradaTanh(input_size=3, output_size=3)
        out_tanh = n_tanh.forward(x)
        assert np.all(out_tanh >= -1.0) and np.all(out_tanh <= 1.0)

        # Sigmoid
        n_sig = NeuronaEntradaSigmoid(input_size=3, output_size=3)
        out_sig = n_sig.forward(x)
        assert np.all(out_sig >= 0.0) and np.all(out_sig <= 1.0)

    def test_enrn_backward_gradientes(self) -> None:
        """Verifica que backward propague gradientes con shape consistente."""
        from LC.celebro.red_neuronal.ENRN import NeuronaEntradaBasica
        n = NeuronaEntradaBasica(input_size=4, output_size=2)
        x = np.random.randn(1, 4).astype(np.float32)
        _ = n.forward(x)
        grad_salida = np.ones((1, 2), dtype=np.float32)
        grad_pesos, grad_sesgo = n.backward(grad_salida, x)
        assert grad_pesos.shape == (4, 2)
        assert grad_sesgo.shape == (1, 2)
        grad_in = n.grad_entrada(grad_salida)
        assert grad_in.shape == (1, 4)


# ============================================================================
# 3. FAMILIA RF_SL (Aprendizaje Supervisado Residual - 10 Neuronas)
# ============================================================================

class TestFamiliaRFSL:
    """Pruebas funcionales para las neuronas de memoria residual RFSL1_1..10."""

    @pytest.mark.parametrize("modulo,clase", [
        ("RF_SL1_1", "GradientBoostingOptimizer"),
        ("RF_SL1_2", "SupportVectorMachineOptimizer"),
        ("RF_SL1_3", "RandomForestOptimizer"),
        ("RF_SL1_4", "NeuralNetworkOptimizer"),
        ("RF_SL1_5", "DecisionTreeOptimizer"),
        ("RF_SL1_6", "NaiveBayesOptimizer"),
        ("RF_SL1_7", "KNearestNeighborsOptimizer"),
        ("RF_SL1_8", "LogisticRegressionOptimizer"),
        ("RF_SL1_9", "LinearRegressionOptimizer"),
        ("RF_SL1_10", "IntegratedSupervisedLearningOptimizer"),
    ])
    def test_import_neuronas_rfsl(self, modulo: str, clase: str) -> None:
        """Verifica la importación de cada optimizador supervisado RF_SL."""
        mod = importlib.import_module(f"LC.celebro.red_neuronal.RF_SL.RFSL1.{modulo}")
        assert hasattr(mod, clase)
        cls = getattr(mod, clase)
        assert callable(cls)

    def test_rfsl1_1_gradient_boosting_forward(self) -> None:
        """Prueba forward y cálculo de pérdida en GradientBoostingOptimizer."""
        from LC.celebro.red_neuronal.RF_SL.RFSL1.RF_SL1_1 import GradientBoostingOptimizer
        opt = GradientBoostingOptimizer(input_size=4, output_size=2)
        x = np.ones((3, 4), dtype=np.float32)
        salida = opt.forward(x)
        assert salida.shape == (3, 2)
        loss = opt.compute_loss(salida, np.zeros((3, 2), dtype=np.float32))
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_rfsl1_1_train_step(self) -> None:
        """Verifica el ciclo de entrenamiento de paso único (train_step)."""
        from LC.celebro.red_neuronal.RF_SL.RFSL1.RF_SL1_1 import GradientBoostingOptimizer
        opt = GradientBoostingOptimizer(input_size=4, output_size=1)
        x = np.random.randn(10, 4).astype(np.float32)
        y = np.random.randn(10, 1).astype(np.float32)
        metricas = opt.train_step(x, y)
        assert "loss_before" in metricas
        assert "loss_after" in metricas
        assert "improvement" in metricas

    def test_rfsl1_10_integrated_optimizer(self) -> None:
        """Verifica la inicialización del optimizador integrado RFSL1_10."""
        from LC.celebro.red_neuronal.RF_SL.RFSL1.RF_SL1_10 import (
            IntegratedSupervisedLearningOptimizer,
        )
        opt = IntegratedSupervisedLearningOptimizer()
        assert opt is not None
        stats = opt.obtener_estadisticas()
        assert isinstance(stats, dict)
        assert "nombre" in stats


# ============================================================================
# 4. FAMILIA RF_EN (Retroalimentación y Refuerzo - 10 Neuronas)
# ============================================================================

class TestFamiliaRFEN:
    """Pruebas funcionales para las neuronas de refuerzo RFEN1_1..10."""

    @pytest.mark.parametrize("idx,clase", [
        (1, "NeuronaRefuerzoQLearning"),
        (2, "NeuronaRefuerzoPolicyGradient"),
        (3, "NeuronaRefuerzoActorCritic"),
        (4, "NeuronaRefuerzoDQN"),
        (5, "NeuronaRefuerzoA3C"),
        (6, "NeuronaRefuerzoPPO"),
        (7, "NeuronaRefuerzoSAC"),
        (8, "NeuronaRefuerzoTD3"),
        (9, "NeuronaRefuerzoRainbowDQN"),
        (10, "NeuronaRefuerzoIMPALA"),
    ])
    def test_import_neuronas_rfen(self, idx: int, clase: str) -> None:
        """Comprueba la disponibilidad de todas las neuronas de refuerzo."""
        mod = importlib.import_module(f"LC.celebro.red_neuronal.RF_EN.RFEN1_RN_{idx}")
        assert hasattr(mod, clase)
        cls = getattr(mod, clase)
        assert callable(cls)

    def test_rfen_qlearning_forward_y_actualizacion(self) -> None:
        """Verifica ciclo de acción y actualización Bellman en Q-Learning (RFEN1)."""
        from LC.celebro.red_neuronal.RF_EN.RFEN1_RN_1 import NeuronaRefuerzoQLearning
        q_agent = NeuronaRefuerzoQLearning(input_size=8, output_size=4, epsilon=0.0)
        q_agent.inicializar_pesos("ceros")

        # Forward determinista (epsilon=0)
        estado = np.array([2])
        accion = q_agent.forward(estado)
        assert isinstance(accion, np.ndarray)
        assert accion.shape == (1,)

        # Actualización Bellman
        delta = q_agent.actualizar_q_value(
            estado=2, accion=0, recompensa=1.0, siguiente_estado=3, terminado=False
        )
        assert isinstance(delta, float)
        assert q_agent.estadisticas_qlearning["actualizaciones_q"] == 1

    def test_rfen_inicializadores_pesos(self) -> None:
        """Verifica que los métodos de inicialización generen tensores válidos."""
        from LC.celebro.red_neuronal.RF_EN import (
            inicializar_pesos_he,
            inicializar_pesos_xavier,
            inicializar_pesos_lecun,
        )
        w_he = inicializar_pesos_he((16, 8), fan_in=16)
        assert_tensor_valido(w_he, esperado_shape=(16, 8), dtype_esperado=np.float32)

        w_xavier = inicializar_pesos_xavier((16, 8), fan_in=16, fan_out=8)
        assert_tensor_valido(w_xavier, esperado_shape=(16, 8), dtype_esperado=np.float32)

        w_lecun = inicializar_pesos_lecun((16, 8), fan_in=16)
        assert_tensor_valido(w_lecun, esperado_shape=(16, 8), dtype_esperado=np.float32)

    def test_rfen_base_not_implemented_raises(self) -> None:
        """Verifica que la clase base abstracta de RF_EN lance NotImplementedError."""
        from LC.celebro.red_neuronal.RF_EN import NeuronaRefuerzoBase
        base = NeuronaRefuerzoBase(input_size=4, output_size=2)
        with pytest.raises(NotImplementedError):
            base.inicializar_pesos()
        with pytest.raises(NotImplementedError):
            base.forward(np.zeros((1, 4)))


# ============================================================================
# 5. FAMILIA RNP (Plasticidad Sináptica / Optimizadores 2026 - 10 Neuronas)
# ============================================================================

class TestFamiliaRNP:
    """Pruebas de inicialización y control de estado para RN1..RN10."""

    @pytest.mark.parametrize("modulo,alias", [
        ("RN1", "adamw"),
        ("RN2", "radam"),
        ("RN3", "lookahead"),
        ("RN4", "nadam"),
        ("RN5", "lamb"),
        ("RN6", "adabelief"),
        ("RN7", "lion"),
        ("RN8", "sam"),
        ("RN9", "swats"),
        ("RN10", "integrado"),
    ])
    def test_import_modulos_rnp(self, modulo: str, alias: str) -> None:
        """Comprueba la existencia de los módulos de optimización RNP."""
        mod = importlib.import_module(f"LC.celebro.red_neuronal.RNP.{modulo}")
        assert mod is not None
        assert hasattr(mod, "NeuralWeightOptimizationConfig")

    def test_rnp_registry_catalogo(self) -> None:
        """Verifica que el registro central de RNP liste todos los algoritmos 2026."""
        from LC.celebro.red_neuronal.RNP import RNPRegistry
        algos = RNPRegistry.list_keys()
        assert len(algos) >= 10
        for clave in ["adamw", "radam", "lookahead", "nadam", "lamb", "lion", "sam"]:
            assert clave in algos

    def test_rnp_factory_creacion(self) -> None:
        """Verifica la factoría de creación de optimizadores RNP."""
        from LC.celebro.red_neuronal.RNP import RNPFactory
        opt = RNPFactory.create("adamw")
        assert opt is not None
        assert hasattr(opt, "config")
        assert hasattr(opt, "optimize_weights")

    def test_rnp_base_not_implemented_handling(self) -> None:
        """Valida que los métodos abstractos de BaseNeuralWeightOptimizer lancen NotImplementedError."""
        from LC.celebro.red_neuronal.RNP.RN1 import (
            BaseNeuralWeightOptimizer,
            NeuralWeightOptimizationConfig,
        )
        cfg = NeuralWeightOptimizationConfig()
        base = BaseNeuralWeightOptimizer(cfg)
        with pytest.raises(NotImplementedError):
            base.create_optimizer(model=None)
        with pytest.raises(NotImplementedError):
            base.optimize_weights(model=None, data_loader=None)


# ============================================================================
# 6. FAMILIA SLRN (Optimizadores Supervisados - 10 Neuronas)
# ============================================================================

class TestFamiliaSLRN:
    """Pruebas funcionales de SL1..SL10 y su meta-ensamble integrado."""

    @pytest.mark.parametrize("idx,nombre_archivo", [
        (1, "SL1"),
        (2, "SL2"),
        (3, "SL3"),
        (4, "SL4"),
        (5, "SL5"),
        (6, "SL6"),
        (7, "SL7"),
        (8, "SL8"),
        (9, "SL9"),
        (10, "SL10"),
    ])
    def test_import_modulos_slrn(self, idx: int, nombre_archivo: str) -> None:
        """Comprueba que todos los archivos SL1..SL10 sean importables."""
        mod = importlib.import_module(f"LC.celebro.red_neuronal.SLRN.{nombre_archivo}")
        assert mod is not None

    def test_slrn_registry_mapeo(self) -> None:
        """Verifica el mapeo en OPTIMIZER_REGISTRY de SLRN."""
        from LC.celebro.red_neuronal.SLRN import OPTIMIZER_REGISTRY
        assert isinstance(OPTIMIZER_REGISTRY, dict)
        assert len(OPTIMIZER_REGISTRY) >= 10
        claves_esperadas = ["backprop", "sgd", "rmsprop", "adagrad", "adadelta", "adamw"]
        for c in claves_esperadas:
            assert c in OPTIMIZER_REGISTRY

    def test_slrn_configuracion_dataclass(self) -> None:
        """Valida valores por defecto de la configuración neuronal supervisada."""
        from LC.celebro.red_neuronal.SLRN import SupervisedLearningNeuralConfig
        cfg = SupervisedLearningNeuralConfig()
        assert cfg.learning_rate > 0.0
        assert 0.0 <= cfg.bp_momentum <= 1.0
        assert cfg.weight_decay >= 0.0

    def test_slrn_sistema_inicializacion(self) -> None:
        """Verifica la inicialización del sistema de ensamble SLRNSystem."""
        from LC.celebro.red_neuronal.SLRN import SLRNSystem
        sistema = SLRNSystem()
        assert sistema is not None
        assert hasattr(sistema, "available")
        assert len(sistema.available) >= 10


# ============================================================================
# 7. INTEGRACIÓN INTER-FAMILIAS Y CONSISTENCIA NUMÉRICA
# ============================================================================

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
            archivos = [f for f in os.listdir(dir_fam) if f.endswith(".py") and f != "__init__.py"]
            assert len(archivos) == 10, f"Familia {fam} tiene {len(archivos)} archivos, esperados 10"
            conteo += len(archivos)

        assert conteo == 50, f"Total de neuronas contabilizadas: {conteo}, esperado: 50"


# Fin de suite exhaustiva test_neuronas.py
