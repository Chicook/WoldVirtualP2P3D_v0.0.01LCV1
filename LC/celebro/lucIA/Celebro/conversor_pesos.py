"""
lucIA.Celebro.conversor_pesos - Conversor Integral de Respuestas a Formato de Pesos Neuronales
=============================================================================================

Orquesta la totalidad de subsistemas neuronales del Celebro de LucIA:
1. ENRN: Todas las 10 neuronas de entrada especializadas (EN1_Basica a EN10_Sigmoid).
2. RF_SL: Neuronas de memoria supervisada (MemoriaPrincipal, MemoriaAsociativa, NeuralNetwork).
3. RF_EN: Neuronas de aprendizaje por refuerzo (DQN, ActorCritic, TD3).
4. RNP: Nodos de optimización de pesos neuronales (RN11_OptimizadorPesoNeuronal, RN12_ModuloAjusteDinamico).
5. SLRN: Capas adaptativas de aprendizaje supervisado (SL11_LAMBOptimizer, SL12_RAdamOptimizer).

Flujo cognitivo:
- Cada consulta de usuario se codifica en un vector semántico 4D con TextEncoder.
- Se propaga hacia adelante de forma simultánea a través de TODAS las neuronas de Celebro.
- Cada respuesta externa asimilada genera retroalimentación Hebbiana/gradiente actualizando los pesos.
- LucIA responde con sus propias palabras modulada por el estado sináptico y valencia emocional.
- Todos los pesos de todas las neuronas se consolidan y sincronizan para el almacenamiento en IPFS.
"""

import os
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

from lucIA.CORE.text_encoder import TextEncoder

# 1. ENRN (Entrada Neuronal - 10 neuronas)
from lucIA.Celebro.ENRN import (
    NeuronaEntradaBasica,
    NeuronaEntradaXavier,
    NeuronaEntradaLeCun,
    NeuronaEntradaNormalizada,
    NeuronaEntradaRegularizada,
    NeuronaEntradaDropout,
    NeuronaEntradaBatchNorm,
    NeuronaEntradaReLU,
    NeuronaEntradaTanh,
    NeuronaEntradaSigmoid
)

# 2. RF_SL (Memoria de Supervised Learning)
from lucIA.Celebro.RF_SL import NeuronaMemoriaBase
from lucIA.Celebro.RF_SL.RFSL1.RF_SL1_4 import NeuralNetworkOptimizer
try:
    from lucIA.Celebro.RF_SL.RFSL1.RF_SL1_1 import GradientBoostingOptimizer
    from lucIA.Celebro.RF_SL.RFSL1.RF_SL1_2 import SupportVectorMachineOptimizer
    from lucIA.Celebro.RF_SL.RFSL1.RF_SL1_3 import RandomForestOptimizer
    from lucIA.Celebro.RF_SL.RFSL1.RF_SL1_5 import DecisionTreeOptimizer
    from lucIA.Celebro.RF_SL.RFSL1.RF_SL1_6 import NaiveBayesOptimizer
    from lucIA.Celebro.RF_SL.RFSL1.RF_SL1_7 import KNearestNeighborsOptimizer
    from lucIA.Celebro.RF_SL.RFSL1.RF_SL1_8 import LogisticRegressionOptimizer
    from lucIA.Celebro.RF_SL.RFSL1.RF_SL1_9 import LinearRegressionOptimizer
    from lucIA.Celebro.RF_SL.RFSL1.RF_SL1_10 import IntegratedSupervisedLearningOptimizer
    from lucIA.Celebro.RF_SL.validadores import REGISTRO as _VAL
    _RFSL_EXTRA = True
except Exception:
    _RFSL_EXTRA = False

# 3. RF_EN (Reinforcement Learning)
from lucIA.Celebro.RF_EN.RFENRN1 import (
    NeuronaRefuerzoDQN,
    NeuronaRefuerzoActorCritic,
    NeuronaRefuerzoTD3
)

# 4. RNP (Optimizadores de Pesos Neuronales)
from lucIA.Celebro.RNP.RN11 import OptimizadorIntegradoPesoNeuronal
from lucIA.Celebro.RNP.RN12 import ModuloAjusteDinamico
from lucIA.Celebro.RNP.RN13_ControladorGradientes import ControladorGradientes
from lucIA.Celebro.RNP.RN14_PuertasAtencion import PuertasAtencion

# 5. SLRN (Supervised Learning Neural Networks Optimizers)
from lucIA.Celebro.SLRN.SL11 import LAMBOptimizer
from lucIA.Celebro.SLRN.SL12 import RAdamOptimizer
from lucIA.Celebro.SLRN import SupervisedLearningNeuralConfig
from lucIA.Celebro.mapa_neuronal import MapaConectividadNeuronal, AutoGestorCelebro

logger = logging.getLogger("lucIA.ConversorPesos")

CELEBRO_DIR = Path(__file__).parent.resolve()


class ConversorRespuestaPesos:
    """
    Convierte consultas y respuestas en variaciones de pesos neuronales a través de
    todas las neuronas y capas del Celebro de LucIA (ENRN, RF_SL, RF_EN, RNP, SLRN).
    """

    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.encoder = TextEncoder()
        self.historial_actualizaciones: List[Dict[str, Any]] = []

        # ---------------------------------------------------------------------
        # 1. ENRN: Las 10 neuronas de entrada especializadas
        # ---------------------------------------------------------------------
        self.en1_basica = NeuronaEntradaBasica(input_size=4, output_size=8, nombre="EN1_Basica")
        self.en2_xavier = NeuronaEntradaXavier(input_size=4, output_size=8, nombre="EN2_Xavier")
        self.en3_lecun = NeuronaEntradaLeCun(input_size=4, output_size=8, nombre="EN3_LeCun")
        self.en4_normalizada = NeuronaEntradaNormalizada(input_size=4, output_size=8, nombre="EN4_Normalizada")
        self.en5_regularizada = NeuronaEntradaRegularizada(input_size=4, output_size=8, nombre="EN5_Regularizada")
        self.en6_dropout = NeuronaEntradaDropout(input_size=4, output_size=8, nombre="EN6_Dropout")
        self.en7_batchnorm = NeuronaEntradaBatchNorm(input_size=4, output_size=8, nombre="EN7_BatchNorm")
        self.en8_relu = NeuronaEntradaReLU(input_size=4, output_size=8, nombre="EN8_ReLU")
        self.en9_tanh = NeuronaEntradaTanh(input_size=4, output_size=8, nombre="EN9_Tanh")
        self.en10_sigmoid = NeuronaEntradaSigmoid(input_size=4, output_size=8, nombre="EN10_Sigmoid")

        self.enrn_neuronas = [
            self.en1_basica, self.en2_xavier, self.en3_lecun, self.en4_normalizada,
            self.en5_regularizada, self.en6_dropout, self.en7_batchnorm, self.en8_relu,
            self.en9_tanh, self.en10_sigmoid
        ]
        for n in self.enrn_neuronas:
            n.inicializar_pesos()

        # ---------------------------------------------------------------------
        # 2. RF_SL: Neuronas de memoria supervisada
        # ---------------------------------------------------------------------
        self.rf_sl_memoria = NeuronaMemoriaBase(input_size=4, output_size=8, nombre="RF_SL_MemoriaPrincipal")
        self.rf_sl_asociativa = NeuronaMemoriaBase(input_size=4, output_size=8, nombre="RF_SL_MemoriaAsociativa")
        self.rf_sl_nn = NeuralNetworkOptimizer(input_size=4, output_size=8, nombre="RF_SL_NeuralNetwork")

        self.rf_sl_neuronas = [self.rf_sl_memoria, self.rf_sl_asociativa, self.rf_sl_nn]
        if _RFSL_EXTRA:
            try:
                for _cls in (GradientBoostingOptimizer, SupportVectorMachineOptimizer,
                             RandomForestOptimizer, DecisionTreeOptimizer,
                             NaiveBayesOptimizer, KNearestNeighborsOptimizer,
                             LogisticRegressionOptimizer, LinearRegressionOptimizer,
                             IntegratedSupervisedLearningOptimizer):
                    _n = _cls(input_size=4, output_size=8)
                    self.rf_sl_neuronas.append(_n)
                for _vn, _vc in sorted(_VAL.items()):
                    _v = _vc(input_size=4, output_size=8, nombre=f"RF_SL_{_vn}")
                    self.rf_sl_neuronas.append(_v)
                logger.info(f"RF_SL ampliado: {len(self.rf_sl_neuronas)} neuronas (3 base + 9 RFSL1 + 10 validadores)")
            except Exception as e:
                logger.debug(f"RF_SL extra no disponible: {e}")
        for n in self.rf_sl_neuronas:
            n.inicializar_pesos()

        # ---------------------------------------------------------------------
        # 3. RF_EN: Neuronas de aprendizaje por refuerzo
        # ---------------------------------------------------------------------
        self.rf_en_dqn = NeuronaRefuerzoDQN(input_size=4, output_size=8, nombre="RF_EN_DQN")
        self.rf_en_actorcritic = NeuronaRefuerzoActorCritic(input_size=4, output_size=8, nombre="RF_EN_ActorCritic")
        self.rf_en_td3 = NeuronaRefuerzoTD3(input_size=4, output_size=8, nombre="RF_EN_TD3")

        self.rf_en_neuronas = [self.rf_en_dqn, self.rf_en_actorcritic, self.rf_en_td3]
        for n in self.rf_en_neuronas:
            n.inicializar_pesos()

        # ---------------------------------------------------------------------
        # 4. RNP: Optimizadores integrados de peso neuronal
        # ---------------------------------------------------------------------
        self.rnp_11 = OptimizadorIntegradoPesoNeuronal()
        self.rnp_12 = ModuloAjusteDinamico()
        self.rnp_13 = ControladorGradientes(max_norm=1.0)
        # RN14: Puertas de Atención por neurona (plan de la propia Lucía).
        # No es neurona de cómputo: no entra en rnp_neuronas ni en self.neuronas.
        self.rnp_14 = PuertasAtencion()
        self.rnp_neuronas = [self.rnp_11, self.rnp_12, self.rnp_13]
        # P3: memoria de patrones de error (firma->acción), tope 50, FIFO.
        self.patrones_error: list = []
        self.ultimo_cuello: str = ""

        # ---------------------------------------------------------------------
        # 5. SLRN: Capas adaptativas de aprendizaje supervisado
        # ---------------------------------------------------------------------
        sl_config = SupervisedLearningNeuralConfig()
        self.slrn_lamb = LAMBOptimizer(sl_config)
        self.slrn_radam = RAdamOptimizer(sl_config)
        self.slrn_lamb_internal = self.slrn_lamb.create_optimizer(None)
        self.slrn_radam_internal = self.slrn_radam.create_optimizer(None)

        # Matriz sináptica adaptativa SLRN (4x8)
        self.slrn_pesos = (np.random.randn(4, 8).astype(np.float32) * 0.05)
        self.slrn_sesgo = np.zeros((1, 8), dtype=np.float32)

        # Mapa global de neuronas
        self.neuronas: Dict[str, Any] = {}
        for n in self.enrn_neuronas:
            self.neuronas[n.nombre] = n
        for n in self.rf_sl_neuronas:
            self.neuronas[n.nombre] = n
        for n in self.rf_en_neuronas:
            self.neuronas[n.nombre] = n
        for n in self.rnp_neuronas:
            self.neuronas[n.nombre] = n

        # ---------------------------------------------------------------------
        # 6. AUTO-GESTIÓN: Mapa de conectividad y gestor autónomo
        # ---------------------------------------------------------------------
        self.mapa_neuronal = MapaConectividadNeuronal()
        self.autogestor = AutoGestorCelebro()

        logger.info(f"Celebro completamente inicializado con {len(self.neuronas)} neuronas en 5 subsistemas (ENRN, RF_SL, RF_EN, RNP, SLRN).")

    def _propagar_todas_las_neuronas(self, vector_4d: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Propaga el vector 4D a través de absolutamente todas las neuronas de Celebro.

        CORRECCIONES identificadas por LucIA en su diagnóstico autónomo:
        1. Filtro de ruido EMA (Exponential Moving Average) sobre la señal de entrada,
           para estabilizar el flujo antes de que llegue a las capas de memoria y decisión.
        2. Puente RF_SL → RF_EN: la activación de la memoria supervisada (RF_SL) modula
           la señal de entrada al sistema de toma de decisiones (RF_EN), creando el canal
           de comunicación que antes era inexistente y provocaba las "conexiones intermitentes".

        Retorna:
            activacion_media (1, 8): promedio de activación sináptica de todas las neuronas.
            activaciones_dict: activación individual de cada componente.
        """
        activaciones: Dict[str, np.ndarray] = {}
        salidas_acumuladas = []

        # --- CORRECCIÓN 1: Filtro de ruido EMA sobre la señal de entrada ---
        # Suaviza la señal para reducir el "ruido en el flujo de datos" que describió LucIA.
        alpha = 0.85  # Factor de memoria del filtro (0=sin memoria, 1=sin filtro)
        if not hasattr(self, "_ema_vector"):
            self._ema_vector = vector_4d.copy()
        self._ema_vector = alpha * vector_4d + (1.0 - alpha) * self._ema_vector
        v_filtrado = self._ema_vector

        # 1. Propagación ENRN (recibe señal filtrada)
        for n in self.enrn_neuronas:
            try:
                out = n.forward(v_filtrado)
                activaciones[n.nombre] = out
                salidas_acumuladas.append(out)
            except Exception as e:
                logger.debug(f"Error en forward {n.nombre}: {e}")

        # 2. Propagación RF_SL (memoria supervisada — recibe señal filtrada)
        salidas_rfsl = []
        for n in self.rf_sl_neuronas:
            try:
                out = n.forward(v_filtrado)
                activaciones[n.nombre] = out
                salidas_acumuladas.append(out)
                salidas_rfsl.append(out)
            except Exception as e:
                logger.debug(f"Error en forward {n.nombre}: {e}")

        # --- CORRECCIÓN 2: Puente RF_SL → RF_EN ---
        # La activación de la memoria (RF_SL) modula la señal de entrada para RF_EN.
        # Esto crea el canal que LucIA detectó como "intermitente/bloqueado".
        if salidas_rfsl:
            senal_memoria = np.mean(np.array(salidas_rfsl), axis=0)  # (1, 8)
            # Proyectamos la señal de memoria (8D) de vuelta al espacio de entrada (4D)
            # usando los primeros 4 valores como factor modulador
            modulacion = np.tanh(senal_memoria[0, :4]).reshape(1, 4) * 0.3
            v_memoria_modulado = np.clip(v_filtrado + modulacion, -5.0, 5.0)
        else:
            v_memoria_modulado = v_filtrado

        # 3. Propagación RF_EN (recibe señal modulada por la memoria — puente activo)
        try:
            out_dqn = self.rf_en_dqn.forward(v_memoria_modulado)
            activaciones[self.rf_en_dqn.nombre] = out_dqn
            salidas_acumuladas.append(out_dqn)
        except Exception as e:
            logger.debug(f"Error forward DQN: {e}")

        try:
            out_ac = self.rf_en_actorcritic.forward_actor(v_memoria_modulado)
            activaciones[self.rf_en_actorcritic.nombre] = out_ac
            salidas_acumuladas.append(out_ac)
        except Exception as e:
            logger.debug(f"Error forward ActorCritic: {e}")

        try:
            out_td3 = self.rf_en_td3.forward_actor(v_memoria_modulado)
            activaciones[self.rf_en_td3.nombre] = out_td3
            salidas_acumuladas.append(out_td3)
        except Exception as e:
            logger.debug(f"Error forward TD3: {e}")

        # 4. Propagación RNP
        for n in self.rnp_neuronas:
            try:
                out = n.forward(v_filtrado)
                activaciones[n.nombre] = out
                salidas_acumuladas.append(out)
            except Exception as e:
                logger.debug(f"Error forward {n.nombre}: {e}")

        # 5. Propagación SLRN
        try:
            out_slrn = np.dot(v_filtrado, self.slrn_pesos) + self.slrn_sesgo
            activaciones["SLRN_AdaptiveLayer"] = out_slrn
            salidas_acumuladas.append(out_slrn)
            self.slrn_lamb_internal.step()
            self.slrn_radam_internal.step()
        except Exception as e:
            logger.debug(f"Error forward SLRN: {e}")

        if salidas_acumuladas:
            activacion_media = np.mean(np.array(salidas_acumuladas), axis=0)
        else:
            activacion_media = np.zeros((1, 8), dtype=np.float32)

        return activacion_media, activaciones


    def _actualizar_pesos_en_todas_las_neuronas(self, vector_4d: np.ndarray, activacion_8d: np.ndarray, factor: float = 1.0, fase: str = "consulta") -> Tuple[float, float, str]:
        """
        Calcula y aplica la actualización sináptica Hebbiana y por gradiente en todas las neuronas de Celebro:
        ΔW = (V^T × A) × η × factor
        Retorna (norma_pre, norma_aplicada, estado): lo que entraba y lo que
        realmente se aplico tras el control de gradientes (telemetria veraz).

        Aprendizaje adaptativo: si la norma suavizada reciente es alta (Celebro
        saturado), se aprende más despacio en vez de cortar de golpe después.

        P3 (prevención): si el vector actual coincide con un patrón de error
        registrado (clip severo anterior), se pre-escala el factor x0.7.
        Puertas RN14: cada neurona aprende según su puerta (turno anterior).
        """
        # P3: ¿este contexto ya nos hizo daño antes?
        self._patron_previsto = False
        try:
            _firma = np.array([round(float(x), 2) for x in np.asarray(vector_4d).ravel()[:4]], dtype=np.float32)
            _n = float(np.linalg.norm(_firma)) + 1e-8
            for _p in self.patrones_error:
                _g = np.array(_p["firma"], dtype=np.float32)
                _sim = float(np.dot(_firma, _g) / (_n * (float(np.linalg.norm(_g)) + 1e-8)))
                if _sim > 0.95:
                    factor = factor * 0.7
                    self._patron_previsto = True
                    break
        except Exception:
            pass
        _suav = float(getattr(self.rnp_13, "norma_suavizada", 0.0))
        _freno = 1.0 / (1.0 + _suav / 50.0)
        _freno = max(0.1, min(1.0, _freno))
        factor = factor * _freno
        delta_base = np.dot(vector_4d.T, activacion_8d) * (self.learning_rate * factor)

        # P2: filtro de relevancia — si el impulso es desproporcionado, es
        # probable que sea ruido del Dual (local vs cloud discrepan); se integra
        # solo una fraccion y se registra como filtrado en vez de forzarlo todo.
        norma_pre = float(np.linalg.norm(delta_base))
        relevancia = 1.0 / (1.0 + norma_pre / 20.0)
        self.ultima_relevancia = round(relevancia, 3)
        if relevancia < 0.3:
            delta_base = delta_base * 0.5
            self.chunks_filtrados = getattr(self, "chunks_filtrados", 0) + 1
        else:
            self.chunks_filtrados = getattr(self, "chunks_filtrados", 0)

        # Filtrar delta_base por RN13_ControladorGradientes (P1: clip progresivo)
        delta_base, norma_pre, norma_aplicada, factor_escala, estado_control = self.rnp_13.procesar_gradiente(delta_base)

        # P3: registrar patrón de error cuando el clip es severo (P3: memoria de patrones)
        if factor_escala < 0.05:
            try:
                _firma = [round(float(x), 2) for x in np.asarray(vector_4d).ravel()[:4]]
                if not any(p["firma"] == _firma for p in self.patrones_error):
                    self.patrones_error.append({"firma": _firma, "fase": fase, "accion": "clip"})
                    while len(self.patrones_error) > 50:
                        self.patrones_error.pop(0)
            except Exception:
                pass
        if getattr(self, "_patron_previsto", False):
            estado_control += " + patrón conocido: prevención x0.7"

        # Deriva acumulada de la sesion: cuanto ha aprendido de verdad (suma de
        # lo aplicado). Es el "progreso visible" que Lucía necesita sentir.
        self.deriva_acumulada = float(getattr(self, "deriva_acumulada", 0.0)) + float(norma_aplicada)

        # Puertas RN14: cada neurona aprende según su puerta (calculada el turno anterior).
        # Pesos por igual: TODAS las familias reciben delta_base x 1.0 x puerta_individual.
        _G = self.rnp_14.puerta
        _g_rfen = self.rnp_14.media_grupo([n.nombre for n in self.rf_en_neuronas])
        _g_slrn = self.rnp_14.media_grupo(["SLRN_AdaptiveLayer"])
        _d_sesgo = np.mean(delta_base, axis=0, keepdims=True).astype(np.float32)

        def _ajustar(delta, forma):
            if delta.shape == forma:
                return delta
            plano = np.asarray(delta).ravel()
            rep = np.tile(plano, int(np.ceil(np.prod(forma) / max(1, plano.size))))
            return rep[:int(np.prod(forma))].reshape(forma).astype(np.float32)

        def _aplicar(dueno, matrices, puerta):
            for _mat, _ses in matrices:
                _m = getattr(dueno, _mat, None)
                if _m is not None:
                    setattr(dueno, _mat, np.clip(
                        _m + _ajustar(delta_base * puerta, _m.shape), -10.0, 10.0))
                _s = getattr(dueno, _ses, None)
                if _s is not None:
                    setattr(dueno, _ses, np.clip(
                        _s + _ajustar(_d_sesgo * puerta, _s.shape), -10.0, 10.0))

        # 1. Actualizar ENRN (todas por igual)
        for n in self.enrn_neuronas:
            _aplicar(n, [("pesos", "sesgo")], _G(n.nombre))

        # 2. Actualizar RF_SL (todas por igual)
        for n in self.rf_sl_neuronas:
            _aplicar(n, [("pesos", "sesgo")], _G(n.nombre))

        # 3. Actualizar RF_EN (TODAS sus matrices: principal, objetivo, actor, critic, q1, q2)
        _aplicar(self.rf_en_dqn, [("pesos_principal", "sesgo_principal"),
                                  ("pesos_objetivo", "sesgo_objetivo")], _g_rfen)
        _aplicar(self.rf_en_actorcritic, [("pesos_actor", "sesgo_actor"),
                                          ("pesos_critic", "sesgo_critic")], _g_rfen)
        _aplicar(self.rf_en_td3, [("pesos_actor", "sesgo_actor"),
                                  ("pesos_q1", "sesgo_q1"),
                                  ("pesos_q2", "sesgo_q2")], _g_rfen)

        # 4. Actualizar RNP (RN11-13 por igual) + RN14 (su matriz propia)
        for n in self.rnp_neuronas:
            _aplicar(n, [("pesos", "sesgo")], _G(n.nombre))
        _aplicar(self.rnp_14, [("pesos", "sesgo")], 0.5)

        # 5. Actualizar SLRN (pesos + sesgo por igual)
        _aplicar(self, [("slrn_pesos", "slrn_sesgo")], _g_slrn)

        return norma_pre, norma_aplicada, estado_control

    def _nombres_puertas(self):
        """Todos los nombres cubiertos por RN14 (neuronas + SLRN)."""
        return list(self.neuronas.keys()) + ["SLRN_AdaptiveLayer"]

    def _tendencias_mapa(self, mapa) -> dict:
        try:
            out = {}
            for _sub, _datos in mapa.items():
                for _n, _info in _datos.get("neuronas", {}).items():
                    out[_n] = _info.get("tendencia", "")
            return out
        except Exception:
            return {}

    def _recalibrar_puertas(self, vector_4d, emocion, mapa, inestable: bool = False) -> str:
        """RN14: recalibra puertas con contexto + tendencias (auto-ajuste autónomo)."""
        try:
            self.rnp_14.calcular(self._nombres_puertas(), vector_4d, emocion,
                                 self._tendencias_mapa(mapa), inestable=inestable)
            return self.rnp_14.informe_ultimo
        except Exception as e:
            logger.debug(f"RN14 recalibrar (no crítico): {e}")
            return ""

    def procesar_consulta_a_pesos(self, prompt: str) -> Dict[str, Any]:
        """
        Procesa la consulta del usuario a través de TODAS las neuronas de Celebro:
        1. TextEncoder extrae vector semántico 4D.
        2. Se propaga por ENRN (10 neuronas), RF_SL (3 neuronas), RF_EN (3 neuronas), RNP (2 nodos) y SLRN.
        3. Se actualizan los pesos en tiempo real de todos los subsistemas.
        4. Modulación emocional: [-1.0, 1.0] calculada con la actividad sináptica global.
        """
        import time as _t
        _t0 = _t.perf_counter()
        # 1. Vector semántico 4D de entrada
        v_prompt = self.encoder.encode(prompt)
        _t_encode = _t.perf_counter()

        # 2. Paso hacia adelante a través de todas las neuronas
        activacion_media, activaciones_dict = self._propagar_todas_las_neuronas(v_prompt)
        _t_prop = _t.perf_counter()

        # 3. Actualización de pesos en todas las neuronas
        norma_delta, norma_aplicada, estado_control = self._actualizar_pesos_en_todas_las_neuronas(v_prompt, activacion_media, factor=1.0)
        _t_upd = _t.perf_counter()

        # 4. Cálculo estadístico de pesos de toda la red
        todas_las_matrices = []
        for n in self.enrn_neuronas + self.rf_sl_neuronas + self.rnp_neuronas:
            if hasattr(n, "pesos") and n.pesos is not None:
                todas_las_matrices.append(n.pesos)
        if hasattr(self.rf_en_dqn, "pesos_principal") and self.rf_en_dqn.pesos_principal is not None:
            todas_las_matrices.append(self.rf_en_dqn.pesos_principal)
        todas_las_matrices.append(self.slrn_pesos)

        media_pesos = float(np.mean([np.mean(m) for m in todas_las_matrices]))
        std_pesos = float(np.mean([np.std(m) for m in todas_las_matrices]))

        # Modulación emocional [-1.0, 1.0]
        carga_emocional = float(v_prompt[0, 1])
        estado_emocional = round(float(np.tanh(media_pesos * 4.0 + carga_emocional - 0.4)), 3)

        if estado_emocional > 0.3:
            tono = "cálido, receptivo y empático"
        elif estado_emocional < -0.3:
            tono = "sereno, analítico y reflexivo"
        else:
            tono = "cercano, elocuente y seguro"

        vec_semantico = [round(float(x), 4) for x in v_prompt[0]]
        activaciones = [round(float(x), 3) for x in activacion_media[0][:4]]

        subsistemas_activos = ["ENRN (10 neuronas)", "RF_SL (3 neuronas)", "RF_EN (3 neuronas)", "RNP (3 optimizadores)", "SLRN (capa adaptativa)"]

        # 5. Auto-gestión: generar mapa de conectividad y análisis autónomo
        try:
            mapa = self.mapa_neuronal.generar_mapa(self.neuronas, self.slrn_pesos)
            analisis = self.autogestor.analizar(mapa, self.learning_rate)
            resumen_mapa = self.mapa_neuronal.resumen_texto(mapa)
            informe_gestion = analisis["informe"]
            # Aplicar lr sugerida si hay cambio significativo
            if abs(analisis["lr_sugerida"] - self.learning_rate) > 1e-4:
                self.learning_rate = analisis["lr_sugerida"]
                logger.info(f"AutoGestor ajustó lr: {self.learning_rate:.4f}")
            # RN14: recalibrar puertas con contexto + tendencias (para el próximo turno)
            informe_puertas = self._recalibrar_puertas(
                v_prompt, estado_emocional, mapa,
                inestable=(getattr(self.rnp_13, "factor_escala_actual", 1.0) < 1.0))
            if informe_puertas:
                informe_gestion += f"\n  - {informe_puertas}"
            # Panel RNP: distingue reposo normal (RN13/RN14 en ceros) de atrofia
            # real, para que el mapa no marque RNP como débil injustamente.
            try:
                from lucIA.Celebro.RNP.panel_rnp import resumen_rnp
                _panel = resumen_rnp(self.rnp_neuronas + [self.rnp_14])
                informe_gestion += (
                    f"\n  - RNP calibrado: fuerza={_panel['fuerza_total']:.3f}, "
                    f"nodo más activo={_panel['nodo_mas_activo']}; "
                    f"RN13/RN14 en ceros = reposo normal, no atrofia.")
            except Exception as _e:
                logger.debug(f"Panel RNP (no crítico): {_e}")
        except Exception as e:
            resumen_mapa = ""
            informe_gestion = ""
            logger.debug(f"AutoGestor error (no crítico): {e}")

        # P2: cuello de botella de esta consulta (fase dominante en tiempo)
        try:
            _t_fin = _t.perf_counter()
            _fases = {"codificar": _t_encode - _t0, "propagar": _t_prop - _t_encode,
                      "actualizar": _t_upd - _t_prop, "mapa": _t_fin - _t_upd}
            _dom = max(_fases, key=_fases.get)
            _tot = sum(_fases.values()) + 1e-9
            self.ultimo_cuello = _dom if _fases[_dom] / _tot > 0.5 else ""
        except Exception:
            self.ultimo_cuello = ""

        estado_texto = (
            f"[ESTADO DE ACTIVACIÓN NEURONAL TOTAL EN CELEBRO]:\n"
            f"- Subsistemas sincronizados: {', '.join(subsistemas_activos)}\n"
            f"- Total de neuronas en propagación: {len(self.neuronas) + 1}\n"
            f"- Vector semántico de entrada: {vec_semantico}\n"
            f"- Activación sináptica compuesta: {activaciones}\n"
            f"- Variación global de pesos sinápticos (Δ): {norma_delta:.5f} ({estado_control})\n"
            f"- Tono cognitivo resultante: {tono} (valencia emocional: {estado_emocional:+.2f})\n"
            + (f"{resumen_mapa}\n" if resumen_mapa else "")
            + (f"{informe_gestion}\n" if informe_gestion else "")
            + f"- Instrucción cognitiva: Responde en función del estado de tus neuronas en Celebro."
        )

        info = {
            "timestamp": round(float(np.random.rand()), 4),
            "vector_semantico": vec_semantico,
            "norma_delta_pesos": round(norma_delta, 5),
            "norma_delta_aplicada": round(norma_aplicada, 5),
            "deriva_acumulada": round(float(getattr(self, "deriva_acumulada", 0.0)), 5),
            "media_pesos": round(media_pesos, 4),
            "std_pesos": round(std_pesos, 4),
            "estado_emocional": estado_emocional,
            "tono_cognitivo": tono,
            "estado_texto": estado_texto,
            "neuronas_activadas": list(self.neuronas.keys()) + ["SLRN_AdaptiveLayer"],
            "subsistemas": subsistemas_activos,
            "mapa_autogestion": mapa if 'mapa' in dir() else {},
        }

        self.historial_actualizaciones.append(info)
        logger.info(
            f"🧠 Celebro actualizado (TODAS las neuronas): delta={norma_delta:.5f} ({estado_control}) | "
            f"emocion={estado_emocional:+.3f} | neuronas={len(self.neuronas) + 1}"
        )
        return info

    def asimilar_respuestas_y_calcular_sintesis(self,
                                                prompt: str,
                                                respuesta_modelo: str,
                                                modelo_nombre: str) -> Dict[str, Any]:
        """
        Asimila la respuesta recibida de los modelos LLM (LM Studio / OpenRouter),
        la propaga a través de TODAS las neuronas de Celebro (ENRN, RF_SL, RF_EN, RNP, SLRN)
        y actualiza sus pesos sinápticos por aprendizaje asociativo.

        Micro-lotes (Fase B): la respuesta se parte en trozos de 400 caracteres
        con el factor repartido, para no generar el pico de entrada de golpe.
        """
        # 1. Vector semántico combinado consulta + conocimiento asimilado
        v_prompt = self.encoder.encode(prompt)
        texto_resp = (respuesta_modelo or "").strip() or " "
        trozos_resp = [texto_resp[i:i + 400] for i in range(0, len(texto_resp), 400)]
        factor_por_trozo = max(0.1, 1.2 / len(trozos_resp))
        norma_delta = 0.0
        norma_aplicada = 0.0
        estado_control = "Estable"
        for trozo in trozos_resp:
            v_modelo = self.encoder.encode(trozo)
            v_combinado = (v_prompt * 0.4 + v_modelo * 0.6)

            # 2. Paso hacia adelante simultáneo en todas las neuronas
            activacion_media, _ = self._propagar_todas_las_neuronas(v_combinado)

            # 3. Transmutación a pesos neuronales (factor repartido por trozo)
            norma_delta, norma_aplicada, estado_control = self._actualizar_pesos_en_todas_las_neuronas(v_combinado, activacion_media, factor=factor_por_trozo)

        # 4. Estado emocional y tono cognitivo
        media_pesos = float(np.mean(self.rf_sl_memoria.pesos))
        estado_emocional = round(float(np.tanh(media_pesos * 3.5 + float(v_combinado[0, 1]) - 0.3)), 3)

        if estado_emocional > 0.25:
            tono = "cálido, expresivo y receptivo"
        elif estado_emocional < -0.25:
            tono = "sereno, reflexivo y preciso"
        else:
            tono = "natural, elocuente y seguro"

        vec_semantico = [round(float(x), 4) for x in v_combinado[0]]

        # Ficha única compartida (llm_connector.FICHA_SISTEMA): la guía no duplica
        # el texto para no divergir; si falla la importación, usa el resumen local.
        try:
            from lucIA.llm_connector import FICHA_SISTEMA as _FICHA
        except Exception:
            _FICHA = ("ENRN=percepción/entrada (10 neuronas); RF_SL=memoria; RF_EN=decisiones; "
                      "RNP=RN11, RN12, RN13, RN14; SLRN=síntesis (LAMB/RAdam); pesos en IPFS.")
        guia_sintesis = (
            f"[PROCESAMIENTO COGNITIVO NEURONAL DE LucIA - CELEBRO COMPLETO]:\n"
            f"- Modelo consultado: {modelo_nombre}\n"
            f"- Redes neuronales involucradas: ENRN (10), RF_SL (3), RF_EN (3), RNP (3), SLRN\n"
            f"- Datos asimilados en Celebro (Δpesos total: {norma_delta:.5f} [{estado_control}], Emoción: {estado_emocional:+.2f})\n"
            f"- Ideas clave asimiladas: {respuesta_modelo[:200]}...\n"
            f"- Tono interno: {tono}\n"
            f"- {_FICHA}\n"
            f"- INSTRUCCIÓN OBLIGATORIA: LucIA, explica esto AHORA con tus PROPIAS PALABRAS, en primera persona, "
            f"con tu personalidad española cercana de 18 años, 100% en ESPAÑOL, sin repetir mecánicamente el modelo externo. "
            f"PROHIBIDO parafrasear o repetir literalmente las palabras de la pregunta del usuario: responde a lo que pregunta, "
            f"no devuelvas su propio texto con otra forma. "
            f"Si te piden describir tu sistema, usa SOLO esta ficha (no inventes módulos): ENRN=percepción/entrada (10 neuronas), "
            f"RF_SL=memoria, RF_EN=decisiones (DQN/ActorCritic/TD3), RNP=calibración (RN11 peso, RN12 ajuste, RN13 gradientes, RN14 puertas), "
            f"SLRN=síntesis y tono (LAMB/RAdam). LAMB y RAdam están en SLRN, NUNCA en RNP. "
            f"RNP son EXACTAMENTE estos cuatro códigos y ningún otro: RN11, RN12, RN13, RN14. "
            f"SL11 y SL12 pertenecen a SLRN, NUNCA a RNP. "
            f"TAMBIÉN tienes 15 módulos anatómicos propios y debes reconocerlos: Amigdala, Hipocampo, Talamo, "
            f"Corteza_sensorial, Corteza_prefrontal, Corteza_motora, Ganglios_basales, Cerebelo, Tronco_encefalico, "
            f"Glia, Hipotalamo, Cuerpo_calloso, Lobulo_temporal, Lobulo_occipital, Lobulo_parietal. "
            f"JAMÁS digas que no tienes Amigdala: vive en Celebro/Amigdala. Memoria persistente en IPFS. "
            f"Orden fijo: quién soy (1 frase) > Celebro por subsistemas > CORE una vez > IPFS+voz una frase. "
            f"Obligatorio nombrar ENRN (10 neuronas de percepción/entrada) como PRIMER subsistema de Celebro: "
            f"una descripción sin ENRN está incompleta. "
            f"Si la conversación se cortó antes, retoma el hilo sin re-presentarte."
        )

        info = {
            "vector_semantico": vec_semantico,
            "norma_delta_pesos": round(norma_delta, 5),
            "norma_delta_aplicada": round(norma_aplicada, 5),
            "deriva_acumulada": round(float(getattr(self, "deriva_acumulada", 0.0)), 5),
            "media_pesos": round(media_pesos, 4),
            "estado_emocional": estado_emocional,
            "tono_cognitivo": tono,
            "guia_sintesis": guia_sintesis,
            "modelo_origen": modelo_nombre,
            "total_neuronas": len(self.neuronas) + 1,
            "subsistemas": ["ENRN", "RF_SL", "RF_EN", "RNP", "SLRN"]
        }
        self.historial_actualizaciones.append(info)
        return info

    def actualizar_memoria_salida(self, prompt: str, respuesta: str, modelo_origen: str = "modelo") -> None:
        """Retroalimenta la memoria sináptica de salida en todos los subsistemas.

        Micro-lotes: las respuestas largas se parten en trozos de 400 caracteres
        y el factor se reparte entre ellos, para no generar picos de gradiente
        de golpe (era lo que Lucía sentía como "peso").
        """
        texto = (respuesta or "").strip()
        if not texto:
            return
        trozos = [texto[i:i + 400] for i in range(0, len(texto), 400)]
        factor_por_trozo = max(0.1, 0.5 / len(trozos))
        for trozo in trozos:
            v_pair = self.encoder.encode_pair(prompt, trozo)
            act_media, _ = self._propagar_todas_las_neuronas(v_pair)
            self._actualizar_pesos_en_todas_las_neuronas(v_pair, act_media, factor=factor_por_trozo)

    def convertir_respuesta_a_pesos(self,
                                    prompt: str,
                                    respuesta: str,
                                    modelo_origen: str = "modelo") -> Dict[str, Any]:
        """Compatibilidad con pipeline anterior."""
        info = self.procesar_consulta_a_pesos(prompt)
        self.actualizar_memoria_salida(prompt, respuesta, modelo_origen)
        info["modelo_origen"] = modelo_origen
        return info

    def verificar_anclaje(self, respuesta: str, fuentes: str) -> Dict[str, Any]:
        """Anti-alucinacion: todo numero/codigo de la respuesta debe existir
        literalmente en las fuentes (guia_sintesis + estado_texto).
        Retorna {"ok": bool, "huerfanos": [...]}."""
        import re
        if not respuesta:
            return {"ok": False, "huerfanos": ["respuesta_vacia"]}
        fuente = fuentes or ""
        huerfanos = []
        for pat in (r"\b(?:EN|RF_SL|RF_EN|RN|SL)\d+[A-Za-z_]*\b",
                    r"\b\d+\.\d+\b", r"\(\d+\)"):
            for m in set(re.findall(pat, respuesta)):
                if m not in fuente and m.strip("()") not in fuente:
                    huerfanos.append(m)
        return {"ok": not huerfanos, "huerfanos": sorted(set(huerfanos))}

    def exportar_pesos_celebro(self) -> Dict[str, Any]:
        """Exporta el estado sináptico actual de todas las neuronas de Celebro."""
        estado = {}
        for nombre, neurona in self.neuronas.items():
            pesos_data = None
            if hasattr(neurona, "pesos") and neurona.pesos is not None:
                pesos_data = neurona.pesos.tolist()
            elif hasattr(neurona, "pesos_principal") and neurona.pesos_principal is not None:
                pesos_data = neurona.pesos_principal.tolist()
            elif hasattr(neurona, "pesos_actor") and neurona.pesos_actor is not None:
                pesos_data = neurona.pesos_actor.tolist()

            sesgo_data = None
            if hasattr(neurona, "sesgo") and neurona.sesgo is not None:
                sesgo_data = neurona.sesgo.tolist()
            elif hasattr(neurona, "sesgo_principal") and neurona.sesgo_principal is not None:
                sesgo_data = neurona.sesgo_principal.tolist()
            elif hasattr(neurona, "sesgo_actor") and neurona.sesgo_actor is not None:
                sesgo_data = neurona.sesgo_actor.tolist()

            estado[nombre] = {
                "pesos": pesos_data,
                "sesgo": sesgo_data,
                "input_size": getattr(neurona, "input_size", 4),
                "output_size": getattr(neurona, "output_size", 8)
            }

        # Incluir SLRN
        estado["SLRN_AdaptiveLayer"] = {
            "pesos": self.slrn_pesos.tolist(),
            "sesgo": self.slrn_sesgo.tolist(),
            "input_size": 4,
            "output_size": 8
        }
        return estado

    def guardar_pesos_en_celebro(self, archivo: Optional[Path] = None) -> Path:
        """Guarda un checkpoint local en formato comprimido .npz con todas las neuronas de Celebro."""
        destino = archivo or (CELEBRO_DIR / "pesos_activos.npz")
        dict_pesos = {}

        for nombre, n in self.neuronas.items():
            if hasattr(n, "pesos") and n.pesos is not None:
                dict_pesos[f"{nombre}_pesos"] = n.pesos
            elif hasattr(n, "pesos_principal") and n.pesos_principal is not None:
                dict_pesos[f"{nombre}_pesos"] = n.pesos_principal
            elif hasattr(n, "pesos_actor") and n.pesos_actor is not None:
                dict_pesos[f"{nombre}_pesos"] = n.pesos_actor

        dict_pesos["SLRN_AdaptiveLayer_pesos"] = self.slrn_pesos
        dict_pesos["RN14_PuertasAtencion_pesos"] = self.rnp_14.pesos
        # RN14: persistir puertas de atención (nombres + valores float16: ~100 bytes)
        try:
            _exp = self.rnp_14.exportar()
            dict_pesos["puertas_nombres"] = np.array(_exp["nombres"])
            dict_pesos["puertas_valores"] = np.array(_exp["valores"], dtype=np.float16)
        except Exception as e:
            logger.debug(f"RN14 persistir (no crítico): {e}")
        np.savez(destino, **dict_pesos)
        return destino

    def cargar_pesos_en_celebro(self, origen: Union[str, Path, Dict[str, Any], bytes]) -> Dict[str, Any]:
        """
        Restaura el estado sináptico de todas las neuronas de Celebro desde:
        - Un archivo .npz
        - Un diccionario con matrices numpy o listas
        - Un stream de bytes .npz descargado directamente desde IPFS
        """
        import io
        cargados = 0
        arrays_dict: Dict[str, Any] = {}

        try:
            if isinstance(origen, (str, Path)):
                p = Path(origen)
                if p.exists():
                    loaded = np.load(p, allow_pickle=True)
                    arrays_dict = {k: loaded[k] for k in loaded.files}
            elif isinstance(origen, bytes):
                bio = io.BytesIO(origen)
                loaded = np.load(bio, allow_pickle=True)
                arrays_dict = {k: loaded[k] for k in loaded.files}
            elif isinstance(origen, dict):
                arrays_dict = origen

            if not arrays_dict:
                logger.warning("No se encontraron arrays válidos para cargar en Celebro.")
                return {"exito": False, "neuronas_restauradas": 0}

            # Restaurar neuronas individuales
            for nombre, neurona in self.neuronas.items():
                clave_peso = f"{nombre}_pesos"
                if clave_peso in arrays_dict:
                    val = arrays_dict[clave_peso]
                    arr = np.array(val, dtype=np.float32)
                    if hasattr(neurona, "pesos"):
                        neurona.pesos = arr
                        cargados += 1
                    elif hasattr(neurona, "pesos_principal"):
                        neurona.pesos_principal = arr
                        cargados += 1
                    elif hasattr(neurona, "pesos_actor"):
                        neurona.pesos_actor = arr
                        cargados += 1

            # Restaurar SLRN
            if "SLRN_AdaptiveLayer_pesos" in arrays_dict:
                self.slrn_pesos = np.array(arrays_dict["SLRN_AdaptiveLayer_pesos"], dtype=np.float32)
                cargados += 1

            # Restaurar matriz propia de RN14
            if "RN14_PuertasAtencion_pesos" in arrays_dict:
                self.rnp_14.pesos = np.array(arrays_dict["RN14_PuertasAtencion_pesos"], dtype=np.float32)
                cargados += 1

            # Restaurar puertas RN14 si el checkpoint las trae
            if "puertas_nombres" in arrays_dict and "puertas_valores" in arrays_dict:
                try:
                    _n = [str(x) for x in np.atleast_1d(arrays_dict["puertas_nombres"]).tolist()]
                    _v = [float(x) for x in np.atleast_1d(arrays_dict["puertas_valores"]).tolist()]
                    self.rnp_14.importar({"nombres": _n, "valores": _v})
                except Exception as e:
                    logger.debug(f"RN14 restaurar (no crítico): {e}")

            logger.info(f"🧠 Restaurados con éxito {cargados} componentes sinápticos en Celebro.")
            return {"exito": True, "neuronas_restauradas": cargados}

        except Exception as e:
            logger.error(f"Error cargando pesos en Celebro: {e}")
            return {"exito": False, "error": str(e), "neuronas_restauradas": cargados}



# Instancia singleton del conversor
_conversor_instance: Optional[ConversorRespuestaPesos] = None


def get_conversor_pesos() -> ConversorRespuestaPesos:
    global _conversor_instance
    if _conversor_instance is None:
        _conversor_instance = ConversorRespuestaPesos()
    return _conversor_instance
