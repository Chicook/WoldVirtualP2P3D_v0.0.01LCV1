"""
PSNRCV.py - Conversor de Respuestas y Consultas a Pesos Neuronales (Arquitectura 2026)
=====================================================================================
Orquesta las 50 neuronas de los 5 subsistemas de WoldVirtualP2P3D_v0.0.01LCV1_DEVPYMD:
1. ENRN (10 neuronas): Percepcion y entrada sensorial (Basica a Sigmoid).
2. RF_SL (10 neuronas): Memoria y optimizacion supervisada (GBM, SVM, RF, NN, DT, NB, KNN, etc.).
3. RF_EN (10 neuronas): Aprendizaje por refuerzo (QLearning, PG, AC, DQN, A3C, PPO, SAC, TD3, etc.).
4. RNP (10 neuronas): Optimizacion de pesos neuronales (AdamW, RAdam, Lookahead, SAM, SWATS, etc.).
5. SLRN (10 neuronas): Optimizadores supervisados 2026 (Backpropagation, SGD, RMSprop, Adam, etc.).
6. PSNRL: Persistencia activa en tiempo real de pesos durante la sesion.
"""
from __future__ import annotations
import os, sys, json, time, math, logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np

CURRENT_FILE = Path(__file__).resolve()
CMFG_DIR = CURRENT_FILE.parent
CELEBRO_DIR = CMFG_DIR.parent
ROOT_DIR = CELEBRO_DIR.parent.parent
PSNRL_DIR = CELEBRO_DIR / "PSNRL"
PSNRL_DIR.mkdir(parents=True, exist_ok=True)

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# 1. ENRN (10 neuronas)
import LC.celebro.red_neuronal.ENRN as enrn_pkg

# 2. RF_SL (10 neuronas)
import LC.celebro.red_neuronal.RF_SL.RFSL1 as rfsl_pkg

# 3. RF_EN (10 neuronas)
import LC.celebro.red_neuronal.RF_EN as rfen_pkg

# 4. RNP (10 neuronas)
import LC.celebro.red_neuronal.RNP as rnp_pkg

# 5. SLRN (10 neuronas)
import LC.celebro.red_neuronal.SLRN as slrn_pkg
from LC.celebro.red_neuronal.SLRN import SupervisedLearningNeuralConfig

logger = logging.getLogger("WoldVirtualP2P3D.PSNRCV")

from LC.celebro.CMFG.neural_math import (
    NeuralMathPrecision2026,
    SemanticEncoderSHA,
    SemanticEncoder4D,
)


# ===========================================================================
# CLASE PRINCIPAL: CONVERSOR RESPUESTA PESOS (PSNRCV) - 50 NEURONAS
# ===========================================================================
class ConversorRespuestaPesos:
    """
    Conversor y orquestador neuronal integral para WoldVirtualP2P3D.
    Integra las 50 neuronas de Celebro (ENRN, RF_SL, RF_EN, RNP, SLRN) y persiste en PSNRL.
    """

    def __init__(self, learning_rate: float = 0.01, session_id: Optional[str] = None):
        self.learning_rate = float(learning_rate)
        self.session_id = session_id or time.strftime("%Y%m%d_%H%M%S")
        self.encoder = SemanticEncoder4D()
        self.historial_actualizaciones: List[Dict[str, Any]] = []
        self.deriva_acumulada: float = 0.0
        self.pasos_sesion: int = 0
        self._ema_vector: Optional[np.ndarray] = None

        # 1. ENRN: 10 neuronas de entrada
        en_clases = [
            enrn_pkg.NeuronaEntradaBasica, enrn_pkg.NeuronaEntradaXavier, enrn_pkg.NeuronaEntradaLeCun,
            enrn_pkg.NeuronaEntradaNormalizada, enrn_pkg.NeuronaEntradaRegularizada, enrn_pkg.NeuronaEntradaDropout,
            enrn_pkg.NeuronaEntradaBatchNorm, enrn_pkg.NeuronaEntradaReLU, enrn_pkg.NeuronaEntradaTanh,
            enrn_pkg.NeuronaEntradaSigmoid
        ]
        self.enrn_neuronas = [cls(4, 8, nombre=f"EN{i+1}_{cls.__name__[14:]}") for i, cls in enumerate(en_clases)]
        for n in self.enrn_neuronas: n.inicializar_pesos()

        # 2. RF_SL: 10 neuronas de memoria supervisada
        rfsl_clases = [
            rfsl_pkg.GradientBoostingOptimizer, rfsl_pkg.SupportVectorMachineOptimizer,
            rfsl_pkg.RandomForestOptimizer, rfsl_pkg.NeuralNetworkOptimizer,
            rfsl_pkg.DecisionTreeOptimizer, rfsl_pkg.NaiveBayesOptimizer,
            rfsl_pkg.KNearestNeighborsOptimizer, rfsl_pkg.LogisticRegressionOptimizer,
            rfsl_pkg.LinearRegressionOptimizer, rfsl_pkg.IntegratedSupervisedLearningOptimizer
        ]
        self.rf_sl_neuronas = [cls(4, 8, nombre=f"RFSL{i+1}_{cls.__name__}") for i, cls in enumerate(rfsl_clases)]
        for n in self.rf_sl_neuronas: n.inicializar_pesos()

        # 3. RF_EN: 10 neuronas de aprendizaje por refuerzo
        rfen_clases = [
            rfen_pkg.NeuronaRefuerzoQLearning, rfen_pkg.NeuronaRefuerzoPolicyGradient,
            rfen_pkg.NeuronaRefuerzoActorCritic, rfen_pkg.NeuronaRefuerzoDQN,
            rfen_pkg.NeuronaRefuerzoA3C, rfen_pkg.NeuronaRefuerzoPPO,
            rfen_pkg.NeuronaRefuerzoSAC, rfen_pkg.NeuronaRefuerzoTD3,
            rfen_pkg.NeuronaRefuerzoRainbowDQN, rfen_pkg.NeuronaRefuerzoIMPALA
        ]
        self.rf_en_neuronas = [cls(4, 8, nombre=f"RFEN{i+1}_{cls.__name__[15:]}") for i, cls in enumerate(rfen_clases)]
        for n in self.rf_en_neuronas: n.inicializar_pesos()

        # 4. RNP: 10 neuronas de optimizacion de pesos
        self.rnp_neuronas = []
        import importlib as _il
        for i, (k, (mod, cls, _, _)) in enumerate(rnp_pkg._MODULE_MAP.items(), 1):
            try:
                m = _il.import_module(f"LC.celebro.red_neuronal.RNP.{mod}")
            except Exception as exc:
                logger.warning("RNP %s no cargado: %s", mod, exc)
                continue
            c, cfg_cls = getattr(m, cls), getattr(m, "NeuralWeightOptimizationConfig")
            inst = c(cfg_cls(learning_rate=self.learning_rate))
            inst.nombre = f"RNP{i}_{k.upper()}"
            inst.pesos = np.random.randn(4, 8).astype(np.float32) * 0.05
            inst.sesgo = np.zeros((1, 8), dtype=np.float32)
            self.rnp_neuronas.append(inst)

        # 5. SLRN: 10 neuronas optimizadoras supervisadas 2026
        cfg_slrn = SupervisedLearningNeuralConfig(learning_rate=self.learning_rate)
        self.slrn_neuronas = []
        for i, (k, mod) in enumerate(slrn_pkg.OPTIMIZER_REGISTRY.items(), 1):
            try:
                m = _il.import_module(f"LC.celebro.red_neuronal.SLRN.{mod}")
            except Exception as exc:
                logger.warning("SLRN %s no cargado: %s", mod, exc)
                continue
            for a in dir(m):
                if a.endswith("Optimizer") and "Base" not in a and "Internal" not in a:
                    inst = getattr(m, a)(cfg_slrn)
                    inst.nombre = f"SLRN{i}_{k.upper()}"
                    inst.pesos = np.random.randn(4, 8).astype(np.float32) * 0.05
                    inst.sesgo = np.zeros((1, 8), dtype=np.float32)
                    self.slrn_neuronas.append(inst)
                    break

        # Matrices de curvatura SOAP (4x4 y 8x8)
        self._soap_L = np.eye(4, dtype=np.float32) * 1e-4
        self._soap_R = np.eye(8, dtype=np.float32) * 1e-4
        self._grad_sum = np.zeros((4, 8), dtype=np.float32)
        self._grad_sq_sum = np.zeros((4, 8), dtype=np.float32)

        # Registro de la totalidad de las 50 neuronas
        self.neuronas: Dict[str, Any] = {}
        for n in self.enrn_neuronas + self.rf_sl_neuronas + self.rf_en_neuronas + self.rnp_neuronas + self.slrn_neuronas:
            self.neuronas[n.nombre] = n
        logger.info("ConversorRespuestaPesos 2026 inicializado | 50 neuronas activas en 5 subsistemas.")

    def _propagar_todas_las_neuronas(self, vector_4d: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Propaga a traves de las 50 neuronas con filtrado EMA y acoplamiento sinaptico."""
        activaciones: Dict[str, np.ndarray] = {}
        salidas = []

        alpha = 0.85
        if self._ema_vector is None: self._ema_vector = vector_4d.copy()
        self._ema_vector = alpha * vector_4d + (1.0 - alpha) * self._ema_vector
        v_in = self._ema_vector

        # 1. ENRN (10 neuronas)
        for n in self.enrn_neuronas:
            try:
                out = NeuralMathPrecision2026.to_8d(n.forward(v_in))
                activaciones[n.nombre] = out; salidas.append(out)
            except Exception as e: logger.debug("Forward %s: %s", n.nombre, e)

        # 2. RF_SL (10 neuronas)
        salidas_sl = []
        for n in self.rf_sl_neuronas:
            try:
                out = NeuralMathPrecision2026.to_8d(n.forward(v_in))
                activaciones[n.nombre] = out; salidas.append(out); salidas_sl.append(out)
            except Exception as e: logger.debug("Forward %s: %s", n.nombre, e)

        # Modulacion de memoria a refuerzo
        mod = np.tanh(np.mean(salidas_sl, axis=0)[0, :4]).reshape(1, 4) * 0.3 if salidas_sl else 0.0
        v_mod = np.clip(v_in + mod, -5.0, 5.0)

        # 3. RF_EN (10 neuronas)
        for n in self.rf_en_neuronas:
            try:
                if hasattr(n, "valores_q"): raw = n.valores_q(v_mod)
                elif hasattr(n, "forward_actor"): raw = n.forward_actor(v_mod)
                elif hasattr(n, "q_table"): raw = v_mod @ n.q_table
                else: raw = n.forward(v_mod)
                out = NeuralMathPrecision2026.to_8d(raw)
                activaciones[n.nombre] = out; salidas.append(out)
            except Exception as e: logger.debug("Forward %s: %s", n.nombre, e)

        # 4 & 5. RNP (10) y SLRN (10) - Capas sinapticas de optimizacion
        for n in self.rnp_neuronas + self.slrn_neuronas:
            try:
                out = NeuralMathPrecision2026.to_8d(np.dot(v_in, n.pesos) + n.sesgo)
                activaciones[n.nombre] = out; salidas.append(out)
            except Exception as e: logger.debug("Forward %s: %s", n.nombre, e)

        act_media = np.mean(salidas, axis=0) if salidas else np.zeros((1, 8), dtype=np.float32)
        return act_media, activaciones

    def _actualizar_pesos_en_todas_las_neuronas(self, vector_4d: np.ndarray, activacion_8d: np.ndarray,
                                               factor: float = 1.0) -> Tuple[float, float, str]:
        """Conversion Hebbiana/gradiente con Muon NS-5, SOAP 2026 y Trust-Ratio a las 50 neuronas."""
        self.pasos_sesion += 1
        t = self.pasos_sesion
        delta_base = np.dot(vector_4d.T, activacion_8d).astype(np.float32) * (self.learning_rate * factor)
        norma_pre = float(np.linalg.norm(delta_base))

        # Muon NS-5 + SOAP Preconditioner
        delta_muon = NeuralMathPrecision2026.newton_schulz5(delta_base, steps=5)
        try: delta_soap = NeuralMathPrecision2026.soap_precondition(delta_muon, self._soap_L, self._soap_R)
        except Exception: delta_soap = delta_muon

        self._grad_sum += delta_soap; self._grad_sq_sum += delta_soap ** 2
        score_gsnr = NeuralMathPrecision2026.gsnr(self._grad_sum, self._grad_sq_sum, t)
        # Trust-ratio por-neurona: cada parametro escala su propio delta ajustado a su forma
        norma_aplicada = 0.0
        sesgo_delta = np.mean(delta_soap, axis=0, keepdims=True)
        for n in self.neuronas.values():
            for attr in ["pesos", "sesgo", "q_online", "q_objetivo", "pesos_actor", "pesos_critic", "q_table"]:
                w = getattr(n, attr, None)
                if isinstance(w, np.ndarray):
                    d = sesgo_delta if "sesgo" in attr else delta_soap
                    d_fit = NeuralMathPrecision2026.ajustar_forma(d, w.shape)
                    d_clip = NeuralMathPrecision2026.trust_ratio_clip(d_fit, w, clip=5.0)
                    setattr(n, attr, np.clip(w + d_fit * 0 + d_clip, -10.0, 10.0))
                    norma_aplicada += float(np.linalg.norm(d_clip))
        norma_aplicada = float(norma_aplicada / max(1, len(self.neuronas)))
        self.deriva_acumulada += norma_aplicada

        return norma_pre, norma_aplicada, f"Muon-NS5+SOAP (GSNR={score_gsnr:.2f})"

    def procesar_consulta_a_pesos(self, prompt: str) -> Dict[str, Any]:
        """Procesa una consulta, propaga por las 50 neuronas, actualiza pesos y persiste en PSNRL."""
        t0 = time.perf_counter()
        v_prompt = self.encoder.encode(prompt)
        act_media, _ = self._propagar_todas_las_neuronas(v_prompt)
        norma_pre, norma_app, estado = self._actualizar_pesos_en_todas_las_neuronas(v_prompt, act_media, factor=1.0)

        emocion = round(float(np.tanh(float(np.mean(self.enrn_neuronas[0].pesos)) * 3.5 + float(v_prompt[0, 1]))), 3)
        tono = "cálido y receptivo" if emocion > 0.2 else ("sereno y reflexivo" if emocion < -0.2 else "analítico y seguro")

        info = {
            "timestamp": time.time(), "prompt": prompt[:100],
            "vector_semantico": [round(float(x), 4) for x in v_prompt[0]],
            "norma_delta_pesos": round(norma_pre, 5), "norma_delta_aplicada": round(norma_app, 5),
            "deriva_acumulada": round(self.deriva_acumulada, 5), "estado_emocional": emocion,
            "tono_cognitivo": tono, "estado_control": estado, "total_neuronas": len(self.neuronas),
            "duracion_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }
        self.historial_actualizaciones.append(info)
        self.persistir_pesos_en_psnrl()
        return info

    def asimilar_respuestas_y_calcular_sintesis(self, prompt: str, respuesta_modelo: str,
                                                modelo_nombre: str) -> Dict[str, Any]:
        """Asimila respuestas en micro-lotes para suavizar el gradiente y actualiza pesos en PSNRL."""
        v_comb = self.encoder.encode_pair(prompt, respuesta_modelo)
        trozos = [respuesta_modelo[i:i + 400] for i in range(0, len(respuesta_modelo or " "), 400)] or [" "]
        factor_lote = max(0.1, 1.0 / len(trozos))

        norma_pre, norma_app, estado = 0.0, 0.0, "Estable"
        for trozo in trozos:
            v_sub = self.encoder.encode_pair(prompt, trozo)
            act_media, _ = self._propagar_todas_las_neuronas(v_sub)
            norma_pre, norma_app, estado = self._actualizar_pesos_en_todas_las_neuronas(v_sub, act_media, factor=factor_lote)

        emocion = round(float(np.tanh(float(np.mean(self.enrn_neuronas[0].pesos)) * 3.0 + float(v_comb[0, 1]))), 3)
        tono = "cálido y comunicativo" if emocion > 0.2 else ("sereno y analítico" if emocion < -0.2 else "preciso y formal")

        guia_sintesis = (
            f"[PROCESAMIENTO COGNITIVO NEURONAL 2026 - 50 NEURONAS]:\n"
            f"- Modelo: {modelo_nombre} | ENRN(10), RF_SL(10), RF_EN(10), RNP(10), SLRN(10)\n"
            f"- Variación sináptica: Δ={norma_app:.5f} ({estado}) | Emoción: {emocion:+.2f}\n"
            f"- Guía: Explica con tus propias palabras en español e integra la síntesis de tus subsistemas."
        )

        info = {
            "modelo_origen": modelo_nombre, "vector_semantico": [round(float(x), 4) for x in v_comb[0]],
            "norma_delta_pesos": round(norma_pre, 5), "norma_delta_aplicada": round(norma_app, 5),
            "deriva_acumulada": round(self.deriva_acumulada, 5), "estado_emocional": emocion,
            "tono_cognitivo": tono, "guia_sintesis": guia_sintesis, "total_neuronas": len(self.neuronas),
        }
        self.historial_actualizaciones.append(info)
        self.persistir_pesos_en_psnrl()
        return info

    def persistir_pesos_en_psnrl(self, etiqueta: Optional[str] = None) -> Tuple[Path, Path]:
        """Persiste el estado de las 50 neuronas en PSNRL (.npz y .json)."""
        tag = etiqueta or f"sesion_{self.session_id}"
        npz_path, json_path = PSNRL_DIR / f"{tag}_pesos.npz", PSNRL_DIR / f"{tag}_metadata.json"
        dict_pesos: Dict[str, np.ndarray] = {}
        metadata: Dict[str, Any] = {
            "session_id": self.session_id, "pasos_sesion": self.pasos_sesion,
            "deriva_acumulada": round(self.deriva_acumulada, 6), "timestamp": time.time(),
            "total_neuronas": len(self.neuronas), "neuronas": {},
        }

        for nombre, n in self.neuronas.items():
            arr = None
            for attr in ["pesos", "q_online", "pesos_actor", "q_table", "pesos_actor_global", "pesos_q_principal"]:
                v = getattr(n, attr, None)
                if isinstance(v, np.ndarray) and v.size > 0:
                    arr = v
                    break
            if arr is not None:
                dict_pesos[f"{nombre}_pesos"] = arr.astype(np.float32)
                metadata["neuronas"][nombre] = {
                    "shape": list(arr.shape), "norma": float(np.linalg.norm(arr)),
                    "media": float(np.mean(arr)), "std": float(np.std(arr)),
                }

        np.savez_compressed(npz_path, **dict_pesos)
        with open(json_path, "w", encoding="utf-8") as f: json.dump(metadata, f, indent=2)
        np.savez_compressed(PSNRL_DIR / "pesos_activos.npz", **dict_pesos)
        with open(PSNRL_DIR / "pesos_activos.json", "w", encoding="utf-8") as f: json.dump(metadata, f, indent=2)

        logger.info("Pesos de 50 neuronas persistidos en PSNRL: %s", npz_path.name)
        return npz_path, json_path

    def cargar_pesos_de_psnrl(self, archivo: Optional[Union[str, Path]] = None) -> int:
        """Carga y sincroniza pesos neuronales de las 50 neuronas desde PSNRL."""
        ruta = Path(archivo) if archivo else (PSNRL_DIR / "pesos_activos.npz")
        if not ruta.exists(): return 0
        cargados, data = 0, np.load(ruta, allow_pickle=True)
        for nombre, n in self.neuronas.items():
            for attr in ["pesos", "q_online", "pesos_actor", "q_table"]:
                clave = f"{nombre}_{attr}"
                if clave in data and hasattr(n, attr):
                    setattr(n, attr, data[clave].astype(np.float32)); cargados += 1
        return cargados


# ===========================================================================
# SINGLETON DE ACCESO GLOBAL
# ===========================================================================
_conversor_global: Optional[ConversorRespuestaPesos] = None


def get_conversor_pesos(learning_rate: float = 0.01) -> ConversorRespuestaPesos:
    """Retorna la instancia global del conversor neuronal de pesos."""
    global _conversor_global
    if _conversor_global is None:
        _conversor_global = ConversorRespuestaPesos(learning_rate=learning_rate)
    return _conversor_global


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Iniciando prueba unitaria de PSNRCV 2026 con 50 neuronas...")
    conversor = get_conversor_pesos()
    res1 = conversor.procesar_consulta_a_pesos("Hola LucIA, ¿cómo funciona tu red neuronal completa?")
    print("Consulta procesada. Delta aplicada:", res1["norma_delta_aplicada"])
    res2 = conversor.asimilar_respuestas_y_calcular_sintesis(
        "Hola LucIA", "Las 50 neuronas en ENRN, RF_SL, RF_EN, RNP y SLRN están sincronizadas.", "qwen2.5:7b"
    )
    print("Respuesta asimilada. Deriva acumulada:", res2["deriva_acumulada"])
    archivos = list(PSNRL_DIR.glob("*.*"))
    print(f"Archivos verificados en PSNRL ({len(archivos)}): {[f.name for f in archivos]}")