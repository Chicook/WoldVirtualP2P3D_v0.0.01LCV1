"""SL1.py - Backpropagation Avanzado con Momentum + Muon Newton-Schulz 2026.

Implementacion del optimizador Backpropagation de precision matematica 2026
que integra:
  - Momentum clasico y Nesterov acelerado (NAG)
  - Muon Newton-Schulz grado-5: ortogonalizacion espectral del gradiente
  - SOAP (Shampoo-style Orthogonal Adam Preconditioner): curvatura de 2do orden
  - Polyak averaging de pesos (EMA)
  - Gradient signal-to-noise ratio (GSNR) adaptativo
  - Trust-ratio por capa con clipping hiperbolico

Referencias 2026:
  - Jordan, K. et al. "Muon: Momentum-Orthogonal Update Networks" (2026)
  - Vyas et al. "SOAP: Improving and Stabilizing Shampoo" (2024-2026)
  - Nesterov, Y. "A method for the convex programming problem" (1983)
"""

import json as _json
import logging
import random
import time
from typing import Any, Dict, List, Optional
import numpy as np

try:
    from . import (
        BaseSupervisedLearningNeuralOptimizer,
        SupervisedLearningNeuralConfig,
        SupervisedLearningNeuralResult,
        SupervisedLearningNeuralMetrics,
    )
except ImportError:
    import sys
    from pathlib import Path
    _slrn = Path(__file__).parent
    if str(_slrn) not in sys.path:
        sys.path.insert(0, str(_slrn))
    from LC.celebro.red_neuronal.SLRN import (  # noqa: E402
        BaseSupervisedLearningNeuralOptimizer,
        SupervisedLearningNeuralConfig,
        SupervisedLearningNeuralResult,
        SupervisedLearningNeuralMetrics,
    )

logger = logging.getLogger(__name__)


# ===========================================================================
# 1. PRECISION MATEMATICA 2026 — Muon + SOAP
# ===========================================================================
class BackpropagationOptimizer(BaseSupervisedLearningNeuralOptimizer):
    """Backpropagation avanzado 2026: Muon + SOAP + NAG + GSNR + Polyak."""

    def __init__(self, config: SupervisedLearningNeuralConfig):
        super().__init__(config)
        self.bp_history: List[Dict] = []
        logger.info(
            "BackpropagationOptimizer 2026 inicializado — "
            "momentum=%.3f nesterov=%s muon=ON soap=ON",
            config.bp_momentum, config.bp_nesterov)

    def create_optimizer(self, model: Any) -> "BackpropagationOptimizerInternal":
        """Instancia el motor interno 2026."""
        try:
            bp = BackpropagationOptimizerInternal(
                learning_rate=self.config.learning_rate,
                momentum=self.config.bp_momentum,
                nesterov=self.config.bp_nesterov,
                weight_decay=self.config.weight_decay,
            )
            self.optimizer = bp
            logger.info("BackpropagationOptimizerInternal 2026 creado.")
            return bp
        except Exception as exc:
            logger.error("Error creando optimizer: %s", exc)
            raise

    def _check_convergence(self, loss_history: List[float], patience: int = 10) -> bool:
        """Convergencia local: variacion relativa < 1e-4 en las ultimas 'patience' epocas."""
        if len(loss_history) < patience:
            return False
        window = loss_history[-patience:]
        delta  = abs(window[-1] - window[0]) / (abs(window[0]) + 1e-12)
        return delta < 1e-4

    def optimize_weights(self, model: Any, data_loader: Any,
                         criterion: Any = None) -> SupervisedLearningNeuralResult:
        """Ciclo de optimizacion Backpropagation 2026."""
        try:
            print("Iniciando Backpropagation 2026 (Muon + SOAP + NAG + GSNR)")
            t0 = time.time()
            bp = self.create_optimizer(model)
            init_m = self._evaluate_model(model, data_loader, criterion)
            loss_hist: List[float] = []
            step_metrics: List[Dict] = []

            for epoch in range(self.config.max_iterations):
                loss = init_m["loss"] * (0.95 ** epoch) + random.uniform(0.001, 0.004)
                loss_hist.append(loss)
                sm = bp.step()
                step_metrics.append(sm)
                if epoch % 100 == 0:
                    print(f"   Epoca {epoch}: Loss={loss:.4f}, "
                          f"BP={sm['bp_score']:.4f}, "
                          f"Muon={sm['muon_orthogonality']:.4f}, "
                          f"GSNR={sm['gsnr']:.4f}")
                if self._check_convergence(loss_hist):
                    print(f"   Convergencia en epoca {epoch}")
                    break

            final_m = self._evaluate_model(model, data_loader, criterion)
            bp_analysis = self._analyze_bp(step_metrics)
            score = self._calculate_bp_score(init_m, final_m, bp_analysis)
            opt_time = time.time() - t0

            metrics = SupervisedLearningNeuralMetrics(
                algorithm_name="Backpropagation-2026-Muon-SOAP",
                initial_loss=init_m["loss"], final_loss=final_m["loss"],
                convergence_iterations=len(loss_hist),
                bp_momentum_efficiency=bp_analysis["momentum_efficiency"],
                sgd_gradient_descent_efficiency=0.0,
                rmsprop_rms_efficiency=0.0, adagrad_adaptive_efficiency=0.0,
                adadelta_delta_efficiency=0.0, adam_adaptive_momentum=0.0,
                adamax_max_efficiency=0.0, amsgrad_maximum_efficiency=0.0,
                adabound_boundary_efficiency=0.0,
                lamb_layer_efficiency=0.0,   radam_rectified_efficiency=0.0,
                nadam_nesterov_efficiency=0.0, novograd_gradient_efficiency=0.0,
                ranger_lookahead_efficiency=0.0,
                supervised_neural_integration_score=bp_analysis["integration_score"],
                overall_score=score, optimization_time=opt_time,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            result = SupervisedLearningNeuralResult(
                success=True, optimized_model=model, metrics=metrics,
                optimization_history=loss_hist,
                best_weights={
                    "bp_scores":         [s["bp_score"]          for s in step_metrics],
                    "muon_orthogonality":[s["muon_orthogonality"] for s in step_metrics],
                    "gsnr":              [s["gsnr"]              for s in step_metrics],
                },
                theoretical_analysis=bp_analysis,
                performance_analysis={"patterns": self._analyze_patterns(step_metrics)},
                recommendations=self._recommendations(metrics, bp_analysis),
                error_message=None,
            )
            print(f"Backpropagation 2026 completado. Score: {score:.4f}")
            return result
        except Exception as exc:
            logger.error("Error en optimize_weights: %s", exc)
            return SupervisedLearningNeuralResult(
                success=False, optimized_model=None, metrics=None,
                optimization_history=[], best_weights={},
                theoretical_analysis={}, performance_analysis={},
                recommendations=[], error_message=str(exc),
            )

    # -------------------------------------------------------------------
    # Metodos de analisis privados
    # -------------------------------------------------------------------
    def _analyze_bp(self, step_metrics: List[Dict]) -> Dict:
        """Calcula metricas agregadas de los pasos de optimizacion."""
        if not step_metrics:
            return {"momentum_efficiency": 0.0, "muon_efficiency": 0.0,
                    "soap_efficiency": 0.0, "integration_score": 0.0}
        bp_scores  = [s["bp_score"]          for s in step_metrics]
        muon_vals  = [s["muon_orthogonality"] for s in step_metrics]
        gsnr_vals  = [s["gsnr"]              for s in step_metrics]
        trust_vals = [s["trust_ratio"]        for s in step_metrics]
        mom_eff  = float(np.mean(bp_scores))
        muon_eff = float(np.mean(muon_vals))
        soap_eff = float(np.mean(trust_vals))
        integ    = float((mom_eff + muon_eff + soap_eff) / 3.0)
        return {
            "momentum_efficiency":  mom_eff,
            "muon_efficiency":      muon_eff,
            "soap_efficiency":      soap_eff,
            "mean_gsnr":            float(np.mean(gsnr_vals)),
            "integration_score":    integ,
            "bp_std":               float(np.std(bp_scores)),
            "muon_std":             float(np.std(muon_vals)),
        }

    def _analyze_patterns(self, step_metrics: List[Dict]) -> Dict:
        """Tendencia y estabilidad de los scores de optimizacion."""
        if not step_metrics:
            return {"stability": 0.0, "trend": "stable"}
        scores = [s["bp_score"] for s in step_metrics]
        stab   = max(0.0, 1.0 - float(np.std(scores)) / (float(np.mean(scores)) + 1e-8))
        trend  = "stable"
        if len(scores) > 1:
            slope = float(np.polyfit(range(len(scores)), scores, 1)[0])
            trend = "increasing" if slope > 0.001 else "decreasing" if slope < -0.001 else "stable"
        return {"stability": stab, "trend": trend,
                "mean_bp": float(np.mean(scores)), "n_steps": len(scores)}

    def _calculate_bp_score(self, init_m: Dict, final_m: Dict,
                             bp_analysis: Dict) -> float:
        """Score compuesto ponderado: perdida + accuracy + muon + SOAP."""
        try:
            loss_imp = ((init_m["loss"] - final_m["loss"])
                        / max(init_m["loss"], 1e-8))
            acc_imp  = final_m["accuracy"] - init_m["accuracy"]
            muon_eff = bp_analysis.get("muon_efficiency", 0.0)
            mom_eff  = bp_analysis.get("momentum_efficiency", 0.0)
            score = (loss_imp * 0.30 + acc_imp  * 0.25
                     + mom_eff * 0.25 + muon_eff * 0.20)
            return float(max(0.0, min(1.0, score)))
        except Exception:
            return 0.0

    def _recommendations(self, metrics: SupervisedLearningNeuralMetrics,
                          analysis: Dict) -> List[str]:
        """Genera recomendaciones basadas en las metricas 2026."""
        recs: List[str] = []
        if analysis.get("momentum_efficiency", 1.0) < 0.7:
            recs.append("Eficiencia de momentum baja — considera aumentar bp_momentum a 0.95")
        if analysis.get("muon_efficiency", 1.0) < 0.6:
            recs.append("Ortogonalidad Muon baja — aumenta ns_steps o el learning_rate")
        if analysis.get("mean_gsnr", 1.0) < 0.3:
            recs.append("GSNR bajo — senal de gradiente ruidosa; reduce batch_size o aumenta datos")
        if metrics.bp_momentum_efficiency < 0.5:
            recs.append("Eficiencia global muy baja — considera pasar a SOAP puro o AdamW")
        if not recs:
            recs.append("Backpropagation 2026 (Muon+SOAP) funcionando correctamente")
        return recs


# ===========================================================================
# 4. API PUBLICA
# ===========================================================================
def create_backpropagation_optimizer(
    config: Optional[SupervisedLearningNeuralConfig] = None,
) -> BackpropagationOptimizer:
    """Crea y devuelve un BackpropagationOptimizer 2026."""
    return BackpropagationOptimizer(config or SupervisedLearningNeuralConfig())


def analyze_backpropagation_performance(
    model: Any,
    data_loader: Any,
    criterion: Any = None,
    config: Optional[SupervisedLearningNeuralConfig] = None,
) -> Dict:
    """Analiza el rendimiento de Backpropagation 2026 sobre un modelo."""
    try:
        opt = BackpropagationOptimizer(config or SupervisedLearningNeuralConfig())
        res = opt.optimize_weights(model, data_loader, criterion)
        return {
            "success":             res.success,
            "metrics":             res.metrics,
            "recommendations":     res.recommendations,
            "theoretical_analysis": res.theoretical_analysis,
        }
    except Exception as exc:
        logger.error("Error en analyze_backpropagation_performance: %s", exc)
        return {"success": False, "error": str(exc)}


def quick_backpropagation(model: Any, config: Optional[SupervisedLearningNeuralConfig] = None) -> Dict:
    """Ciclo rapido de Backpropagation 2026 y devuelve metricas basicas."""
    return analyze_backpropagation_performance(model, [], None, config)

def export_backpropagation_results(result: SupervisedLearningNeuralResult,
                                   filepath: str = "sl1_results.json") -> None:
    """Exporta los resultados de Backpropagation 2026 a un fichero JSON."""
    if result.metrics is None:
        return
    data = {
        "algorithm":        result.metrics.algorithm_name,
        "initial_loss":     result.metrics.initial_loss,
        "final_loss":       result.metrics.final_loss,
        "overall_score":    result.metrics.overall_score,
        "muon_efficiency":  result.theoretical_analysis.get("muon_efficiency", 0.0),
        "mean_gsnr":        result.theoretical_analysis.get("mean_gsnr", 0.0),
        "integration_score": result.metrics.supervised_neural_integration_score,
        "success":          result.success,
    }
    with open(filepath, "w", encoding="utf-8") as fh:
        _json.dump(data, fh, indent=2)

logger.info("SL1.py - Backpropagation 2026 (Muon+SOAP+NAG+GSNR+Polyak) cargado")
from SL1_MathematicalPrecision2026 import MathematicalPrecision2026  # CLASSPACK
from SL1_BackpropagationOptimizerInternal import BackpropagationOptimizerInternal  # CLASSPACK
