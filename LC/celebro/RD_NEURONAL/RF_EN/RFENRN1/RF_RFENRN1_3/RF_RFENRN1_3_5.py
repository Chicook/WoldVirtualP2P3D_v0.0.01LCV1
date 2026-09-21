"""
RF_RFENRN1_3_5.py - Gestor de Balanceado de Pesos
=================================================

Implementa técnicas avanzadas de balanceado de pesos para redes neuronales
de aprendizaje por refuerzo. Incluye balanceado por capas, balanceado por
importancia, balanceado por gradientes y técnicas de equilibrio automático
para optimizar la distribución de pesos en el modelo.

Características:
- Balanceado automático por capas con diferentes estrategias
- Balanceado basado en importancia de pesos
- Balanceado por magnitud de gradientes
- Equilibrio automático de pesos entre capas
- Balanceado adaptativo durante el entrenamiento
- Detección de desequilibrios en distribución de pesos
- Corrección automática de pesos desbalanceados

Autor: LucIA Development Team
Versión: 3.0.0
"""

try:
    import torch
    import torch.nn as nn
except ImportError:
    pass  # dependencia pesada opcional
try:
    torch
except NameError:
    import types as _t
    torch = _t.SimpleNamespace(
        no_grad=lambda *a, **k: (lambda f: f) if a and callable(a[0]) else (lambda f: f),
        optim=_t.SimpleNamespace(Optimizer=object),
        Tensor=object,
    )
try:
    nn
except NameError:
    import types as _t2
    nn = _t2.SimpleNamespace(Module=object)
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import math
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger('RFENRN1.RF_RFENRN1_3_5')


@dataclass
class WeightBalancingConfig:
    """Configuración para balanceado de pesos"""
    use_layer_balancing: bool = True
    use_importance_balancing: bool = True
    use_gradient_balancing: bool = True
    use_adaptive_balancing: bool = True
    balancing_frequency: int = 20
    importance_threshold: float = 0.1
    gradient_threshold: float = 0.5
    balancing_strength: float = 0.1
    equilibrium_tolerance: float = 0.05
    max_balancing_iterations: int = 5


class WeightBalancingManager:
    """
    Gestor de balanceado de pesos para redes de refuerzo.

    Implementa técnicas avanzadas de balanceado que mantienen
    una distribución equilibrada de pesos en el modelo.
    """

    def __init__(self, config: Optional[WeightBalancingConfig] = None):
        """
        Inicializa el gestor de balanceado.

        Args:
            config: Configuración de balanceado (opcional)
        """
        self.config = config or WeightBalancingConfig()
        self.layer_weights = defaultdict(list)
        self.balancing_stats = {
            'total_balancings': 0,
            'layer_balancings': 0,
            'importance_balancings': 0,
            'gradient_balancings': 0,
            'equilibrium_score': 0.0,
            'balancing_efficiency': 0.0
        }
        self.equilibrium_history = []

        logger.info("WeightBalancingManager inicializado")

    def balance_model_weights(self, model: nn.Module, epoch: int) -> nn.Module:
        """
        Balancea pesos del modelo.

        Args:
            model: Modelo PyTorch
            epoch: Época actual

        Returns:
            Modelo con pesos balanceados
        """
        if epoch % self.config.balancing_frequency != 0:
            return model

        # Calcular equilibrio actual
        equilibrium_score = self._calculate_equilibrium_score(model)
        self.equilibrium_history.append(equilibrium_score)

        # Determinar estrategia de balanceado
        if equilibrium_score < self.config.equilibrium_tolerance:
            if self.config.use_layer_balancing:
                model = self._balance_by_layers(model)

            if self.config.use_importance_balancing:
                model = self._balance_by_importance(model)

            if self.config.use_gradient_balancing:
                model = self._balance_by_gradients(model)

            if self.config.use_adaptive_balancing:
                model = self._adaptive_balancing(model)

        self.balancing_stats['total_balancings'] += 1

        return model

    def _calculate_equilibrium_score(self, model: nn.Module) -> float:
        """
        Calcula el score de equilibrio del modelo.

        Args:
            model: Modelo PyTorch

        Returns:
            Score de equilibrio (0-1, mayor es mejor)
        """
        layer_norms = []

        for name, param in model.named_parameters():
            if 'weight' in name:
                norm = torch.norm(param.data).item()
                layer_norms.append(norm)
                self.layer_weights[name].append(norm)

        if not layer_norms:
            return 1.0

        # Calcular equilibrio basándose en variabilidad de normas
        mean_norm = np.mean(layer_norms)
        std_norm = np.std(layer_norms)

        if mean_norm == 0:
            return 1.0

        equilibrium_score = 1.0 / (1.0 + std_norm / mean_norm)

        return equilibrium_score

    def _balance_by_layers(self, model: nn.Module) -> nn.Module:
        """
        Balancea pesos por capas.

        Args:
            model: Modelo PyTorch

        Returns:
            Modelo con pesos balanceados por capas
        """
        layer_norms = {}

        # Calcular normas por capa
        for name, param in model.named_parameters():
            if 'weight' in name:
                layer_norms[name] = torch.norm(param.data).item()

        if not layer_norms:
            return model

        # Calcular norma objetivo
        target_norm = np.mean(list(layer_norms.values()))

        # Balancear cada capa hacia la norma objetivo
        for name, param in model.named_parameters():
            if 'weight' in name:
                current_norm = layer_norms[name]
                if current_norm > 0:
                    scale_factor = target_norm / current_norm
                    # Aplicar balanceado gradual
                    adjusted_factor = 1.0 + self.config.balancing_strength * (scale_factor - 1.0)
                    param.data *= adjusted_factor

        self.balancing_stats['layer_balancings'] += 1
        logger.info(f"Balanceado por capas aplicado (norma objetivo: {target_norm:.4f})")

        return model

    def _balance_by_importance(self, model: nn.Module) -> nn.Module:
        """
        Balancea pesos por importancia.

        Args:
            model: Modelo PyTorch

        Returns:
            Modelo con pesos balanceados por importancia
        """
        importance_scores = {}

        # Calcular importancia de cada peso
        for name, param in model.named_parameters():
            if 'weight' in name:
                # Importancia basada en magnitud y gradientes
                magnitude = torch.norm(param.data).item()
                if param.grad is not None:
                    gradient_magnitude = torch.norm(param.grad).item()
                    importance = magnitude * gradient_magnitude
                else:
                    importance = magnitude

                importance_scores[name] = importance

        if not importance_scores:
            return model

        # Balancear pesos de baja importancia
        mean_importance = np.mean(list(importance_scores.values()))

        for name, param in model.named_parameters():
            if 'weight' in name and name in importance_scores:
                importance = importance_scores[name]
                if importance < mean_importance * self.config.importance_threshold:
                    # Aumentar magnitud de pesos de baja importancia
                    boost_factor = 1.0 + self.config.balancing_strength
                    param.data *= boost_factor

        self.balancing_stats['importance_balancings'] += 1
        logger.info(f"Balanceado por importancia aplicado")

        return model

    def _balance_by_gradients(self, model: nn.Module) -> nn.Module:
        """
        Balancea pesos por gradientes.

        Args:
            model: Modelo PyTorch

        Returns:
            Modelo con pesos balanceados por gradientes
        """
        gradient_norms = {}

        # Calcular normas de gradientes
        for name, param in model.named_parameters():
            if 'weight' in name and param.grad is not None:
                gradient_norms[name] = torch.norm(param.grad).item()

        if not gradient_norms:
            return model

        # Balancear pesos con gradientes pequeños
        mean_gradient = np.mean(list(gradient_norms.values()))

        for name, param in model.named_parameters():
            if 'weight' in name and name in gradient_norms:
                grad_norm = gradient_norms[name]
                if grad_norm < mean_gradient * self.config.gradient_threshold:
                    # Ajustar pesos con gradientes pequeños
                    adjustment_factor = 1.0 + self.config.balancing_strength
                    param.data *= adjustment_factor

        self.balancing_stats['gradient_balancings'] += 1
        logger.info(f"Balanceado por gradientes aplicado")

        return model

    def _adaptive_balancing(self, model: nn.Module) -> nn.Module:
        """
        Aplica balanceado adaptativo.

        Args:
            model: Modelo PyTorch

        Returns:
            Modelo con balanceado adaptativo
        """
        # Análisis histórico de equilibrio
        if len(self.equilibrium_history) >= 5:
            recent_equilibrium = self.equilibrium_history[-5:]
            equilibrium_trend = np.mean(np.diff(recent_equilibrium))

            # Ajustar fuerza de balanceado basándose en tendencia
            if equilibrium_trend < 0:
                # Equilibrio empeorando, aumentar fuerza
                balancing_strength = self.config.balancing_strength * 1.5
            else:
                # Equilibrio mejorando, mantener fuerza
                balancing_strength = self.config.balancing_strength
        else:
            balancing_strength = self.config.balancing_strength

        # Aplicar balanceado adaptativo
        for name, param in model.named_parameters():
            if 'weight' in name:
                # Balanceado basado en desviación de la media
                weight_norm = torch.norm(param.data).item()

                # Calcular desviación promedio
                if name in self.layer_weights and len(self.layer_weights[name]) >= 3:
                    historical_norms = self.layer_weights[name][-3:]
                    mean_historical = np.mean(historical_norms)

                    if mean_historical > 0:
                        deviation = abs(weight_norm - mean_historical) / mean_historical

                        if deviation > 0.1:  # Desviación significativa
                            # Corregir hacia la media histórica
                            correction_factor = mean_historical / weight_norm
                            adjusted_factor = 1.0 + balancing_strength * (correction_factor - 1.0)
                            param.data *= adjusted_factor

        logger.info(f"Balanceado adaptativo aplicado")

        return model

    def calculate_balancing_efficiency(self) -> float:
        """
        Calcula la eficiencia del balanceado.

        Returns:
            Eficiencia del balanceado (0-1)
        """
        if len(self.equilibrium_history) < 2:
            return 0.0

        # Calcular mejora en equilibrio
        initial_equilibrium = self.equilibrium_history[0]
        final_equilibrium = self.equilibrium_history[-1]

        if initial_equilibrium == 0:
            return 0.0

        efficiency = (final_equilibrium - initial_equilibrium) / initial_equilibrium
        efficiency = max(0.0, min(1.0, efficiency))

        self.balancing_stats['balancing_efficiency'] = efficiency

        return efficiency

    def get_balancing_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de balanceado.

        Returns:
            Diccionario con estadísticas
        """
        return {
            'balancing_stats': self.balancing_stats.copy(),
            'equilibrium_history': self.equilibrium_history.copy(),
            'efficiency': self.calculate_balancing_efficiency()
        }

    def reset_stats(self) -> None:
        """Reinicia las estadísticas de balanceado."""
        self.layer_weights.clear()
        self.equilibrium_history.clear()
        self.balancing_stats = {
            'total_balancings': 0,
            'layer_balancings': 0,
            'importance_balancings': 0,
            'gradient_balancings': 0,
            'equilibrium_score': 0.0,
            'balancing_efficiency': 0.0
        }

        logger.info("Estadísticas de balanceado reiniciadas")
