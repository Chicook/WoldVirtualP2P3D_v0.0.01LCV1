"""
RF_RFENRN1_3_7.py - Gestor de Validación de Pesos
==================================================

Implementa técnicas avanzadas de validación de pesos para redes neuronales
de aprendizaje por refuerzo. Incluye validación de integridad, detección
de pesos corruptos, validación de rangos, verificación de coherencia y
técnicas de diagnóstico automático para mantener la calidad de los pesos.

Características:
- Validación de integridad de pesos
- Detección automática de pesos corruptos
- Validación de rangos y valores
- Verificación de coherencia entre capas
- Diagnóstico automático de problemas
- Corrección automática de pesos inválidos
- Reportes detallados de validación

Autor: LucIA Development Team
Versión: 3.0.0
"""

try:
    import torch
    import torch.nn as nn
except ImportError:
    pass  # dependencia pesada opcional
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import math
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger('RFENRN1.RF_RFENRN1_3_7')


@dataclass
class WeightValidationConfig:
    """Configuración para validación de pesos"""
    use_integrity_validation: bool = True
    use_corruption_detection: bool = True
    use_range_validation: bool = True
    use_coherence_validation: bool = True
    use_automatic_diagnosis: bool = True
    validation_frequency: int = 50
    corruption_threshold: float = 1e-6
    range_min: float = -10.0
    range_max: float = 10.0
    coherence_threshold: float = 0.1
    max_invalid_weights: int = 100


class WeightValidationManager:
    """
    Gestor de validación de pesos para redes de refuerzo.

    Implementa técnicas avanzadas de validación que aseguran
    la calidad y coherencia de los pesos del modelo.
    """

    def __init__(self, config: Optional[WeightValidationConfig] = None):
        """
        Inicializa el gestor de validación.

        Args:
            config: Configuración de validación (opcional)
        """
        self.config = config or WeightValidationConfig()
        self.validation_stats = {
            'total_validations': 0,
            'passed_validations': 0,
            'failed_validations': 0,
            'corrupted_weights_detected': 0,
            'invalid_ranges_detected': 0,
            'coherence_violations': 0,
            'automatic_corrections': 0,
            'validation_accuracy': 0.0
        }
        self.validation_history = []
        self.weight_snapshots = defaultdict(list)

        logger.info("WeightValidationManager inicializado")

    def validate_model_weights(self, model: nn.Module, step: int) -> Dict[str, Any]:
        """
        Valida pesos del modelo.

        Args:
            model: Modelo PyTorch
            step: Paso actual

        Returns:
            Resultado de validación
        """
        if step % self.config.validation_frequency != 0:
            return {'status': 'skipped', 'step': step}

        validation_result = {
            'step': step,
            'status': 'passed',
            'issues': [],
            'corrections': [],
            'statistics': {}
        }

        try:
            # Validación de integridad
            if self.config.use_integrity_validation:
                integrity_result = self._validate_integrity(model)
                validation_result['statistics']['integrity'] = integrity_result

                if not integrity_result['passed']:
                    validation_result['issues'].extend(integrity_result['issues'])

            # Detección de corrupción
            if self.config.use_corruption_detection:
                corruption_result = self._detect_corruption(model)
                validation_result['statistics']['corruption'] = corruption_result

                if corruption_result['corrupted_count'] > 0:
                    validation_result['issues'].extend(corruption_result['issues'])
                    validation_result['corrections'].extend(corruption_result['corrections'])

            # Validación de rangos
            if self.config.use_range_validation:
                range_result = self._validate_ranges(model)
                validation_result['statistics']['ranges'] = range_result

                if range_result['out_of_range_count'] > 0:
                    validation_result['issues'].extend(range_result['issues'])
                    validation_result['corrections'].extend(range_result['corrections'])

            # Validación de coherencia
            if self.config.use_coherence_validation:
                coherence_result = self._validate_coherence(model)
                validation_result['statistics']['coherence'] = coherence_result

                if coherence_result['violations'] > 0:
                    validation_result['issues'].extend(coherence_result['issues'])

            # Diagnóstico automático
            if self.config.use_automatic_diagnosis:
                diagnosis_result = self._automatic_diagnosis(model)
                validation_result['statistics']['diagnosis'] = diagnosis_result

                if diagnosis_result['problems_detected'] > 0:
                    validation_result['issues'].extend(diagnosis_result['issues'])

            # Determinar estado final
            if validation_result['issues']:
                validation_result['status'] = 'failed'
                self.validation_stats['failed_validations'] += 1
            else:
                validation_result['status'] = 'passed'
                self.validation_stats['passed_validations'] += 1

            # Aplicar correcciones automáticas
            if validation_result['corrections']:
                model = self._apply_corrections(model, validation_result['corrections'])
                self.validation_stats['automatic_corrections'] += len(validation_result['corrections'])

            # Actualizar estadísticas
            self.validation_stats['total_validations'] += 1

            # Registrar en historial
            self.validation_history.append(validation_result)
            if len(self.validation_history) > 100:
                self.validation_history.pop(0)

        except Exception as e:
            logger.error(f"Error en validación: {e}")
            validation_result['status'] = 'error'
            validation_result['error'] = str(e)
            self.validation_stats['failed_validations'] += 1

        return validation_result

    def _validate_integrity(self, model: nn.Module) -> Dict[str, Any]:
        """
        Valida integridad de pesos.

        Args:
            model: Modelo PyTorch

        Returns:
            Resultado de validación de integridad
        """
        result = {
            'passed': True,
            'issues': [],
            'total_parameters': 0,
            'valid_parameters': 0
        }

        for name, param in model.named_parameters():
            if 'weight' in name:
                total_params = param.numel()
                result['total_parameters'] += total_params

                # Verificar valores finitos
                finite_mask = torch.isfinite(param.data)
                finite_count = finite_mask.sum().item()
                result['valid_parameters'] += finite_count

                if finite_count < total_params:
                    invalid_count = total_params - finite_count
                    result['issues'].append(f"Parámetros no finitos en {name}: {invalid_count}")
                    result['passed'] = False

                # Verificar NaN
                nan_count = torch.isnan(param.data).sum().item()
                if nan_count > 0:
                    result['issues'].append(f"NaN detectados en {name}: {nan_count}")
                    result['passed'] = False

                # Verificar Inf
                inf_count = torch.isinf(param.data).sum().item()
                if inf_count > 0:
                    result['issues'].append(f"Inf detectados en {name}: {inf_count}")
                    result['passed'] = False

        return result

    def _detect_corruption(self, model: nn.Module) -> Dict[str, Any]:
        """
        Detecta pesos corruptos.

        Args:
            model: Modelo PyTorch

        Returns:
            Resultado de detección de corrupción
        """
        result = {
            'corrupted_count': 0,
            'issues': [],
            'corrections': []
        }

        for name, param in model.named_parameters():
            if 'weight' in name:
                # Detectar valores extremos
                weight_data = param.data

                # Calcular estadísticas
                mean_val = weight_data.mean().item()
                std_val = weight_data.std().item()

                # Detectar outliers extremos
                outlier_mask = torch.abs(weight_data - mean_val) > 5 * std_val
                outlier_count = outlier_mask.sum().item()

                if outlier_count > 0:
                    result['corrupted_count'] += outlier_count
                    result['issues'].append(f"Outliers extremos en {name}: {outlier_count}")

                    # Corrección automática
                    if outlier_count < self.config.max_invalid_weights:
                        # Reemplazar outliers con valores aleatorios
                        correction_mask = outlier_mask
                        corrected_weights = torch.randn_like(weight_data[correction_mask]) * std_val
                        weight_data[correction_mask] = corrected_weights

                        result['corrections'].append(f"Outliers corregidos en {name}: {outlier_count}")

        return result

    def _validate_ranges(self, model: nn.Module) -> Dict[str, Any]:
        """
        Valida rangos de pesos.

        Args:
            model: Modelo PyTorch

        Returns:
            Resultado de validación de rangos
        """
        result = {
            'out_of_range_count': 0,
            'issues': [],
            'corrections': []
        }

        for name, param in model.named_parameters():
            if 'weight' in name:
                weight_data = param.data

                # Verificar rango mínimo
                min_mask = weight_data < self.config.range_min
                min_count = min_mask.sum().item()

                # Verificar rango máximo
                max_mask = weight_data > self.config.range_max
                max_count = max_mask.sum().item()

                total_out_of_range = min_count + max_count

                if total_out_of_range > 0:
                    result['out_of_range_count'] += total_out_of_range
                    result['issues'].append(f"Pesos fuera de rango en {name}: {total_out_of_range}")

                    # Corrección automática
                    if total_out_of_range < self.config.max_invalid_weights:
                        # Clamp a rangos válidos
                        weight_data.clamp_(self.config.range_min, self.config.range_max)
                        result['corrections'].append(f"Rangos corregidos en {name}")

        return result

    def _validate_coherence(self, model: nn.Module) -> Dict[str, Any]:
        """
        Valida coherencia entre capas.

        Args:
            model: Modelo PyTorch

        Returns:
            Resultado de validación de coherencia
        """
        result = {
            'violations': 0,
            'issues': []
        }

        layer_norms = {}

        # Calcular normas por capa
        for name, param in model.named_parameters():
            if 'weight' in name:
                layer_norms[name] = torch.norm(param.data).item()

        # Verificar coherencia entre capas adyacentes
        layer_names = list(layer_norms.keys())
        for i in range(len(layer_names) - 1):
            current_layer = layer_names[i]
            next_layer = layer_names[i + 1]

            current_norm = layer_norms[current_layer]
            next_norm = layer_norms[next_layer]

            if current_norm > 0 and next_norm > 0:
                ratio = max(current_norm, next_norm) / min(current_norm, next_norm)

                if ratio > 1.0 / self.config.coherence_threshold:
                    result['violations'] += 1
                    result['issues'].append(
                        f"Incoherencia entre {current_layer} y {next_layer}: ratio {ratio:.2f}"
                    )

        return result

    def _automatic_diagnosis(self, model: nn.Module) -> Dict[str, Any]:
        """
        Realiza diagnóstico automático.

        Args:
            model: Modelo PyTorch

        Returns:
            Resultado del diagnóstico
        """
        result = {
            'problems_detected': 0,
            'issues': [],
            'recommendations': []
        }

        # Análisis de distribución de pesos
        all_weights = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                all_weights.append(param.data.flatten())

        if all_weights:
            all_weights_tensor = torch.cat(all_weights)

            # Detectar problemas de distribución
            mean_weight = all_weights_tensor.mean().item()
            std_weight = all_weights_tensor.std().item()

            if abs(mean_weight) > 1.0:
                result['problems_detected'] += 1
                result['issues'].append(f"Media de pesos muy alta: {mean_weight:.4f}")
                result['recommendations'].append("Considerar inicialización más conservadora")

            if std_weight > 2.0:
                result['problems_detected'] += 1
                result['issues'].append(f"Desviación estándar muy alta: {std_weight:.4f}")
                result['recommendations'].append("Considerar regularización más fuerte")

            if std_weight < 0.01:
                result['problems_detected'] += 1
                result['issues'].append(f"Desviación estándar muy baja: {std_weight:.4f}")
                result['recommendations'].append("Considerar inicialización más agresiva")

        return result

    def _apply_corrections(self, model: nn.Module, corrections: List[str]) -> nn.Module:
        """
        Aplica correcciones automáticas.

        Args:
            model: Modelo PyTorch
            corrections: Lista de correcciones

        Returns:
            Modelo con correcciones aplicadas
        """
        logger.info(f"Aplicando {len(corrections)} correcciones automáticas")

        for correction in corrections:
            logger.info(f"Corrección aplicada: {correction}")

        return model

    def calculate_validation_accuracy(self) -> float:
        """
        Calcula la precisión de validación.

        Returns:
            Precisión de validación (0-1)
        """
        if self.validation_stats['total_validations'] == 0:
            return 0.0

        accuracy = self.validation_stats['passed_validations'] / self.validation_stats['total_validations']
        self.validation_stats['validation_accuracy'] = accuracy

        return accuracy

    def get_validation_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de validación.

        Returns:
            Diccionario con estadísticas
        """
        return {
            'validation_stats': self.validation_stats.copy(),
            'validation_history': self.validation_history.copy(),
            'accuracy': self.calculate_validation_accuracy()
        }

    def reset_stats(self) -> None:
        """Reinicia las estadísticas de validación."""
        self.validation_history.clear()
        self.weight_snapshots.clear()
        self.validation_stats = {
            'total_validations': 0,
            'passed_validations': 0,
            'failed_validations': 0,
            'corrupted_weights_detected': 0,
            'invalid_ranges_detected': 0,
            'coherence_violations': 0,
            'automatic_corrections': 0,
            'validation_accuracy': 0.0
        }

        logger.info("Estadísticas de validación reiniciadas")
