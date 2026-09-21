"""
Neurona de Optimización de Código - RF_RFEN1_RN_9_4
Especializada en optimizar rendimiento y eficiencia de código
"""

import ast
import numpy as np
from typing import Dict, List, Any, Tuple


class CodeOptimizationNeuron:
    """
    Neurona especializada en optimizar código para mejor rendimiento
    Detecta cuellos de botella y sugiere optimizaciones
    """

    def __init__(self, input_size: int = 100, learning_rate: float = 0.001):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.weights = np.random.normal(0, 0.1, input_size)
        self.bias = np.random.normal(0, 0.1)
        self.optimization_suggestions = []
        self.performance_improvements = {}

    def forward(self, code_features: np.ndarray) -> np.ndarray:
        """
        Identifica oportunidades de optimización
        """
        if len(code_features) != self.input_size:
            raise ValueError(f"Expected {self.input_size} features, got {len(code_features)}")

        weighted_sum = np.dot(code_features, self.weights) + self.bias
        return self._softmax(weighted_sum)

    def backward(self, error: np.ndarray, code_features: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Ajusta pesos basado en éxito de optimizaciones
        """
        gradient = error
        weight_gradient = np.outer(gradient, code_features)
        bias_gradient = gradient

        self.weights -= self.learning_rate * weight_gradient.sum(axis=0)
        self.bias -= self.learning_rate * bias_gradient.sum()

        return weight_gradient, bias_gradient

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Función de activación softmax"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def analyze_performance(self, code: str) -> Dict[str, Any]:
        """
        Analiza rendimiento del código y detecta cuellos de botella

        Returns:
            Dict con sugerencias de optimización
        """
        try:
            tree = ast.parse(code)
            analyzer = PerformanceAnalyzer()
            analyzer.visit(tree)

            suggestions = self._generate_optimization_suggestions(analyzer)

            return {
                'bottlenecks': analyzer.bottlenecks,
                'suggestions': suggestions,
                'estimated_improvement': self._estimate_improvement(suggestions)
            }

        except SyntaxError as e:
            return {'error': f'Syntax error: {str(e)}'}

    def _generate_optimization_suggestions(self, analyzer: 'PerformanceAnalyzer') -> List[Dict[str, str]]:
        """
        Genera sugerencias de optimización basadas en el análisis
        """
        suggestions = []

        # Detectar bucles lentos
        if analyzer.has_nested_loops:
            suggestions.append({
                'type': 'nested_loops',
                'suggestion': 'Considerar vectorización con NumPy o uso de map/filter',
                'impact': 'high'
            })

        # Detectar búsquedas lineales
        if analyzer.has_linear_search:
            suggestions.append({
                'type': 'linear_search',
                'suggestion': 'Usar sets o dicts para búsquedas O(1)',
                'impact': 'high'
            })

        # Detectar recreación de objetos
        if analyzer.has_object_recreation:
            suggestions.append({
                'type': 'object_recreation',
                'suggestion': 'Reutilizar objetos o usar caching',
                'impact': 'medium'
            })

        # Detectar llamadas repetidas
        if analyzer.has_repeated_calls:
            suggestions.append({
                'type': 'repeated_calls',
                'suggestion': 'Cachear resultados de funciones costosas',
                'impact': 'medium'
            })

        return suggestions

    def _estimate_improvement(self, suggestions: List[Dict[str, Any]]) -> float:
        """
        Estima el porcentaje de mejoría con las optimizaciones sugeridas
        """
        total_improvement = 0.0

        for suggestion in suggestions:
            if suggestion.get('impact') == 'high':
                total_improvement += 0.3
            elif suggestion.get('impact') == 'medium':
                total_improvement += 0.15
            else:
                total_improvement += 0.05

        # Cap en 80% de mejora máxima estimada
        return min(total_improvement * 100, 80.0)

    def optimize_code(self, code: str, suggestions: List[Dict[str, Any]]) -> str:
        """
        Aplica optimizaciones automáticas al código
        """
        # Esta función se completaría con transformaciones AST reales
        # Por ahora retorna el código original con comentarios de mejoras
        optimized = code

        for suggestion in suggestions:
            suggestion_text = f"# OPTIMIZATION: {suggestion.get('suggestion')}\n"
            optimized = suggestion_text + optimized

        return optimized


class PerformanceAnalyzer(ast.NodeVisitor):
    """Analiza código para detectar problemas de rendimiento"""

    def __init__(self):
        self.bottlenecks = []
        self.has_nested_loops = False
        self.has_linear_search = False
        self.has_object_recreation = False
        self.has_repeated_calls = False
        self.loop_depth = 0
        self.in_loop = False

    def visit_For(self, node: ast.For):
        self.loop_depth += 1
        self.in_loop = True

        if self.loop_depth > 1:
            self.has_nested_loops = True

        # Detectar búsquedas lineales
        for child in ast.walk(node):
            if isinstance(child, ast.If):
                self.has_linear_search = True

        self.generic_visit(node)
        self.loop_depth -= 1
        self.in_loop = False

    def visit_While(self, node: ast.While):
        self.loop_depth += 1
        self.in_loop = True

        if self.loop_depth > 1:
            self.has_nested_loops = True

        self.generic_visit(node)
        self.loop_depth -= 1
        self.in_loop = False

    def visit_Assign(self, node: ast.Assign):
        if self.in_loop:
            # Detectar recreación de objetos en loops
            self.has_object_recreation = True
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if self.in_loop and isinstance(node.func, ast.Name):
            # Detectar llamadas repetidas en loops
            if not node.func.id.startswith('__'):
                self.has_repeated_calls = True
        self.generic_visit(node)
