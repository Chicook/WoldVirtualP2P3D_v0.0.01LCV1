"""
Neurona de Análisis de Complejidad - RF_RFEN1_RN_9_6
Especializada en medir y reducir complejidad de código
"""

import ast
import numpy as np
from typing import Dict, List, Any, Tuple


class ComplexityAnalyzerNeuron:
    """
    Neurona especializada en analizar complejidad ciclomática y cognitiva
    Ayuda a mantener código simple y mantenible
    """

    def __init__(self, input_size: int = 100, learning_rate: float = 0.001):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.weights = np.random.normal(0, 0.1, input_size)
        self.bias = np.random.normal(0, 0.1)
        self.complexity_history = []
        self.refactoring_suggestions = []

    def forward(self, code_features: np.ndarray) -> np.ndarray:
        """
        Evalúa la complejidad del código
        """
        if len(code_features) != self.input_size:
            raise ValueError(f"Expected {self.input_size} features, got {len(code_features)}")

        weighted_sum = np.dot(code_features, self.weights) + self.bias
        return self._swish(weighted_sum)

    def backward(self, error: np.ndarray, code_features: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Ajusta pesos basado en reducción de complejidad
        """
        s = self._sigmoid(np.dot(code_features, self.weights) + self.bias)
        gradient = error * (s + (np.dot(code_features, self.weights) + self.bias) * s * (1 - s))

        weight_gradient = gradient * code_features
        bias_gradient = gradient

        self.weights -= self.learning_rate * weight_gradient
        self.bias -= self.learning_rate * bias_gradient

        return weight_gradient, bias_gradient

    def _swish(self, x: np.ndarray) -> np.ndarray:
        """Función de activación Swish"""
        return x * self._sigmoid(x)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Función sigmoid"""
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

    def analyze_complexity(self, code: str) -> Dict[str, Any]:
        """
        Analiza complejidad del código

        Returns:
            Dict con métricas de complejidad
        """
        try:
            tree = ast.parse(code)

            # Análisis de complejidad
            analyzer = ComplexityAnalyzer()
            analyzer.visit(tree)

            metrics = {
                'cyclomatic_complexity': analyzer.cyclomatic_complexity,
                'cognitive_complexity': analyzer.cognitive_complexity,
                'nesting_depth': analyzer.max_nesting_depth,
                'function_count': len(analyzer.functions),
                'average_function_length': analyzer.get_avg_function_length(),
                'max_function_complexity': analyzer.get_max_function_complexity(),
                'maintainability_index': self._calculate_maintainability_index(analyzer)
            }

            self.complexity_history.append(metrics)

            return metrics

        except SyntaxError as e:
            return {'error': f'Syntax error: {str(e)}'}

    def suggest_simplifications(self, complexity_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Sugiere simplificaciones basadas en métricas de complejidad
        """
        suggestions = []

        # Complejidad ciclomática alta
        if complexity_metrics.get('cyclomatic_complexity', 0) > 10:
            suggestions.append({
                'type': 'reduce_cyclomatic_complexity',
                'description': 'La complejidad ciclomática es alta. Considerar extraer funciones.',
                'severity': 'high'
            })

        # Profundidad de anidación alta
        if complexity_metrics.get('nesting_depth', 0) > 3:
            suggestions.append({
                'type': 'reduce_nesting',
                'description': 'Demasiadas capas de anidación. Usar early returns.',
                'severity': 'medium'
            })

        # Función muy compleja
        if complexity_metrics.get('max_function_complexity', 0) > 15:
            suggestions.append({
                'type': 'split_large_function',
                'description': 'Dividir función grande en funciones más pequeñas.',
                'severity': 'high'
            })

        # Bajo índice de mantenibilidad
        if complexity_metrics.get('maintainability_index', 100) < 50:
            suggestions.append({
                'type': 'improve_maintainability',
                'description': 'Mejorar mantenibilidad del código mediante refactorización.',
                'severity': 'high'
            })

        return suggestions

    def _calculate_maintainability_index(self, analyzer: 'ComplexityAnalyzer') -> float:
        """
        Calcula índice de mantenibilidad (0-100)
        """
        # Fórmula simplificada de mantenibilidad
        complexity_penalty = analyzer.cyclomatic_complexity * 2
        nesting_penalty = analyzer.max_nesting_depth * 5
        function_penalty = len(analyzer.functions) * 0.5

        maintainability = 100 - complexity_penalty - nesting_penalty - function_penalty

        return max(0, min(100, maintainability))


class ComplexityAnalyzer(ast.NodeVisitor):
    """Analiza complejidad del código"""

    def __init__(self):
        self.cyclomatic_complexity = 1  # Base complexity
        self.cognitive_complexity = 1
        self.max_nesting_depth = 0
        self.current_nesting = 0
        self.functions = []
        self.function_lengths = {}
        self.current_function = None
        self.current_function_lines = 0

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.functions.append(node.name)
        self.current_function = node.name
        self.current_function_lines = node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0
        self.function_lengths[node.name] = self.current_function_lines

        # Calcular complejidad de la función
        func_complexity = 1
        for stmt in ast.walk(node):
            if isinstance(stmt, (ast.If, ast.While, ast.For, ast.BoolOp, ast.ExceptHandler)):
                func_complexity += 1

        self.generic_visit(node)
        self.current_function = None
        self.current_function_lines = 0

    def visit_If(self, node: ast.If):
        self.cyclomatic_complexity += 1
        self.cognitive_complexity += 1
        self.current_nesting += 1
        if self.current_nesting > self.max_nesting_depth:
            self.max_nesting_depth = self.current_nesting
        self.generic_visit(node)
        self.current_nesting -= 1

    def visit_While(self, node: ast.While):
        self.cyclomatic_complexity += 1
        self.current_nesting += 1
        if self.current_nesting > self.max_nesting_depth:
            self.max_nesting_depth = self.current_nesting
        self.generic_visit(node)
        self.current_nesting -= 1

    def visit_For(self, node: ast.For):
        self.cyclomatic_complexity += 1
        self.current_nesting += 1
        if self.current_nesting > self.max_nesting_depth:
            self.max_nesting_depth = self.current_nesting
        self.generic_visit(node)
        self.current_nesting -= 1

    def visit_BoolOp(self, node: ast.BoolOp):
        self.cyclomatic_complexity += len(node.values) - 1
        self.cognitive_complexity += 1
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler):
        self.cyclomatic_complexity += 1
        self.generic_visit(node)

    def get_avg_function_length(self) -> float:
        """Calcula promedio de líneas por función"""
        if not self.function_lengths:
            return 0
        return sum(self.function_lengths.values()) / len(self.function_lengths)

    def get_max_function_complexity(self) -> int:
        """Retorna la complejidad máxima"""
        return self.cyclomatic_complexity
