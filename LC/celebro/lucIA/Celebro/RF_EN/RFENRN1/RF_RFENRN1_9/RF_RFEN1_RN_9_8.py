"""
Neurona de Análisis de Rendimiento - RF_RFEN1_RN_9_8
Especializada en detectar problemas de rendimiento y optimización
"""

import ast
import numpy as np
from typing import Dict, List, Any, Tuple
import time


class PerformanceAnalyzerNeuron:
    """
    Neurona especializada en análisis profundo de rendimiento
    Detecta cuellos de botella y sugiere optimizaciones
    """

    def __init__(self, input_size: int = 100, learning_rate: float = 0.001):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.weights = np.random.normal(0, 0.1, input_size)
        self.bias = np.random.normal(0, 0.1)
        self.performance_metrics = {}
        self.bottleneck_history = []

    def forward(self, code_features: np.ndarray) -> np.ndarray:
        """
        Identifica problemas de rendimiento potenciales
        """
        if len(code_features) != self.input_size:
            raise ValueError(f"Expected {self.input_size} features, got {len(code_features)}")

        weighted_sum = np.dot(code_features, self.weights) + self.bias
        return self._leaky_relu(weighted_sum)

    def backward(self, error: np.ndarray, code_features: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Ajusta pesos basado en análisis de rendimiento
        """
        gradient = error * self._leaky_relu_derivative(
            np.dot(code_features, self.weights) + self.bias
        )

        weight_gradient = gradient * code_features
        bias_gradient = gradient

        self.weights -= self.learning_rate * weight_gradient
        self.bias -= self.learning_rate * bias_gradient

        return weight_gradient, bias_gradient

    def _leaky_relu(self, x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Función de activación Leaky ReLU"""
        return np.where(x > 0, x, alpha * x)

    def _leaky_relu_derivative(self, x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Derivada de Leaky ReLU"""
        return np.where(x > 0, 1, alpha)

    def analyze_performance_metrics(self, code: str) -> Dict[str, Any]:
        """
        Analiza métricas de rendimiento del código

        Returns:
            Dict con métricas y sugerencias de optimización
        """
        try:
            tree = ast.parse(code)
            analyzer = PerformanceASTAnalyzer()
            analyzer.visit(tree)

            metrics = {
                'operations_count': analyzer.total_operations,
                'memory_operations': analyzer.memory_ops,
                'iterations': analyzer.iterations,
                'recursive_calls': analyzer.recursive_calls,
                'external_calls': analyzer.external_calls,
                'estimated_complexity': self._estimate_time_complexity(analyzer),
                'bottlenecks': analyzer.get_bottlenecks(),
                'optimization_suggestions': self._generate_optimization_suggestions(analyzer)
            }

            self.performance_metrics = metrics
            return metrics

        except SyntaxError as e:
            return {'error': f'Syntax error: {str(e)}'}

    def _estimate_time_complexity(self, analyzer: 'PerformanceASTAnalyzer') -> str:
        """
        Estima la complejidad temporal del código
        """
        # Análisis simplificado
        if analyzer.iterations == 0 and analyzer.recursive_calls == 0:
            return 'O(1)'
        elif analyzer.iterations <= 10:
            return 'O(n)'
        elif analyzer.iterations > 10:
            return 'O(n²) o peor'
        else:
            return 'Variable'

    def _generate_optimization_suggestions(self, analyzer: 'PerformanceASTAnalyzer') -> List[Dict[str, Any]]:
        """
        Genera sugerencias específicas de optimización
        """
        suggestions = []

        # Muchas iteraciones
        if analyzer.iterations > 5:
            suggestions.append({
                'type': 'reduce_iterations',
                'description': 'Usar vectorización con NumPy en lugar de loops',
                'potential_improvement': '20-50%'
            })

        # Llamadas recursivas profundas
        if analyzer.recursive_calls > 3:
            suggestions.append({
                'type': 'tail_recursion',
                'description': 'Considerar iteración en lugar de recursión profunda',
                'potential_improvement': '10-30%'
            })

        # Muchas operaciones de memoria
        if analyzer.memory_ops > 10:
            suggestions.append({
                'type': 'memory_optimization',
                'description': 'Reducir operaciones de asignación de memoria',
                'potential_improvement': '15-40%'
            })

        return suggestions

    def profile_code(self, code: str, iterations: int = 100) -> Dict[str, float]:
        """
        Hace profiling del código ejecutándolo múltiples veces

        Returns:
            Dict con métricas de tiempo de ejecución
        """
        # Compilar y ejecutar código de prueba
        try:
            compiled = compile(code, '<string>', 'exec')

            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                exec(compiled)
                end = time.perf_counter()
                times.append(end - start)

            return {
                'min_time': min(times),
                'max_time': max(times),
                'avg_time': sum(times) / len(times),
                'total_time': sum(times),
                'variance': np.var(times),
                'std_deviation': np.std(times)
            }

        except Exception as e:
            return {'error': str(e)}


class PerformanceASTAnalyzer(ast.NodeVisitor):
    """Analizador de AST para métricas de rendimiento"""

    def __init__(self):
        self.total_operations = 0
        self.memory_ops = 0
        self.iterations = 0
        self.recursive_calls = 0
        self.external_calls = 0
        self.nested_loops = 0
        self.loop_depth = 0

    def visit_For(self, node: ast.For):
        self.iterations += 1
        self.loop_depth += 1
        if self.loop_depth > self.nested_loops:
            self.nested_loops = self.loop_depth

        self.generic_visit(node)
        self.loop_depth -= 1

    def visit_While(self, node: ast.While):
        self.iterations += 1
        self.loop_depth += 1
        if self.loop_depth > self.nested_loops:
            self.nested_loops = self.loop_depth

        self.generic_visit(node)
        self.loop_depth -= 1

    def visit_Call(self, node: ast.Call):
        self.external_calls += 1

        # Detectar recursión
        if isinstance(node.func, ast.Name):
            self.recursive_calls += 1

        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        self.memory_ops += 1
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp):
        self.total_operations += 1
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare):
        self.total_operations += 1
        self.generic_visit(node)

    def get_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identifica cuellos de botella"""
        bottlenecks = []

        if self.iterations > 10:
            bottlenecks.append({
                'type': 'loop_performance',
                'description': f'{self.iterations} loops detectados',
                'impact': 'high'
            })

        if self.nested_loops > 2:
            bottlenecks.append({
                'type': 'nested_loops',
                'description': f'Loops anidados a nivel {self.nested_loops}',
                'impact': 'critical'
            })

        if self.recursive_calls > 5:
            bottlenecks.append({
                'type': 'recursion',
                'description': f'{self.recursive_calls} llamadas recursivas detectadas',
                'impact': 'medium'
            })

        return bottlenecks
