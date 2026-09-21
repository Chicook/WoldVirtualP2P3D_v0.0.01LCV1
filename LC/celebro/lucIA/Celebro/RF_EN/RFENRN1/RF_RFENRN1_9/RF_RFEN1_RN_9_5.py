"""
Neurona de Detección de Patrones - RF_RFEN1_RN_9_5
Especializada en detectar y aplicar patrones de diseño
"""

import ast
import numpy as np
from typing import Dict, List, Any, Tuple
import re


class PatternMatchingNeuron:
    """
    Neurona especializada en detectar y sugerir patrones de diseño
    Identifica anti-patrones y sugiere soluciones
    """

    def __init__(self, input_size: int = 100, learning_rate: float = 0.001):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.weights = np.random.normal(0, 0.1, input_size)
        self.bias = np.random.normal(0, 0.1)
        self.design_patterns = {}
        self.antipatterns_detected = []

    def forward(self, code_features: np.ndarray) -> np.ndarray:
        """
        Detecta patrones y anti-patrones en el código
        """
        if len(code_features) != self.input_size:
            raise ValueError(f"Expected {self.input_size} features, got {len(code_features)}")

        weighted_sum = np.dot(code_features, self.weights) + self.bias
        return self._elu(weighted_sum)

    def backward(self, error: np.ndarray, code_features: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Ajusta pesos basado en detección de patrones
        """
        gradient = error * self._elu_derivative(
            np.dot(code_features, self.weights) + self.bias
        )

        weight_gradient = gradient * code_features
        bias_gradient = gradient

        self.weights -= self.learning_rate * weight_gradient
        self.bias -= self.learning_rate * bias_gradient

        return weight_gradient, bias_gradient

    def _elu(self, x: np.ndarray) -> np.ndarray:
        """Función de activación ELU"""
        return np.where(x > 0, x, np.exp(x) - 1)

    def _elu_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivada de ELU"""
        return np.where(x > 0, 1, np.exp(x))

    def detect_patterns(self, code: str) -> Dict[str, Any]:
        """
        Detecta patrones de diseño y anti-patrones en el código

        Returns:
            Dict con patrones detectados y sugerencias
        """
        try:
            tree = ast.parse(code)
            detector = PatternDetector()
            detector.visit(tree)

            patterns = {
                'singleton': detector.has_singleton,
                'factory': detector.has_factory,
                'observer': detector.has_observer,
                'strategy': detector.has_strategy,
                'decorator': detector.has_decorator
            }

            antipatterns = detector.get_antipatterns()

            return {
                'patterns': patterns,
                'antipatterns': antipatterns,
                'suggestions': self._generate_pattern_suggestions(patterns, antipatterns)
            }

        except SyntaxError as e:
            return {'error': f'Syntax error: {str(e)}'}

    def _generate_pattern_suggestions(self, patterns: Dict[str, bool],
                                      antipatterns: List[str]) -> List[Dict[str, str]]:
        """
        Genera sugerencias de patrones a aplicar
        """
        suggestions = []

        # Sugerir Factory Pattern si hay mucha creación de objetos
        if 'god_object' in antipatterns:
            suggestions.append({
                'pattern': 'factory',
                'description': 'Usar Factory Pattern para crear objetos',
                'benefit': 'Reduce acoplamiento y facilita testing'
            })

        # Sugerir Observer Pattern si hay mucha comunicación entre objetos
        if 'tight_coupling' in antipatterns:
            suggestions.append({
                'pattern': 'observer',
                'description': 'Usar Observer Pattern para comunicación desacoplada',
                'benefit': 'Reduce dependencias entre componentes'
            })

        # Sugerir Strategy Pattern si hay muchos condicionales
        if 'conditional_complexity' in antipatterns:
            suggestions.append({
                'pattern': 'strategy',
                'description': 'Usar Strategy Pattern para reemplazar condicionales',
                'benefit': 'Mejora extensibilidad y mantenibilidad'
            })

        return suggestions

    def suggest_refactoring(self, code: str) -> str:
        """
        Sugiere refactorización aplicando patrones de diseño
        """
        detection_result = self.detect_patterns(code)

        if 'error' in detection_result:
            return code

        suggestions_text = "# SUGGERENCIAS DE PATRONES DE DISEÑO:\n\n"

        for suggestion in detection_result.get('suggestions', []):
            suggestions_text += f"# {suggestion['description']}\n"
            suggestions_text += f"# Beneficio: {suggestion['benefit']}\n\n"

        return suggestions_text + code


class PatternDetector(ast.NodeVisitor):
    """Detecta patrones de diseño en el código"""

    def __init__(self):
        self.has_singleton = False
        self.has_factory = False
        self.has_observer = False
        self.has_strategy = False
        self.has_decorator = False
        self.class_definitions = []
        self.method_definitions = []
        self.conditional_count = 0
        self.coupling_score = 0

    def visit_ClassDef(self, node: ast.ClassDef):
        self.class_definitions.append(node.name)

        # Detectar Singleton Pattern
        if self._is_singleton(node):
            self.has_singleton = True

        # Detectar Factory Pattern
        if self._is_factory(node):
            self.has_factory = True

        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.method_definitions.append(node.name)

        # Detectar Decorator Pattern
        if any(ast.Constant(s=str(node.decorator_list)) for decorator in node.decorator_list):
            self.has_decorator = True

        # Contar condicionales para detectar necesidad de Strategy
        for n in ast.walk(node):
            if isinstance(n, ast.If):
                self.conditional_count += 1

        self.generic_visit(node)

    def _is_singleton(self, node: ast.ClassDef) -> bool:
        """Detecta si una clase implementa Singleton Pattern"""
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                if item.name == '__new__' or item.name == '__instance__':
                    return True
        return False

    def _is_factory(self, node: ast.ClassDef) -> bool:
        """Detecta si una clase implementa Factory Pattern"""
        methods = [item for item in node.body if isinstance(item, ast.FunctionDef)]
        for method in methods:
            if 'create' in method.name.lower() or 'factory' in method.name.lower():
                return True
        return False

    def get_antipatterns(self) -> List[str]:
        """Identifica anti-patrones en el código"""
        antipatterns = []

        # God Object
        if len(self.class_definitions) == 1 and len(self.method_definitions) > 20:
            antipatterns.append('god_object')

        # Conditional Complexity
        if self.conditional_count > 10:
            antipatterns.append('conditional_complexity')

        # Tight Coupling (simplificado)
        if self.coupling_score > 50:
            antipatterns.append('tight_coupling')

        return antipatterns
