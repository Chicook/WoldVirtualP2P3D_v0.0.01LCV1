"""
Neurona de Análisis de Código - RF_RFEN1_RN_9_1
Especializada en análisis estático de código Python usando AST
"""

import ast
import numpy as np
from typing import Dict, List, Any, Tuple
from collections import defaultdict


class CodeAnalysisNeuron:
    """
    Neurona especializada en análisis de código usando AST (Abstract Syntax Tree)
    Analiza estructura, complejidad y patrones de código
    """

    def __init__(self, input_size: int = 100, learning_rate: float = 0.001):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.weights = np.random.normal(0, 0.1, input_size)
        self.bias = np.random.normal(0, 0.1)
        self.code_patterns = defaultdict(int)
        self.complexity_metrics = {}

    def analyze_code_structure(self, code: str) -> Dict[str, Any]:
        """
        Analiza la estructura del código usando AST

        Returns:
            Dict con métricas de análisis
        """
        try:
            tree = ast.parse(code)
            analyzer = ASTAnalyzer()
            analyzer.visit(tree)

            metrics = {
                'functions': analyzer.functions,
                'classes': analyzer.classes,
                'imports': analyzer.imports,
                'variables': analyzer.variables,
                'loops': analyzer.loops,
                'conditionals': analyzer.conditionals,
                'nesting_depth': analyzer.max_depth,
                'lines_of_code': len(code.split('\n')),
                'cyclomatic_complexity': self._calculate_cyclomatic_complexity(tree)
            }

            return metrics
        except SyntaxError as e:
            return {'error': f'Syntax error: {str(e)}'}

    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """
        Calcula complejidad ciclomática del código
        """
        complexity = 1  # Base complexity

        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.With, ast.Try)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1

        return complexity

    def forward(self, code_features: np.ndarray) -> np.ndarray:
        """
        Propagación hacia adelante
        """
        if len(code_features) != self.input_size:
            raise ValueError(f"Expected {self.input_size} features, got {len(code_features)}")

        weighted_sum = np.dot(code_features, self.weights) + self.bias
        return self._sigmoid(weighted_sum)

    def backward(self, error: np.ndarray, code_features: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Propagación hacia atrás - ajusta pesos y bias
        """
        gradient = error * self._sigmoid_derivative(
            np.dot(code_features, self.weights) + self.bias
        )

        weight_gradient = gradient * code_features
        bias_gradient = gradient

        self.weights -= self.learning_rate * weight_gradient
        self.bias -= self.learning_rate * bias_gradient

        return weight_gradient, bias_gradient

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Función de activación sigmoid"""
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

    def _sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivada de sigmoid"""
        s = self._sigmoid(x)
        return s * (1 - s)

    def extract_code_features(self, code: str) -> np.ndarray:
        """
        Extrae features numéricos del código
        """
        metrics = self.analyze_code_structure(code)

        if 'error' in metrics:
            return np.zeros(self.input_size)

        # Convertir métricas a vector numérico
        features = [
            len(metrics.get('functions', [])),
            len(metrics.get('classes', [])),
            len(metrics.get('imports', [])),
            len(metrics.get('variables', [])),
            len(metrics.get('loops', [])),
            len(metrics.get('conditionals', [])),
            metrics.get('nesting_depth', 0),
            metrics.get('lines_of_code', 0),
            metrics.get('cyclomatic_complexity', 0)
        ]

        # Completar hasta input_size con padding
        if len(features) < self.input_size:
            features.extend([0] * (self.input_size - len(features)))
        else:
            features = features[:self.input_size]

        return np.array(features, dtype=np.float32)


class ASTAnalyzer(ast.NodeVisitor):
    """
    Visitador de AST para análisis de código
    """

    def __init__(self):
        self.functions = []
        self.classes = []
        self.imports = []
        self.variables = []
        self.loops = []
        self.conditionals = []
        self.max_depth = 0
        self.current_depth = 0

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.functions.append(node.name)
        self.current_depth += 1
        if self.current_depth > self.max_depth:
            self.max_depth = self.current_depth
        self.generic_visit(node)
        self.current_depth -= 1

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.functions.append(node.name)
        self.current_depth += 1
        if self.current_depth > self.max_depth:
            self.max_depth = self.current_depth
        self.generic_visit(node)
        self.current_depth -= 1

    def visit_ClassDef(self, node: ast.ClassDef):
        self.classes.append(node.name)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import):
        self.imports.extend([alias.name for alias in node.names])
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            self.imports.append(node.module)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.variables.append(target.id)
        self.generic_visit(node)

    def visit_For(self, node: ast.For):
        self.loops.append(type(node).__name__)
        self.generic_visit(node)

    def visit_While(self, node: ast.While):
        self.loops.append(type(node).__name__)
        self.generic_visit(node)

    def visit_If(self, node: ast.If):
        self.conditionals.append(1)
        self.generic_visit(node)
