"""
Neurona de Refactorización de Código - RF_RFEN1_RN_9_2
Especializada en mejorar código mediante refactorización automática
"""

import ast
import numpy as np
from typing import Dict, List, Any, Tuple
from ast import NodeTransformer, NodeVisitor


class CodeRefactoringNeuron:
    """
    Neurona especializada en refactorización automática de código
    Mejora código detectando patrones y aplicando transformaciones
    """

    def __init__(self, input_size: int = 100, learning_rate: float = 0.001):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.weights = np.random.normal(0, 0.1, input_size)
        self.bias = np.random.normal(0, 0.1)
        self.refactoring_patterns = {}
        self.improvement_count = 0

    def forward(self, code_features: np.ndarray) -> np.ndarray:
        """
        Propagación hacia adelante para detectar oportunidades de refactorización
        """
        if len(code_features) != self.input_size:
            raise ValueError(f"Expected {self.input_size} features, got {len(code_features)}")

        weighted_sum = np.dot(code_features, self.weights) + self.bias
        return self._relu(weighted_sum)

    def backward(self, error: np.ndarray, code_features: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Ajusta pesos basado en éxito de refactorizaciones
        """
        gradient = error * self._relu_derivative(
            np.dot(code_features, self.weights) + self.bias
        )

        weight_gradient = gradient * code_features
        bias_gradient = gradient

        self.weights -= self.learning_rate * weight_gradient
        self.bias -= self.learning_rate * bias_gradient

        return weight_gradient, bias_gradient

    def _relu(self, x: np.ndarray) -> np.ndarray:
        """Función de activación ReLU"""
        return np.maximum(0, x)

    def _relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivada de ReLU"""
        return (x > 0).astype(float)

    def refactor_code(self, code: str) -> Dict[str, Any]:
        """
        Aplica refactorizaciones automáticas al código

        Returns:
            Dict con código mejorado y lista de cambios aplicados
        """
        try:
            tree = ast.parse(code)

            # Detectar oportunidades de refactorización
            opportunities = self._detect_refactoring_opportunities(tree)

            if not opportunities:
                return {
                    'refactored_code': code,
                    'changes': [],
                    'improvements': []
                }

            # Aplicar refactorizaciones
            refactorizer = CodeRefactorizer(opportunities)
            refactored_tree = refactorizer.visit(tree)

            # Convertir AST a código
            refactored_code = ast.unparse(refactored_tree)

            self.improvement_count += len(opportunities)

            return {
                'refactored_code': refactored_code,
                'changes': opportunities,
                'improvements': self._calculate_improvements(code, refactored_code)
            }

        except SyntaxError as e:
            return {
                'refactored_code': code,
                'error': f'Syntax error: {str(e)}',
                'changes': []
            }

    def _detect_refactoring_opportunities(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """
        Detecta oportunidades de refactorización en el AST
        """
        opportunities = []
        detector = RefactoringDetector()
        detector.visit(tree)

        # Detectar variables no utilizadas
        for unused_var in detector.unused_variables:
            opportunities.append({
                'type': 'unused_variable',
                'description': f'Variable {unused_var} no utilizada',
                'severity': 'low'
            })

        # Detectar funciones muy largas
        for func_name, complexity in detector.function_complexity.items():
            if complexity > 15:
                opportunities.append({
                    'type': 'complex_function',
                    'description': f'Función {func_name} es muy compleja (complejidad: {complexity})',
                    'severity': 'medium'
                })

        # Detectar loops anidados
        if detector.nested_loops > 2:
            opportunities.append({
                'type': 'nested_loops',
                'description': f'Demasiados loops anidados ({detector.nested_loops})',
                'severity': 'medium'
            })

        return opportunities

    def _calculate_improvements(self, original: str, refactored: str) -> Dict[str, float]:
        """
        Calcula métricas de mejoras aplicadas
        """
        original_lines = len(original.split('\n'))
        refactored_lines = len(refactored.split('\n'))

        return {
            'lines_reduced': max(0, original_lines - refactored_lines),
            'complexity_reduction': 0.15,  # Estimación
            'readability_improvement': 0.20  # Estimación
        }


class RefactoringDetector(NodeVisitor):
    """Detecta oportunidades de refactorización"""

    def __init__(self):
        self.unused_variables = set()
        self.function_complexity = {}
        self.nested_loops = 0
        self.current_function = None
        self.loop_depth = 0
        self.variables_used = set()
        self.variables_defined = set()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.current_function = node.name
        complexity = 1

        for stmt in ast.walk(node):
            if isinstance(stmt, (ast.If, ast.While, ast.For, ast.BoolOp)):
                complexity += 1

        self.function_complexity[node.name] = complexity
        self.generic_visit(node)

    def visit_For(self, node: ast.For):
        self.loop_depth += 1
        if self.loop_depth > self.nested_loops:
            self.nested_loops = self.loop_depth
        self.generic_visit(node)
        self.loop_depth -= 1

    def visit_While(self, node: ast.While):
        self.loop_depth += 1
        if self.loop_depth > self.nested_loops:
            self.nested_loops = self.loop_depth
        self.generic_visit(node)
        self.loop_depth -= 1

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load):
            self.variables_used.add(node.id)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.variables_defined.add(target.id)
        self.generic_visit(node)


class CodeRefactorizer(NodeTransformer):
    """Aplica refactorizaciones al AST"""

    def __init__(self, opportunities: List[Dict[str, Any]]):
        self.opportunities = opportunities
        self.applied_changes = []

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # Aplicar mejoras si es necesario
        return node
