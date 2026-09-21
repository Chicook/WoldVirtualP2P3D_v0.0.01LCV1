"""
Neurona de Generación de Tests - RF_RFEN1_RN_9_3
Especializada en generar tests unitarios automáticamente
"""

import ast
import numpy as np
from typing import Dict, List, Any, Tuple


class TestGenerationNeuron:
    """
    Neurona especializada en generar tests unitarios automáticamente
    Analiza funciones y clases para crear tests de cobertura
    """

    def __init__(self, input_size: int = 100, learning_rate: float = 0.001):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.weights = np.random.normal(0, 0.1, input_size)
        self.bias = np.random.normal(0, 0.1)
        self.generated_tests = []
        self.test_coverage = {}

    def forward(self, code_features: np.ndarray) -> np.ndarray:
        """
        Propaga para identificar funciones que necesitan tests
        """
        if len(code_features) != self.input_size:
            raise ValueError(f"Expected {self.input_size} features, got {len(code_features)}")

        weighted_sum = np.dot(code_features, self.weights) + self.bias
        return self._tanh(weighted_sum)

    def backward(self, error: np.ndarray, code_features: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Ajusta pesos basado en calidad de tests generados
        """
        gradient = error * (1 - np.tanh(np.dot(code_features, self.weights) + self.bias) ** 2)

        weight_gradient = gradient * code_features
        bias_gradient = gradient

        self.weights -= self.learning_rate * weight_gradient
        self.bias -= self.learning_rate * bias_gradient

        return weight_gradient, bias_gradient

    def _tanh(self, x: np.ndarray) -> np.ndarray:
        """Función de activación tanh"""
        return np.tanh(x)

    def generate_tests(self, code: str) -> str:
        """
        Genera código de tests unitarios para el código dado

        Returns:
            String con código de tests generado
        """
        try:
            tree = ast.parse(code)
            extractor = TestExtractor()
            extractor.visit(tree)

            tests_code = self._generate_test_template(extractor.functions, extractor.classes)

            self.generated_tests.append({
                'original_code': code,
                'tests': tests_code,
                'coverage_estimate': len(extractor.functions) + len(extractor.classes)
            })

            return tests_code

        except SyntaxError as e:
            return f"# Error parsing code: {str(e)}\n\n# No tests generated"

    def _generate_test_template(self, functions: List[str], classes: List[str]) -> str:
        """
        Genera plantilla de tests para las funciones y clases detectadas
        """
        test_code = """import unittest
import sys
from typing import Any, List, Dict

class TestSuite(unittest.TestCase):
    \"\"\"Tests unitarios generados automáticamente\"\"\"
    
    def setUp(self):
        \"\"\"Configuración inicial para cada test\"\"\"
        pass
    
    def tearDown(self):
        \"\"\"Limpieza después de cada test\"\"\"
        pass
    
"""

        # Generar tests para funciones
        for func_name in functions:
            test_code += f"""    def test_{func_name}(self):
        \"\"\"Test para función {func_name}\"\"\"
        # TODO: Implementar test específico
        # Ejemplo:
        # result = {func_name}(test_input)
        # self.assertIsNotNone(result)
        pass
    
"""

        # Generar tests para clases
        for class_name in classes:
            test_code += f"""    def test_{class_name}_initialization(self):
        \"\"\"Test para inicialización de {class_name}\"\"\"
        # TODO: Implementar test específico
        # Ejemplo:
        # instance = {class_name}()
        # self.assertIsNotNone(instance)
        pass
    
    def test_{class_name}_methods(self):
        \"\"\"Test para métodos de {class_name}\"\"\"
        # TODO: Implementar tests para métodos
        pass
    
"""

        test_code += """if __name__ == '__main__':
    unittest.main()
"""

        return test_code

    def generate_parametrized_test(self, func_info: Dict[str, Any]) -> str:
        """
        Genera test parametrizado para una función específica
        """
        func_name = func_info.get('name', 'unknown_function')
        params = func_info.get('params', [])

        test_code = f"""    def test_{func_name}_parametrized(self):
        \"\"\"Test parametrizado para {func_name}\"\"\"
        test_cases = [
            # TODO: Agregar casos de test
            # {{'input': ..., 'expected': ...}},
        ]
        
        for i, test_case in enumerate(test_cases):
            with self.subTest(case=i):
                # result = {func_name}(test_case['input'])
                # self.assertEqual(result, test_case['expected'])
                pass
"""
        return test_code


class TestExtractor(ast.NodeVisitor):
    """Extrae información necesaria para generar tests"""

    def __init__(self):
        self.functions = []
        self.classes = []
        self.current_class = None

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if self.current_class:
            # Es un método
            self.functions.append(f"{self.current_class}.{node.name}")
        else:
            # Es una función
            self.functions.append(node.name)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        if self.current_class:
            self.functions.append(f"{self.current_class}.{node.name}")
        else:
            self.functions.append(node.name)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        self.classes.append(node.name)
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class
