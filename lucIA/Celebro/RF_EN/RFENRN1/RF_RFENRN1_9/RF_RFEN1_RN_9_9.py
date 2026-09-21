"""
Neurona de Documentación - RF_RFEN1_RN_9_9
Especializada en generar y mejorar documentación de código
"""

import ast
import numpy as np
from typing import Dict, List, Any, Tuple


class DocumentationNeuron:
    """
    Neurona especializada en generar documentación automática
    Crea docstrings, comentarios y documentación técnica
    """

    def __init__(self, input_size: int = 100, learning_rate: float = 0.001):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.weights = np.random.normal(0, 0.1, input_size)
        self.bias = np.random.normal(0, 0.1)
        self.documents_generated = []
        self.documentation_quality = {}

    def forward(self, code_features: np.ndarray) -> np.ndarray:
        """
        Evalúa necesidad de documentación
        """
        if len(code_features) != self.input_size:
            raise ValueError(f"Expected {self.input_size} features, got {len(code_features)}")

        weighted_sum = np.dot(code_features, self.weights) + self.bias
        return self._softplus(weighted_sum)

    def backward(self, error: np.ndarray, code_features: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Ajusta pesos basado en calidad de documentación generada
        """
        gradient = error * self._softplus_derivative(
            np.dot(code_features, self.weights) + self.bias
        )

        weight_gradient = gradient * code_features
        bias_gradient = gradient

        self.weights -= self.learning_rate * weight_gradient
        self.bias -= self.learning_rate * bias_gradient

        return weight_gradient, bias_gradient

    def _softplus(self, x: np.ndarray) -> np.ndarray:
        """Función de activación Softplus"""
        return np.log(1 + np.exp(x))

    def _softplus_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivada de Softplus"""
        return 1 / (1 + np.exp(-x))

    def analyze_documentation(self, code: str) -> Dict[str, Any]:
        """
        Analiza la documentación existente en el código

        Returns:
            Dict con métricas de documentación
        """
        try:
            tree = ast.parse(code)
            analyzer = DocumentationAnalyzer()
            analyzer.visit(tree)

            metrics = {
                'has_docstrings': analyzer.has_docstrings,
                'missing_docstrings': analyzer.missing_docstrings,
                'has_type_hints': analyzer.has_type_hints,
                'has_comments': analyzer.has_comments,
                'docstring_coverage': analyzer.get_docstring_coverage(),
                'type_hint_coverage': analyzer.get_type_hint_coverage(),
                'suggestions': self._generate_documentation_suggestions(analyzer)
            }

            return metrics

        except SyntaxError as e:
            return {'error': f'Syntax error: {str(e)}'}

    def generate_docstrings(self, code: str) -> str:
        """
        Genera docstrings automáticamente para funciones y clases sin documentar

        Returns:
            Código con docstrings añadidos
        """
        try:
            tree = ast.parse(code)
            generator = DocstringGenerator()
            modified_tree = generator.visit(tree)

            docstrings_added = generator.docstrings_added

            # Convertir AST modificado a código
            import astor  # Si está disponible
            try:
                docstringed_code = astor.to_source(modified_tree)
            except:
                # Fallback: añadir docstrings manualmente
                docstringed_code = self._add_docstrings_manually(code, docstrings_added)

            return docstringed_code

        except SyntaxError as e:
            return code  # Retornar código original si hay error

    def _add_docstrings_manually(self, code: str, docstrings: List[Dict[str, Any]]) -> str:
        """
        Añade docstrings manualmente al código
        """
        lines = code.split('\n')
        offset = 0

        for doc_info in sorted(docstrings, key=lambda x: x['line'], reverse=True):
            line_num = doc_info['line'] - 1 + offset
            docstring = doc_info['docstring']

            # Insertar docstring después de la definición
            lines.insert(line_num + 1, f'    """{docstring}"""')
            offset += 1

        return '\n'.join(lines)

    def _generate_documentation_suggestions(self, analyzer: 'DocumentationAnalyzer') -> List[str]:
        """
        Genera sugerencias para mejorar documentación
        """
        suggestions = []

        if len(analyzer.missing_docstrings) > 0:
            suggestions.append(f'Añadir docstrings a {len(analyzer.missing_docstrings)} funciones/clases')

        if analyzer.has_type_hints < len(analyzer.all_functions) * 0.5:
            suggestions.append('Añadir type hints a más del 50% de las funciones')

        if analyzer.has_comments < 10:
            suggestions.append('Añadir más comentarios explicativos en secciones complejas')

        return suggestions

    def generate_readme_content(self, code_info: Dict[str, Any]) -> str:
        """
        Genera contenido de README basado en el análisis del código
        """
        readme = f"""# {code_info.get('project_name', 'Project')}

## Descripción
{code_info.get('description', 'Proyecto desarrollado con Python')}

## Características
- {len(code_info.get('functions', []))} funciones implementadas
- {len(code_info.get('classes', []))} clases definidas
- Cobertura de documentación: {code_info.get('docstring_coverage', 0)}%

## Instalación
```bash
pip install -r requirements.txt
```

## Uso
```python
# Ejemplo de uso
from module import main
main()
```

## Desarrollo
Este proyecto fue generado y mejorado usando LucIA Neural Network System.

## Licencia
MIT License
"""
        return readme


class DocumentationAnalyzer(ast.NodeVisitor):
    """Analiza documentación en el código"""

    def __init__(self):
        self.has_docstrings = False
        self.missing_docstrings = []
        self.has_type_hints = 0
        self.has_comments = 0
        self.all_functions = []
        self.all_classes = []

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.all_functions.append(node.name)

        # Verificar docstring
        if not ast.get_docstring(node):
            self.missing_docstrings.append(node.name)
        else:
            self.has_docstrings = True

        # Verificar type hints
        if node.returns is not None or any(arg.annotation for arg in node.args.args):
            self.has_type_hints += 1

        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        self.all_classes.append(node.name)

        # Verificar docstring
        if not ast.get_docstring(node):
            self.missing_docstrings.append(f'class_{node.name}')
        else:
            self.has_docstrings = True

        self.generic_visit(node)

    def visit_Comment(self, node):
        self.has_comments += 1

    def get_docstring_coverage(self) -> float:
        """Calcula porcentaje de cobertura de docstrings"""
        total = len(self.all_functions) + len(self.all_classes)
        if total == 0:
            return 100.0
        documented = total - len(self.missing_docstrings)
        return (documented / total) * 100

    def get_type_hint_coverage(self) -> float:
        """Calcula porcentaje de cobertura de type hints"""
        if len(self.all_functions) == 0:
            return 100.0
        return (self.has_type_hints / len(self.all_functions)) * 100


class DocstringGenerator(ast.NodeTransformer):
    """Genera docstrings automáticamente"""

    def __init__(self):
        self.docstrings_added = []

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # Añadir docstring si no existe
        if not ast.get_docstring(node):
            docstring = self._generate_function_docstring(node)
            doc = ast.Constant(value=docstring)
            node.body.insert(0, ast.Expr(value=doc))

            self.docstrings_added.append({
                'name': node.name,
                'type': 'function',
                'line': node.lineno,
                'docstring': docstring
            })

        return node

    def visit_ClassDef(self, node: ast.ClassDef):
        # Añadir docstring si no existe
        if not ast.get_docstring(node):
            docstring = self._generate_class_docstring(node)
            doc = ast.Constant(value=docstring)
            node.body.insert(0, ast.Expr(value=doc))

            self.docstrings_added.append({
                'name': node.name,
                'type': 'class',
                'line': node.lineno,
                'docstring': docstring
            })

        return node

    def _generate_function_docstring(self, node: ast.FunctionDef) -> str:
        """Genera docstring para una función"""
        args = [arg.arg for arg in node.args.args]

        doc = f"\"\"\"\n    {node.name}\n\n"
        doc += "    Args:\n"
        for arg in args:
            doc += f"        {arg}: Description here\n"

        if node.returns:
            doc += "\n    Returns:\n        Description here\n"

        doc += "    \"\"\""
        return doc

    def _generate_class_docstring(self, node: ast.ClassDef) -> str:
        """Genera docstring para una clase"""
        methods = [item.name for item in node.body if isinstance(item, ast.FunctionDef)]

        doc = f"\"\"\"\n    {node.name}\n\n    Description here\n\n"

        if methods:
            doc += "    Methods:\n"
            for method in methods[:5]:  # Limitar a 5 métodos
                doc += f"        {method}: Description here\n"

        doc += "    \"\"\""
        return doc
