"""
Neurona de Calidad de Código - RF_RFEN1_RN_9_10
Especializada en evaluar calidad general del código y sugerir mejoras
"""

import ast
import numpy as np
from typing import Dict, List, Any, Tuple


class CodeQualityNeuron:
    """
    Neurona especializada en evaluación de calidad general del código
    Integra análisis de complejidad, seguridad, rendimiento y documentación
    """

    def __init__(self, input_size: int = 100, learning_rate: float = 0.001):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.weights = np.random.normal(0, 0.1, input_size)
        self.bias = np.random.normal(0, 0.1)
        self.quality_scores = {}
        self.improvement_history = []

    def forward(self, code_features: np.ndarray) -> np.ndarray:
        """
        Evalúa calidad general del código
        """
        if len(code_features) != self.input_size:
            raise ValueError(f"Expected {self.input_size} features, got {len(code_features)}")

        weighted_sum = np.dot(code_features, self.weights) + self.bias
        return self._mish(weighted_sum)

    def backward(self, error: np.ndarray, code_features: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Ajusta pesos basado en mejoras de calidad aplicadas
        """
        x = np.dot(code_features, self.weights) + self.bias
        gradient = error * self._mish_derivative(x)

        weight_gradient = gradient * code_features
        bias_gradient = gradient

        self.weights -= self.learning_rate * weight_gradient
        self.bias -= self.learning_rate * bias_gradient

        return weight_gradient, bias_gradient

    def _mish(self, x: np.ndarray) -> np.ndarray:
        """Función de activación Mish"""
        return x * np.tanh(np.log(1 + np.exp(x)))

    def _mish_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivada de Mish"""
        omega = 1 + np.exp(x)
        delta = 2 * np.log(omega) + (x * omega) / omega
        return np.tanh(np.log(omega)) + (x * omega) / (1 + omega**2)

    def evaluate_code_quality(self, code: str) -> Dict[str, Any]:
        """
        Evalúa calidad general del código y genera reporte completo

        Returns:
            Dict con score de calidad y desglose por categorías
        """
        try:
            tree = ast.parse(code)

            # Múltiples análisis
            complexity_analyzer = QualityASTAnalyzer()
            complexity_analyzer.visit(tree)

            metrics = {
                'structure_score': self._evaluate_structure(complexity_analyzer),
                'readability_score': self._evaluate_readability(complexity_analyzer),
                'maintainability_score': self._evaluate_maintainability(complexity_analyzer),
                'testability_score': self._evaluate_testability(complexity_analyzer),
                'overall_score': 0.0,  # Se calculará
                'issues': complexity_analyzer.get_issues(),
                'recommendations': []
            }

            # Calcular score general
            metrics['overall_score'] = self._calculate_overall_score(metrics)

            # Generar recomendaciones
            metrics['recommendations'] = self._generate_quality_recommendations(metrics)

            self.quality_scores = metrics
            return metrics

        except SyntaxError as e:
            return {'error': f'Syntax error: {str(e)}'}

    def _evaluate_structure(self, analyzer: 'QualityASTAnalyzer') -> float:
        """
        Evalúa estructura del código (0-100)
        """
        score = 100.0

        # Penalizar estructura irregular
        if analyzer.long_lines > 5:
            score -= 10
        if analyzer.deep_nesting > 3:
            score -= 15
        if not analyzer.has_main_function:
            score -= 5
        if analyzer.imports_at_end:
            score -= 10

        return max(0, score)

    def _evaluate_readability(self, analyzer: 'QualityASTAnalyzer') -> float:
        """
        Evalúa legibilidad del código (0-100)
        """
        score = 100.0

        # Penalizar falta de documentación
        if len(analyzer.functions_without_docs) > 0:
            score -= 5 * min(len(analyzer.functions_without_docs), 10)

        # Penalizar código denso
        if analyzer.avg_function_length > 50:
            score -= 20
        elif analyzer.avg_function_length > 30:
            score -= 10

        # Penalizar nombres pobres
        if analyzer.short_names > 3:
            score -= 15

        return max(0, score)

    def _evaluate_maintainability(self, analyzer: 'QualityASTAnalyzer') -> float:
        """
        Evalúa mantenibilidad del código (0-100)
        """
        score = 100.0

        # Penalizar alta complejidad
        if analyzer.complexity > 20:
            score -= 25
        elif analyzer.complexity > 10:
            score -= 15

        # Penalizar código duplicado
        if analyzer.duplicate_patterns > 3:
            score -= 20

        # Penalizar muchos argumentos
        if analyzer.max_arguments > 5:
            score -= 10

        return max(0, score)

    def _evaluate_testability(self, analyzer: 'QualityASTAnalyzer') -> float:
        """
        Evalúa testabilidad del código (0-100)
        """
        score = 100.0

        # Penalizar dependencias externas
        if analyzer.external_dependencies > 10:
            score -= 15

        # Penalizar funciones muy largas
        if analyzer.max_function_length > 100:
            score -= 25

        # Penalizar código no modular
        if analyzer.tightly_coupled:
            score -= 20

        return max(0, score)

    def _calculate_overall_score(self, metrics: Dict[str, float]) -> float:
        """
        Calcula score general ponderado
        """
        weights = {
            'structure_score': 0.20,
            'readability_score': 0.25,
            'maintainability_score': 0.30,
            'testability_score': 0.25
        }

        overall = sum(
            metrics.get(key, 0) * weight
            for key, weight in weights.items()
        )

        return round(overall, 2)

    def _generate_quality_recommendations(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Genera recomendaciones basadas en métricas de calidad
        """
        recommendations = []

        if metrics.get('overall_score', 0) < 70:
            recommendations.append({
                'priority': 'high',
                'category': 'general',
                'suggestion': 'Aplicar refactorización general para mejorar calidad',
                'estimated_impact': '30-50%'
            })

        if metrics.get('readability_score', 0) < 70:
            recommendations.append({
                'priority': 'medium',
                'category': 'readability',
                'suggestion': 'Mejorar nombres de variables y funciones',
                'estimated_impact': '20-30%'
            })

        if metrics.get('maintainability_score', 0) < 70:
            recommendations.append({
                'priority': 'high',
                'category': 'maintainability',
                'suggestion': 'Reducir complejidad y extraer funciones',
                'estimated_impact': '25-40%'
            })

        return recommendations

    def get_quality_report(self, code: str) -> str:
        """
        Genera reporte de calidad en formato texto
        """
        metrics = self.evaluate_code_quality(code)

        report = f"""
╔═══════════════════════════════════════════════════╗
║       REPORTE DE CALIDAD DE CÓDIGO               ║
╚═══════════════════════════════════════════════════╝

Score General: {metrics.get('overall_score', 0)}/100

Desglose por Categorías:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • Estructura:        {metrics.get('structure_score', 0)}/100
  • Legibilidad:       {metrics.get('readability_score', 0)}/100
  • Mantenibilidad:    {metrics.get('maintainability_score', 0)}/100
  • Testabilidad:      {metrics.get('testability_score', 0)}/100

Problemas Detectados: {len(metrics.get('issues', []))}
"""

        if metrics.get('recommendations'):
            report += "\nRecomendaciones Prioritarias:\n"
            for i, rec in enumerate(metrics.get('recommendations', [])[:3], 1):
                report += f"\n{i}. [{rec.get('priority', 'medium').upper()}] {rec.get('suggestion', '')}\n"
                report += f"   Impacto estimado: {rec.get('estimated_impact', 'N/A')}\n"

        return report


class QualityASTAnalyzer(ast.NodeVisitor):
    """Analizador de calidad general del código"""

    def __init__(self):
        self.long_lines = 0
        self.deep_nesting = 0
        self.has_main_function = False
        self.imports_at_end = False
        self.functions_without_docs = []
        self.avg_function_length = 0
        self.short_names = 0
        self.complexity = 0
        self.duplicate_patterns = 0
        self.max_arguments = 0
        self.external_dependencies = 0
        self.max_function_length = 0
        self.tightly_coupled = False
        self.current_nesting = 0
        self.functions = []

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.functions.append(node)

        # Detectar función main
        if node.name == 'main' or '__main__' in str(node.decorator_list):
            self.has_main_function = True

        # Contar argumentos
        if len(node.args.args) > self.max_arguments:
            self.max_arguments = len(node.args.args)

        # Verificar documentación
        if not ast.get_docstring(node):
            self.functions_without_docs.append(node.name)

        # Verificar longitud de nombres
        if len(node.name) < 3:
            self.short_names += 1

        self.generic_visit(node)

    def visit_If(self, node: ast.If):
        self.complexity += 1
        self.current_nesting += 1
        if self.current_nesting > self.deep_nesting:
            self.deep_nesting = self.current_nesting
        self.generic_visit(node)
        self.current_nesting -= 1

    def visit_Call(self, node: ast.Call):
        self.external_dependencies += 1
        self.generic_visit(node)

    def visit_For(self, node: ast.For):
        self.complexity += 1
        self.generic_visit(node)

    def visit_While(self, node: ast.While):
        self.complexity += 1
        self.generic_visit(node)

    def get_issues(self) -> List[str]:
        """Retorna lista de problemas detectados"""
        issues = []

        if self.deep_nesting > 3:
            issues.append('Demasiadas capas de anidación')
        if len(self.functions_without_docs) > 3:
            issues.append(f'{len(self.functions_without_docs)} funciones sin documentar')
        if self.short_names > 5:
            issues.append('Muchos nombres de variables muy cortos')

        return issues
