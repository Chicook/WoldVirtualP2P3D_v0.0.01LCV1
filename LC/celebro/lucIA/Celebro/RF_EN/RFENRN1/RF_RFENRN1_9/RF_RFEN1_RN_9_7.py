"""
Neurona de Análisis de Seguridad - RF_RFEN1_RN_9_7
Especializada en detectar vulnerabilidades y problemas de seguridad
"""

import ast
import numpy as np
from typing import Dict, List, Any, Tuple
import re


class SecurityAnalyzerNeuron:
    """
    Neurona especializada en detectar vulnerabilidades de seguridad
    Identifica patrones inseguros y sugiere soluciones
    """

    def __init__(self, input_size: int = 100, learning_rate: float = 0.001):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.weights = np.random.normal(0, 0.1, input_size)
        self.bias = np.random.normal(0, 0.1)
        self.security_issues = []
        self.vulnerability_patterns = self._init_vulnerability_patterns()

    def _init_vulnerability_patterns(self) -> Dict[str, List[str]]:
        """
        Inicializa patrones de vulnerabilidades conocidas
        """
        return {
            'sql_injection': ['execute', 'executemany', 'query', 'cursor', '%s', '?'],
            'command_injection': ['os.system', 'subprocess.call', 'eval', 'exec', 'shell=True'],
            'path_traversal': ['../', '..\\', 'open', 'file', 'path'],
            'hardcoded_secrets': ['password', 'api_key', 'secret', 'token', 'auth'],
            'weak_crypto': ['md5', 'sha1', 'DES', 'RC4'],
            'xss': ['innerHTML', 'document.write', 'eval'],
            'xxe': ['lxml', 'xml.etree', 'XMLParser'],
            'ssrf': ['urllib', 'requests.get', 'httplib'],
            'mass_assignment': ['**kwargs', '.update', '__dict__'],
            'log_injection': ['log', 'logger', 'print']
        }

    def forward(self, code_features: np.ndarray) -> np.ndarray:
        """
        Detecta vulnerabilidades en el código
        """
        if len(code_features) != self.input_size:
            raise ValueError(f"Expected {self.input_size} features, got {len(code_features)}")

        weighted_sum = np.dot(code_features, self.weights) + self.bias
        return self._gelu(weighted_sum)

    def backward(self, error: np.ndarray, code_features: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Ajusta pesos basado en detección de vulnerabilidades
        """
        gradient = error * self._gelu_derivative(
            np.dot(code_features, self.weights) + self.bias
        )

        weight_gradient = gradient * code_features
        bias_gradient = gradient

        self.weights -= self.learning_rate * weight_gradient
        self.bias -= self.learning_rate * bias_gradient

        return weight_gradient, bias_gradient

    def _gelu(self, x: np.ndarray) -> np.ndarray:
        """Función de activación GELU"""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    def _gelu_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivada de GELU"""
        return 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    def scan_for_vulnerabilities(self, code: str) -> Dict[str, Any]:
        """
        Escanea código en busca de vulnerabilidades de seguridad

        Returns:
            Dict con vulnerabilidades encontradas
        """
        issues = []

        # Analizar código fuente
        for vuln_type, patterns in self.vulnerability_patterns.items():
            for pattern in patterns:
                if pattern in code:
                    issues.append({
                        'type': vuln_type,
                        'severity': self._get_severity(vuln_type),
                        'description': self._get_vulnerability_description(vuln_type),
                        'recommendation': self._get_recommendation(vuln_type),
                        'line': self._find_line(code, pattern)
                    })

        # Análisis AST para detección más precisa
        try:
            tree = ast.parse(code)
            ast_scanner = ASTSecurityScanner()
            ast_scanner.visit(tree)
            issues.extend(ast_scanner.security_issues)
        except SyntaxError:
            pass

        return {
            'total_issues': len(issues),
            'issues': issues,
            'risk_score': self._calculate_risk_score(issues),
            'security_level': self._determine_security_level(issues)
        }

    def _get_severity(self, vuln_type: str) -> str:
        """Determina severidad de una vulnerabilidad"""
        critical = ['sql_injection', 'command_injection', 'xxe']
        high = ['hardcoded_secrets', 'path_traversal', 'weak_crypto']

        if vuln_type in critical:
            return 'critical'
        elif vuln_type in high:
            return 'high'
        else:
            return 'medium'

    def _get_vulnerability_description(self, vuln_type: str) -> str:
        """Retorna descripción de la vulnerabilidad"""
        descriptions = {
            'sql_injection': 'Posible inyección SQL sin parametrización correcta',
            'command_injection': 'Ejecución de comandos del sistema sin validación',
            'hardcoded_secrets': 'Credenciales o secretos hardcodeados en el código',
            'weak_crypto': 'Uso de algoritmos de cifrado débiles o obsoletos',
            'path_traversal': 'Vulnerable a ataques de directory traversal',
            'xss': 'Posible Cross-Site Scripting si se usa HTML dinámico',
            'xxe': 'XML External Entity injection posible'
        }
        return descriptions.get(vuln_type, 'Problema de seguridad detectado')

    def _get_recommendation(self, vuln_type: str) -> str:
        """Retorna recomendación para solucionar vulnerabilidad"""
        recommendations = {
            'sql_injection': 'Usar parámetros preparados (prepared statements)',
            'command_injection': 'Validar y sanitizar entradas antes de ejecutar',
            'hardcoded_secrets': 'Usar variables de entorno o vault para secretos',
            'weak_crypto': 'Usar algoritmos modernos: SHA-256, AES-256, bcrypt',
            'path_traversal': 'Validar y sanitizar rutas de archivos',
            'xss': 'Escapar output HTML o usar templates seguros',
            'xxe': 'Deshabilitar procesamiento de entidades externas en XML'
        }
        return recommendations.get(vuln_type, 'Implementar validación y sanitización apropiada')

    def _find_line(self, code: str, pattern: str) -> int:
        """Encuentra línea donde aparece el patrón"""
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            if pattern in line:
                return i
        return 0

    def _calculate_risk_score(self, issues: List[Dict[str, Any]]) -> float:
        """Calcula score de riesgo (0-100)"""
        score = 0
        for issue in issues:
            severity = issue.get('severity', 'low')
            if severity == 'critical':
                score += 10
            elif severity == 'high':
                score += 5
            elif severity == 'medium':
                score += 2
            else:
                score += 1

        return min(100, score)

    def _determine_security_level(self, issues: List[Dict[str, Any]]) -> str:
        """Determina nivel de seguridad general"""
        critical_count = sum(1 for i in issues if i.get('severity') == 'critical')
        high_count = sum(1 for i in issues if i.get('severity') == 'high')

        if critical_count > 0:
            return 'CRITICAL'
        elif high_count > 2:
            return 'HIGH'
        elif high_count > 0 or len(issues) > 5:
            return 'MEDIUM'
        elif len(issues) > 0:
            return 'LOW'
        else:
            return 'SAFE'


class ASTSecurityScanner(ast.NodeVisitor):
    """Scanner de seguridad usando AST"""

    def __init__(self):
        self.security_issues = []

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Attribute):
            # Detectar llamadas peligrosas
            if node.func.attr == 'system':
                self.security_issues.append({
                    'type': 'command_injection',
                    'severity': 'critical',
                    'description': 'Uso de os.system() - vulnerable a command injection',
                    'recommendation': 'Usar subprocess con validación de argumentos'
                })

            if node.func.attr in ['eval', 'exec']:
                self.security_issues.append({
                    'type': 'code_injection',
                    'severity': 'critical',
                    'description': f'Uso de {node.func.attr}() - vulnerable a code injection',
                    'recommendation': 'Evitar eval/exec, usar alternativas seguras'
                })

        self.generic_visit(node)

    def visit_Str(self, node: ast.Str):
        # Detectar posibles credenciales hardcodeadas
        value = node.s.lower()
        suspicious_keywords = ['password=', 'secret=', 'api_key=', 'token=']

        for keyword in suspicious_keywords:
            if keyword in value:
                self.security_issues.append({
                    'type': 'hardcoded_secrets',
                    'severity': 'high',
                    'description': 'Posibles credenciales hardcodeadas detectadas',
                    'recommendation': 'Mover credenciales a variables de entorno'
                })

        self.generic_visit(node)
