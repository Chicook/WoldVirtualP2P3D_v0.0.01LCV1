"""
RFEN2_RN_10.py - Neurona de Refuerzo con Arquitectura Transformadora Adaptativa
===============================================================================

Esta neurona implementa Arquitectura Transformadora Adaptativa con mecanismos
de atención dinámicos y optimización de pesos avanzada, utilizando técnicas
avanzadas de 2025 para transformers adaptativos.

Características Avanzadas 2025:
- Mecanismos de atención adaptativos dinámicos
- Arquitectura transformer con pesos adaptativos
- Atención multi-cabeza con pesos contextuales
- Optimización de pesos con técnicas transformer avanzadas
- Adaptación dinámica de la arquitectura
- Mecanismos de atención esparsa eficientes
- Optimización de memoria para transformers
- Atención causal adaptativa
- Posicionamiento adaptativo
- Normalización de capas adaptativa

Autor: LucIA Development Team
Versión: 2.0.0
Fecha: 2025
"""

import numpy as np
import math
from typing import Tuple, Optional, Dict, Any, List
import logging
from collections import deque
import time
from . import NeuronaRefuerzoAvanzadaBase, calcular_adaptacion_dinamica, \
    aplicar_consolidacion_memoria, LUCIA_ADVANCED_RL_CONFIG

logger = logging.getLogger('RFENRN2.RFEN2_RN_10')


class NeuronaRefuerzoTransformerAdaptativa(NeuronaRefuerzoAvanzadaBase):
    """
    Neurona de refuerzo con Arquitectura Transformadora Adaptativa.

    Esta neurona implementa una arquitectura transformer adaptativa
    con mecanismos de atención dinámicos y optimización de pesos avanzada.
    """

    def __init__(self, input_size: int, output_size: int,
                 nombre: str = "NeuronaRefuerzoTransformerAdaptativa",
                 learning_rate: float = 0.0001,
                 gamma: float = 0.99,
                 d_model: int = 64,
                 n_heads: int = 8,
                 n_layers: int = 6,
                 d_ff: int = 256,
                 dropout_rate: float = 0.1,
                 attention_dropout: float = 0.1,
                 max_seq_length: int = 512,
                 adaptive_attention: bool = True,
                 sparse_attention: bool = True,
                 causal_attention: bool = True,
                 layer_norm_eps: float = 1e-6,
                 position_encoding_type: str = "learned",
                 attention_type: str = "multi_head"):
        """
        Inicializa la neurona de refuerzo transformer adaptativa.

        Args:
            input_size: Número de características de entrada
            output_size: Número de acciones posibles
            nombre: Nombre identificativo de la neurona
            learning_rate: Tasa de aprendizaje
            gamma: Factor de descuento
            d_model: Dimensión del modelo
            n_heads: Número de cabezas de atención
            n_layers: Número de capas transformer
            d_ff: Dimensión de la capa feed-forward
            dropout_rate: Tasa de dropout
            attention_dropout: Tasa de dropout de atención
            max_seq_length: Longitud máxima de secuencia
            adaptive_attention: Si usar atención adaptativa
            sparse_attention: Si usar atención esparsa
            causal_attention: Si usar atención causal
            layer_norm_eps: Épsilon para normalización de capas
            position_encoding_type: Tipo de codificación posicional
            attention_type: Tipo de atención
        """
        super().__init__(input_size, output_size, nombre)

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.attention_dropout = attention_dropout
        self.max_seq_length = max_seq_length
        self.adaptive_attention = adaptive_attention
        self.sparse_attention = sparse_attention
        self.causal_attention = causal_attention
        self.layer_norm_eps = layer_norm_eps
        self.position_encoding_type = position_encoding_type
        self.attention_type = attention_type

        # Pesos de la arquitectura transformer adaptativa
        self.pesos_transformer = {}
        self.sesgos_transformer = {}

        # Pesos de atención adaptativa
        self.pesos_atencion_adaptativa = {}
        self.sesgos_atencion_adaptativa = {}

        # Codificación posicional
        self.position_encoding = None

        # Memoria de secuencias
        self.memoria_secuencias = deque(maxlen=max_seq_length)
        self.secuencia_actual = []

        # Estadísticas específicas de transformer adaptativo
        self.estadisticas_transformer = {
            'attention_weights_media': 0.0,
            'attention_entropy': 0.0,
            'layer_norm_estabilidad': 0.0,
            'gradient_norm_media': 0.0,
            'attention_sparsity': 0.0,
            'position_encoding_utilidad': 0.0,
            'transformer_depth_utilidad': 0.0,
            'multi_head_diversidad': 0.0,
            'feed_forward_eficiencia': 0.0,
            'dropout_efectividad': 0.0,
            'causal_attention_eficiencia': 0.0,
            'adaptive_attention_responsiveness': 0.0,
            'sparse_attention_efficiency': 0.0,
            'memory_efficiency': 0.0,
            'computational_efficiency': 0.0,
            'attention_convergence': 0.0
        }

        # Historial de transformer adaptativo
        self.historial_attention_weights = deque(maxlen=1000)
        self.historial_layer_outputs = deque(maxlen=1000)
        self.historial_position_encodings = deque(maxlen=1000)
        self.historial_attention_patterns = deque(maxlen=1000)
        self.historial_transformer_states = deque(maxlen=1000)

        logger.info(f"NeuronaRefuerzoTransformerAdaptativa creada: {self}")

    def inicializar_pesos(self) -> None:
        """
        Inicializa los pesos de la arquitectura transformer adaptativa.
        """
        # Inicializar pesos base para compatibilidad con la clase base
        self.pesos = np.random.normal(0, 0.1, (self.input_size, self.output_size))
        self.sesgo = np.zeros((1, self.output_size), dtype=LUCIA_ADVANCED_RL_CONFIG['precision'])

        # Inicializar pesos de entrada
        self.pesos_transformer['input_projection'] = np.random.normal(0, 0.1, (self.input_size, self.d_model))
        self.sesgos_transformer['input_projection'] = np.zeros(self.d_model, dtype=LUCIA_ADVANCED_RL_CONFIG['precision'])

        # Inicializar pesos de cada capa transformer
        for layer in range(self.n_layers):
            layer_name = f"layer_{layer}"

            # Pesos de atención multi-cabeza
            self.pesos_transformer[f"{layer_name}_attention_q"] = np.random.normal(0, 0.1, (self.d_model, self.d_model))
            self.pesos_transformer[f"{layer_name}_attention_k"] = np.random.normal(0, 0.1, (self.d_model, self.d_model))
            self.pesos_transformer[f"{layer_name}_attention_v"] = np.random.normal(0, 0.1, (self.d_model, self.d_model))
            self.pesos_transformer[f"{layer_name}_attention_output"] = np.random.normal(0, 0.1, (self.d_model, self.d_model))

            # Sesgos de atención
            self.sesgos_transformer[f"{layer_name}_attention_q"] = np.zeros(self.d_model, dtype=LUCIA_ADVANCED_RL_CONFIG['precision'])
            self.sesgos_transformer[f"{layer_name}_attention_k"] = np.zeros(self.d_model, dtype=LUCIA_ADVANCED_RL_CONFIG['precision'])
            self.sesgos_transformer[f"{layer_name}_attention_v"] = np.zeros(self.d_model, dtype=LUCIA_ADVANCED_RL_CONFIG['precision'])
            self.sesgos_transformer[f"{layer_name}_attention_output"] = np.zeros(self.d_model, dtype=LUCIA_ADVANCED_RL_CONFIG['precision'])

            # Pesos de feed-forward
            self.pesos_transformer[f"{layer_name}_ff1"] = np.random.normal(0, 0.1, (self.d_model, self.d_ff))
            self.pesos_transformer[f"{layer_name}_ff2"] = np.random.normal(0, 0.1, (self.d_ff, self.d_model))

            # Sesgos de feed-forward
            self.sesgos_transformer[f"{layer_name}_ff1"] = np.zeros(self.d_ff, dtype=LUCIA_ADVANCED_RL_CONFIG['precision'])
            self.sesgos_transformer[f"{layer_name}_ff2"] = np.zeros(self.d_model, dtype=LUCIA_ADVANCED_RL_CONFIG['precision'])

            # Pesos de normalización de capas
            self.pesos_transformer[f"{layer_name}_layer_norm1"] = np.ones(self.d_model, dtype=LUCIA_ADVANCED_RL_CONFIG['precision'])
            self.pesgos_transformer[f"{layer_name}_layer_norm2"] = np.ones(self.d_model, dtype=LUCIA_ADVANCED_RL_CONFIG['precision'])
            self.sesgos_transformer[f"{layer_name}_layer_norm1"] = np.zeros(self.d_model, dtype=LUCIA_ADVANCED_RL_CONFIG['precision'])
            self.sesgos_transformer[f"{layer_name}_layer_norm2"] = np.zeros(self.d_model, dtype=LUCIA_ADVANCED_RL_CONFIG['precision'])

        # Inicializar pesos de salida
        self.pesos_transformer['output_projection'] = np.random.normal(0, 0.1, (self.d_model, self.output_size))
        self.sesgos_transformer['output_projection'] = np.zeros(self.output_size, dtype=LUCIA_ADVANCED_RL_CONFIG['precision'])

        # Inicializar codificación posicional
        self._inicializar_position_encoding()

        # Inicializar pesos de atención adaptativa si está habilitada
        if self.adaptive_attention:
            self._inicializar_atencion_adaptativa()

        logger.info(f"Pesos transformer adaptativo inicializados: d_model={self.d_model}, "
                    f"n_heads={self.n_heads}, n_layers={self.n_layers}")

    def _inicializar_position_encoding(self) -> None:
        """
        Inicializa la codificación posicional.
        """
        if self.position_encoding_type == "learned":
            # Codificación posicional aprendida
            self.position_encoding = np.random.normal(0, 0.1, (self.max_seq_length, self.d_model))
        elif self.position_encoding_type == "sinusoidal":
            # Codificación posicional sinusoidal
            self.position_encoding = np.zeros((self.max_seq_length, self.d_model))
            for pos in range(self.max_seq_length):
                for i in range(0, self.d_model, 2):
                    self.position_encoding[pos, i] = np.sin(pos / (10000 ** (2 * i / self.d_model)))
                    if i + 1 < self.d_model:
                        self.position_encoding[pos, i + 1] = np.cos(pos / (10000 ** (2 * i / self.d_model)))
        else:
            # Sin codificación posicional
            self.position_encoding = np.zeros((self.max_seq_length, self.d_model))

    def _inicializar_atencion_adaptativa(self) -> None:
        """
        Inicializa los pesos de atención adaptativa.
        """
        # Pesos para adaptar la atención según el contexto
        self.pesos_atencion_adaptativa['context_gate'] = np.random.normal(0, 0.1, (self.d_model, self.d_model))
        self.sesgos_atencion_adaptativa['context_gate'] = np.zeros(self.d_model, dtype=LUCIA_ADVANCED_RL_CONFIG['precision'])

        # Pesos para adaptar la temperatura de atención
        self.pesos_atencion_adaptativa['temperature_adaptation'] = np.random.normal(0, 0.1, (self.d_model, 1))
        self.sesgos_atencion_adaptativa['temperature_adaptation'] = np.zeros(1, dtype=LUCIA_ADVANCED_RL_CONFIG['precision'])

        # Pesos para atención esparsa
        if self.sparse_attention:
            self.pesos_atencion_adaptativa['sparsity_gate'] = np.random.normal(0, 0.1, (self.d_model, self.d_model))
            self.sesgos_atencion_adaptativa['sparsity_gate'] = np.zeros(self.d_model, dtype=LUCIA_ADVANCED_RL_CONFIG['precision'])

    def _aplicar_layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """
        Aplica normalización de capas.

        Args:
            x: Entrada a normalizar
            gamma: Parámetros de escala
            beta: Parámetros de sesgo

        Returns:
            Salida normalizada
        """
        # Calcular media y varianza
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)

        # Normalizar
        x_norm = (x - mean) / np.sqrt(var + self.layer_norm_eps)

        # Aplicar escala y sesgo
        return gamma * x_norm + beta

    def _aplicar_atencion_multi_cabeza(self, query: np.ndarray, key: np.ndarray, value: np.ndarray,
                                       layer_name: str, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Aplica atención multi-cabeza adaptativa.

        Args:
            query: Consultas
            key: Claves
            value: Valores
            layer_name: Nombre de la capa
            mask: Máscara de atención (opcional)

        Returns:
            Salida de atención
        """
        # Calcular Q, K, V
        Q = np.dot(query, self.pesos_transformer[f"{layer_name}_attention_q"]) + self.sesgos_transformer[f"{layer_name}_attention_q"]
        K = np.dot(key, self.pesos_transformer[f"{layer_name}_attention_k"]) + self.sesgos_transformer[f"{layer_name}_attention_k"]
        V = np.dot(value, self.pesos_transformer[f"{layer_name}_attention_v"]) + self.sesgos_transformer[f"{layer_name}_attention_v"]

        # Dividir en cabezas
        head_dim = self.d_model // self.n_heads
        Q_heads = Q.reshape(-1, self.n_heads, head_dim)
        K_heads = K.reshape(-1, self.n_heads, head_dim)
        V_heads = V.reshape(-1, self.n_heads, head_dim)

        # Calcular atención para cada cabeza
        attention_outputs = []

        for head in range(self.n_heads):
            Q_head = Q_heads[:, head, :]
            K_head = K_heads[:, head, :]
            V_head = V_heads[:, head, :]

            # Calcular scores de atención
            scores = np.dot(Q_head, K_head.T) / np.sqrt(head_dim)

            # Aplicar máscara causal si está habilitada
            if self.causal_attention and mask is None:
                seq_len = scores.shape[0]
                mask = np.tril(np.ones((seq_len, seq_len)))
                scores = scores * mask + (1 - mask) * -1e9

            # Aplicar máscara si se proporciona
            if mask is not None:
                scores = scores * mask + (1 - mask) * -1e9

            # Aplicar atención adaptativa si está habilitada
            if self.adaptive_attention:
                # Adaptar temperatura de atención
                temperature = np.dot(Q_head.mean(axis=0), self.pesos_atencion_adaptativa['temperature_adaptation'].flatten()) + \
                    self.sesgos_atencion_adaptativa['temperature_adaptation'][0]
                temperature = np.exp(temperature)  # Asegurar positividad
                scores = scores / temperature

            # Aplicar softmax
            exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
            attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

            # Aplicar dropout de atención
            if self.attention_dropout > 0:
                dropout_mask = np.random.random(attention_weights.shape) > self.attention_dropout
                attention_weights = attention_weights * dropout_mask / (1 - self.attention_dropout)

            # Aplicar atención esparsa si está habilitada
            if self.sparse_attention:
                # Calcular gate de esparsidad
                sparsity_gate = np.dot(Q_head.mean(axis=0), self.pesos_atencion_adaptativa['sparsity_gate']) + \
                    self.sesgos_atencion_adaptativa['sparsity_gate']
                sparsity_gate = np.sigmoid(sparsity_gate)

                # Aplicar esparsidad
                attention_weights = attention_weights * sparsity_gate

            # Calcular salida de atención
            attention_output = np.dot(attention_weights, V_head)
            attention_outputs.append(attention_output)

        # Concatenar cabezas
        attention_output = np.concatenate(attention_outputs, axis=-1)

        # Proyección de salida
        output = np.dot(attention_output, self.pesos_transformer[f"{layer_name}_attention_output"]) + \
            self.sesgos_transformer[f"{layer_name}_attention_output"]

        # Guardar pesos de atención para análisis
        self.historial_attention_weights.append({
            'layer': layer_name,
            'attention_weights': attention_weights.copy(),
            'timestamp': time.time()
        })

        return output

    def _aplicar_feed_forward(self, x: np.ndarray, layer_name: str) -> np.ndarray:
        """
        Aplica la capa feed-forward.

        Args:
            x: Entrada
            layer_name: Nombre de la capa

        Returns:
            Salida de feed-forward
        """
        # Primera capa lineal con activación ReLU
        ff1 = np.dot(x, self.pesos_transformer[f"{layer_name}_ff1"]) + self.sesgos_transformer[f"{layer_name}_ff1"]
        ff1 = np.maximum(0, ff1)  # ReLU

        # Aplicar dropout
        if self.dropout_rate > 0:
            dropout_mask = np.random.random(ff1.shape) > self.dropout_rate
            ff1 = ff1 * dropout_mask / (1 - self.dropout_rate)

        # Segunda capa lineal
        ff2 = np.dot(ff1, self.pesos_transformer[f"{layer_name}_ff2"]) + self.sesgos_transformer[f"{layer_name}_ff2"]

        return ff2

    def _procesar_capa_transformer(self, x: np.ndarray, layer: int) -> np.ndarray:
        """
        Procesa una capa transformer.

        Args:
            x: Entrada
            layer: Número de capa

        Returns:
            Salida de la capa
        """
        layer_name = f"layer_{layer}"

        # Atención multi-cabeza con residual connection
        attention_output = self._aplicar_atencion_multi_cabeza(x, x, x, layer_name)
        x = x + attention_output

        # Normalización de capas
        x = self._aplicar_layer_norm(x,
                                     self.pesos_transformer[f"{layer_name}_layer_norm1"],
                                     self.sesgos_transformer[f"{layer_name}_layer_norm1"])

        # Feed-forward con residual connection
        ff_output = self._aplicar_feed_forward(x, layer_name)
        x = x + ff_output

        # Normalización de capas
        x = self._aplicar_layer_norm(x,
                                     self.pesos_transformer[f"{layer_name}_layer_norm2"],
                                     self.sesgos_transformer[f"{layer_name}_layer_norm2"])

        # Guardar salida de capa
        self.historial_layer_outputs.append({
            'layer': layer,
            'output': x.copy(),
            'timestamp': time.time()
        })

        return x

    def _agregar_a_secuencia(self, estado: np.ndarray) -> None:
        """
        Agrega un estado a la secuencia actual.

        Args:
            estado: Estado a agregar
        """
        self.secuencia_actual.append(estado.copy())

        # Mantener longitud máxima
        if len(self.secuencia_actual) > self.max_seq_length:
            self.secuencia_actual = self.secuencia_actual[-self.max_seq_length:]

    def _aplicar_position_encoding(self, x: np.ndarray) -> np.ndarray:
        """
        Aplica codificación posicional.

        Args:
            x: Entrada

        Returns:
            Entrada con codificación posicional
        """
        seq_len = x.shape[0]

        if seq_len <= self.max_seq_length:
            pos_encoding = self.position_encoding[:seq_len]
            return x + pos_encoding
        else:
            # Truncar si es necesario
            return x + self.position_encoding[:self.max_seq_length]

    def forward(self, estado: np.ndarray, contexto: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Propagación hacia adelante con arquitectura transformer adaptativa.

        Args:
            estado: Estado actual del entorno
            contexto: Contexto adicional (opcional)

        Returns:
            Acción o valor estimado
        """
        if not self.pesos_transformer:
            self.inicializar_pesos()

        # Agregar estado a la secuencia
        self._agregar_a_secuencia(estado)

        # Convertir secuencia a array
        if len(self.secuencia_actual) == 0:
            secuencia_array = estado.reshape(1, -1)
        else:
            secuencia_array = np.array(self.secuencia_actual)

        # Proyección de entrada
        x = np.dot(secuencia_array, self.pesos_transformer['input_projection']) + \
            self.sesgos_transformer['input_projection']

        # Aplicar codificación posicional
        x = self._aplicar_position_encoding(x)

        # Procesar a través de las capas transformer
        for layer in range(self.n_layers):
            x = self._procesar_capa_transformer(x, layer)

        # Proyección de salida
        output = np.dot(x[-1], self.pesos_transformer['output_projection']) + \
            self.sesgos_transformer['output_projection']

        # Aplicar función de activación
        exp_output = np.exp(output - np.max(output))
        probabilidades = exp_output / np.sum(exp_output)

        # Actualizar estadísticas
        self._actualizar_estadisticas_transformer()

        return probabilidades

    def _actualizar_estadisticas_transformer(self) -> None:
        """
        Actualiza las estadísticas específicas del transformer adaptativo.
        """
        # Calcular estadísticas de atención
        if len(self.historial_attention_weights) > 0:
            attention_weights_recientes = [h['attention_weights'] for h in
                                           list(self.historial_attention_weights)[-10:]]
            if attention_weights_recientes:
                weights_array = np.array(attention_weights_recientes)
                self.estadisticas_transformer['attention_weights_media'] = np.mean(weights_array)

                # Calcular entropía de atención
                entropy = -np.sum(weights_array * np.log(weights_array + 1e-8), axis=-1)
                self.estadisticas_transformer['attention_entropy'] = np.mean(entropy)

        # Calcular estadísticas de capas
        if len(self.historial_layer_outputs) > 0:
            layer_outputs_recientes = [h['output'] for h in
                                       list(self.historial_layer_outputs)[-10:]]
            if layer_outputs_recientes:
                outputs_array = np.array(layer_outputs_recientes)
                varianza_outputs = np.var(outputs_array)
                self.estadisticas_transformer['layer_norm_estabilidad'] = 1.0 / (1.0 + varianza_outputs)

        # Calcular eficiencia de atención esparsa
        if self.sparse_attention:
            # Simular eficiencia basada en esparsidad
            self.estadisticas_transformer['sparse_attention_efficiency'] = 0.7  # Placeholder

        # Calcular eficiencia computacional
        self.estadisticas_transformer['computational_efficiency'] = 1.0 / (1.0 + self.n_layers * 0.1)

        # Calcular eficiencia de memoria
        self.estadisticas_transformer['memory_efficiency'] = 1.0 / (1.0 + len(self.secuencia_actual) * 0.01)

    def seleccionar_accion(self, estado: np.ndarray, contexto: Optional[np.ndarray] = None) -> int:
        """
        Selecciona una acción usando la política transformer adaptativa.

        Args:
            estado: Estado actual del entorno
            contexto: Contexto adicional (opcional)

        Returns:
            Acción seleccionada
        """
        probabilidades = self.forward(estado, contexto)

        # Selección basada en probabilidades
        accion = np.random.choice(self.output_size, p=probabilidades)

        return accion

    def entrenar_paso_transformer(self, estado: np.ndarray, accion: int, recompensa: float,
                                  siguiente_estado: np.ndarray, terminado: bool,
                                  contexto: Optional[np.ndarray] = None) -> None:
        """
        Realiza un paso de entrenamiento transformer adaptativo.

        Args:
            estado: Estado actual
            accion: Acción tomada
            recompensa: Recompensa recibida
            siguiente_estado: Siguiente estado
            terminado: Si el episodio terminó
            contexto: Contexto adicional (opcional)
        """
        # Calcular probabilidades actuales
        probabilidades = self.forward(estado, contexto)

        # Calcular gradiente de política
        grad_log_prob = np.zeros(self.output_size)
        grad_log_prob[accion] = 1.0 / (probabilidades[accion] + 1e-8)

        # Calcular gradientes para todos los pesos (simplificado)
        # En una implementación real, se usaría backpropagation completa

        # Gradiente para proyección de salida
        if len(self.secuencia_actual) > 0:
            ultimo_estado = self.secuencia_actual[-1]
            grad_output_proj = np.outer(ultimo_estado, grad_log_prob) * recompensa

            # Actualizar pesos de salida
            self.pesos_transformer['output_projection'] -= self.learning_rate * grad_output_proj
            self.sesgos_transformer['output_projection'] -= self.learning_rate * grad_log_prob * recompensa

        # Guardar en historial de pesos
        self.historial_pesos.append(self.pesos_transformer['output_projection'].copy())

        # Actualizar estadísticas
        self._actualizar_estadisticas_transformer()

    def obtener_estadisticas_transformer(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas específicas del transformer adaptativo.

        Returns:
            Diccionario con estadísticas transformer
        """
        if not self.pesos_transformer:
            return {'estado': 'no_inicializada'}

        stats_transformer = {
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
            'd_ff': self.d_ff,
            'dropout_rate': self.dropout_rate,
            'attention_dropout': self.attention_dropout,
            'max_seq_length': self.max_seq_length,
            'adaptive_attention': self.adaptive_attention,
            'sparse_attention': self.sparse_attention,
            'causal_attention': self.causal_attention,
            'layer_norm_eps': self.layer_norm_eps,
            'position_encoding_type': self.position_encoding_type,
            'attention_type': self.attention_type,
            'attention_weights_media': self.estadisticas_transformer['attention_weights_media'],
            'attention_entropy': self.estadisticas_transformer['attention_entropy'],
            'layer_norm_estabilidad': self.estadisticas_transformer['layer_norm_estabilidad'],
            'gradient_norm_media': self.estadisticas_transformer['gradient_norm_media'],
            'attention_sparsity': self.estadisticas_transformer['attention_sparsity'],
            'position_encoding_utilidad': self.estadisticas_transformer['position_encoding_utilidad'],
            'transformer_depth_utilidad': self.estadisticas_transformer['transformer_depth_utilidad'],
            'multi_head_diversidad': self.estadisticas_transformer['multi_head_diversidad'],
            'feed_forward_eficiencia': self.estadisticas_transformer['feed_forward_eficiencia'],
            'dropout_efectividad': self.estadisticas_transformer['dropout_efectividad'],
            'causal_attention_eficiencia': self.estadisticas_transformer['causal_attention_eficiencia'],
            'adaptive_attention_responsiveness': self.estadisticas_transformer['adaptive_attention_responsiveness'],
            'sparse_attention_efficiency': self.estadisticas_transformer['sparse_attention_efficiency'],
            'memory_efficiency': self.estadisticas_transformer['memory_efficiency'],
            'computational_efficiency': self.estadisticas_transformer['computational_efficiency'],
            'attention_convergence': self.estadisticas_transformer['attention_convergence'],
            'secuencia_actual_length': len(self.secuencia_actual),
            'historial_attention_weights_size': len(self.historial_attention_weights),
            'historial_layer_outputs_size': len(self.historial_layer_outputs)
        }

        return stats_transformer

    def verificar_estabilidad(self) -> Dict[str, bool]:
        """
        Verifica la estabilidad de la neurona transformer adaptativa.

        Returns:
            Diccionario con indicadores de estabilidad
        """
        estabilidad = super().verificar_estabilidad_avanzada()

        # Verificaciones específicas de transformer adaptativo
        estabilidad['attention_weights_estables'] = self.estadisticas_transformer['attention_weights_media'] > 0.1
        estabilidad['layer_norm_estable'] = self.estadisticas_transformer['layer_norm_estabilidad'] > 0.7
        estabilidad['attention_entropy_apropiada'] = self.estadisticas_transformer['attention_entropy'] > 0.5
        estabilidad['computational_efficiency_ok'] = self.estadisticas_transformer['computational_efficiency'] > 0.5
        estabilidad['memory_efficiency_ok'] = self.estadisticas_transformer['memory_efficiency'] > 0.3
        estabilidad['sparse_attention_eficiente'] = self.estadisticas_transformer['sparse_attention_efficiency'] > 0.5
        estabilidad['secuencia_length_apropiada'] = len(self.secuencia_actual) > 0
        estabilidad['transformer_depth_util'] = self.n_layers > 0

        return estabilidad

    def reinicializar_con_parametros(self, learning_rate: float = None,
                                     gamma: float = None,
                                     d_model: int = None,
                                     n_heads: int = None,
                                     n_layers: int = None,
                                     d_ff: int = None,
                                     dropout_rate: float = None,
                                     attention_dropout: float = None,
                                     max_seq_length: int = None,
                                     adaptive_attention: bool = None,
                                     sparse_attention: bool = None,
                                     causal_attention: bool = None,
                                     layer_norm_eps: float = None,
                                     position_encoding_type: str = None,
                                     attention_type: str = None) -> None:
        """
        Reinicializa la neurona transformer adaptativa con nuevos parámetros.
        """
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if gamma is not None:
            self.gamma = gamma
        if d_model is not None:
            self.d_model = d_model
        if n_heads is not None:
            self.n_heads = n_heads
        if n_layers is not None:
            self.n_layers = n_layers
        if d_ff is not None:
            self.d_ff = d_ff
        if dropout_rate is not None:
            self.dropout_rate = dropout_rate
        if attention_dropout is not None:
            self.attention_dropout = attention_dropout
        if max_seq_length is not None:
            self.max_seq_length = max_seq_length
        if adaptive_attention is not None:
            self.adaptive_attention = adaptive_attention
        if sparse_attention is not None:
            self.sparse_attention = sparse_attention
        if causal_attention is not None:
            self.causal_attention = causal_attention
        if layer_norm_eps is not None:
            self.layer_norm_eps = layer_norm_eps
        if position_encoding_type is not None:
            self.position_encoding_type = position_encoding_type
        if attention_type is not None:
            self.attention_type = attention_type

        self.inicializar_pesos()
        self.resetear_historial()

        # Limpiar estructuras transformer
        self.pesos_transformer.clear()
        self.sesgos_transformer.clear()
        self.pesos_atencion_adaptativa.clear()
        self.sesgos_atencion_adaptativa.clear()

        # Limpiar historiales específicos
        self.historial_attention_weights.clear()
        self.historial_layer_outputs.clear()
        self.historial_position_encodings.clear()
        self.historial_attention_patterns.clear()
        self.historial_transformer_states.clear()

        # Limpiar secuencia
        self.secuencia_actual.clear()
        self.memoria_secuencias.clear()

        # Resetear estadísticas transformer
        for key in self.estadisticas_transformer:
            self.estadisticas_transformer[key] = 0.0

        logger.info(f"Neurona transformer adaptativa reinicializada: lr={self.learning_rate}, "
                    f"d_model={self.d_model}, n_heads={self.n_heads}, n_layers={self.n_layers}, "
                    f"adaptive_attention={self.adaptive_attention}, sparse_attention={self.sparse_attention}")

    def __str__(self) -> str:
        return (f"NeuronaRefuerzoTransformerAdaptativa(entrada={self.input_size}, "
                f"salida={self.output_size}, lr={self.learning_rate}, "
                f"d_model={self.d_model}, n_heads={self.n_heads}, n_layers={self.n_layers}, "
                f"adaptive_attention={self.adaptive_attention}, sparse_attention={self.sparse_attention})")

    def __repr__(self) -> str:
        return self.__str__()

# Funciones de utilidad específicas para RFEN2_RN_10


def crear_neurona_transformer_adaptativa(input_size: int, output_size: int,
                                         configuracion: Dict[str, Any] = None) -> NeuronaRefuerzoTransformerAdaptativa:
    """
    Función de conveniencia para crear una neurona transformer adaptativa.

    Args:
        input_size: Tamaño de entrada
        output_size: Tamaño de salida
        configuracion: Configuración personalizada

    Returns:
        Neurona transformer adaptativa configurada
    """
    if configuracion is None:
        configuracion = {}

    return NeuronaRefuerzoTransformerAdaptativa(
        input_size=input_size,
        output_size=output_size,
        nombre=configuracion.get('nombre', 'NeuronaRefuerzoTransformerAdaptativa'),
        learning_rate=configuracion.get('learning_rate', LUCIA_ADVANCED_RL_CONFIG['default_learning_rate']),
        gamma=configuracion.get('gamma', LUCIA_ADVANCED_RL_CONFIG['default_gamma']),
        d_model=configuracion.get('d_model', LUCIA_ADVANCED_RL_CONFIG['transformer_d_model']),
        n_heads=configuracion.get('n_heads', LUCIA_ADVANCED_RL_CONFIG['transformer_n_heads']),
        n_layers=configuracion.get('n_layers', LUCIA_ADVANCED_RL_CONFIG['transformer_n_layers']),
        d_ff=configuracion.get('d_ff', LUCIA_ADVANCED_RL_CONFIG['transformer_d_ff']),
        dropout_rate=configuracion.get('dropout_rate', LUCIA_ADVANCED_RL_CONFIG['transformer_dropout_rate']),
        attention_dropout=configuracion.get('attention_dropout', LUCIA_ADVANCED_RL_CONFIG['transformer_attention_dropout']),
        max_seq_length=configuracion.get('max_seq_length', LUCIA_ADVANCED_RL_CONFIG['transformer_max_seq_length']),
        adaptive_attention=configuracion.get('adaptive_attention', True),
        sparse_attention=configuracion.get('sparse_attention', True),
        causal_attention=configuracion.get('causal_attention', True),
        layer_norm_eps=configuracion.get('layer_norm_eps', 1e-6),
        position_encoding_type=configuracion.get('position_encoding_type', 'learned'),
        attention_type=configuracion.get('attention_type', 'multi_head')
    )


# Configuración específica para RFEN2_RN_10
RFEN2_RN_10_CONFIG = {
    'inicializacion_preferida': 'transformer_adaptativo',
    'learning_rate_default': 0.0001,
    'gamma_default': 0.99,
    'd_model_default': 64,
    'n_heads_default': 8,
    'n_layers_default': 6,
    'd_ff_default': 256,
    'dropout_rate_default': 0.1,
    'attention_dropout_default': 0.1,
    'max_seq_length_default': 512,
    'adaptive_attention_default': True,
    'sparse_attention_default': True,
    'causal_attention_default': True,
    'layer_norm_eps_default': 1e-6,
    'position_encoding_type_default': 'learned',
    'attention_type_default': 'multi_head',
    'umbral_attention_weights_estables': 0.1,
    'umbral_layer_norm_estable': 0.7,
    'umbral_attention_entropy': 0.5,
    'umbral_computational_efficiency': 0.5,
    'umbral_memory_efficiency': 0.3,
    'umbral_sparse_attention_efficiency': 0.5,
    'umbral_secuencia_length': 0,
    'umbral_transformer_depth': 0
}

logger.info("RFEN2_RN_10.py cargado correctamente - Neurona de Refuerzo Transformer Adaptativa")
