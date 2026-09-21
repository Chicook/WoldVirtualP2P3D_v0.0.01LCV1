import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import random
import math
from collections import defaultdict, deque
import copy
import time
import threading

# Configuración del logger
logger = logging.getLogger(__name__)

class KnowledgeDistillationWeightOptimizer(ABC):
    """
    Clase base abstracta para optimizadores de destilación de conocimiento de pesos.
    Define la interfaz común para todas las estrategias de destilación de conocimiento.
    """
    def __init__(self, config=None):
        self.config = config if config is not None else {}
        logger.info("KnowledgeDistillationWeightOptimizer base inicializado.")

    @abstractmethod
    def distill_knowledge_weights(self, teacher_model: nn.Module, student_model: nn.Module, 
                                 data_loader=None) -> nn.Module:
        """
        Método abstracto para destilar conocimiento de pesos del modelo maestro al estudiante.
        Debe ser implementado por las subclases.
        """
        pass

class TraditionalKnowledgeDistillationOptimizer(KnowledgeDistillationWeightOptimizer):
    """
    Optimizador de pesos basado en destilación de conocimiento tradicional.
    Utiliza destilación de conocimiento tradicional para transferir conocimiento de pesos.
    """
    def __init__(self, temperature: float = 3.0, alpha: float = 0.7,
                 distillation_loss_weight: float = 1.0, student_loss_weight: float = 0.3,
                 config=None):
        super().__init__(config)
        self.temperature = self.config.get('temperature', temperature)
        self.alpha = self.config.get('alpha', alpha)
        self.distillation_loss_weight = self.config.get('distillation_loss_weight', distillation_loss_weight)
        self.student_loss_weight = self.config.get('student_loss_weight', student_loss_weight)
        self.distillation_history = []
        logger.info(f"TraditionalKnowledgeDistillationOptimizer inicializado: temperature={self.temperature}, alpha={self.alpha}")

    def _compute_distillation_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        """
        Calcula la pérdida de destilación de conocimiento.
        """
        # Aplicar temperatura a los logits
        student_soft = F.softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        
        # Calcular pérdida de destilación (KL divergence)
        distillation_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            teacher_soft,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        return distillation_loss

    def _compute_student_loss(self, student_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calcula la pérdida del estudiante.
        """
        return F.cross_entropy(student_logits, targets)

    def _compute_total_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, 
                           targets: torch.Tensor) -> torch.Tensor:
        """
        Calcula la pérdida total combinando destilación y pérdida del estudiante.
        """
        distillation_loss = self._compute_distillation_loss(student_logits, teacher_logits)
        student_loss = self._compute_student_loss(student_logits, targets)
        
        total_loss = (self.alpha * distillation_loss + 
                     (1 - self.alpha) * student_loss)
        
        return total_loss

    def distill_knowledge_weights(self, teacher_model: nn.Module, student_model: nn.Module, 
                                 data_loader=None) -> nn.Module:
        logger.info("Iniciando destilación de conocimiento tradicional de pesos.")
        
        if data_loader is None:
            logger.warning("No se proporcionó data_loader para destilación de conocimiento. Saltando.")
            return student_model
        
        # Configurar optimizador para el modelo estudiante
        optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
        
        # Entrenamiento con destilación
        teacher_model.eval()
        student_model.train()
        
        max_epochs = self.config.get('max_epochs', 50)
        
        for epoch in range(max_epochs):
            total_loss = 0.0
            num_batches = 0
            
            for inputs, targets in data_loader:
                # Forward pass del maestro
                with torch.no_grad():
                    teacher_logits = teacher_model(inputs)
                
                # Forward pass del estudiante
                student_logits = student_model(inputs)
                
                # Calcular pérdida total
                loss = self._compute_total_loss(student_logits, teacher_logits, targets)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            
            # Guardar estadísticas de la época
            epoch_stats = {
                'epoch': epoch,
                'total_loss': avg_loss,
                'temperature': self.temperature,
                'alpha': self.alpha
            }
            self.distillation_history.append(epoch_stats)
            
            # Log de progreso
            if epoch % 10 == 0:
                logger.info(f"Época {epoch}: Pérdida total = {avg_loss:.4f}")
        
        logger.info("Destilación de conocimiento tradicional completada.")
        return student_model

class AttentionTransferKnowledgeDistillationOptimizer(KnowledgeDistillationWeightOptimizer):
    """
    Optimizador de pesos basado en destilación de conocimiento con transferencia de atención.
    Utiliza transferencia de atención para destilar conocimiento de pesos.
    """
    def __init__(self, attention_transfer_weight: float = 1.0, 
                 feature_transfer_weight: float = 0.5,
                 config=None):
        super().__init__(config)
        self.attention_transfer_weight = self.config.get('attention_transfer_weight', attention_transfer_weight)
        self.feature_transfer_weight = self.config.get('feature_transfer_weight', feature_transfer_weight)
        self.distillation_history = []
        logger.info(f"AttentionTransferKnowledgeDistillationOptimizer inicializado: attention_weight={self.attention_transfer_weight}, feature_weight={self.feature_transfer_weight}")

    def _extract_attention_maps(self, model: nn.Module, inputs: torch.Tensor) -> List[torch.Tensor]:
        """
        Extrae mapas de atención del modelo.
        """
        attention_maps = []
        
        def hook_fn(module, input, output):
            if hasattr(module, 'attention_weights'):
                attention_maps.append(module.attention_weights)
            elif isinstance(module, (nn.Linear, nn.Conv2d)):
                # Simular mapas de atención para capas lineales y convolucionales
                attention_map = torch.mean(torch.abs(output), dim=1, keepdim=True)
                attention_maps.append(attention_map)
        
        hooks = []
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.MultiheadAttention)):
                hook = module.register_forward_hook(hook_fn)
                hooks.append(hook)
        
        # Forward pass
        _ = model(inputs)
        
        # Remover hooks
        for hook in hooks:
            hook.remove()
        
        return attention_maps

    def _compute_attention_transfer_loss(self, teacher_attention: List[torch.Tensor], 
                                        student_attention: List[torch.Tensor]) -> torch.Tensor:
        """
        Calcula la pérdida de transferencia de atención.
        """
        total_loss = 0.0
        
        for t_att, s_att in zip(teacher_attention, student_attention):
            # Asegurar que los mapas de atención tengan el mismo tamaño
            if t_att.shape != s_att.shape:
                # Redimensionar si es necesario
                s_att = F.interpolate(s_att, size=t_att.shape[-2:], mode='bilinear', align_corners=False)
            
            # Calcular pérdida de transferencia de atención
            loss = F.mse_loss(s_att, t_att)
            total_loss += loss
        
        return total_loss / len(teacher_attention) if teacher_attention else torch.tensor(0.0)

    def _compute_feature_transfer_loss(self, teacher_features: List[torch.Tensor], 
                                      student_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Calcula la pérdida de transferencia de características.
        """
        total_loss = 0.0
        
        for t_feat, s_feat in zip(teacher_features, student_features):
            # Asegurar que las características tengan el mismo tamaño
            if t_feat.shape != s_feat.shape:
                # Redimensionar si es necesario
                s_feat = F.interpolate(s_feat, size=t_feat.shape[-2:], mode='bilinear', align_corners=False)
            
            # Calcular pérdida de transferencia de características
            loss = F.mse_loss(s_feat, t_feat)
            total_loss += loss
        
        return total_loss / len(teacher_features) if teacher_features else torch.tensor(0.0)

    def distill_knowledge_weights(self, teacher_model: nn.Module, student_model: nn.Module, 
                                 data_loader=None) -> nn.Module:
        logger.info("Iniciando destilación de conocimiento con transferencia de atención.")
        
        if data_loader is None:
            logger.warning("No se proporcionó data_loader para destilación de conocimiento con atención. Saltando.")
            return student_model
        
        # Configurar optimizador para el modelo estudiante
        optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
        
        # Entrenamiento con transferencia de atención
        teacher_model.eval()
        student_model.train()
        
        max_epochs = self.config.get('max_epochs', 50)
        
        for epoch in range(max_epochs):
            total_loss = 0.0
            num_batches = 0
            
            for inputs, targets in data_loader:
                # Extraer mapas de atención del maestro
                with torch.no_grad():
                    teacher_attention = self._extract_attention_maps(teacher_model, inputs)
                
                # Extraer mapas de atención del estudiante
                student_attention = self._extract_attention_maps(student_model, inputs)
                
                # Calcular pérdida de transferencia de atención
                attention_loss = self._compute_attention_transfer_loss(teacher_attention, student_attention)
                
                # Calcular pérdida de transferencia de características
                feature_loss = self._compute_feature_transfer_loss(teacher_attention, student_attention)
                
                # Calcular pérdida total
                total_loss_batch = (self.attention_transfer_weight * attention_loss + 
                                  self.feature_transfer_weight * feature_loss)
                
                # Backward pass
                optimizer.zero_grad()
                total_loss_batch.backward()
                optimizer.step()
                
                total_loss += total_loss_batch.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            
            # Guardar estadísticas de la época
            epoch_stats = {
                'epoch': epoch,
                'total_loss': avg_loss,
                'attention_transfer_weight': self.attention_transfer_weight,
                'feature_transfer_weight': self.feature_transfer_weight
            }
            self.distillation_history.append(epoch_stats)
            
            # Log de progreso
            if epoch % 10 == 0:
                logger.info(f"Época {epoch}: Pérdida total = {avg_loss:.4f}")
        
        logger.info("Destilación de conocimiento con transferencia de atención completada.")
        return student_model

class ProgressiveKnowledgeDistillationOptimizer(KnowledgeDistillationWeightOptimizer):
    """
    Optimizador de pesos basado en destilación de conocimiento progresiva.
    Utiliza destilación de conocimiento progresiva para transferir conocimiento de pesos.
    """
    def __init__(self, progressive_stages: int = 3, stage_temperature: float = 3.0,
                 stage_alpha: float = 0.7, config=None):
        super().__init__(config)
        self.progressive_stages = self.config.get('progressive_stages', progressive_stages)
        self.stage_temperature = self.config.get('stage_temperature', stage_temperature)
        self.stage_alpha = self.config.get('stage_alpha', stage_alpha)
        self.distillation_history = []
        logger.info(f"ProgressiveKnowledgeDistillationOptimizer inicializado: stages={self.progressive_stages}, temperature={self.stage_temperature}")

    def _compute_stage_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, 
                           targets: torch.Tensor, stage: int) -> torch.Tensor:
        """
        Calcula la pérdida para una etapa específica de destilación progresiva.
        """
        # Ajustar parámetros según la etapa
        temperature = self.stage_temperature * (1.0 - stage / self.progressive_stages)
        alpha = self.stage_alpha * (stage / self.progressive_stages)
        
        # Calcular pérdida de destilación
        student_soft = F.softmax(student_logits / temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / temperature, dim=1)
        
        distillation_loss = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=1),
            teacher_soft,
            reduction='batchmean'
        ) * (temperature ** 2)
        
        # Calcular pérdida del estudiante
        student_loss = F.cross_entropy(student_logits, targets)
        
        # Calcular pérdida total
        total_loss = (alpha * distillation_loss + 
                     (1 - alpha) * student_loss)
        
        return total_loss

    def distill_knowledge_weights(self, teacher_model: nn.Module, student_model: nn.Module, 
                                 data_loader=None) -> nn.Module:
        logger.info("Iniciando destilación de conocimiento progresiva de pesos.")
        
        if data_loader is None:
            logger.warning("No se proporcionó data_loader para destilación de conocimiento progresiva. Saltando.")
            return student_model
        
        # Configurar optimizador para el modelo estudiante
        optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
        
        # Entrenamiento progresivo
        teacher_model.eval()
        student_model.train()
        
        max_epochs_per_stage = self.config.get('max_epochs_per_stage', 20)
        
        for stage in range(self.progressive_stages):
            logger.info(f"Iniciando etapa {stage + 1}/{self.progressive_stages} de destilación progresiva.")
            
            for epoch in range(max_epochs_per_stage):
                total_loss = 0.0
                num_batches = 0
                
                for inputs, targets in data_loader:
                    # Forward pass del maestro
                    with torch.no_grad():
                        teacher_logits = teacher_model(inputs)
                    
                    # Forward pass del estudiante
                    student_logits = student_model(inputs)
                    
                    # Calcular pérdida para esta etapa
                    loss = self._compute_stage_loss(student_logits, teacher_logits, targets, stage)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                
                avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
                
                # Guardar estadísticas de la época
                epoch_stats = {
                    'stage': stage,
                    'epoch': epoch,
                    'total_loss': avg_loss,
                    'temperature': self.stage_temperature * (1.0 - stage / self.progressive_stages),
                    'alpha': self.stage_alpha * (stage / self.progressive_stages)
                }
                self.distillation_history.append(epoch_stats)
                
                # Log de progreso
                if epoch % 5 == 0:
                    logger.info(f"Etapa {stage + 1}, Época {epoch}: Pérdida total = {avg_loss:.4f}")
        
        logger.info("Destilación de conocimiento progresiva completada.")
        return student_model

class KnowledgeDistillationWeightAnalyzer:
    """
    Analizador para evaluar el rendimiento de la destilación de conocimiento de pesos.
    """
    def __init__(self):
        logger.info("KnowledgeDistillationWeightAnalyzer inicializado.")

    def analyze_knowledge_distillation_optimization(self, teacher_model: nn.Module, 
                                                   student_model: nn.Module, 
                                                   test_data_loader=None) -> Dict[str, float]:
        """
        Analiza el rendimiento de la destilación de conocimiento.
        """
        analysis_results = {}
        
        # Evaluar rendimiento del maestro
        teacher_performance = self._evaluate_model_performance(teacher_model, test_data_loader)
        
        # Evaluar rendimiento del estudiante
        student_performance = self._evaluate_model_performance(student_model, test_data_loader)
        
        # Calcular mejora
        improvement = teacher_performance - student_performance
        improvement_percentage = (improvement / teacher_performance) * 100
        
        analysis_results['teacher_performance'] = teacher_performance
        analysis_results['student_performance'] = student_performance
        analysis_results['improvement'] = improvement
        analysis_results['improvement_percentage'] = improvement_percentage
        
        # Analizar características de destilación de conocimiento
        analysis_results['knowledge_transfer_efficiency'] = self._analyze_knowledge_transfer_efficiency(teacher_model, student_model)
        analysis_results['model_compression_ratio'] = self._analyze_model_compression_ratio(teacher_model, student_model)
        analysis_results['distillation_quality'] = self._analyze_distillation_quality(teacher_model, student_model)
        
        logger.info(f"Análisis de destilación de conocimiento: Mejora = {improvement_percentage:.2f}%")
        return analysis_results

    def _evaluate_model_performance(self, model: nn.Module, data_loader) -> float:
        """
        Evalúa el rendimiento del modelo.
        """
        if data_loader is None:
            return random.random()
        
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, targets in data_loader:
                outputs = model(inputs)
                loss = F.mse_loss(outputs, targets)
                total_loss += loss.item()
        
        return total_loss

    def _analyze_knowledge_transfer_efficiency(self, teacher_model: nn.Module, student_model: nn.Module) -> float:
        """
        Analiza la eficiencia de transferencia de conocimiento.
        """
        # Simular eficiencia de transferencia basándose en la similitud de pesos
        total_similarity = 0.0
        total_params = 0
        
        for (name1, param1), (name2, param2) in zip(teacher_model.named_parameters(), student_model.named_parameters()):
            if param1.requires_grad and param2.requires_grad:
                # Calcular similitud de pesos
                similarity = torch.cosine_similarity(param1.data.flatten(), param2.data.flatten(), dim=0)
                total_similarity += similarity.item()
                total_params += 1
        
        knowledge_transfer_efficiency = total_similarity / max(total_params, 1)
        return knowledge_transfer_efficiency

    def _analyze_model_compression_ratio(self, teacher_model: nn.Module, student_model: nn.Module) -> float:
        """
        Analiza la relación de compresión del modelo.
        """
        # Calcular tamaño del maestro
        teacher_size = sum(p.numel() for p in teacher_model.parameters() if p.requires_grad)
        
        # Calcular tamaño del estudiante
        student_size = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
        
        # Calcular relación de compresión
        compression_ratio = teacher_size / max(student_size, 1)
        return compression_ratio

    def _analyze_distillation_quality(self, teacher_model: nn.Module, student_model: nn.Module) -> float:
        """
        Analiza la calidad de la destilación.
        """
        # Simular calidad de destilación basándose en la distribución de pesos
        teacher_weight_std = 0.0
        student_weight_std = 0.0
        
        for param in teacher_model.parameters():
            if param.requires_grad:
                teacher_weight_std += torch.std(param.data).item()
        
        for param in student_model.parameters():
            if param.requires_grad:
                student_weight_std += torch.std(param.data).item()
        
        distillation_quality = 1.0 / (1.0 + abs(teacher_weight_std - student_weight_std))
        return distillation_quality

def create_knowledge_distillation_weight_optimizer(optimizer_type: str, **kwargs) -> KnowledgeDistillationWeightOptimizer:
    """
    Factoría para crear diferentes tipos de optimizadores de destilación de conocimiento.
    """
    if optimizer_type == "traditional_knowledge_distillation":
        return TraditionalKnowledgeDistillationOptimizer(**kwargs)
    elif optimizer_type == "attention_transfer_knowledge_distillation":
        return AttentionTransferKnowledgeDistillationOptimizer(**kwargs)
    elif optimizer_type == "progressive_knowledge_distillation":
        return ProgressiveKnowledgeDistillationOptimizer(**kwargs)
    else:
        raise ValueError(f"Tipo de optimizador de destilación de conocimiento no soportado: {optimizer_type}")

def distill_knowledge_model_weights(teacher_model: nn.Module, student_model: nn.Module, 
                                   optimizer_type: str, data_loader=None, **kwargs) -> Tuple[nn.Module, Dict[str, float]]:
    """
    Función de conveniencia para aplicar destilación de conocimiento a los pesos de un modelo.
    """
    optimizer = create_knowledge_distillation_weight_optimizer(optimizer_type, **kwargs)
    
    # Aplicar destilación de conocimiento
    optimized_student_model = optimizer.distill_knowledge_weights(teacher_model, student_model, data_loader)
    
    # Analizar rendimiento
    analyzer = KnowledgeDistillationWeightAnalyzer()
    analysis = analyzer.analyze_knowledge_distillation_optimization(teacher_model, optimized_student_model, data_loader)
    
    return optimized_student_model, analysis

# Exportar clases y funciones principales
__all__ = [
    'KnowledgeDistillationWeightOptimizer',
    'TraditionalKnowledgeDistillationOptimizer',
    'AttentionTransferKnowledgeDistillationOptimizer',
    'ProgressiveKnowledgeDistillationOptimizer',
    'KnowledgeDistillationWeightAnalyzer',
    'create_knowledge_distillation_weight_optimizer',
    'distill_knowledge_model_weights'
]

logger.info("RFEN7_RN_9 - Destilación de Conocimiento de Pesos cargada correctamente")
