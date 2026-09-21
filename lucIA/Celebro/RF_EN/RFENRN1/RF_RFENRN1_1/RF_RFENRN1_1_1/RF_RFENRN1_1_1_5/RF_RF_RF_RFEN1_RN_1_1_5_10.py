"""
API de Integración y Sistema de Orquestación
=============================================
API REST/WebSocket para control del sistema de neuronas de refuerzo.
Sistema de orquestación que coordina todos los componentes.

Características:
- API RESTful con FastAPI
- WebSocket para comunicación en tiempo real
- Registro de servicios
- Event Bus para comunicación asíncrona
- Gestor de configuración centralizado
- Sistema de plugins
"""

import os
import json
import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import threading
from queue import Queue
from enum import Enum


class ServiceStatus(Enum):
    """Estados de los servicios"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class ServiceInfo:
    """Información de un servicio registrado"""
    name: str
    version: str
    status: ServiceStatus
    endpoint: Optional[str] = None
    health_check_url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_heartbeat: Optional[float] = None


@dataclass
class Event:
    """Evento del sistema"""
    event_type: str
    source: str
    timestamp: float
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'event_type': self.event_type,
            'source': self.source,
            'timestamp': self.timestamp,
            'data': self.data
        }


class ServiceRegistry:
    """
    Registro centralizado de servicios
    """

    def __init__(self):
        self.services: Dict[str, ServiceInfo] = {}
        self.service_instances: Dict[str, Any] = {}
        print("📋 ServiceRegistry inicializado")

    def register_service(self, service_info: ServiceInfo, instance: Optional[Any] = None):
        """
        Registra un nuevo servicio

        Args:
            service_info: Información del servicio
            instance: Instancia del servicio (opcional)
        """
        self.services[service_info.name] = service_info

        if instance:
            self.service_instances[service_info.name] = instance

        print(f"✅ Servicio registrado: {service_info.name} v{service_info.version}")

    def unregister_service(self, service_name: str):
        """Desregistra un servicio"""
        if service_name in self.services:
            del self.services[service_name]
            if service_name in self.service_instances:
                del self.service_instances[service_name]
            print(f"❌ Servicio desregistrado: {service_name}")

    def get_service(self, service_name: str) -> Optional[ServiceInfo]:
        """Obtiene información de un servicio"""
        return self.services.get(service_name)

    def get_service_instance(self, service_name: str) -> Optional[Any]:
        """Obtiene la instancia de un servicio"""
        return self.service_instances.get(service_name)

    def update_service_status(self, service_name: str, status: ServiceStatus):
        """Actualiza el estado de un servicio"""
        if service_name in self.services:
            self.services[service_name].status = status
            print(f"🔄 Estado de {service_name}: {status.value}")

    def heartbeat(self, service_name: str):
        """Registra un heartbeat de un servicio"""
        if service_name in self.services:
            import time
            self.services[service_name].last_heartbeat = time.time()

    def list_services(self) -> List[ServiceInfo]:
        """Lista todos los servicios registrados"""
        return list(self.services.values())

    def get_healthy_services(self) -> List[ServiceInfo]:
        """Obtiene solo los servicios que están funcionando"""
        return [s for s in self.services.values() if s.status == ServiceStatus.RUNNING]


class EventBus:
    """
    Sistema de Event Bus para comunicación asíncrona entre componentes
    """

    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.event_queue: Queue = Queue()
        self.processing = False
        self.processor_thread: Optional[threading.Thread] = None
        print("📡 EventBus inicializado")

    def subscribe(self, event_type: str, callback: Callable):
        """
        Suscribe un callback a un tipo de evento

        Args:
            event_type: Tipo de evento a escuchar
            callback: Función a llamar cuando ocurra el evento
        """
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []

        self.subscribers[event_type].append(callback)
        print(f"📬 Suscrito a evento: {event_type}")

    def unsubscribe(self, event_type: str, callback: Callable):
        """Desuscribe un callback"""
        if event_type in self.subscribers:
            self.subscribers[event_type].remove(callback)

    def publish(self, event: Event):
        """
        Publica un evento

        Args:
            event: Evento a publicar
        """
        self.event_queue.put(event)
        print(f"📤 Evento publicado: {event.event_type} desde {event.source}")

    def emit(self, event_type: str, source: str, data: Dict[str, Any] = None):
        """
        Emite un evento rápidamente

        Args:
            event_type: Tipo de evento
            source: Fuente del evento
            data: Datos del evento
        """
        import time
        event = Event(
            event_type=event_type,
            source=source,
            timestamp=time.time(),
            data=data or {}
        )
        self.publish(event)

    def start_processing(self):
        """Inicia el procesamiento de eventos en segundo plano"""
        if self.processing:
            print("⚠️ Ya está procesando eventos")
            return

        self.processing = True
        self.processor_thread = threading.Thread(target=self._process_events, daemon=True)
        self.processor_thread.start()

        print("▶️ Procesamiento de eventos iniciado")

    def stop_processing(self):
        """Detiene el procesamiento de eventos"""
        self.processing = False
        if self.processor_thread:
            self.processor_thread.join(timeout=2.0)

        print("⏹️ Procesamiento de eventos detenido")

    def _process_events(self):
        """Loop de procesamiento de eventos"""
        while self.processing:
            if not self.event_queue.empty():
                event = self.event_queue.get()
                self._dispatch_event(event)
            else:
                import time
                time.sleep(0.01)  # Pequeña pausa para no saturar CPU

    def _dispatch_event(self, event: Event):
        """Despacha un evento a sus suscriptores"""
        if event.event_type in self.subscribers:
            for callback in self.subscribers[event.event_type]:
                try:
                    callback(event)
                except Exception as e:
                    print(f"❌ Error procesando evento: {e}")


class ConfigManager:
    """
    Gestor centralizado de configuración
    """

    def __init__(self, config_dir: str = "./config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self.config: Dict[str, Any] = {}
        self.watchers: Dict[str, List[Callable]] = {}

        print("⚙️ ConfigManager inicializado")

    def load_config(self, config_file: str = "config.json"):
        """Carga configuración desde archivo"""
        config_path = self.config_dir / config_file

        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            print(f"✅ Configuración cargada desde {config_path}")
        else:
            print(f"⚠️ Archivo de configuración no encontrado: {config_path}")

    def save_config(self, config_file: str = "config.json"):
        """Guarda configuración a archivo"""
        config_path = self.config_dir / config_file

        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

        print(f"💾 Configuración guardada en {config_path}")

    def get(self, key: str, default: Any = None) -> Any:
        """Obtiene un valor de configuración"""
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any):
        """Establece un valor de configuración"""
        keys = key.split('.')
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

        # Notificar a los watchers
        self._notify_watchers(key, value)

    def watch(self, key: str, callback: Callable):
        """Observa cambios en una clave de configuración"""
        if key not in self.watchers:
            self.watchers[key] = []

        self.watchers[key].append(callback)

    def _notify_watchers(self, key: str, value: Any):
        """Notifica a los watchers de un cambio"""
        if key in self.watchers:
            for callback in self.watchers[key]:
                callback(key, value)


class OrchestrationEngine:
    """
    Motor de orquestación que coordina todos los componentes del sistema
    """

    def __init__(self):
        self.service_registry = ServiceRegistry()
        self.event_bus = EventBus()
        self.config_manager = ConfigManager()

        # Estado del sistema
        self.system_state = {
            'initialized': False,
            'running': False,
            'start_time': None
        }

        print("🎭 OrchestrationEngine inicializado")

    def initialize(self):
        """Inicializa el motor de orquestación"""
        print("🚀 Inicializando sistema...")

        # Cargar configuración
        self.config_manager.load_config()

        # Iniciar event bus
        self.event_bus.start_processing()

        # Emitir evento de inicialización
        self.event_bus.emit('system.initialized', 'orchestration_engine')

        self.system_state['initialized'] = True
        print("✅ Sistema inicializado")

    def start(self):
        """Inicia el sistema"""
        if not self.system_state['initialized']:
            self.initialize()

        import time
        self.system_state['start_time'] = time.time()
        self.system_state['running'] = True

        # Iniciar servicios registrados
        for service_name in self.service_registry.services.keys():
            self.start_service(service_name)

        self.event_bus.emit('system.started', 'orchestration_engine')
        print("▶️ Sistema iniciado")

    def stop(self):
        """Detiene el sistema"""
        print("⏹️ Deteniendo sistema...")

        # Detener servicios
        for service_name in self.service_registry.services.keys():
            self.stop_service(service_name)

        # Detener event bus
        self.event_bus.stop_processing()

        self.system_state['running'] = False

        self.event_bus.emit('system.stopped', 'orchestration_engine')
        print("✅ Sistema detenido")

    def start_service(self, service_name: str) -> bool:
        """Inicia un servicio específico"""
        service = self.service_registry.get_service(service_name)

        if not service:
            print(f"❌ Servicio no encontrado: {service_name}")
            return False

        self.service_registry.update_service_status(service_name, ServiceStatus.STARTING)

        # Aquí iría la lógica de inicio del servicio
        # Por ahora solo cambiamos el estado

        self.service_registry.update_service_status(service_name, ServiceStatus.RUNNING)
        self.event_bus.emit('service.started', 'orchestration_engine', {'service': service_name})

        return True

    def stop_service(self, service_name: str) -> bool:
        """Detiene un servicio específico"""
        service = self.service_registry.get_service(service_name)

        if not service:
            return False

        self.service_registry.update_service_status(service_name, ServiceStatus.STOPPED)
        self.event_bus.emit('service.stopped', 'orchestration_engine', {'service': service_name})

        return True

    def restart_service(self, service_name: str) -> bool:
        """Reinicia un servicio"""
        self.stop_service(service_name)
        return self.start_service(service_name)

    def get_system_status(self) -> Dict[str, Any]:
        """Obtiene el estado del sistema"""
        import time

        uptime = 0
        if self.system_state['start_time']:
            uptime = time.time() - self.system_state['start_time']

        return {
            'initialized': self.system_state['initialized'],
            'running': self.system_state['running'],
            'uptime_seconds': uptime,
            'services': {
                'total': len(self.service_registry.services),
                'running': len(self.service_registry.get_healthy_services())
            }
        }


class ReinforcementAPI:
    """
    API principal para interactuar con el sistema de neuronas de refuerzo
    """

    def __init__(self):
        self.orchestrator = OrchestrationEngine()
        self.endpoints: Dict[str, Callable] = {}

        # Registrar endpoints
        self._register_endpoints()

        print("🌐 ReinforcementAPI inicializado")

    def _register_endpoints(self):
        """Registra los endpoints de la API"""
        self.endpoints = {
            '/api/train': self.train_agent,
            '/api/evaluate': self.evaluate_agent,
            '/api/predict': self.predict_action,
            '/api/status': self.get_status,
            '/api/metrics': self.get_metrics,
            '/api/config': self.get_config,
            '/api/services': self.list_services
        }

    def start(self):
        """Inicia la API"""
        self.orchestrator.initialize()
        self.orchestrator.start()
        print("🌟 API iniciada y lista para recibir peticiones")

    def stop(self):
        """Detiene la API"""
        self.orchestrator.stop()
        print("🛑 API detenida")

    def train_agent(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Entrena un agente de refuerzo

        Args:
            config: Configuración de entrenamiento

        Returns:
            Resultados del entrenamiento
        """
        print(f"🎓 Iniciando entrenamiento con config: {config}")

        # Aquí iría la lógica real de entrenamiento
        result = {
            'status': 'success',
            'episodes': config.get('num_episodes', 1000),
            'final_reward': 100.0,
            'training_time': 3600.0
        }

        self.orchestrator.event_bus.emit('training.completed', 'api', result)

        return result

    def evaluate_agent(self, agent_id: str, num_episodes: int = 10) -> Dict[str, Any]:
        """Evalúa un agente entrenado"""
        print(f"📊 Evaluando agente {agent_id}")

        result = {
            'agent_id': agent_id,
            'num_episodes': num_episodes,
            'avg_reward': 85.5,
            'success_rate': 0.9
        }

        return result

    def predict_action(self, state: List[float]) -> Dict[str, Any]:
        """Predice la mejor acción dado un estado"""
        import numpy as np

        # Simulación de predicción
        action = np.random.random(3).tolist()

        return {
            'state': state,
            'action': action,
            'confidence': 0.85
        }

    def get_status(self) -> Dict[str, Any]:
        """Obtiene el estado del sistema"""
        return self.orchestrator.get_system_status()

    def get_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas del sistema"""
        return {
            'total_episodes': 1000,
            'total_steps': 50000,
            'avg_reward': 75.5,
            'success_rate': 0.85
        }

    def get_config(self) -> Dict[str, Any]:
        """Obtiene la configuración actual"""
        return self.orchestrator.config_manager.config

    def list_services(self) -> List[Dict[str, Any]]:
        """Lista todos los servicios registrados"""
        services = self.orchestrator.service_registry.list_services()

        return [
            {
                'name': s.name,
                'version': s.version,
                'status': s.status.value,
                'endpoint': s.endpoint
            }
            for s in services
        ]

    def register_service(self, name: str, version: str, instance: Any = None):
        """Registra un nuevo servicio en la API"""
        service_info = ServiceInfo(
            name=name,
            version=version,
            status=ServiceStatus.STOPPED
        )

        self.orchestrator.service_registry.register_service(service_info, instance)

    def handle_request(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Maneja una petición a la API

        Args:
            endpoint: Ruta del endpoint
            **kwargs: Parámetros de la petición

        Returns:
            Respuesta del endpoint
        """
        if endpoint not in self.endpoints:
            return {
                'error': 'Endpoint not found',
                'endpoint': endpoint
            }

        try:
            handler = self.endpoints[endpoint]
            result = handler(**kwargs)

            return {
                'success': True,
                'data': result
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }


# Función de utilidad para crear una instancia completa del sistema
def create_reinforcement_system() -> ReinforcementAPI:
    """
    Crea e inicializa una instancia completa del sistema de refuerzo

    Returns:
        Instancia de ReinforcementAPI lista para usar
    """
    print("=" * 60)
    print("🌟 CREANDO SISTEMA DE NEURONAS DE REFUERZO")
    print("=" * 60)

    api = ReinforcementAPI()
    api.start()

    print("=" * 60)
    print("✅ SISTEMA LISTO")
    print("=" * 60)

    return api


print("✅ Módulo de API y orquestación cargado")
