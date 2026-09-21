"""
Neurona Especializada 6: Protocolo OpenSim y Networking de Metaversos
Implementación de protocolos LLUDP, LLSD y comunicación con grids
WoldVirtual3DlucIA v0.6.0
"""

import json
import struct
import socket
import threading
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import time
import uuid


class PacketType(Enum):
    """Tipos de paquetes de protocolo OpenSim/Second Life"""
    AGENT_UPDATE = 0x04
    CHAT_FROM_VIEWER = 0x50
    OBJECT_UPDATE = 0x0C
    AVATAR_APPEARANCE = 0x9E
    COMPLETE_AGENT_MOVEMENT = 0xF9
    REGION_HANDSHAKE = 0x94
    PACKET_ACK = 0xFB


class MessagePriority(Enum):
    """Prioridades de mensajes"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class NetworkPacket:
    """Paquete de red LLUDP"""
    packet_type: PacketType
    sequence_number: int
    payload: bytes
    reliable: bool = True
    priority: MessagePriority = MessagePriority.MEDIUM
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentInfo:
    """Información de agente (avatar) conectado"""
    agent_id: str
    session_id: str
    circuit_code: int
    first_name: str
    last_name: str
    position: Tuple[float, float, float] = (128.0, 128.0, 25.0)
    look_at: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    connected: bool = False


class LLSDParser:
    """
    Parser para LLSD (Linden Lab Structured Data)
    Formato de serialización usado en OpenSim/Second Life
    """

    @staticmethod
    def parse_xml_llsd(xml_string: str) -> Dict:
        """
        Parsea LLSD en formato XML

        Args:
            xml_string: String XML LLSD

        Returns:
            Diccionario Python con datos
        """
        # Implementación simplificada
        # En producción usar librería como pyogp.lib.base.llsd

        result = {}

        # Parser básico para demostración
        if "<key>" in xml_string and "<string>" in xml_string:
            # Extraer pares clave-valor simples
            lines = xml_string.split('\n')
            current_key = None

            for line in lines:
                line = line.strip()

                if line.startswith("<key>"):
                    current_key = line.replace("<key>", "").replace("</key>", "")
                elif line.startswith("<string>") and current_key:
                    value = line.replace("<string>", "").replace("</string>", "")
                    result[current_key] = value
                    current_key = None
                elif line.startswith("<integer>") and current_key:
                    value = int(line.replace("<integer>", "").replace("</integer>", ""))
                    result[current_key] = value
                    current_key = None

        return result

    @staticmethod
    def parse_binary_llsd(binary_data: bytes) -> Dict:
        """Parsea LLSD en formato binario"""
        result = {}

        # Implementación simplificada
        # Verificar header
        if len(binary_data) < 4:
            return result

        header = binary_data[:4]
        if header == b'<?ll':
            # Es LLSD binario válido
            # Implementar parser completo en producción
            pass

        return result

    @staticmethod
    def to_xml_llsd(data: Dict) -> str:
        """
        Convierte datos Python a LLSD XML

        Args:
            data: Diccionario de datos

        Returns:
            String XML LLSD
        """
        xml_lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<llsd>']

        def serialize_value(value):
            if isinstance(value, str):
                return f"<string>{value}</string>"
            elif isinstance(value, int):
                return f"<integer>{value}</integer>"
            elif isinstance(value, float):
                return f"<real>{value}</real>"
            elif isinstance(value, bool):
                return f"<boolean>{1 if value else 0}</boolean>"
            elif isinstance(value, dict):
                lines = ["<map>"]
                for k, v in value.items():
                    lines.append(f"<key>{k}</key>")
                    lines.append(serialize_value(v))
                lines.append("</map>")
                return "\n".join(lines)
            elif isinstance(value, list):
                lines = ["<array>"]
                for item in value:
                    lines.append(serialize_value(item))
                lines.append("</array>")
                return "\n".join(lines)
            else:
                return "<undef />"

        xml_lines.append(serialize_value(data))
        xml_lines.append('</llsd>')

        return "\n".join(xml_lines)

    @staticmethod
    def to_json_llsd(data: Dict) -> str:
        """Convierte a formato JSON (más simple)"""
        return json.dumps(data, indent=2)


class OpenSimProtocol:
    """
    Implementación del protocolo LLUDP de OpenSimulator
    Compatible con Second Life
    """

    def __init__(self, grid_uri: str = "localhost", port: int = 9000):
        self.grid_uri = grid_uri
        self.port = port
        self.socket = None
        self.sequence_number = 0
        self.pending_acks = {}
        self.agents = {}
        self.running = False

    def create_packet(
        self,
        packet_type: PacketType,
        payload: bytes,
        reliable: bool = True
    ) -> NetworkPacket:
        """
        Crea un paquete de red

        Args:
            packet_type: Tipo de paquete
            payload: Datos del paquete
            reliable: Si requiere ACK

        Returns:
            NetworkPacket creado
        """
        self.sequence_number += 1

        return NetworkPacket(
            packet_type=packet_type,
            sequence_number=self.sequence_number,
            payload=payload,
            reliable=reliable
        )

    def encode_packet(self, packet: NetworkPacket) -> bytes:
        """
        Codifica paquete a formato binario LLUDP

        Args:
            packet: Paquete a codificar

        Returns:
            Bytes del paquete
        """
        # Header LLUDP (simplificado)
        # Byte 0: Flags
        flags = 0x00
        if packet.reliable:
            flags |= 0x40  # Flag de fiabilidad

        # Construir paquete
        packet_data = bytearray()
        packet_data.append(flags)

        # Número de secuencia (4 bytes)
        packet_data.extend(struct.pack('>I', packet.sequence_number))

        # Tipo de paquete (1 byte para frecuente, 2 para no frecuente)
        packet_type_id = packet.packet_type.value
        if packet_type_id < 0xFF:
            packet_data.append(packet_type_id)
        else:
            packet_data.append(0xFF)
            packet_data.extend(struct.pack('>H', packet_type_id))

        # Payload
        packet_data.extend(packet.payload)

        return bytes(packet_data)

    def decode_packet(self, data: bytes) -> Optional[NetworkPacket]:
        """
        Decodifica paquete binario a NetworkPacket

        Args:
            data: Datos binarios

        Returns:
            NetworkPacket decodificado o None
        """
        if len(data) < 6:
            return None

        # Leer flags
        flags = data[0]
        reliable = bool(flags & 0x40)

        # Leer número de secuencia
        seq_num = struct.unpack('>I', data[1:5])[0]

        # Leer tipo de paquete
        if data[5] == 0xFF and len(data) > 7:
            packet_type_id = struct.unpack('>H', data[6:8])[0]
            payload = data[8:]
        else:
            packet_type_id = data[5]
            payload = data[6:]

        # Mapear tipo de paquete
        try:
            packet_type = PacketType(packet_type_id)
        except ValueError:
            return None

        return NetworkPacket(
            packet_type=packet_type,
            sequence_number=seq_num,
            payload=payload,
            reliable=reliable
        )

    def send_packet(self, packet: NetworkPacket, address: Tuple[str, int]):
        """Envía paquete por UDP"""
        if not self.socket:
            return

        data = self.encode_packet(packet)
        self.socket.sendto(data, address)

        # Guardar para reenvío si es fiable
        if packet.reliable:
            self.pending_acks[packet.sequence_number] = {
                "packet": packet,
                "address": address,
                "attempts": 0,
                "timestamp": time.time()
            }

    def send_ack(self, sequence_number: int, address: Tuple[str, int]):
        """Envía acknowledgment de paquete recibido"""
        ack_packet = self.create_packet(
            PacketType.PACKET_ACK,
            struct.pack('>I', sequence_number),
            reliable=False
        )
        self.send_packet(ack_packet, address)

    def handle_region_handshake(
        self,
        agent_id: str,
        session_id: str,
        region_info: Dict
    ) -> bytes:
        """
        Maneja handshake de región

        Args:
            agent_id: ID del agente
            session_id: ID de sesión
            region_info: Información de la región

        Returns:
            Payload del paquete de respuesta
        """
        # Construir respuesta de handshake
        response = {
            "RegionFlags": region_info.get("flags", 0),
            "SimAccess": 21,  # PG rating
            "SimName": region_info.get("name", "WoldVirtual Region"),
            "RegionID": region_info.get("id", str(uuid.uuid4())),
            "TerrainBase0": region_info.get("terrain_base0", str(uuid.uuid4())),
            "TerrainBase1": region_info.get("terrain_base1", str(uuid.uuid4())),
            "TerrainBase2": region_info.get("terrain_base2", str(uuid.uuid4())),
            "TerrainBase3": region_info.get("terrain_base3", str(uuid.uuid4())),
            "WaterHeight": region_info.get("water_height", 20.0)
        }

        # Serializar a bytes (simplificado)
        return json.dumps(response).encode('utf-8')

    def handle_agent_update(self, payload: bytes) -> Dict:
        """
        Procesa actualización de agente (movimiento, rotación)

        Args:
            payload: Datos del paquete

        Returns:
            Información de actualización
        """
        if len(payload) < 64:
            return {}

        # Extraer datos (formato simplificado)
        offset = 0

        # Agent ID (16 bytes UUID)
        agent_id = payload[offset:offset+16].hex()
        offset += 16

        # Session ID (16 bytes UUID)
        session_id = payload[offset:offset+16].hex()
        offset += 16

        # Body Rotation (quaternion, 4 floats)
        body_rotation = struct.unpack('>ffff', payload[offset:offset+16])
        offset += 16

        # Head Rotation (quaternion, 4 floats)
        head_rotation = struct.unpack('>ffff', payload[offset:offset+16])
        offset += 16

        return {
            "agent_id": agent_id,
            "session_id": session_id,
            "body_rotation": body_rotation,
            "head_rotation": head_rotation
        }

    def create_object_update_packet(
        self,
        object_id: str,
        position: Tuple[float, float, float],
        rotation: Tuple[float, float, float, float],
        velocity: Tuple[float, float, float]
    ) -> bytes:
        """
        Crea paquete de actualización de objeto

        Args:
            object_id: ID del objeto
            position: Posición (x, y, z)
            rotation: Rotación quaternion (x, y, z, w)
            velocity: Velocidad (x, y, z)

        Returns:
            Payload del paquete
        """
        payload = bytearray()

        # Object ID (16 bytes)
        object_uuid = uuid.UUID(object_id)
        payload.extend(object_uuid.bytes)

        # Position (3 floats)
        payload.extend(struct.pack('>fff', *position))

        # Rotation (4 floats)
        payload.extend(struct.pack('>ffff', *rotation))

        # Velocity (3 floats)
        payload.extend(struct.pack('>fff', *velocity))

        return bytes(payload)


class MetaverseNetworking:
    """
    Sistema de networking de alto nivel para metaversos
    Gestiona conexiones, agentes y sincronización
    """

    def __init__(self):
        self.protocol = OpenSimProtocol()
        self.llsd_parser = LLSDParser()
        self.connected_agents = {}
        self.region_objects = {}
        self.event_handlers = {}

    def connect_to_grid(
        self,
        grid_uri: str,
        login_uri: str,
        first_name: str,
        last_name: str,
        password: str
    ) -> Dict:
        """
        Conecta a un grid de OpenSimulator

        Args:
            grid_uri: URI del grid
            login_uri: URI de login
            first_name, last_name: Nombre del avatar
            password: Contraseña

        Returns:
            Datos de login
        """
        # Preparar request de login
        login_params = {
            "first": first_name,
            "last": last_name,
            "passwd": self._hash_password(password),
            "start": "last",
            "channel": "WoldVirtual3D",
            "version": "0.6.0",
            "platform": "Win",
            "mac": "false",
            "options": ["inventory-root", "inventory-skeleton"],
            "agree_to_tos": "true",
            "read_critical": "true"
        }

        # Simular respuesta de login (en producción hacer HTTP POST)
        login_response = {
            "success": True,
            "agent_id": str(uuid.uuid4()),
            "session_id": str(uuid.uuid4()),
            "secure_session_id": str(uuid.uuid4()),
            "first_name": first_name,
            "last_name": last_name,
            "circuit_code": 12345,
            "sim_ip": "127.0.0.1",
            "sim_port": 9000,
            "region_x": 256000,
            "region_y": 256000,
            "seed_capability": f"http://{grid_uri}:9000/CAPS/",
            "look_at": "[r1,r1,r0]",
            "home": "{'region_handle':[r256000,r256000], 'position':[r128,r128,r25]}"
        }

        # Registrar agente
        agent = AgentInfo(
            agent_id=login_response["agent_id"],
            session_id=login_response["session_id"],
            circuit_code=login_response["circuit_code"],
            first_name=first_name,
            last_name=last_name
        )

        self.connected_agents[agent.agent_id] = agent

        return login_response

    def _hash_password(self, password: str) -> str:
        """Hashea contraseña con MD5 (compatible con SL/OpenSim)"""
        return hashlib.md5(password.encode()).hexdigest()

    def teleport_agent(
        self,
        agent_id: str,
        region_handle: int,
        position: Tuple[float, float, float]
    ) -> bool:
        """
        Teleporta agente a nueva ubicación

        Args:
            agent_id: ID del agente
            region_handle: Handle de región destino
            position: Posición destino

        Returns:
            True si exitoso
        """
        agent = self.connected_agents.get(agent_id)
        if not agent:
            return False

        # Crear mensaje de teleport
        teleport_data = {
            "AgentID": agent_id,
            "SessionID": agent.session_id,
            "RegionHandle": region_handle,
            "Position": position
        }

        # En producción, enviar via CAPS
        agent.position = position

        return True

    def send_chat_message(
        self,
        agent_id: str,
        message: str,
        chat_type: int = 0  # 0=normal, 1=whisper, 2=shout
    ):
        """
        Envía mensaje de chat

        Args:
            agent_id: ID del agente emisor
            message: Texto del mensaje
            chat_type: Tipo de chat
        """
        agent = self.connected_agents.get(agent_id)
        if not agent:
            return

        # Construir payload de chat
        chat_payload = {
            "AgentID": agent_id,
            "SessionID": agent.session_id,
            "Message": message,
            "Type": chat_type,
            "Channel": 0
        }

        # Crear paquete
        payload_bytes = json.dumps(chat_payload).encode('utf-8')
        packet = self.protocol.create_packet(
            PacketType.CHAT_FROM_VIEWER,
            payload_bytes
        )

        # En producción, enviar a servidor
        print(f"[CHAT] {agent.first_name} {agent.last_name}: {message}")

    def update_object(
        self,
        object_id: str,
        position: Optional[Tuple[float, float, float]] = None,
        rotation: Optional[Tuple[float, float, float, float]] = None,
        velocity: Optional[Tuple[float, float, float]] = None
    ):
        """Actualiza objeto en el mundo"""
        obj = self.region_objects.get(object_id, {})

        if position:
            obj["position"] = position
        if rotation:
            obj["rotation"] = rotation
        if velocity:
            obj["velocity"] = velocity

        obj["timestamp"] = time.time()
        self.region_objects[object_id] = obj

        # Crear paquete de actualización
        payload = self.protocol.create_object_update_packet(
            object_id,
            obj.get("position", (0, 0, 0)),
            obj.get("rotation", (0, 0, 0, 1)),
            obj.get("velocity", (0, 0, 0))
        )

        packet = self.protocol.create_packet(
            PacketType.OBJECT_UPDATE,
            payload
        )

    def register_event_handler(self, event_type: str, handler_func):
        """Registra manejador de eventos"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler_func)

    def trigger_event(self, event_type: str, event_data: Dict):
        """Dispara evento"""
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    handler(event_data)
                except Exception as e:
                    print(f"Error in event handler: {e}")

    def get_region_info(self) -> Dict:
        """Obtiene información de región actual"""
        return {
            "name": "WoldVirtual Region",
            "size": (256, 256),
            "water_height": 20.0,
            "connected_agents": len(self.connected_agents),
            "objects": len(self.region_objects),
            "protocol_version": "0.6.0"
        }
