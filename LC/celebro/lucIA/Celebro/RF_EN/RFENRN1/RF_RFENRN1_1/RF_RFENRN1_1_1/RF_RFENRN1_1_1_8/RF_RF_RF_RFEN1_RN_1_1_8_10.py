"""
Neurona Especializada 10: Integración con Second Life y OpenSimulator
Sistema unificador de metaversos y compatibilidad cross-platform
WoldVirtual3DlucIA v0.6.0
"""

import json
import hashlib
import uuid
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import base64


class PlatformType(Enum):
    """Plataformas de metaverso soportadas"""
    OPENSIMULATOR = "opensim"
    SECOND_LIFE = "secondlife"
    WOLDVIRTUAL = "woldvirtual"
    UNITY_BASED = "unity"
    UNREAL_BASED = "unreal"


class AssetType(Enum):
    """Tipos de assets"""
    TEXTURE = 0
    SOUND = 1
    CALLING_CARD = 2
    LANDMARK = 3
    CLOTHING = 5
    OBJECT = 6
    NOTECARD = 7
    SCRIPT = 10
    BODYPART = 13
    ANIMATION = 20
    GESTURE = 21
    MESH = 49


@dataclass
class AssetDescriptor:
    """Descriptor de asset compatible multi-plataforma"""
    asset_id: str
    name: str
    asset_type: AssetType
    creator_id: str
    owner_id: str
    data: Optional[bytes] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    platform: PlatformType = PlatformType.WOLDVIRTUAL


@dataclass
class InventoryItem:
    """Item de inventario compatible con SL/OpenSim"""
    item_id: str
    name: str
    description: str
    asset_id: str
    asset_type: AssetType
    folder_id: str
    permissions: Dict[str, int] = field(default_factory=dict)
    flags: int = 0


class SecondLifeIntegration:
    """
    Integración con Second Life
    Maneja assets, inventario y comunicación con grid SL
    """

    def __init__(self, grid_uri: str = "https://login.agni.lindenlab.com"):
        self.grid_uri = grid_uri
        self.session_id = None
        self.agent_id = None
        self.inventory_cache = {}
        self.asset_cache = {}

    def login(
        self,
        first_name: str,
        last_name: str,
        password: str,
        start_location: str = "last"
    ) -> Dict:
        """
        Login a Second Life grid

        Args:
            first_name, last_name: Nombre del avatar
            password: Contraseña
            start_location: Ubicación de inicio

        Returns:
            Datos de sesión
        """
        # Hash de contraseña MD5 (formato SL)
        password_hash = hashlib.md5(password.encode()).hexdigest()
        password_hash = "$1$" + hashlib.md5(password_hash.encode()).hexdigest()

        login_params = {
            "first": first_name,
            "last": last_name,
            "passwd": password_hash,
            "start": start_location,
            "channel": "WoldVirtual3D",
            "version": "0.6.0",
            "platform": "Win",
            "mac": "false",
            "id0": self._generate_mac_address(),
            "agree_to_tos": "true",
            "read_critical": "true",
            "viewer_digest": hashlib.md5(b"WoldVirtual3D_0.6.0").hexdigest(),
            "options": [
                "inventory-root",
                "inventory-skeleton",
                "inventory-lib-root",
                "inventory-lib-owner",
                "inventory-skel-lib",
                "gestures",
                "event_categories",
                "event_notifications",
                "classified_categories",
                "buddy-list",
                "ui-config"
            ]
        }

        # Simular respuesta de login (en producción hacer XML-RPC)
        login_response = {
            "success": True,
            "agent_id": str(uuid.uuid4()),
            "session_id": str(uuid.uuid4()),
            "secure_session_id": str(uuid.uuid4()),
            "first_name": first_name,
            "last_name": last_name,
            "start_location": start_location,
            "circuit_code": 12345,
            "sim_ip": "127.0.0.1",
            "sim_port": 13000,
            "region_x": 256000,
            "region_y": 256000,
            "seed_capability": f"{self.grid_uri}/CAPS/{uuid.uuid4()}/",
            "inventory_root": [{"folder_id": str(uuid.uuid4())}],
            "inventory_skeleton": []
        }

        self.session_id = login_response["session_id"]
        self.agent_id = login_response["agent_id"]

        return login_response

    def _generate_mac_address(self) -> str:
        """Genera MAC address para identificación"""
        return ":".join([f"{i:02x}" for i in [0x00, 0x11, 0x22, 0x33, 0x44, 0x55]])

    def fetch_inventory(self) -> Dict:
        """
        Obtiene inventario del agente

        Returns:
            Estructura de carpetas e items
        """
        # Estructura de inventario compatible con SL
        inventory = {
            "root_folder": str(uuid.uuid4()),
            "folders": [
                {
                    "folder_id": str(uuid.uuid4()),
                    "parent_id": "root",
                    "name": "Animations",
                    "type": 20
                },
                {
                    "folder_id": str(uuid.uuid4()),
                    "parent_id": "root",
                    "name": "Body Parts",
                    "type": 13
                },
                {
                    "folder_id": str(uuid.uuid4()),
                    "parent_id": "root",
                    "name": "Clothing",
                    "type": 5
                },
                {
                    "folder_id": str(uuid.uuid4()),
                    "parent_id": "root",
                    "name": "Gestures",
                    "type": 21
                },
                {
                    "folder_id": str(uuid.uuid4()),
                    "parent_id": "root",
                    "name": "Landmarks",
                    "type": 3
                },
                {
                    "folder_id": str(uuid.uuid4()),
                    "parent_id": "root",
                    "name": "Notecards",
                    "type": 7
                },
                {
                    "folder_id": str(uuid.uuid4()),
                    "parent_id": "root",
                    "name": "Objects",
                    "type": 6
                },
                {
                    "folder_id": str(uuid.uuid4()),
                    "parent_id": "root",
                    "name": "Scripts",
                    "type": 10
                },
                {
                    "folder_id": str(uuid.uuid4()),
                    "parent_id": "root",
                    "name": "Textures",
                    "type": 0
                }
            ],
            "items": []
        }

        self.inventory_cache = inventory
        return inventory

    def upload_asset(
        self,
        asset_data: bytes,
        asset_type: AssetType,
        name: str,
        description: str = ""
    ) -> str:
        """
        Sube asset al grid

        Args:
            asset_data: Datos del asset
            asset_type: Tipo de asset
            name: Nombre
            description: Descripción

        Returns:
            Asset ID
        """
        asset_id = str(uuid.uuid4())

        asset = AssetDescriptor(
            asset_id=asset_id,
            name=name,
            asset_type=asset_type,
            creator_id=self.agent_id or "unknown",
            owner_id=self.agent_id or "unknown",
            data=asset_data,
            metadata={"description": description},
            platform=PlatformType.SECOND_LIFE
        )

        self.asset_cache[asset_id] = asset

        # En producción, enviar vía HTTP upload capability
        return asset_id

    def download_asset(self, asset_id: str) -> Optional[bytes]:
        """
        Descarga asset del grid

        Args:
            asset_id: ID del asset

        Returns:
            Datos del asset
        """
        asset = self.asset_cache.get(asset_id)
        if asset:
            return asset.data

        # En producción, descargar vía HTTP asset service
        return None

    def parse_llsd_xml(self, xml_string: str) -> Dict:
        """Parsea LLSD XML (usado en capabilities)"""
        # Parser simplificado
        result = {}

        lines = xml_string.strip().split('\n')
        for line in lines:
            line = line.strip()
            if "<key>" in line and "<string>" in line:
                key_start = line.find("<key>") + 5
                key_end = line.find("</key>")
                val_start = line.find("<string>") + 8
                val_end = line.find("</string>")

                if key_end > key_start and val_end > val_start:
                    key = line[key_start:key_end]
                    value = line[val_start:val_end]
                    result[key] = value

        return result


class OpenSimConnector:
    """
    Conector para OpenSimulator
    Más flexible que SL, soporta features adicionales
    """

    def __init__(self, grid_uri: str = "http://localhost:8002"):
        self.grid_uri = grid_uri
        self.robust_services = {}
        self.region_info = {}

    def connect_to_grid(
        self,
        grid_name: str,
        login_uri: str
    ) -> Dict:
        """
        Conecta a un grid OpenSimulator

        Args:
            grid_name: Nombre del grid
            login_uri: URI de login

        Returns:
            Información del grid
        """
        grid_info = {
            "grid_name": grid_name,
            "login_uri": login_uri,
            "grid_nick": grid_name.lower().replace(" ", "_"),
            "platform": "OpenSimulator",
            "version": "0.9.2",
            "services": {
                "asset": f"{self.grid_uri}:8003",
                "inventory": f"{self.grid_uri}:8003",
                "grid": f"{self.grid_uri}:8003",
                "presence": f"{self.grid_uri}:8003",
                "avatar": f"{self.grid_uri}:8003"
            }
        }

        self.robust_services = grid_info["services"]
        return grid_info

    def create_region(
        self,
        region_name: str,
        location: Tuple[int, int],
        size: Tuple[int, int] = (256, 256),
        port: int = 9000
    ) -> Dict:
        """
        Crea una nueva región en el grid

        Args:
            region_name: Nombre de la región
            location: Coordenadas (x, y) en el grid
            size: Tamaño de la región
            port: Puerto UDP

        Returns:
            Información de la región
        """
        region_id = str(uuid.uuid4())

        region = {
            "region_id": region_id,
            "region_name": region_name,
            "location_x": location[0],
            "location_y": location[1],
            "size_x": size[0],
            "size_y": size[1],
            "internal_port": port,
            "external_hostname": "127.0.0.1",
            "owner_uuid": str(uuid.uuid4()),
            "estate_id": 1,
            "region_type": "Mainland"
        }

        self.region_info[region_id] = region
        return region

    def load_oar(self, oar_path: str, region_id: str) -> bool:
        """
        Carga archivo OAR (OpenSim Archive)

        Args:
            oar_path: Ruta al archivo OAR
            region_id: ID de región destino

        Returns:
            True si exitoso
        """
        # OAR es un archivo tar.gz con estructura específica
        # Contiene terreno, objetos, scripts, assets

        # En producción, extraer y procesar OAR
        return True

    def save_oar(self, region_id: str, output_path: str) -> bool:
        """
        Guarda región como archivo OAR

        Args:
            region_id: ID de región
            output_path: Ruta de salida

        Returns:
            True si exitoso
        """
        region = self.region_info.get(region_id)
        if not region:
            return False

        # En producción, crear archivo OAR con estructura:
        # - assets/ (texturas, meshes, scripts)
        # - objects/ (prims y objetos)
        # - terrains/ (heightmaps)
        # - landdata/ (parcels)
        # - settings/ (configuración)

        return True

    def convert_from_secondlife(self, sl_asset: AssetDescriptor) -> AssetDescriptor:
        """
        Convierte asset de Second Life a formato OpenSim

        Args:
            sl_asset: Asset de SL

        Returns:
            Asset convertido
        """
        # OpenSim es mayormente compatible con SL
        # Algunas diferencias en permisos y metadata

        opensim_asset = AssetDescriptor(
            asset_id=sl_asset.asset_id,
            name=sl_asset.name,
            asset_type=sl_asset.asset_type,
            creator_id=sl_asset.creator_id,
            owner_id=sl_asset.owner_id,
            data=sl_asset.data,
            metadata=sl_asset.metadata.copy(),
            platform=PlatformType.OPENSIMULATOR
        )

        # Ajustar permisos si es necesario
        opensim_asset.metadata["opensim_compatible"] = True

        return opensim_asset


class MetaverseUnifier:
    """
    Sistema unificador de metaversos
    Permite interoperabilidad entre plataformas
    """

    def __init__(self):
        self.platforms = {}
        self.asset_registry = {}
        self.conversion_cache = {}

    def register_platform(
        self,
        platform_type: PlatformType,
        connector: Any
    ):
        """Registra un conector de plataforma"""
        self.platforms[platform_type] = connector

    def convert_asset(
        self,
        asset: AssetDescriptor,
        target_platform: PlatformType
    ) -> Optional[AssetDescriptor]:
        """
        Convierte asset entre plataformas

        Args:
            asset: Asset origen
            target_platform: Plataforma destino

        Returns:
            Asset convertido
        """
        # Verificar cache
        cache_key = f"{asset.asset_id}_{target_platform.value}"
        if cache_key in self.conversion_cache:
            return self.conversion_cache[cache_key]

        # Convertir según plataformas
        converted = None

        if asset.platform == PlatformType.SECOND_LIFE and target_platform == PlatformType.OPENSIMULATOR:
            converted = self._convert_sl_to_opensim(asset)
        elif asset.platform == PlatformType.OPENSIMULATOR and target_platform == PlatformType.SECOND_LIFE:
            converted = self._convert_opensim_to_sl(asset)
        elif target_platform == PlatformType.WOLDVIRTUAL:
            converted = self._convert_to_woldvirtual(asset)

        if converted:
            self.conversion_cache[cache_key] = converted

        return converted

    def _convert_sl_to_opensim(self, asset: AssetDescriptor) -> AssetDescriptor:
        """Convierte de Second Life a OpenSim"""
        return AssetDescriptor(
            asset_id=asset.asset_id,
            name=asset.name,
            asset_type=asset.asset_type,
            creator_id=asset.creator_id,
            owner_id=asset.owner_id,
            data=asset.data,
            metadata={**asset.metadata, "converted_from": "secondlife"},
            platform=PlatformType.OPENSIMULATOR
        )

    def _convert_opensim_to_sl(self, asset: AssetDescriptor) -> AssetDescriptor:
        """Convierte de OpenSim a Second Life"""
        # Verificar compatibilidad
        if asset.asset_type == AssetType.MESH:
            # Meshes pueden requerir conversión especial
            pass

        return AssetDescriptor(
            asset_id=asset.asset_id,
            name=asset.name,
            asset_type=asset.asset_type,
            creator_id=asset.creator_id,
            owner_id=asset.owner_id,
            data=asset.data,
            metadata={**asset.metadata, "converted_from": "opensim"},
            platform=PlatformType.SECOND_LIFE
        )

    def _convert_to_woldvirtual(self, asset: AssetDescriptor) -> AssetDescriptor:
        """Convierte cualquier asset a formato WoldVirtual"""
        # WoldVirtual soporta formato extendido con más metadata
        enhanced_metadata = {
            **asset.metadata,
            "woldvirtual_version": "0.6.0",
            "original_platform": asset.platform.value,
            "enhanced": True
        }

        return AssetDescriptor(
            asset_id=asset.asset_id,
            name=asset.name,
            asset_type=asset.asset_type,
            creator_id=asset.creator_id,
            owner_id=asset.owner_id,
            data=asset.data,
            metadata=enhanced_metadata,
            platform=PlatformType.WOLDVIRTUAL
        )

    def synchronize_inventory(
        self,
        source_platform: PlatformType,
        target_platform: PlatformType,
        agent_id: str
    ) -> Dict:
        """
        Sincroniza inventario entre plataformas

        Args:
            source_platform: Plataforma origen
            target_platform: Plataforma destino
            agent_id: ID del agente

        Returns:
            Resultado de sincronización
        """
        source_connector = self.platforms.get(source_platform)
        target_connector = self.platforms.get(target_platform)

        if not source_connector or not target_connector:
            return {"success": False, "error": "Platform connector not found"}

        # Obtener inventario de origen
        source_inventory = source_connector.fetch_inventory()

        # Convertir items
        converted_items = []
        for item in source_inventory.get("items", []):
            # Descargar asset
            asset_data = source_connector.download_asset(item["asset_id"])

            if asset_data:
                # Crear descriptor
                asset = AssetDescriptor(
                    asset_id=item["asset_id"],
                    name=item["name"],
                    asset_type=AssetType(item["asset_type"]),
                    creator_id=item.get("creator_id", "unknown"),
                    owner_id=agent_id,
                    data=asset_data,
                    platform=source_platform
                )

                # Convertir
                converted = self.convert_asset(asset, target_platform)
                if converted:
                    converted_items.append(converted)

        return {
            "success": True,
            "items_converted": len(converted_items),
            "items": converted_items
        }

    def create_hypergrid_link(
        self,
        source_grid: str,
        target_grid: str,
        region_name: str
    ) -> str:
        """
        Crea link hypergrid entre grids OpenSim

        Args:
            source_grid: URI del grid origen
            target_grid: URI del grid destino
            region_name: Nombre de región

        Returns:
            URI de hypergrid
        """
        # Formato hypergrid: hg.example.com:8002:RegionName
        hypergrid_uri = f"{target_grid}:{region_name}"

        return hypergrid_uri

    def export_for_unity(self, assets: List[AssetDescriptor]) -> Dict:
        """
        Exporta assets en formato compatible con Unity

        Args:
            assets: Lista de assets

        Returns:
            Bundle de assets para Unity
        """
        unity_bundle = {
            "version": "2021.3",
            "assets": [],
            "metadata": {
                "source": "WoldVirtual3D",
                "export_date": "2025-11-05"
            }
        }

        for asset in assets:
            unity_asset = {
                "id": asset.asset_id,
                "name": asset.name,
                "type": self._map_asset_type_to_unity(asset.asset_type),
                "data": base64.b64encode(asset.data).decode() if asset.data else None
            }
            unity_bundle["assets"].append(unity_asset)

        return unity_bundle

    def _map_asset_type_to_unity(self, asset_type: AssetType) -> str:
        """Mapea tipos de asset a categorías Unity"""
        mapping = {
            AssetType.TEXTURE: "Texture2D",
            AssetType.MESH: "Mesh",
            AssetType.ANIMATION: "AnimationClip",
            AssetType.SOUND: "AudioClip",
            AssetType.SCRIPT: "MonoScript"
        }
        return mapping.get(asset_type, "Asset")

    def get_platform_statistics(self) -> Dict:
        """Obtiene estadísticas de todas las plataformas"""
        stats = {
            "registered_platforms": len(self.platforms),
            "cached_assets": len(self.asset_registry),
            "conversions_cached": len(self.conversion_cache),
            "platforms": {}
        }

        for platform_type, connector in self.platforms.items():
            stats["platforms"][platform_type.value] = {
                "connected": True,
                "type": platform_type.value
            }

        return stats
