"""Pruebas de la frontera de compatibilidad y configuracion local."""

from __future__ import annotations

import json
from pathlib import Path

from LC.compatibility import resolve_module_name


def test_aliases_legacy_apuntan_al_paquete_activo() -> None:
    assert resolve_module_name("BKSVCB") == "LC.celebro.BKSVCB"
    assert resolve_module_name("ipfs_manager") == "LC.celebro.CMFG.ipfs_manager"
    assert resolve_module_name("SNSBSTNPRB") == "LC.celebro.CMFG.SBSTM.SNSBSTNPRB"


def test_configuracion_local_tiene_esquema_y_backends() -> None:
    config_path = Path(__file__).parents[2] / "modelosIAlocal" / "IAlocal.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["schema_version"] == 1
    assert {item["type"] for item in config["backends"]} == {"ollama", "lmstudio"}
    assert config["fallback"]["offline"] is True
