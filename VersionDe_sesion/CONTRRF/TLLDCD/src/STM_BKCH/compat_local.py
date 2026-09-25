"""Shims locales BKSVCB: conversor PSNRCV e IPFSManager mínimos.

Evitan el crash cuando los módulos originales LC.celebro no existen en src/.
El conversor expone la interfaz usada por mainLCSTM y BKSVCB:
neuronas, deriva_acumulada, procesar_consulta_a_pesos().
"""
from __future__ import annotations
import hashlib
import time
from typing import Any, Dict, List, Optional


class ConversorRespuestaPesos:
    """Conversor local de 50 neuronas (5 capas x 10)."""

    def __init__(self) -> None:
        self.neuronas: List[Dict[str, Any]] = [
            {"id": f"{capa}{i}", "peso": 0.0}
            for capa in ("ENRN", "RF_EN", "RF_SL", "RNP", "SLRN")
            for i in range(1, 11)
        ]
        self.deriva_acumulada: float = 0.0

    def persistir_pesos_en_psnrl(self, etiqueta: str = "checkpoint"):
        from pathlib import Path
        return (Path(f"{etiqueta}.npz"), Path(f"{etiqueta}.json"))

    def procesar_consulta_a_pesos(self, prompt: str) -> Dict[str, Any]:
        h = int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16)
        val = ((h % 2000) / 1000.0) - 1.0
        norma = round((h % 100000) / 100000.0, 5)
        self.deriva_acumulada = round(self.deriva_acumulada + norma * 0.001, 6)
        tonos = ("reflexivo", "analitico", "creativo", "pragmatico")
        return {
            "tono_cognitivo": tonos[h % len(tonos)],
            "estado_emocional": round(val, 3),
            "norma_delta_aplicada": norma,
        }


_conversor_inst: Optional[ConversorRespuestaPesos] = None


def get_conversor_pesos() -> ConversorRespuestaPesos:
    global _conversor_inst
    if _conversor_inst is None:
        _conversor_inst = ConversorRespuestaPesos()
    return _conversor_inst


class IPFSManager:
    """IPFS no-op: devuelve CIDs sintéticos sin requerir daemon."""

    def almacenar_pesos(self, datos: Any = None, **kw: Any) -> Dict[str, Any]:
        sello = hashlib.sha256(repr((str(datos)[:200], time.time())).encode()).hexdigest()[:16]
        return {"ipfs_cid": f"QmLocal{sello}", "ok": True}

    def subir_y_limpiar_psnrl(self, forzar_borrado_sin_daemon: bool = False) -> Dict[str, Any]:
        return {"ok": True, "modo": "local-sin-daemon"}


_ipfs_inst: Optional[IPFSManager] = None


def get_ipfs_manager() -> IPFSManager:
    global _ipfs_inst
    if _ipfs_inst is None:
        _ipfs_inst = IPFSManager()
    return _ipfs_inst
