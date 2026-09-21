"""mainLC - Servidor blockchain de neuronas activas de LucIA."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Bloque:
    index: int
    timestamp: float
    neurona_id: str
    modulo: str
    prev_hash: str
    hash: str = field(default="")

    def calcular_hash(self) -> str:
        base = f"{self.index}|{self.timestamp}|{self.neurona_id}|{self.modulo}|{self.prev_hash}"
        return hashlib.sha256(base.encode("utf-8")).hexdigest()


class BlockchainServer:
    """Registra cada neurona activa con un hash unico por bloque."""

    PAQUETES = ("ENRN", "RF_SL", "RF_EN", "RNP", "SLRN")

    def __init__(self, raiz: str = "LC/celebro/RD_NEURONAL") -> None:
        self.raiz = Path(raiz)
        self.cadena: list[Bloque] = []
        self._vistos: set[str] = set()
        self._genesis()

    def _genesis(self) -> Bloque:
        b = Bloque(0, time.time(), "GENESIS", "mainLC", "0" * 64)
        b.hash = b.calcular_hash()
        self.cadena.append(b)
        return b

    def _nuevo_bloque(self, neurona_id: str, modulo: str) -> Bloque:
        b = Bloque(len(self.cadena), time.time(), neurona_id, modulo, self.cadena[-1].hash)
        b.hash = b.calcular_hash()
        self.cadena.append(b)
        self._vistos.add(neurona_id)
        return b

    @staticmethod
    def _activa(path: Path) -> str | None:
        """Activa si se puede cargar; si no, si al menos compila (activa-ligera)."""
        try:
            spec = importlib.util.spec_from_file_location(f"lucia_{path.stem}", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            clases = [n for n, o in vars(mod).items() if isinstance(o, type)]
            return clases[0] if clases else "modulo"
        except Exception:
            try:
                ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
                return "sintaxis-ok"
            except Exception:
                return None

    def mostrar(self, b: Bloque) -> None:
        print(f"[Bloque {b.index:04d}] {b.neurona_id} | hash={b.hash[:16]}... | prev={b.prev_hash[:8]}...")

    def registrar_todas(self, verbose: bool = True) -> list[Bloque]:
        """Un bloque con hash unico por cada neurona activa en disco."""
        nuevos: list[Bloque] = []
        for fam in self.PAQUETES:
            d = self.raiz / fam
            if not d.is_dir():
                continue
            for path in sorted(d.rglob("*.py")):
                if path.name.startswith("_") and path.name != "__init__.py":
                    continue
                tag = self._activa(path)
                if tag is None:
                    continue  # inactiva: no entra en cadena
                nid = f"{fam}.{path.relative_to(d).with_suffix('')}:{tag}"
                if nid in self._vistos:
                    continue
                b = self._nuevo_bloque(nid, fam)
                nuevos.append(b)
                if verbose:
                    self.mostrar(b)
        return nuevos

    def verificar(self) -> bool:
        for i, b in enumerate(self.cadena):
            if b.hash != b.calcular_hash():
                return False
            if i and b.prev_hash != self.cadena[i - 1].hash:
                return False
        return True

if __name__ == "__main__":
    srv = BlockchainServer()
    srv.mostrar(srv.cadena[0])
    nuevos = srv.registrar_todas()
    print(f"Bloques: {len(srv.cadena)} (nuevos: {len(nuevos)}) | integra: {srv.verificar()}")
