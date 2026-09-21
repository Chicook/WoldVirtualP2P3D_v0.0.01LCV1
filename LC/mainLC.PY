"""mainLC - Servidor blockchain de neuronas activas de LucIA."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# --- Conexión modular lucIA / RD_Neuronal (imports perezosos vía PuenteRDNeuronal) ---
_LUCIA_DIR = Path(__file__).parent / "celebro" / "lucIA"
if str(_LUCIA_DIR) not in sys.path:
    sys.path.insert(0, str(_LUCIA_DIR))
if str(_LUCIA_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_LUCIA_DIR.parent))


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

    def __init__(self, raiz: str | Path | None = None) -> None:
        if raiz is None:
            candidatos = [
                Path("LC/celebro/RD_Neuronal"),
                Path("celebro/RD_Neuronal"),
                Path(__file__).parent / "celebro" / "RD_Neuronal",
            ]
            self.raiz = next((p for p in candidatos if p.is_dir()), Path("LC/celebro/RD_Neuronal"))
        else:
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

    SEP = "+" + "-" * 78 + "+"
    MID = "+" + "-" * 78 + "+"
    TITULO = "LucIA - Blockchain de Neuronas Activas - servidor mainLC"

    def cabecera(self) -> None:
        print(self.SEP)
        print(f"|{self.TITULO.center(78)}|")
        print(self.MID)
        print(f"| {'BLOQUE':<8} {'NEURONA':<38} {'FAMILIA':<8} {'HASH (sha256)':<17} |")
        print(self.MID)

    def mostrar(self, b: Bloque) -> None:
        neurona = b.neurona_id if len(b.neurona_id) <= 38 else b.neurona_id[:35] + "~"
        print(f"| {b.index:<8} {neurona:<38} {b.modulo:<8} {(b.hash[:16] + '~'):<17} |")

    def seccion(self, fam: str) -> None:
        print(self.MID)
        print(f"| > Familia {fam:<66} |")
        print(self.MID)

    def resumen(self, nuevos: list[Bloque], segundos: float) -> None:
        from collections import Counter
        por_fam = Counter(b.modulo for b in nuevos if b.modulo != "mainLC")
        print(self.MID)
        print(f"| Total bloques: {len(self.cadena)} ({len(nuevos)} neuronas)  "
              f"{' '.join(f'{f}={por_fam.get(f, 0)}' for f in self.PAQUETES)}")
        ok = "[OK] INTEGRA" if self.verificar() else "[FALLO] ROTA"
        print(f"| Estado cadena: {ok}   Tiempo registro: {segundos:.2f} s")
        print(self.SEP)

    def registrar_todas(self, verbose: bool = True) -> list[Bloque]:
        """Un bloque con hash unico por cada neurona activa en disco."""
        nuevos: list[Bloque] = []
        if verbose:
            self.cabecera()
            self.mostrar(self.cadena[0])
        for fam in self.PAQUETES:
            d = self.raiz / fam
            if not d.is_dir():
                continue
            if verbose:
                self.seccion(fam)
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

class LucIAKernel:
    """Fachada modular: blockchain + sesión lucIA + cerebro + red neuronal.

    RD_Neuronal real vive en lucIA/Celebro/ y se resuelve vía PuenteRDNeuronal.
    """

    def __init__(self, raiz: str | Path | None = None) -> None:
        from lucIA.main import PuenteRDNeuronal
        PuenteRDNeuronal.instalar()
        self.puente = PuenteRDNeuronal
        self.blockchain = BlockchainServer(raiz)
        self.sesion = None
        self.conector = None
        self.conversor = None
        self.memoria = None
        self.cerebro = None

    def arrancar(self, verbose: bool = True) -> list[Bloque]:
        from lucIA import iniciar_sesion, HybridLLMConnector, get_conversor_pesos
        from lucIA.CORE.memory_manager import get_memory_manager
        from lucIA.main import SistemaCerebral
        self.sesion = iniciar_sesion()
        self.conector = HybridLLMConnector()
        self.conversor = get_conversor_pesos()
        self.memoria = get_memory_manager()
        self.cerebro = SistemaCerebral()
        self.cerebro.iniciar()
        t0 = time.time()
        nuevos = self.blockchain.registrar_todas(verbose=verbose)
        if verbose:
            self.blockchain.resumen(nuevos, time.time() - t0)
        return nuevos

    def turno(self, texto: str, hablar: bool = False) -> dict:
        assert self.conector is not None, "llamar a arrancar() primero"
        resp = self.conector.interactuar(texto)
        if isinstance(resp, tuple):
            respuesta, modelo = resp[0], (resp[1] if len(resp) > 1 else "?")
        elif isinstance(resp, dict):
            respuesta = resp.get("respuesta", str(resp))
            modelo = resp.get("modelo", "?")
        else:
            respuesta, modelo = str(resp), "?"
        info = self.conversor.procesar_consulta_a_pesos(texto) if self.conversor else {}
        if self.cerebro is not None:
            self.cerebro.tras_turno(texto, respuesta, modelo, info, self.memoria, self.conversor)
        if hablar:
            try:
                from lucIA import speak
                speak(respuesta)
            except Exception:
                pass
        return {"respuesta": respuesta, "modelo": modelo, "pesos": info}

    def chat(self) -> None:
        from lucIA.main import ejecutar_chat_principal
        ejecutar_chat_principal()

    def estado(self) -> dict:
        n = len(self.blockchain.cadena)
        return {"bloques": n, "integra": self.blockchain.verificar(),
                "cerebro": bool(getattr(self.cerebro, "activo", False))}

    def apagar(self) -> dict:
        from lucIA import cerrar_sesion
        try:
            if self.cerebro is not None:
                self.cerebro.volcar_cierre()
        except Exception:
            pass
        return cerrar_sesion()


if __name__ == "__main__":
    kernel = LucIAKernel()
    kernel.arrancar()
    kernel.chat()
    kernel.apagar()