"""Corteza_motora - Acciones de LucIA: comandos como movimiento (basico)."""
from lucIA.Celebro.bases_neuro import ModuloCerebral


class CortezaMotora(ModuloCerebral):
    nombre = "Motora"
    ACCIONES = ("/voz", "/modo", "/descansa", "/bien", "/mal", "/estado", "/memoria", "/olvida")

    def __init__(self):
        super().__init__()
        self.ultima_accion = ""

    def actuar(self, comando: str) -> str:
        cmd = (comando or "").split()[0] if comando else ""
        if cmd in self.ACCIONES:
            self.ultima_accion = cmd
            self._notar(f"accion {cmd}")
            return cmd
        return ""

    def resumen(self) -> str:
        return f"Motora: ultima={self.ultima_accion or '-'} | {super().resumen()}"


_motora_global = None

def get_motora() -> CortezaMotora:
    global _motora_global
    if _motora_global is None:
        _motora_global = CortezaMotora()
    return _motora_global
