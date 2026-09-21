"""Corteza_motora/planificador.py - De la intencion a la accion."""
import re


class PlanificadorMotor:
    """Convierte intenciones del dialogo en secuencias de comandos validos."""

    COMANDOS = ("/voz", "/modo", "/descansa", "/bien", "/mal",
                "/estado", "/memoria", "/olvida")

    def __init__(self):
        self.plan = []
        self.ejecutados = 0

    def planificar(self, intencion: str) -> list:
        t = (intencion or "").lower()
        pasos = []
        if re.search(r"descans|respir|pausa", t):
            pasos.append("/descansa")
        if re.search(r"callate|silencio|sin voz", t):
            pasos.append("/voz off")
        if re.search(r"habla|voz alta", t):
            pasos.append("/voz on")
        if re.search(r"olvida|borra.*memoria|empezar de cero", t):
            pasos.append("/olvida")
        if re.search(r"estado|como estas|diagnost", t):
            pasos.append("/estado")
        self.plan = pasos
        return list(pasos)

    def siguiente(self) -> str:
        if not self.plan:
            return ""
        paso = self.plan.pop(0)
        self.ejecutados += 1
        return paso

    def es_comando(self, texto: str) -> bool:
        partes = (texto or "").strip().split()
        return bool(partes) and partes[0] in self.COMANDOS
