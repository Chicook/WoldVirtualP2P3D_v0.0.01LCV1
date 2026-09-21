"""Lobulo_parietal/mapa.py - Mapa contextual quien/donde/cuando."""
import re
import time


class MapaContextual:
    """Fusiona cada turno en un mapa: interlocutor, temas, modelo, momento."""

    def __init__(self, max_turnos: int = 20):
        self.max_turnos = max_turnos
        self.turnos = []

    def situar(self, pregunta: str, modelo: str = "", turno: int = 0) -> dict:
        temas = [w.lower() for w in re.findall(r"[a-zA-Záéíóúñ]{5,}", pregunta or "")][:6]
        mapa = {"turno": turno, "modelo": modelo, "cuando": round(time.time(), 0),
                "temas": temas, "chars": len(pregunta or "")}
        self.turnos.append(mapa)
        if len(self.turnos) > self.max_turnos:
            self.turnos.pop(0)
        return dict(mapa)

    def hilo(self) -> dict:
        """Devuelve el hilo: temas recurrentes de la sesion."""
        frec = {}
        for t in self.turnos:
            for w in t["temas"]:
                frec[w] = frec.get(w, 0) + 1
        top = sorted(frec.items(), key=lambda kv: kv[1], reverse=True)[:5]
        return {"turnos": len(self.turnos), "temas_top": top,
                "modelos": sorted({t["modelo"] for t in self.turnos if t["modelo"]})}
