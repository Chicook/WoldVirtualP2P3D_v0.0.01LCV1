"""lucIA.model_router - Rotacion estricta Local -> Cloud -> Local (1 modelo por pregunta)."""
import json, time
from pathlib import Path
from typing import List, Tuple, Optional

ESTADO_FILE = Path(__file__).parent / "config" / "router_state.json"
SUFIJO_ES = "\n[Responde únicamente en español de España]"
PALABRAS_INGLES = {"the","is","and","to","of","in","you","that","with","this","from","user","asking","response","thinking","suggests","what","will","about"}

class ModelRouter:
    def __init__(self, pool_lm: List[str], pool_cloud: List[str], pool_ollama: Optional[List[str]] = None, cooldown_s: int = 60, estado_file=None):
        self.pool_lm = [m for m in pool_lm if m]
        self.pool_cloud = [m for m in pool_cloud if m]
        self.pool_ollama = [m for m in (pool_ollama or []) if m]
        self.pool_local = self.pool_lm  # compat
        self.estado_file = estado_file or ESTADO_FILE
        self.cooldown_s = cooldown_s
        self.turno = 0
        self.fallos: dict = {}  # modelo -> timestamp fin cooldown
        self.historial: list = []
        self._cargar()

    def _cargar(self):
        try:
            if self.estado_file.exists():
                d = json.loads(self.estado_file.read_text(encoding="utf-8"))
                self.turno = int(d.get("turno", 0))
                self.historial = d.get("historial", [])[-50:]
        except Exception:
            pass

    def _guardar(self):
        try:
            self.estado_file.parent.mkdir(parents=True, exist_ok=True)
            self.estado_file.write_text(json.dumps({"turno": self.turno, "historial": self.historial[-50:]}, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def _en_cooldown(self, m: str) -> bool:
        return time.time() < self.fallos.get(m, 0)

    def marcar_fallo(self, modelo: str):
        self.fallos[modelo] = time.time() + self.cooldown_s

    def _elegir(self, pool: List[str], k: int) -> Optional[str]:
        disp = [m for m in pool if not self._en_cooldown(m)] or pool
        if not disp:
            return None
        return disp[k % len(disp)]

    def siguiente(self, modo: str = "auto") -> Tuple[str, str]:
        """Retorna (modelo, origen). Rotación L->O->C en auto; 1 modelo por llamada."""
        if modo == "local":
            m = self._elegir(self.pool_lm, self.turno)
            return (m or "qwen2.5-0.5b-instruct", "LMStudio_Local")
        if modo == "ollama":
            m = self._elegir(self.pool_ollama, self.turno)
            return (m or "qwen3:1.7b", "Ollama_Local")
        if modo == "cloud":
            m = self._elegir(self.pool_cloud, self.turno)
            return (m or "nvidia/nemotron-3.5-lightning:free", "OpenRouter_Free")
        # auto: rotación triple LM -> Ollama -> Cloud
        k = self.turno % 3
        if k == 0:
            m = self._elegir(self.pool_lm, self.turno // 3)
            return (m or "qwen2.5-0.5b-instruct", "LMStudio_Local")
        if k == 1:
            m = self._elegir(self.pool_ollama, self.turno // 3)
            return (m or "qwen3:1.7b", "Ollama_Local")
        m = self._elegir(self.pool_cloud, self.turno // 3)
        return (m or "nvidia/nemotron-3.5-lightning:free", "OpenRouter_Free")

    def toca(self) -> str:
        return ("LOCAL" if self.turno % 3 == 0 else "OLLAMA" if self.turno % 3 == 1 else "CLOUD")

    def avanzar(self, modelo: str, origen: str, ok: bool = True):
        self.historial.append({"turno": self.turno, "modelo": modelo, "origen": origen, "ok": ok})
        if not ok:
            self.marcar_fallo(modelo)
        self.turno += 1
        self._guardar()

def es_ingles(texto: str) -> bool:
    import re
    toks = [w.lower() for w in re.findall(r"[a-zA-Záéíóúñ]+", texto or "")]
    if len(toks) < 4:
        return False
    en = sum(1 for t in toks if t in PALABRAS_INGLES)
    return en > 3
