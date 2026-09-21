"""lucIA.chat - LEGACY: delega al entry-point único main.py."""
import sys, os
from pathlib import Path
ROOT_DIR = Path(__file__).parent.resolve()
CACHE_DIR = ROOT_DIR / "Celebro" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
sys.pycache_prefix = str(CACHE_DIR)
os.environ["PYTHONPYCACHEPREFIX"] = str(CACHE_DIR)
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ROOT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR.parent))

def iniciar_chat_interactivo(modo_inicio: str = "auto"):
    from lucIA.main import ejecutar_chat_principal
    return ejecutar_chat_principal()

if __name__ == "__main__":
    iniciar_chat_interactivo()
