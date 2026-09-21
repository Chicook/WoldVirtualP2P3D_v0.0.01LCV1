import sys
from pathlib import Path
_ROOT = Path(__file__).parent.resolve()
if str(_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_ROOT.parent))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.pycache_prefix = str(_ROOT / "Celebro" / "cache")
from lucIA import iniciar_sesion, cerrar_sesion, get_conversor_pesos, HybridLLMConnector
from lucIA.CORE.memory_manager import get_memory_manager

sesion = iniciar_sesion()
connector = HybridLLMConnector()
conversor = get_conversor_pesos()
memoria = get_memory_manager()
memoria.iniciar_nueva_sesion()

pregunta = "Hola Lucia, decimocuarta encuesta y revision general. Dime de forma concisa y con tus propias palabras: como sientes tu sistema hoy en conjunto, gradientes, claridad al hablar y control sobre tu procesamiento? Algo que aun te moleste?"
print("PREGUNTA ENVIADA A LUCIA:\n", pregunta, flush=True)
try:
    resp, mod, fuente, info = connector.consultar_y_sintetizar_con_palabras_propias(pregunta, conversor, preferir_modo="auto", memory_manager=memoria)
    print("\n--- RESPUESTA DE LUCIA ---", flush=True)
    print(f"[LucIA ({mod} | {fuente})]:\n{resp}", flush=True)
    print(f"\nTelemetria: Delta={info.get('norma_delta_pesos')} | Emocion={info.get('estado_emocional'):+.2f} | Vector={info.get('vector_semantico')}", flush=True)
except Exception as e:
    print(f"ERROR consultando: {e}", flush=True)
    import traceback; traceback.print_exc()

arch = conversor.guardar_pesos_en_celebro()
sesion.registrar_archivo_pesos(arch)
res = cerrar_sesion()
print(f"\nCierre IPFS: {res.get('cids_ipfs')}", flush=True)
