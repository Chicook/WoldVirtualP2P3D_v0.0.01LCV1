"""
test_system.py - Pruebas automáticas de verificación del sistema lucIA
======================================================================
Valida:
1. Redirección de pycache a Celebro/cache.
2. Vaciado completo de Celebro/cache al cerrar sesión.
3. Envío de pesos a IPFS y borrado local estricto.
4. Límite de tamaño <= 125 MB.
"""

import os
import sys
import unittest
import numpy as np
from pathlib import Path

# Asegurar codificación utf-8
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).parent.resolve()
CACHE_DIR = ROOT / "Celebro" / "cache"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from lucIA import (
    iniciar_sesion,
    cerrar_sesion,
    IPFSManager,
    verificar_limite_espacio,
    vaciar_directorio_cache
)
from lucIA.CORE.base import LucIANeuronBase, LucIASystem


class TestLucIASystem(unittest.TestCase):

    def setUp(self):
        vaciar_directorio_cache()

    def test_01_tamano_sistema(self):
        """Verifica que el tamaño total del sistema no exceda los 125 MB."""
        info = verificar_limite_espacio()
        self.assertTrue(info["valido"], f"Tamaño ({info['tamano_mb']} MB) supera 125 MB")
        self.assertLessEqual(info["tamano_mb"], 125.0)

    def test_02_redireccion_pycache(self):
        """Verifica que sys.pycache_prefix apunte a Celebro/cache."""
        iniciar_sesion()
        self.assertEqual(Path(sys.pycache_prefix).resolve(), CACHE_DIR.resolve())
        cerrar_sesion()

    def test_03_vaciado_cache_al_cerrar(self):
        """Verifica que Celebro/cache quede 100% vacía tras cerrar sesión."""
        iniciar_sesion()
        # Crear un archivo de prueba en cache
        dummy_file = CACHE_DIR / "dummy_bytecode.pyc"
        dummy_file.write_bytes(b"bytecode_simulado")
        self.assertTrue(dummy_file.exists())

        # Cerrar sesión
        cerrar_sesion()
        self.assertFalse(dummy_file.exists())
        self.assertEqual(len(list(CACHE_DIR.iterdir())), 0)

    def test_04_pesos_ipfs_y_borrado_local(self):
        """Verifica que los pesos se transfieran a IPFS y se borren del sistema local."""
        sesion = iniciar_sesion()

        # Crear archivo local de pesos
        archivo_peso = ROOT / "test_pesos_temporales.npz"
        np.savez(archivo_peso, w=np.random.randn(5, 5))
        self.assertTrue(archivo_peso.exists())

        # Registrar en sesión
        sesion.registrar_archivo_pesos(archivo_peso)

        # Cerrar sesión
        resultado = sesion.cerrar()

        # 1. El archivo local de pesos DEBE estar borrado
        self.assertFalse(archivo_peso.exists(), "El archivo local de pesos no fue eliminado tras IPFS")

        # 2. Debe haberse generado CID IPFS
        self.assertGreater(len(resultado["cids_ipfs"]), 0)
        cid = resultado["cids_ipfs"][0]
        self.assertTrue(cid.startswith("Qm") or cid.startswith("bafy"))

        # 3. La cache debe estar vacía
        self.assertEqual(len(list(CACHE_DIR.iterdir())), 0)

    def test_05_recuperacion_desde_ipfs(self):
        """Verifica que los pesos almacenados en IPFS se puedan recuperar por su CID."""
        ipfs = IPFSManager()
        datos_prueba = {"capa1": [0.1, 0.2, 0.3], "neuronas": 3}
        res = ipfs.almacenar_pesos(origen=datos_prueba, nombre_modelo="modelo_prueba")
        cid = res["cid"]

        recuperado = ipfs.recuperar_pesos(cid)
        self.assertEqual(recuperado["capa1"], [0.1, 0.2, 0.3])

    def test_06_asimilacion_pesos_y_sintesis_propia(self):
        """Verifica la conversión de respuestas a pesos en Celebro y el filtrado en español."""
        from lucIA.Celebro.conversor_pesos import get_conversor_pesos
        from lucIA.llm_connector import _limpiar_respuesta_modelo

        conversor = get_conversor_pesos()
        info = conversor.asimilar_respuestas_y_calcular_sintesis(
            prompt="¿Qué es una neurona?",
            respuesta_modelo="Una neurona artificial procesa señales numéricas mediante ponderaciones y funciones de activación.",
            modelo_nombre="modelo_test"
        )
        self.assertIn("vector_semantico", info)
        self.assertIn("norma_delta_pesos", info)
        self.assertGreater(info["norma_delta_pesos"], 0.0)
        self.assertIn("guia_sintesis", info)

        # Probar limpieza estricta de español sin residuos de monólogo en inglés
        texto_sucio = '- User asked: "Test"\n- Persona: LucIA\n"Mi red neuronal en Celebro se adapta en tiempo real a cada pregunta que recibo."'
        limpio = _limpiar_respuesta_modelo(texto_sucio)
        self.assertNotIn("User asked", limpio)
        self.assertIn("red neuronal en Celebro", limpio)

    def test_07_auto_refactor_engine(self):
        """Verifica el motor autónomo de auto-inspección y diagnóstico de arquitectura."""
        from lucIA.CORE.auto_refactor import get_autorefactor_engine
        engine = get_autorefactor_engine()
        arqui = engine.inspeccionar_arquitectura()
        self.assertGreater(arqui["archivos_totales"], 10)
        self.assertGreater(arqui["lineas_totales"], 1000)
        self.assertLessEqual(arqui["tamano_mb"], 125.0)

        diag = engine.generar_diagnostico()
        self.assertIn("salud", diag)
        self.assertIn("prioridades", diag)

    def test_08_codificar_memoria_y_cache_en_pesos_e_ipfs(self):
        """Verifica que memoria_conversacion.json y archivos en cache se codifiquen en pesos neuronales, suban a IPFS y se borren localmente."""
        import json
        from lucIA import codificar_memoria_en_pesos, codificar_cache_en_pesos
        from lucIA.Celebro.conversor_pesos import get_conversor_pesos

        sesion = iniciar_sesion()
        conversor = get_conversor_pesos()

        # 1. Crear memoria_conversacion.json simulada
        memoria_path = ROOT / "Celebro" / "memoria_conversacion.json"
        datos_memoria = {
            "version": "1.0",
            "turnos": [
                {
                    "ts": 123456789.0,
                    "user": "Hola LucIA, ¿cómo funciona tu memoria?",
                    "lucia": "Mi memoria conversacional se asimila directamente como pesos sinápticos.",
                    "modelo": "test_model",
                    "emocion": 0.2
                }
            ],
            "resumen_contexto": "",
            "total_sesiones": 1,
            "ultima_sesion": "2026-09-06 20:00:00"
        }
        memoria_path.write_text(json.dumps(datos_memoria, ensure_ascii=False), encoding="utf-8")
        self.assertTrue(memoria_path.exists())

        # 2. Crear archivo temporal en Celebro/cache
        archivo_cache_dummy = CACHE_DIR / "datos_temporales_sesion.json"
        archivo_cache_dummy.write_text(json.dumps({"evento": "analisis_cache"}), encoding="utf-8")
        self.assertTrue(archivo_cache_dummy.exists())

        # 3. Codificar memoria en pesos
        archivo_npz = codificar_memoria_en_pesos(sesion=sesion, conversor=conversor)
        self.assertIsNotNone(archivo_npz)
        self.assertTrue(archivo_npz.exists())
        # El JSON original debe haberse eliminado
        self.assertFalse(memoria_path.exists(), "El JSON de memoria debe eliminarse tras codificarse")

        # 4. Codificar cache en pesos
        archivos_cache_npz = codificar_cache_en_pesos(sesion=sesion, conversor=conversor)
        self.assertGreater(len(archivos_cache_npz), 0)
        self.assertFalse(archivo_cache_dummy.exists(), "El archivo de cache debe eliminarse tras codificarse")

        # 5. Cerrar sesión y verificar transferencia a IPFS y limpieza
        res_cierre = sesion.cerrar()
        self.assertTrue(res_cierre["exito"])
        # El archivo .npz local debe haberse borrado tras subirse a IPFS
        self.assertFalse(archivo_npz.exists(), "El archivo .npz de memoria debe borrarse del disco tras enviarse a IPFS")
        self.assertGreater(len(res_cierre["cids_ipfs"]), 0)

    def test_09_restaurar_pesos_desde_ipfs(self):
        """Verifica la descarga y restauración de pesos neuronales desde IPFS hacia Celebro al iniciar sesión."""
        from lucIA import restaurar_pesos_desde_ipfs
        from lucIA.Celebro.conversor_pesos import get_conversor_pesos

        conversor = get_conversor_pesos()

        # 1. Simular un estado sináptico específico
        matriz_test = np.full((4, 8), 0.777, dtype=np.float32)
        conversor.slrn_pesos = matriz_test.copy()

        # 2. Guardar pesos de sesión y transferir a IPFS al cerrar
        sesion_salida = iniciar_sesion()
        archivo_salida = conversor.guardar_pesos_en_celebro()
        sesion_salida.registrar_archivo_pesos(archivo_salida)
        cierre = sesion_salida.cerrar()
        self.assertTrue(cierre["exito"])

        # 3. Alterar intencionalmente los pesos locales para comprobar la restauración
        conversor.slrn_pesos = np.zeros((4, 8), dtype=np.float32)
        self.assertFalse(np.allclose(conversor.slrn_pesos, matriz_test))

        # 4. Iniciar nueva sesión y restaurar pesos desde IPFS
        sesion_nueva = iniciar_sesion()
        res_restauracion = restaurar_pesos_desde_ipfs(sesion=sesion_nueva, conversor=conversor)
        self.assertTrue(res_restauracion["exito"])
        self.assertTrue(res_restauracion["restaurado"])
        self.assertIsNotNone(res_restauracion["cid"])
        self.assertGreater(res_restauracion["componentes_restaurados"], 0)

        # 5. Comprobar que los pesos restaurados en SLRN coincidan exactamente con los que se enviaron a IPFS
        self.assertTrue(np.allclose(conversor.slrn_pesos, matriz_test, atol=1e-4), "Los pesos restaurados desde IPFS no coinciden con los originales")
        sesion_nueva.cerrar()

    def test_10_rotacion_un_modelo_por_turno(self):
        """Verifica rotacion triple LM->Ollama->Cloud."""
        from lucIA.model_router import ModelRouter
        from lucIA.llm_connector import OPENROUTER_FREE_MODELS
        import tempfile
        # Fichero temporal: los tests no deben contaminar el router real.
        tmp_estado = Path(tempfile.gettempdir()) / "router_state_test.json"
        try:
            tmp_estado.unlink()
        except Exception:
            pass
        r = ModelRouter(["L1", "L2"], list(OPENROUTER_FREE_MODELS[:4]), ["O1", "O2"], estado_file=tmp_estado)
        seq = []
        for _ in range(6):
            m, o = r.siguiente("auto")
            seq.append((m, o))
            r.avanzar(m, o)
        origenes = [o for _, o in seq]
        self.assertEqual(origenes, ["LMStudio_Local", "Ollama_Local", "OpenRouter_Free"] * 2)


    def test_11_env_cargado(self):
        """Verifica que .env de la raíz se carga al importar lucIA."""
        import lucIA  # noqa: F401 — el import ejecuta la carga .env
        self.assertTrue(os.environ.get("OPENROUTER_API_KEY", ""), "OPENROUTER_API_KEY no cargada desde .env")

    def test_12_providers_factory(self):
        """Verifica adaptador único: lm/ollama/openrouter/offline."""
        from lucIA.providers.factory import get_provider
        for kind in ("lm", "ollama", "offline"):
            p = get_provider(kind)
            self.assertTrue(hasattr(p, "chat"))
        self.assertEqual(get_provider("offline").name, "echo-offline")

    def test_13_sincronizar_cerebro(self):
        """Verifica F2: sincronizar_desde_conversor siembra módulos antes de la 1ª consulta."""
        from lucIA.main import SistemaCerebral
        from lucIA.Celebro.conversor_pesos import get_conversor_pesos
        from lucIA.CORE.memory_manager import get_memory_manager
        c = SistemaCerebral()
        c.iniciar()
        conv = get_conversor_pesos()
        mem = get_memory_manager()
        n = c.sincronizar_desde_conversor(conv, mem, {"cid": "QmTest", "componentes_restaurados": 1})
        self.assertGreaterEqual(n, 1)


if __name__ == "__main__":
    unittest.main()



