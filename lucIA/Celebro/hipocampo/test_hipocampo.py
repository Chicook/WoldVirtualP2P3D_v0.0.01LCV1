"""Tests del hipocampo (viven dentro de Celebro/hipocampo/)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.resolve()))

from lucIA.Celebro.hipocampo import Hipocampo, get_hipocampo


class FakeConversor:
    def __init__(self):
        from lucIA.Celebro.conversor_pesos import get_conversor_pesos
        self.real = get_conversor_pesos()
        self.encoder = self.real.encoder
    def _propagar_todas_las_neuronas(self, v):
        return self.real._propagar_todas_las_neuronas(v)
    def _actualizar_pesos_en_todas_las_neuronas(self, v, act, factor=1.0, fase="x"):
        return self.real._actualizar_pesos_en_todas_las_neuronas(v, act, factor=factor, fase=fase)
    def guardar_pesos_en_celebro(self, archivo=None):
        return self.real.guardar_pesos_en_celebro(archivo=archivo)


class TestHipocampo(unittest.TestCase):
    def test_importancia_emotivo_mayor(self):
        h = Hipocampo()
        t1 = h.registrar("hola", "hola, que tal", emocion=0.8, delta=1.0)
        t2 = h.registrar("hola", "hola, que tal", emocion=0.0, delta=0.01)
        self.assertGreater(t1["importancia"], t2["importancia"])

    def test_buffer_capado(self):
        h = Hipocampo(max_buffer=5)
        for i in range(8):
            h.registrar(f"p{i}", f"r{i}")
        self.assertEqual(len(h.buffer), 5)

    def test_consolidar_mueve_pesos(self):
        import numpy as np
        h = Hipocampo()
        conv = FakeConversor()
        antes = conv.real.slrn_pesos.copy()
        h.registrar("Como estas hoy?", "Estoy bien, aprendiendo en tiempo real.",
                    emocion=0.5, delta=2.0)
        hechos = h.consolidar(conv, n=5)
        self.assertEqual(hechos, 1)
        self.assertFalse(np.allclose(antes, conv.real.slrn_pesos))

    def test_volcar_cierre_vacia_buffer(self):
        h = Hipocampo()
        conv = FakeConversor()
        h.registrar("p1", "r1 r1 r1", emocion=0.1, delta=0.5)
        h.registrar("p2", "r2 r2 r2", emocion=0.6, delta=1.0)
        res = h.volcar_cierre(conv, sesion=None)
        self.assertEqual(res["buffer_restante"], 0)
        self.assertEqual(len(h.buffer), 0)
        self.assertEqual(res["consolidados"], 2)
        import os
        if res["archivo"] and os.path.exists(res["archivo"]):
            os.remove(res["archivo"])

    def test_singleton(self):
        self.assertIs(get_hipocampo(), get_hipocampo())


if __name__ == "__main__":
    unittest.main()
