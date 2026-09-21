"""Tests basicos de los 14 modulos cerebrales (1 fichero central, ligero)."""
import unittest


class TestCerebroBasico(unittest.TestCase):
    def test_modulos_resumen(self):
        from lucIA.Celebro.hipocampo import get_hipocampo
        from lucIA.Celebro.Talamo import get_talamo
        from lucIA.Celebro.Amigdala import get_amigdala
        from lucIA.Celebro.Corteza_sensorial import get_sensorial
        from lucIA.Celebro.Corteza_prefrontal import get_prefrontal
        from lucIA.Celebro.Ganglios_basales import get_ganglios
        from lucIA.Celebro.Cerebelo import get_cerebelo
        from lucIA.Celebro.Tronco_encefalico import get_tronco
        from lucIA.Celebro.Glia import get_glia
        from lucIA.Celebro.Hipotalamo import get_hipotalamo
        from lucIA.Celebro.Cuerpo_calloso import get_calloso
        from lucIA.Celebro.Corteza_motora import get_motora
        from lucIA.Celebro.Lobulo_temporal import get_temporal
        from lucIA.Celebro.Lobulo_occipital import get_occipital
        from lucIA.Celebro.Lobulo_parietal import get_parietal
        mods = [get_hipocampo(), get_talamo(), get_amigdala(), get_sensorial(),
                get_prefrontal(), get_ganglios(), get_cerebelo(), get_tronco(),
                get_glia(), get_hipotalamo(), get_calloso(), get_motora(),
                get_temporal(), get_occipital(), get_parietal()]
        self.assertEqual(len(mods), 15)
        for m in mods:
            r = m.resumen() if hasattr(m, "resumen") else m.resumen_estado()
            self.assertTrue(len(r) > 5, type(m).__name__)

    def test_sistema_cerebral_agrupa(self):
        from lucIA.main import SistemaCerebral
        c = SistemaCerebral()
        self.assertTrue(c.iniciar())
        self.assertEqual(len(c.modulos), 13)
        self.assertGreaterEqual(len(c.lineas_estado()), 8)


if __name__ == "__main__":
    unittest.main()
