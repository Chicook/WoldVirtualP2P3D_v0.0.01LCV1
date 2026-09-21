"""Test rapido de modulos mejorados."""
import sys, os, logging
logging.disable(logging.WARNING)
sys.path.insert(0, '.')
sys.path.insert(0, 'lucIA')

from lucIA.Celebro.Lobulo_temporal import get_temporal
t = get_temporal()
r = t.oir("Hola, me pregunto como funciona la red neuronal de LucIA y sus modulos anatomicos")
print("Temporal OK: tono={} lexico={}".format(r["tono"], r["lexicon"]))
top = t.conceptos_top(5)
print("  Top conceptos: {}".format(top))
print("  Riqueza lexica: {}".format(t.riqueza_lexica()))

from lucIA.Celebro.Hipotalamo import get_hipotalamo
h = get_hipotalamo()
for _ in range(5):
    info = h.latido(0.5)
print("Hipotalamo OK: energia={} estado={}".format(info["energia"], info["estado"]))
h.reponer(parcial=True)
print("Hipotalamo reponer OK: energia={}".format(h.energia))
ritmo = h.ritmo_sesion()
print("  Ritmo: turnos={} delta_medio={} alertas={}".format(ritmo["turnos"], ritmo["delta_medio"], ritmo["alertas"]))

from lucIA.Celebro.Cuerpo_calloso import get_calloso
c = get_calloso()
r2 = c.integrar(
    "Esta es la respuesta del modelo local sobre redes neuronales.",
    "Esta es la respuesta del modelo cloud con informacion adicional sobre sinapsis."
)
print("Calloso OK: via={} sim={} chars={}".format(r2["via"], r2["similitud"], r2["chars"]))

from lucIA.Celebro.Glia import get_glia
g = get_glia()
v = g.verificar()
print("Glia OK: limpio={}".format(v["limpio"]))
g.limpiar_eventos_antiguos(max_eventos=10)
print("Glia limpiar OK")

print()
print("TODOS LOS MODULOS MEJORADOS FUNCIONAN CORRECTAMENTE")
