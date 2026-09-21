"""
lucIA.Celebro.hipocampo - Hipocampo artificial de LucIA
========================================================
Consolidador entre memoria a corto plazo (buffer en RAM) y memoria a largo
plazo (pesos neuronales en Celebro + IPFS). No almacena: decide QUE se
consolida, CUANDO y CON QUE FUERZA, como el hipocampo biologico.

Diseno autocontenido: este paquete SOLO usa la API publica del conversor
(actualizar_memoria_salida, guardar_pesos_en_celebro) y tipos estandar.
No importa ni modifica ningun otro modulo de LucIA.

Uso (cuando main.py lo cablee):
    from lucIA.Celebro.hipocampo import get_hipocampo
    h = get_hipocampo()
    h.registrar(pregunta, respuesta, modelo, emocion, delta, vector=None)
    h.consolidar(conversor)          # cada K turnos o en /descansa (replay)
    h.volcar_cierre(conversor)       # al cerrar sesion: todo a pesos
"""

from .hipocampo import Hipocampo, get_hipocampo

__all__ = ["Hipocampo", "get_hipocampo"]
