"""Cerebelo/visemas.py - Tabla fonema espanol -> visema Godot."""
# 8 visemas basicos para lip-sync futuro (puertos 9876/9877).

VISEMAS = {
    "A": ("a", "á"),
    "E": ("e", "é"),
    "I": ("i", "í"),
    "O": ("o", "ó"),
    "U": ("u", "ú", "ü", "w"),
    "M": ("m", "b", "p"),
    "F": ("f", "v"),
    "L": ("l", "n", "d", "t", "s", "z", "c", "r", "y", "g", "j", "k", "x", "h", "q"),
    "RESTO": (),
}

_MS_POR_FONEMA = 90


def texto_a_visemas(texto: str) -> list:
    """Convierte texto en secuencia [{t_ms, visema}] para el avatar."""
    seq = []
    t = 0
    for ch in (texto or "").lower():
        if ch in " ,.;:!?¿¡\n\t":
            t += 60
            continue
        for vis, fons in VISEMAS.items():
            if ch in fons:
                seq.append({"t_ms": t, "visema": vis})
                t += _MS_POR_FONEMA
                break
    return seq
