"""SLRN.optimizadores_legado - Registro liviano de _legado/SLRN (SL1-10, SL13-15).

Mismo criterio que RNP: se conserva el indice sin pandas ni simulaciones.
"""
REGISTRO_LEGADO_SLRN = {
    "SL1": ("BackpropagationOptimizer", "Backpropagation con momentum"),
    "SL2": ("Optimizador SLRN 2", "legado"),
    "SL3": ("Optimizador SLRN 3", "legado"),
    "SL4": ("Optimizador SLRN 4", "legado"),
    "SL5": ("Optimizador SLRN 5", "legado"),
    "SL6": ("Optimizador SLRN 6", "legado"),
    "SL7": ("Optimizador SLRN 7", "legado"),
    "SL8": ("Optimizador SLRN 8", "legado"),
    "SL9": ("Optimizador SLRN 9", "legado"),
    "SL10": ("Optimizador SLRN 10", "legado"),
    "SL13": ("Optimizador SLRN 13", "legado"),
    "SL14": ("Optimizador SLRN 14", "legado"),
    "SL15": ("Optimizador SLRN 15", "legado"),
}

def listar_legado():
    return dict(REGISTRO_LEGADO_SLRN)
