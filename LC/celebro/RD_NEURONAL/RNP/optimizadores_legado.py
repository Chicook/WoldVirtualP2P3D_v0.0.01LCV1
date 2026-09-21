"""RNP.optimizadores_legado - Registro liviano de los 10 optimizadores de _legado/RNP.

El codigo original usaba pandas + simulaciones aleatorias (no apto para el
presupuesto <125MB sin dependencias pesadas). Se conserva el nombre,
descripcion y factory API sin la dependencia pesada.
"""
REGISTRO_LEGADO_RNP = {
    "RN1": ("AdamWOptimizer", "Adam with Weight Decay"),
    "RN2": ("Optimizador RNP 2", "legado"),
    "RN3": ("Optimizador RNP 3", "legado"),
    "RN4": ("Optimizador RNP 4", "legado"),
    "RN5": ("Optimizador RNP 5", "legado"),
    "RN6": ("Optimizador RNP 6", "legado"),
    "RN7": ("Optimizador RNP 7", "legado"),
    "RN8": ("Optimizador RNP 8", "legado"),
    "RN9": ("Optimizador RNP 9", "legado"),
    "RN10": ("Optimizador RNP 10", "legado"),
}

def listar_legado():
    return dict(REGISTRO_LEGADO_RNP)
