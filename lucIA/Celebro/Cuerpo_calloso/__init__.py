"""Cuerpo_calloso - Puente de coherencia entre hemisferio local y cloud.

Rol biologico: sincroniza informacion entre los hemisferios cerebrales
izquierdo y derecho. En LucIA es el arbitro que decide como combinar
la respuesta del modelo local con la del modelo cloud, evalua la
coherencia semantica entre ambas y puede elegir, mezclar o rechazar
segun similitud y calidad.
"""
import re
import math
from lucIA.Celebro.bases_neuro import ModuloCerebral


def _tokenizar(texto: str) -> set:
    """Tokens significativos de un texto (>= 4 letras, sin stopwords)."""
    STOP = frozenset({"este", "esta", "esto", "esos", "esas", "donde",
                      "cuando", "porque", "para", "como", "pero", "tiene",
                      "tengo", "muy", "mas", "una", "unos", "unas"})
    return {
        w.lower() for w in re.findall(r"[a-zA-Záéíóúñü]{4,}", texto or "")
        if w.lower() not in STOP
    }


def _similitud_jaccard(a: str, b: str) -> float:
    """Similitud Jaccard de tokens entre dos textos (0.0 - 1.0)."""
    ta, tb = _tokenizar(a), _tokenizar(b)
    if not ta or not tb:
        return 0.0
    intersec = ta & tb
    union = ta | tb
    return round(len(intersec) / len(union), 3)


class CuerpoCalloso(ModuloCerebral):
    nombre = "CuerpoCalloso"

    def __init__(self):
        super().__init__()
        self.ultima_integracion: str = ""
        self._ultima_similitud: float = 0.0
        self._via_elegida: str = "ninguna"
        self._integraciones: int = 0

    # -- Integracion principal -----------------------------------------

    def integrar(self, via_local: str, via_cloud: str,
                 umbral_baja: float = 0.15,
                 umbral_alta: float = 0.55) -> dict:
        """Arbitra y fusiona las dos vias segun similitud semantica.

        Estrategias:
          - similitud >= umbral_alta  -> mezcla simetrica (complementan)
          - similitud < umbral_baja   -> elige la mas larga (mas completa)
          - entre umbrales            -> prioriza local + fragmento cloud
        """
        local = (via_local or "").strip()
        cloud = (via_cloud or "").strip()

        if not local and not cloud:
            self._via_elegida = "ninguna"
            self.ultima_integracion = ""
            return self._resultado()

        if not cloud:
            self._via_elegida = "local"
            self.ultima_integracion = local
            return self._resultado()

        if not local:
            self._via_elegida = "cloud"
            self.ultima_integracion = cloud
            return self._resultado()

        sim = _similitud_jaccard(local, cloud)
        self._ultima_similitud = sim
        self._integraciones += 1

        if sim >= umbral_alta:
            # Vias muy similares: mezcla simetrica (primera mitad local + segunda cloud)
            mid_l = len(local) // 2
            mid_c = len(cloud) // 2
            self.ultima_integracion = local[:mid_l].rstrip() + " " + cloud[mid_c:].lstrip()
            self._via_elegida = "fusion"
        elif sim < umbral_baja:
            # Muy distintas: tomar la mas larga (suele ser la mas rica)
            if len(cloud) > len(local) * 1.2:
                self.ultima_integracion = cloud
                self._via_elegida = "cloud"
            else:
                self.ultima_integracion = local
                self._via_elegida = "local"
        else:
            # Complementarias: local completo + complemento cloud (sin solapamiento)
            complemento = self._extraer_complemento(local, cloud)
            self.ultima_integracion = (
                local.rstrip(".,; ") + (". " + complemento if complemento else "")
            )
            self._via_elegida = "local+cloud"

        # Limitar longitud maxima (evitar respuestas excesivamente largas)
        if len(self.ultima_integracion) > 3000:
            self.ultima_integracion = self.ultima_integracion[:3000].rstrip() + "..."

        self._notar(
            f"sim={sim} via={self._via_elegida} "
            f"chars={len(self.ultima_integracion)}"
        )
        return self._resultado()

    def _extraer_complemento(self, base: str, extra: str,
                              max_chars: int = 400) -> str:
        """Extrae del extra las frases que no esten ya en base."""
        tokens_base = _tokenizar(base)
        partes = re.split(r"(?<=[.!?])\s+", extra)
        complemento = []
        for frase in partes:
            tokens_frase = _tokenizar(frase)
            if not tokens_frase:
                continue
            solapamiento = len(tokens_frase & tokens_base) / len(tokens_frase)
            if solapamiento < 0.5:   # menos de la mitad solapada = informacion nueva
                complemento.append(frase.strip())
            if sum(len(f) for f in complemento) >= max_chars:
                break
        return " ".join(complemento)

    def _resultado(self) -> dict:
        return {
            "texto": self.ultima_integracion,
            "similitud": self._ultima_similitud,
            "via": self._via_elegida,
            "chars": len(self.ultima_integracion),
        }

    # -- Estado y pesos -----------------------------------------------

    def a_pesos(self, conversor) -> None:
        """Exporta estadisticas de integracion a pesos neuronales."""
        try:
            resumen = (
                f"integraciones={self._integraciones} "
                f"ultima_via={self._via_elegida} "
                f"ultima_similitud={self._ultima_similitud}"
            )
            conversor.actualizar_memoria_salida(
                "[Calloso] arbitraje local-cloud", resumen, "CuerpoCalloso"
            )
        except Exception:
            pass

    def resumen(self) -> str:
        return (
            f"CuerpoCalloso: via={self._via_elegida} "
            f"sim={self._ultima_similitud} "
            f"integraciones={self._integraciones} | {super().resumen()}"
        )


_calloso_global = None


def get_calloso() -> CuerpoCalloso:
    global _calloso_global
    if _calloso_global is None:
        _calloso_global = CuerpoCalloso()
    return _calloso_global
