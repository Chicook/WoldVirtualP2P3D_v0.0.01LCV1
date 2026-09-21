"""Corteza_prefrontal/offline.py - Respuestas sin modelos externos."""
import json
import re
from pathlib import Path

_CORPUS = Path(__file__).parent / "corpus_offline.json"
_CACHE = {}
STOP = {"que", "como", "con", "para", "este", "esta", "esto", "eso",
        "esos", "esas", "del", "las", "los", "una", "unos", "unas",
        "este", "donde", "cuando", "porque", "pues", "alla"}


def _cargar():
    if "qs" not in _CACHE:
        try:
            _CACHE["qs"] = json.loads(_CORPUS.read_text(encoding="utf-8"))
        except Exception:
            _CACHE["qs"] = []
        df = {}
        for item in _CACHE["qs"]:
            for c in set(item.get("claves", [])):
                df[c] = df.get(c, 0) + 1
        _CACHE["df"] = df
    return _CACHE["qs"]


def buscar(pregunta: str, minimo: float = 0.2) -> dict:
    """Devuelve la mejor respuesta offline {texto, score} o {}.
    Pondera por rareza (IDF): 'miedo' vale mas que 'hoy'."""
    toks = set(re.findall(r"[a-záéíóúñ]{3,}", (pregunta or "").lower())) - STOP
    if not toks:
        return {}
    items = _cargar()
    df = _CACHE.get("df", {})
    mejor, mejor_k = None, (-1, -1.0)
    for item in items:
        claves = set(item.get("claves", []))
        hit = toks & claves
        if not hit:
            continue
        n = len(hit)
        s = sum(1.0 / df.get(c, 1) for c in hit)
        if (n, s) > mejor_k:
            mejor, mejor_k = item, (n, s)
    if mejor and mejor_k[1] >= minimo:
        return {"texto": mejor["respuesta"], "score": round(mejor_k[1], 3),
                "tema": mejor.get("tema", "")}
    return {}
