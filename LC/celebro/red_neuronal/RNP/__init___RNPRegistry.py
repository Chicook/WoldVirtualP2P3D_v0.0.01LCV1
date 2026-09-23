class RNPRegistry:
    """Registro central de todas las neuronas disponibles en el paquete RNP.

    Proporciona acceso perezoso a modulos, clases y funciones de cada RN.
    """

    _cache: Dict[str, _LazyModule] = {}

    @classmethod
    def _resolve_key(cls, key: str) -> str:
        key = key.lower().strip()
        return _ALIASES.get(key, key)

    @classmethod
    def _get_lazy(cls, key: str) -> _LazyModule:
        key = cls._resolve_key(key)
        if key not in _MODULE_MAP:
            available = cls.list_keys()
            raise KeyError(
                f"RNPRegistry: neurona {key!r} no encontrada. Disponibles: {available}"
            )
        if key not in cls._cache:
            module_name = _MODULE_MAP[key][0]
            cls._cache[key] = _LazyModule(module_name)
        return cls._cache[key]

    @classmethod
    def get_module(cls, key: str) -> Any:
        """Devuelve el modulo importado para el algoritmo indicado."""
        return cls._get_lazy(key)._load()

    @classmethod
    def get_class(cls, key: str) -> Type:
        """Devuelve la clase optimizadora principal del algoritmo indicado."""
        key = cls._resolve_key(key)
        class_name = _MODULE_MAP[key][1]
        return getattr(cls.get_module(key), class_name)

    @classmethod
    def get_factory_fn(cls, key: str) -> Callable:
        """Devuelve la funcion factory del algoritmo indicado."""
        key = cls._resolve_key(key)
        fn_name = _MODULE_MAP[key][2]
        return getattr(cls.get_module(key), fn_name)

    @classmethod
    def get_quick_fn(cls, key: str) -> Callable:
        """Devuelve la funcion quick_* del algoritmo indicado."""
        key = cls._resolve_key(key)
        fn_name = _MODULE_MAP[key][3]
        return getattr(cls.get_module(key), fn_name)

    @classmethod
    def list_keys(cls) -> List[str]:
        """Lista todos los alias canonicos disponibles."""
        return sorted(_MODULE_MAP.keys())

    @classmethod
    def list_all(cls) -> Dict[str, Dict[str, str]]:
        """Devuelve un diccionario con todos los metadatos registrados."""
        result: Dict[str, Dict[str, str]] = {}
        for key, (mod, cls_name, factory_fn, quick_fn) in _MODULE_MAP.items():
            result[key] = {
                "module":     mod,
                "class":      cls_name,
                "factory_fn": factory_fn,
                "quick_fn":   quick_fn,
            }
        return result

    @classmethod
    def probe(cls, key: str) -> Dict[str, Any]:
        """Comprueba si un modulo puede importarse y devuelve su estado."""
        k = cls._resolve_key(key)
        info: Dict[str, Any] = {"key": k, "available": False, "error": None}
        try:
            mod = cls.get_module(k)
            info["available"] = True
            info["module"]    = repr(mod)
        except Exception as exc:
            info["error"] = str(exc)
        return info

    @classmethod
    def probe_all(cls) -> Dict[str, Dict[str, Any]]:
        """Comprueba todos los modulos y devuelve un reporte de disponibilidad."""
        return {k: cls.probe(k) for k in cls.list_keys()}

# ===========================================================================
# Fabrica de optimizadores (RNPFactory)
# ===========================================================================
