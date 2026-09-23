class RNPBenchmark:
    """Sistema de benchmarking para comparar todos los algoritmos del paquete RNP.

    Permite evaluar y comparar el rendimiento de todas las neuronas disponibles
    con un modelo de prueba generico o con un modelo proporcionado por el usuario.
    """

    @staticmethod
    def _default_model(layers: int = 3, size: int = 4) -> List[Any]:
        """Genera un modelo sintetico de prueba (lista de matrices numpy)."""
        try:
            import numpy as np  # noqa: PLC0415
            return [np.random.randn(size, size).astype(np.float64) for _ in range(layers)]
        except ImportError:
            return [[[0.0] * size for _ in range(size)] for _ in range(layers)]

    @staticmethod
    def run_single(
        key: str,
        model: Optional[Any] = None,
        config: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Ejecuta el benchmark para un algoritmo individual.

        Returns:
            Diccionario con: key, success, elapsed_s, error, result.
        """
        if model is None:
            model = RNPBenchmark._default_model()
        entry: Dict[str, Any] = {
            "key": key, "success": False,
            "elapsed_s": 0.0, "error": None, "result": None,
        }
        t0 = time.perf_counter()
        try:
            result = RNPFactory.quick_run(key, model=model, config=config)
            entry["success"] = (
                result.get("success", False)
                if isinstance(result, dict) else bool(result)
            )
            entry["result"] = result
        except Exception as exc:
            entry["error"] = traceback.format_exc(limit=4)
            logger.warning("RNPBenchmark[%s] error: %s", key, exc)
        finally:
            entry["elapsed_s"] = round(time.perf_counter() - t0, 6)
        return entry

    @staticmethod
    def run_all(
        model: Optional[Any] = None,
        keys: Optional[List[str]] = None,
        config: Optional[Any] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Ejecuta el benchmark para todos los algoritmos (o la lista indicada).

        Args:
            model:   Modelo de prueba; se genera uno sintetico si es None.
            keys:    Lista de algoritmos a evaluar; todos si es None.
            config:  Configuracion compartida opcional.
            verbose: Si es True, imprime un resumen por consola.

        Returns:
            Diccionario con claves summary, results, rankings.
        """
        if model is None:
            model = RNPBenchmark._default_model()
        targets = keys if keys is not None else RNPRegistry.list_keys()
        results: Dict[str, Dict[str, Any]] = {}
        for k in targets:
            results[k] = RNPBenchmark.run_single(k, model=model, config=config)
        successful = [k for k, v in results.items() if v["success"]]
        failed     = [k for k, v in results.items() if not v["success"]]
        rankings   = sorted(successful, key=lambda kk: results[kk]["elapsed_s"])
        summary: Dict[str, Any] = {
            "total":      len(targets),
            "successful": len(successful),
            "failed":     len(failed),
            "fastest":    rankings[0] if rankings else None,
            "rankings":   rankings,
        }
        if verbose:
            sep = "=" * 60
            print(f"\n{sep}")
            print(f"  RNP Benchmark v{__version__} (API {__api_level__})")
            print(sep)
            for k in targets:
                r = results[k]
                tick = "OK " if r["success"] else "ERR"
                err_hint = ("  // " + str(r["error"])[:50]) if r["error"] else ""
                print(f"  [{tick}] [{k:12s}] {r['elapsed_s']:.4f}s{err_hint}")
            print(
                f"\n  OK: {len(successful)}/{len(targets)} "
                f"| Mas rapido: {summary['fastest']}"
            )
            print(f"{sep}\n")
        return {"summary": summary, "results": results, "rankings": rankings}

    @staticmethod
    def compare(
        keys: List[str],
        model: Optional[Any] = None,
        config: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Compara un subconjunto de algoritmos y devuelve el ranking."""
        return RNPBenchmark.run_all(model=model, keys=keys, config=config)

# ===========================================================================
# Utilidades del paquete
# ===========================================================================
