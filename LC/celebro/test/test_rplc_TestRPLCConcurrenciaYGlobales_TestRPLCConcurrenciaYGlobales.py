class TestRPLCConcurrenciaYGlobales:
    """Pruebas de concurrencia multi-hilo y atajos globales."""

    def test_singleton_get_procesador_rplc(self) -> None:
        """Verifica que get_procesador_rplc devuelva la misma instancia singleton."""
        p1 = get_procesador_rplc()
        p2 = get_procesador_rplc()
        assert p1 is p2

    def test_reprocesar_con_metricas_global(self) -> None:
        """Comprueba la función de nivel de módulo reprocesar_con_metricas."""
        salida, met = reprocesar_con_metricas("Prueba de integración directa del subsistema")
        assert isinstance(salida, str)
        assert "repeticion_detectada" in met
        assert "items_en_cache" in met

    def test_reprocesar_para_lucia_shortcut(self) -> None:
        """Comprueba el atajo reprocesar_para_lucia()."""
        txt = reprocesar_para_lucia("Texto directo sin métricas")
        assert isinstance(txt, str)
        assert len(txt) > 0

    def test_obtener_estadisticas_rplc(self) -> None:
        """Verifica el consolidado de telemetría de RPLC."""
        stats = obtener_estadisticas_rplc()
        assert isinstance(stats, dict)
        assert "total" in stats
        assert "tiempo_medio_ms" in stats
        assert "memoria_cache_items" in stats

    def test_concurrencia_multi_hilo(self) -> None:
        """Verifica la thread-safety de RPLC ejecutando 10 hilos simultáneos."""
        proc = get_procesador_rplc()
        errores: List[Exception] = []

        def worker(idx: int) -> None:
            try:
                for _ in range(5):
                    proc.procesar(f"Mensaje de hilo concurrente #{idx} con datos técnicos")
            except Exception as e:
                errores.append(e)

        hilos = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for h in hilos:
            h.start()
        for h in hilos:
            h.join()

        assert len(errores) == 0, f"Ocurrieron excepciones concurrentes: {errores}"


# Fin de suite exhaustiva test_rplc.py
