class CustodiaIPFS:
    """Sube todos los archivos de PSNRL a IPFS antes de cualquier limpieza."""

    def __init__(self, psnrl_dir: Optional[Path] = None) -> None:
        self._psnrl_dir = psnrl_dir or PSNRL_DIR
        self._psnrl_dir.mkdir(parents=True, exist_ok=True)
        self._mgr = None

    def _gestor(self):
        """Importación diferida de IPFSManager para evitar dependencias circulares."""
        if self._mgr is None:
            if str(ROOT_DIR) not in sys.path:
                sys.path.insert(0, str(ROOT_DIR))
            try:
                from LC.celebro.CMFG.ipfs_manager import get_ipfs_manager
                self._mgr = get_ipfs_manager()
            except Exception as exc:
                raise RuntimeError(f"IPFSManager no disponible: {exc}") from exc
        return self._mgr

    @staticmethod
    def _sha256(ruta: Path) -> str:
        h = hashlib.sha256()
        with open(ruta, "rb") as fh:
            for bloque in iter(lambda: fh.read(65536), b""):
                h.update(bloque)
        return h.hexdigest()

    def ejecutar(self, forzar_borrado: bool = False) -> ResultadoCustodia:
        """Sube cada archivo de PSNRL a IPFS y registra CID + sha256."""
        res = ResultadoCustodia()
        t0 = time.perf_counter()
        try: mgr = self._gestor()
        except RuntimeError as exc:
            res.errores.append(str(exc)); res.duracion_seg = time.perf_counter() - t0; return res

        archivos = sorted(
            f for f in self._psnrl_dir.iterdir()
            if f.is_file() and not f.name.startswith(".")
        )
        if not archivos:
            logger.info("PSNRL vacío — sin pesos que subir.")
            res.duracion_seg = time.perf_counter() - t0
            return res

        estado = mgr.estado_conexion()
        res.daemon_real = estado.get("daemon_activo", False)
        res.cli_real    = estado.get("cli_disponible", False)
        borrar_ok = res.daemon_real or res.cli_real or forzar_borrado

        logger.info("CustodiaIPFS: %d arch | daemon=%s | cli=%s", len(archivos), res.daemon_real, res.cli_real)
        for arch in archivos:
            sha = self._sha256(arch)
            try:
                r = mgr.almacenar_pesos(
                    origen=arch, nombre_modelo=arch.stem, eliminar_local=borrar_ok,
                    metadatos={"sha256_original": sha, "purgador": __version__,
                               "ts_purga": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                )
                res.archivos_procesados += 1
                res.archivos_pinados += 1
                res.cids_obtenidos.append(r["cid"])
                res.bytes_subidos += r.get("tamano_bytes", arch.stat().st_size)
                if r.get("borrado_local"):
                    res.borrados_locales.append(arch.name)
                logger.info("  ✓ %s → %s | %s", arch.name, r["cid"][:20], r["nodo"])
            except Exception as exc:
                res.archivos_fallidos += 1
                res.errores.append(f"{arch.name}: {exc}")
                logger.error("  ✗ %s: %s", arch.name, exc)

        res.duracion_seg = time.perf_counter() - t0
        logger.info("CustodiaIPFS: %d/%d pinados | %d err | %.2fs",
                    res.archivos_pinados, res.archivos_procesados, res.archivos_fallidos, res.duracion_seg)
        return res

# ── FASE 2 — PURGA DEL FILESYSTEM ────────────────────────────────────────────
