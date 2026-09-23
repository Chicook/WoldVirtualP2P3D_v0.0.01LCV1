class PurgaFilesystem:
from typing import Final
    """Elimina __pycache__, .pyc/.pyo/.pyd, temporales IPFS y contenido de CHG."""

    _PYC_EXT:    Final[Tuple[str, ...]] = (".pyc", ".pyo", ".pyd")
    _TMP_PREFIX: Final[str]             = "_tmp_"
    _EXCLUIR_DEFAULT: Final[Tuple[str, ...]] = (".git", "node_modules", ".venv", "venv", "env")

    def __init__(
        self,
        raiz: Optional[Path] = None,
        chg_dir: Optional[Path] = None,
        excluir: Optional[List[str]] = None,
    ) -> None:
        self._raiz    = raiz    or ROOT_DIR
        self._chg_dir = chg_dir or CHG_DIR
        self._excluir = tuple(excluir) if excluir else self._EXCLUIR_DEFAULT

    def _excluida(self, ruta: Path) -> bool:
        return any(e in ruta.parts for e in self._excluir)

    def _tamaño(self, ruta: Path, es_dir: bool) -> int:
        if not es_dir:
            try:
                return ruta.stat().st_size
            except Exception:
                return 0
        total = 0
        try:
            for f in ruta.rglob("*"):
                if f.is_file():
                    total += f.stat().st_size
        except Exception:
            pass
        return total

    def _borrar(self, ruta: Path, res: ResultadoPurga, es_dir: bool = False) -> bool:
        try:
            tamaño = self._tamaño(ruta, es_dir)
            shutil.rmtree(ruta, ignore_errors=False) if es_dir else ruta.unlink()
            res.bytes_liberados += tamaño
            res.rutas_eliminadas.append(str(ruta))
            return True
        except FileNotFoundError:
            return False
        except Exception as exc:
            msg = f"{'[Dir]' if es_dir else '[File]'} {ruta}: {exc}"
            res.errores.append(msg)
            logger.warning(msg)
            return False

    def limpiar_pycache(self, res: ResultadoPurga) -> None:
        for d in self._raiz.rglob("__pycache__"):
            if d.is_dir() and not self._excluida(d):
                if self._borrar(d, res, es_dir=True):
                    res.pycache_eliminados += 1
                    logger.info("  🗑 __pycache__: %s", d.relative_to(self._raiz))

    def limpiar_pyc(self, res: ResultadoPurga) -> None:
        for ext in self._PYC_EXT:
            for f in self._raiz.rglob(f"*{ext}"):
                if f.is_file() and not self._excluida(f) and "__pycache__" not in f.parts:
                    if self._borrar(f, res):
                        res.pyc_eliminados += 1

    def limpiar_tmp_ipfs(self, res: ResultadoPurga) -> None:
        for f in self._raiz.rglob(f"{self._TMP_PREFIX}*.bin"):
            if f.is_file() and not self._excluida(f):
                if self._borrar(f, res):
                    res.tmp_ipfs_eliminados += 1
                    logger.info("  🗑 tmp IPFS: %s", f.relative_to(self._raiz))

    def limpiar_chg(self, res: ResultadoPurga) -> None:
        if not self._chg_dir.exists():
            return
        for item in self._chg_dir.iterdir():
            if item.name in (".gitkeep", "README.md"):
                continue
            if not self._excluida(item):
                es_dir = item.is_dir()
                if self._borrar(item, res, es_dir=es_dir):
                    res.chg_eliminados += 1
                    logger.info("  🗑 CHG: %s", item.name)

    def limpiar_pytest_cache(self, res: ResultadoPurga) -> None:
        for d in self._raiz.rglob(".pytest_cache"):
            if d.is_dir() and not self._excluida(d):
                if self._borrar(d, res, es_dir=True):
                    res.pytest_cache_eliminados += 1
                    logger.info("  🗑 pytest_cache: %s", d.relative_to(self._raiz))

    def limpiar_logs_tmp(self, res: ResultadoPurga) -> None:
        for pat in ("*.log", "*.tmp", "*.bak", "*.orig", "*.rej"):
            for f in self._raiz.rglob(pat):
                if f.is_file() and not self._excluida(f) and ".git" not in f.parts:
                    if self._borrar(f, res):
                        res.logs_tmp_eliminados += 1

    def limpiar_bak_ledger(self, res: ResultadoPurga) -> None:
        """Limpia .json.tmp huérfanos (los .bak del ledger se conservan)."""
        for f in self._raiz.rglob("*.json.tmp"):
            if f.is_file() and not self._excluida(f):
                if self._borrar(f, res):
                    res.bak_tmp_eliminados += 1

    def ejecutar(self) -> ResultadoPurga:
        res = ResultadoPurga()
        t0 = time.perf_counter()
        logger.info("PurgaFilesystem: raíz=%s", self._raiz)
        self.limpiar_pycache(res)
        self.limpiar_pyc(res)
        self.limpiar_tmp_ipfs(res)
        self.limpiar_chg(res)
        self.limpiar_pytest_cache(res)
        self.limpiar_logs_tmp(res)
        self.limpiar_bak_ledger(res)
        res.duracion_seg = time.perf_counter() - t0
        logger.info(
            "Purga OK: pycache=%d | pyc=%d | tmp=%d | chg=%d | pytest=%d | logs=%d | bak=%d | %.2f KB | %.2fs",
            res.pycache_eliminados, res.pyc_eliminados, res.tmp_ipfs_eliminados,
            res.chg_eliminados, res.pytest_cache_eliminados, res.logs_tmp_eliminados,
            res.bak_tmp_eliminados, res.bytes_liberados / 1024, res.duracion_seg,
        )
        return res

# ── ORQUESTADOR ──────────────────────────────────────────────────────────────
