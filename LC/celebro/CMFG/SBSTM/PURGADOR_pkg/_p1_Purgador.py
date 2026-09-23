class Purgador:
from typing import Final
    """
    Orquesta las dos fases:
      1. CustodiaIPFS  → sube PSNRL a IPFS
      2. PurgaFilesystem → elimina __pycache__ y residuos Python
    Si la custodia falla y no se fuerza, la purga se cancela para proteger los datos.
    """

    _INFORME: Final[str] = "purgador_informe.json"

    def __init__(
        self,
        raiz: Optional[Path] = None,
        omitir_ipfs: bool = False,
        forzar: bool = False,
        guardar_informe: bool = True,
        excluir: Optional[List[str]] = None,
    ) -> None:
        self._raiz          = raiz or ROOT_DIR
        self._omitir_ipfs   = omitir_ipfs
        self._forzar        = forzar
        self._guardar       = guardar_informe
        self._custodia      = CustodiaIPFS()
        self._purga_fs      = PurgaFilesystem(raiz=self._raiz, excluir=excluir)

    def _guardar_informe(self, informe: InformePurgador) -> None:
        ruta = self._raiz / self._INFORME
        try:
            with open(ruta, "w", encoding="utf-8") as fh:
                json.dump(informe.como_dict(), fh, indent=2, ensure_ascii=False)
            logger.info("Informe guardado: %s", ruta)
        except Exception as exc:
            logger.error("No se pudo guardar el informe: %s", exc)

    def ejecutar(self) -> InformePurgador:
        t0 = time.perf_counter()
        inf = InformePurgador(
            timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            raiz_escaneada=str(self._raiz),
            fase1_omitida=self._omitir_ipfs,
            purga_forzada=self._forzar,
        )
        # Fase 1
        if not self._omitir_ipfs:
            logger.info("═══ FASE 1: Custodia IPFS ═══")
            inf.custodia = self._custodia.ejecutar(forzar_borrado=self._forzar)
            if not inf.custodia.exito and not self._forzar:
                logger.warning("Custodia incompleta. Purga cancelada (usa forzar=True).")
                inf.duracion_total_seg = time.perf_counter() - t0
                if self._guardar:
                    self._guardar_informe(inf)
                return inf
        # Fase 2
        logger.info("═══ FASE 2: Purga Filesystem ═══")
        inf.purga = self._purga_fs.ejecutar()
        inf.duracion_total_seg = time.perf_counter() - t0
        if self._guardar:
            self._guardar_informe(inf)
        return inf

# ── API PÚBLICA ──────────────────────────────────────────────────────────────
