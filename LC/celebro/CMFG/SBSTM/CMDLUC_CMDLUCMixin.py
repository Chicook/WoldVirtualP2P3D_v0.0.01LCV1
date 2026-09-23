class CMDLUCMixin:
    """Mezcla de consola: comandos, estado y cierre ordenado."""

    def mostrar_estado_sistema(self) -> None:
        """Despliega en terminal un reporte detallado del estado de todos los componentes."""
        valida, err = self.servidor_bks.validar_cadena()
        cadena_str = "VALIDA E INMUTABLE" if valida else f"ERROR: {err}"
        mod_act = self.cliente_iafree.gestor.obtener_modelo_activo()["id"] if self.cliente_iafree else "N/A"
        metricas = self.cliente_iafree.obtener_metricas_consumo() if self.cliente_iafree else {}
        costo_usd = float(metricas.get("costo_acumulado_usd", 0.0))

        datos = {
            "Cadena de Bloques": f"{len(self.servidor_bks.cadena)} bloques [{cadena_str}]",
            "Txs en Espera": str(len(self.servidor_bks.transacciones_pendientes)),
            "Neuronas Activas": f"{len(self.conversor_psn.neuronas)} (ENRN, RF_EN, RF_SL, RNP, SLRN)",
            "Deriva Muon": f"{self.conversor_psn.deriva_acumulada:.6f}",
            "Turnos de Sesion": str(self.turno_actual),
            "Modelo Activo": mod_act,
            "Costo Acumulado": f"${costo_usd:.2f} USD (100% Free)",
        }
        if EstiloTerminalLucIA:
            EstiloTerminalLucIA.renderizar_tarjeta_sistema("ESTADO INTEGRAL DE LUCIA (2026)", datos)
            print()
        else:
            for k, v in datos.items():
                print(f"  {k}: {v}")

    def listar_modelos_gratuitos(self) -> None:
        """Muestra los modelos disponibles sin costo del subsistema IAFREE."""
        if not self.cliente_iafree:
            print("Subsistema IAFREE no disponible.")
            return
        modelos = self.cliente_iafree.gestor.listar_modelos()
        print("\n\033[38;5;51m" + "=" * 68 + "\033[0m")
        print(f"  \033[1;37mCATALOGO DE MODELOS GRATUITOS IAFREE ({len(modelos)} disponibles):\033[0m")
        print("\033[38;5;51m" + "=" * 68 + "\033[0m")
        for m in modelos:
            pref = "\033[38;5;48m▶\033[0m " if m["id"] == self.cliente_iafree.gestor.obtener_modelo_activo()["id"] else "  "
            print(f"{pref}\033[38;5;51m{m['id']:<42}\033[0m | {m['contexto']} tok | {m['nombre']}")
        print("\033[38;5;51m" + "-" * 68 + "\033[0m\n")

    def ejecutar_bucle_interactivo(self) -> None:
        """Bucle principal de recepcion e interpretacion de comandos de consola."""
        while self.activa:
            try:
                entrada = input("\033[38;5;214mLucIA> Tu:\033[0m ").strip()
            except (KeyboardInterrupt, EOFError):
                print()
                break

            if not entrada:
                continue
            if entrada.lower() in ("salir", "exit", "quit"):
                print("\n\033[38;5;214mIniciando protocolo de salida y sincronizacion...\033[0m")
                break
            if entrada.lower() in ("estado", "status"):
                self.mostrar_estado_sistema()
                continue
            if entrada.lower() in ("ayuda", "help"):
                if EstiloTerminalLucIA:
                    panel_ayuda_comandos()
                continue
            if entrada.lower() in ("modelos", "models", "free"):
                self.listar_modelos_gratuitos()
                continue
            if entrada.lower() in ("minar", "mine"):
                blk = self.servidor_bks.minar_transacciones_pendientes()
                if blk:
                    if EstiloTerminalLucIA:
                        EstiloTerminalLucIA.renderizar_notificacion_bloque(blk.indice, blk.hash_bloque)
                    else:
                        print(f"Bloque #{blk.indice} minado: {blk.hash_bloque}")
                else:
                    print("  No hay transacciones pendientes para minar.")
                continue
            if entrada.lower().startswith("modelo "):
                nuevo_mod = entrada[7:].strip()
                if self.cliente_iafree and self.cliente_iafree.gestor.seleccionar_por_id(nuevo_mod):
                    print(f"  \033[38;5;48mModelo gratuito seleccionado: {nuevo_mod}\033[0m")
                else:
                    print(f"  \033[38;5;214mModelo '{nuevo_mod}' no encontrado en catalogo :free.\033[0m")
                continue
            if entrada.lower() in ("pin",):
                self._cmd_pin()
                continue
            if entrada.lower().startswith("restaurar "):
                self._cmd_restaurar(entrada[10:].strip())
                continue
            if entrada.lower() in ("benchmark",):
                self._cmd_benchmark()
                continue
            if entrada.lower() in ("exportar-pesos", "exportar_pesos"):
                self._cmd_exportar_pesos()
                continue
            if entrada.lower() in ("ia-local", "modelos-locales", "estado-local"):
                self._cmd_estado_ia_local()
                continue
            if entrada.lower().startswith("descargar-local"):
                partes = entrada.split()
                lim = int(partes[1]) if len(partes) > 1 and partes[1].isdigit() else 2
                self._cmd_descargar_ia_local(limite=lim)
                continue
            if entrada.lower() in ("constructor", "constructor-estado"):
                self._cmd_estado_constructor()
                continue
            if entrada.lower() in ("unificar", "unificar-constructor"):
                self._cmd_unificar_constructor()
                continue
            if entrada.lower() in ("refactorizar", "version-sesion", "version_sesion"):
                self._cmd_version_sesion()
                continue
            if entrada.lower() in ("integrar", "bitacora", "cierre-completo"):
                self._cmd_integrar()
                continue
            if entrada.lower() in ("refactor-neural", "hrctnr", "refactor-hrctnr"):
                self._cmd_hrctnr()
                continue

            self.procesar_turno_dialogo(entrada)

        self.activa = False

    def _cmd_pin(self) -> None:
        """Sube PSNRL a IPFS sin borrar sin confirmación."""
        try:
            from LC.celebro.CMFG.ipfs_manager import get_ipfs_manager
            res = get_ipfs_manager().subir_y_limpiar_psnrl(forzar_borrado_sin_daemon=False)
            print(f"  Pin: {len(res.get('cids', []))} CIDs | borrados={len(res.get('borrados', []))} | pendientes={len(res.get('pendiente_pin', []))}")
        except Exception as exc:
            print(f"  [pin] {exc}")

    def _cmd_restaurar(self, cid: str) -> None:
        if not cid:
            print("  Uso: restaurar <CID>")
            return
        try:
            from LC.celebro.CMFG.ipfs_manager import get_ipfs_manager
            data = get_ipfs_manager().recuperar_pesos(cid)
            n = len(data) if isinstance(data, (bytes, list)) else len(str(data))
            print(f"  Restaurado CID {cid[:24]}... ({n} B)")
        except Exception as exc:
            print(f"  [restaurar] {exc}")

    def _cmd_benchmark(self) -> None:
        if not self.cliente_iafree:
            print("  IAFREE no disponible.")
            return
        res = self.cliente_iafree.benchmark_rapido_modelos(max_modelos=3)
        for mid, lat in res.items():
            print(f"  {mid}: {lat} ms")

    def _cmd_exportar_pesos(self) -> None:
        if self.conversor_psn is None:
            print("  Conversor no inicializado.")
            return
        npz, js = self.conversor_psn.persistir_pesos_en_psnrl(etiqueta=f"export_{self.sesion_id}")
        print(f"  Exportado: {npz.name} + {js.name}")

    def _cmd_estado_ia_local(self) -> None:
        """Muestra perfil HW + modelos en LC/modelosIAlocal + recomendados."""
        if not _IALOCAL_DISPONIBLE or _estado_ia_local is None:
            print("  DSIALCLGRG no disponible.")
            return
        try:
            est = _estado_ia_local()
            perf = est.get("perfil", {})
            print("\n\033[38;5;51m" + "=" * 68 + "\033[0m")
            print("  \033[1;37mIA LOCAL DSIALCLGRG (LC/modelosIAlocal):\033[0m")
            print(f"  RAM: {perf.get('ram_total_gb')}GB | VRAM: {perf.get('vram_total_gb')}GB "
                  f"({perf.get('gpu')}) | Ollama: {est.get('ollama_online')}")
            print(f"  Recomendados: {', '.join(est.get('recomendados', []))}")
            for m in est.get("locales", []):
                marca = "\033[38;5;48mOK\033[0m" if m.get("descargado") else "\033[38;5;214m--\033[0m"
                print(f"  [{marca}] {m['id']} ({m.get('tamano_gb')}GB, {m.get('origen')})")
            print("\033[38;5;51m" + "-" * 68 + "\033[0m\n")
        except Exception as exc:
            print(f"  [ia-local] {exc}")

    def _cmd_descargar_ia_local(self, limite: int = 2) -> None:
        """Descarga autonoma de modelos ligeros (uso: descargar-local [N])."""
        print(f"  Descargando {limite} modelo(s) ligero(s) segun tu RAM/VRAM...")
        for rep in self.descargar_ia_local_autonomo(limite=limite):
            marca = "\033[38;5;48mOK\033[0m" if rep.get("exito") else "\033[38;5;203mFALLO\033[0m"
            print(f"  [{marca}] {rep.get('modelo', '?')}: {rep.get('mensaje')}")

    def _cmd_estado_constructor(self) -> None:
        """Muestra overlay activo, pendientes y si Constructor esta vacia."""
        if self.gestor_hrctrc is None:
            print("  HRCTRC no disponible.")
            return
        try:
            est = self.gestor_hrctrc.estado()
            print("\n\033[38;5;51m" + "=" * 68 + "\033[0m")
            print("  \033[1;37mCONSTRUCTOR HRCTRC (LC/Constructor):\033[0m")
            print(f"  Activo: {est['activa']} | Sesion: {est['sesion']} | "
                  f"Vacia: {est['constructor_vacia']}")
            for p in est.get("pendientes", [])[:10]:
                print(f"  [~] {p}")
            print("\033[38;5;51m" + "-" * 68 + "\033[0m\n")
        except Exception as exc:
            print(f"  [constructor] {exc}")

    def _cmd_unificar_constructor(self) -> None:
        """Unifica el overlay en rutas reales y vacia Constructor."""
        if self.gestor_hrctrc is None:
            print("  HRCTRC no disponible.")
            return
        rep = self.gestor_hrctrc.finalizar_sesion(aplicar=True)
        print(f"  {rep.get('mensaje')}")

    def _cmd_version_sesion(self) -> None:
        """Re-ejecuta la version de sesion 400/450 con barra de progreso."""
        if self.refactorizador is None:
            print("  Refactorizador no disponible.")
            return
        rep = self.refactorizador.ejecutar(mostrar_barra=True)
        print(f"  {rep.get('mensaje')}")

    def _cmd_integrar(self) -> None:
        """Pipeline manual: refactor -> .md -> pesos -> IPFS -> devopencode."""
        if self.integrador is None:
            print("  INTEGRACIONRF no disponible.")
            return
        rep = self.integrador.cierre_completo()
        print(f"  {self.integrador.resumen_cierre_txt(rep)}")

    def _cmd_hrctnr(self) -> None:
        """Monitor y refactorizador neural del sistema."""
        try:
            from LC.celebro.CMFG.SBSTM.HRCNTR import (
                ejecutar_hrctnr, confirmar_actualizacion_hrctnr,
                actualizar_sistema_hrctnr, estado_hrctnr,
            )
            print("  HRCNTR - Monitor y Refactorizador Neural")
            est = estado_hrctnr()
            print(f"  Archivos: {est['monitoreo']['archivos']}"
                  f" | Oversized: {est['monitoreo']['oversized']}"
                  f" | Generados: {est['generados']}")
            confirmacion = confirmar_actualizacion_hrctnr()
            print(confirmacion)
            entrada = input("  Confirmar [S/N]? ").strip().lower()
            if entrada in ("s", "si", "sí", "y", "yes"):
                rep = actualizar_sistema_hrctnr(confirmar=True)
                print(f"  {rep.get('mensaje')}")
            else:
                print("  Actualización cancelada.")
        except Exception as exc:
            print(f"  [hrctnr] {exc}")

    def cerrar_sistema(self) -> None:
        """Cierre ordenado: minado final, persistencia IPFS y purga de residuos (CHG/__pycache__)."""
        with self.lock:
            if not self.servidor_bks:
                return
            print("\n\033[38;5;51m" + "=" * 76 + "\033[0m")
            print("  \033[1;37mCONSOLIDANDO ESTADO COGNITIVO Y PERSISTENCIA FINAL...\033[0m")

            try:
                blk = self.servidor_bks.minar_transacciones_pendientes()
                if blk:
                    print(f"  Bloque final consolidado: \033[38;5;220m{blk.hash_bloque[:28]}...\033[0m")
            except Exception:
                pass

            # HRCNTR: confirmar actualizacion antes de INTEGRACIONRF
            try:
                from LC.celebro.CMFG.SBSTM.HRCNTR import (
                    get_gestor_hrctnr, confirmar_actualizacion_hrctnr,
                    actualizar_sistema_hrctnr,
                )
                g = get_gestor_hrctnr()
                mon = g.monitor_sistema()
                vivos = {"mainLCSTM.py", "__init__.py", "HRCTRC.py",
                         "HRCTRC_RFCT.py", "HRCNTR.py", "BASELUC.py",
                         "INTEGRACIONRF.py", "SNSBSTNPRB.py", "BKSVCB.py",
                         "ipfs_manager.py"}
                n_listos = len([e for e in mon["archivos"]
                                if e["sobreescrito"]
                                and e["ruta"] not in vivos])
                if n_listos > 0:
                    print("\n  HRCNTR - Confirmar actualizacion:")
                    print(confirmar_actualizacion_hrctnr())
                    entrada = input("  Confirmar [S/N]? ").strip().lower()
                    if entrada in ("s", "si", "sí", "y", "yes"):
                        rep_u = actualizar_sistema_hrctnr(confirmar=True)
                        print(f"  \033[38;5;48m{rep_u.get('mensaje')}\033[0m")
            except Exception as e_hc:
                print(f"  \033[38;5;214mHRCNTR cierre: {e_hc}\033[0m")

            # INTEGRACIONRF: refactor -> .md -> pesos -> IPFS -> devopencode
            # (incluye la unificacion HRCTRC; si no esta, fallback directo)
            try:
                if self.integrador is not None:
                    rep_i = self.integrador.cierre_completo()
                    print(f"  \033[38;5;48m{self.integrador.resumen_cierre_txt(rep_i)}\033[0m")
                elif self.gestor_hrctrc is not None and self.gestor_hrctrc.esta_activa():
                    rep_h = self.gestor_hrctrc.finalizar_sesion(aplicar=True)
                    print(f"  \033[38;5;48mConstructor unificado: {rep_h.get('aplicados', 0)} archivo(s) "
                          f"en su ruta real; Constructor vacia.\033[0m")
            except Exception as e_hrc:
                print(f"  \033[38;5;214mIntegracionRF: {e_hrc}\033[0m")

            try:
                self.servidor_bks.cerrar_sesion_y_subir_ipfs()
                print("  \033[38;5;48mSincronizacion IPFS de bloques y pesos completada exitosamente.\033[0m")
            except Exception as e_close:
                print(f"  \033[38;5;214mCierre IPFS: {e_close}\033[0m")

            try:
                from LC.celebro.CMFG.SBSTM.PURGADOR import (
                    recolectar_pycache_en_chg,
                    solo_limpiar_pycache,
                )
                recolectar_pycache_en_chg()  # junta lo último en CHG…
                r = solo_limpiar_pycache()   # …y lo vacía todo
                print(f"  \033[38;5;48mPurga residuos: __pycache__={r.pycache_eliminados} chg={r.chg_eliminados} "
                      f"pytest={r.pytest_cache_eliminados} logs={r.logs_tmp_eliminados} "
                      f"({r.bytes_liberados / 1024:.1f} KB)\033[0m")
            except Exception as e_purga:
                print(f"  \033[38;5;214mPurga: {e_purga}\033[0m")

            print("\033[38;5;51m" + "=" * 76 + "\033[0m\n")
            self.servidor_bks = None
