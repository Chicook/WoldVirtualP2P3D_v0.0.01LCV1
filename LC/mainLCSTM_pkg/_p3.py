"""
mainLCSTM - parte 3/4 (version de sesion LucIA).
Este código Python realiza tareas relacionadas con la red y la red social en línea. A continuación, te proporciono una breve explicación y u
"""
from __future__ import annotations

"""
mainLCSTM.py - Orquestador Central del Sistema Cognitivo LucIA (WoldVirtualP2P3D 2026)
=====================================================================================
Coordinacion maestro: Blockchain BKSVCB, Sesion P2P, IAFREE ($0.00), STYLOS y RPLC.
"""
from __future__ import annotations

import atexit
import json
import logging
import os
import signal
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Final, List, Optional, Tuple, Union

# ─── JERARQUIA DE DIRECTORIOS Y CONFIGURACION DE ENTORNO ─────────────────────

            if entrada.lower() in ("refactorizar", "version-sesion", "version_sesion"):
                self._cmd_version_sesion()
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

            # HRCTRC: unificar overlay en rutas reales y dejar Constructor vacia
            try:
                if self.gestor_hrctrc is not None and self.gestor_hrctrc.esta_activa():
                    rep_h = self.gestor_hrctrc.finalizar_sesion(aplicar=True)
                    print(f"  \033[38;5;48mConstructor unificado: {rep_h.get('aplicados', 0)} archivo(s) "
                          f"en su ruta real; Constructor vacia.\033[0m")
            except Exception as e_hrc:
                print(f"  \033[38;5;214mConstructor: {e_hrc}\033[0m")

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


# ─── GESTOR DE CONTEXTO PARA INTEGRACIONES EXTERNAS ─────────────────────────
