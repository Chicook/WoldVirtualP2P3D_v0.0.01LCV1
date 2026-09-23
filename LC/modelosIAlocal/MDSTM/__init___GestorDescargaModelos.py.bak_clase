class GestorDescargaModelos:
    """Clase principal: LucIA descarga modelos ligeros de forma autonoma."""

    def __init__(self) -> None:
        MODELOS_DIR.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._hilo_fondo: Optional[threading.Thread] = None
        self.perfil: Dict[str, Any] = self.perfilar_hardware()

    # ── Hardware ──────────────────────────────────────────────
    def perfilar_hardware(self, guardar: bool = True) -> Dict[str, Any]:
        ram_total, ram_libre = _ram_gb()
        vram_total, gpu = _vram_gb()
        try:
            disco = round(shutil.disk_usage(str(MODELOS_DIR)).free / 1e9, 2)
        except Exception:
            disco = 0.0
        perfil = {"timestamp": time.time(), "ram_total_gb": ram_total,
                  "ram_libre_gb": ram_libre, "vram_total_gb": vram_total,
                  "gpu": gpu, "disco_libre_gb": disco,
                  "cpu_nucleos": os.cpu_count() or 4, "sistema": platform.system(),
                  "ollama_host": OLLAMA_HOST, "mdstm_version": __version__}
        if guardar:
            try:
                PERFIL_HW.write_text(json.dumps(perfil, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception as exc:
                logger.warning("MDSTM no pudo guardar perfil: %s", exc)
        self.perfil = perfil
        return perfil

    def capacidad_real(self) -> str:
        """Frase factual que demuestra a LucIA que SI puede descargar."""
        p = self.perfil
        return (f"Puedo descargar modelos YA: dispongo de {p['ram_total_gb']}GB RAM, "
                f"{p['vram_total_gb']}GB VRAM ({p['gpu']}), {p['disco_libre_gb']}GB libres "
                f"en {MODELOS_DIR} y modulo MDSTM v{__version__} operativo.")

    # ── Catalogo ──────────────────────────────────────────────
    def recomendar(self, perfil: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        p = perfil or self.perfil
        ram, vram = float(p.get("ram_total_gb", 8.0)), float(p.get("vram_total_gb", 0.0))
        disco = float(p.get("disco_libre_gb", 10.0))
        aptos = [m for m in CATALOGO if m["ram"] <= ram and m["vram"] <= vram + 0.01 and m["gb"] <= disco]
        return sorted(aptos or [CATALOGO[0]], key=lambda m: m["gb"])

    def buscar(self, texto: str) -> Optional[Dict[str, Any]]:
        t = texto.lower().strip()
        for m in CATALOGO:
            if m["id"] in t or m["tag"] in t:
                return m
        corto = re.sub(r"[^a-z0-9.+ ]", " ", t)
        for m in CATALOGO:
            base = m["tag"].split(":")[0]
            if base in corto:
                return m
        return None

    # ── Registro ──────────────────────────────────────────────
    def _leer_registro(self) -> Dict[str, Any]:
        if REGISTRO.exists():
            try:
                data = json.loads(REGISTRO.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
        return {"descargas": {}}

    def _anotar(self, modelo_id: str, origen: str, exito: bool) -> None:
        with self._lock:
            reg = self._leer_registro()
            reg.setdefault("descargas", {})[modelo_id] = {
                "origen": origen, "exito": exito, "fecha": time.time()}
            try:
                REGISTRO.write_text(json.dumps(reg, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass

    def listar_descargados(self) -> List[Dict[str, Any]]:
        reg = self._leer_registro().get("descargas", {})
        out: List[Dict[str, Any]] = []
        for mid, meta in reg.items():
            out.append({"id": mid, "origen": meta.get("origen", "?"),
                        "exito": meta.get("exito", False), "fecha": meta.get("fecha", 0.0)})
        for f in sorted(MODELOS_DIR.glob("*.gguf")):
            if not any(o["id"] == f.stem for o in out):
                out.append({"id": f.stem, "origen": "gguf-manual",
                            "exito": True, "fecha": f.stat().st_mtime})
        return out

    # ── Motores de descarga ───────────────────────────────────
    def ollama_online(self) -> bool:
        try:
            req = urllib.request.Request(f"{OLLAMA_HOST.rstrip('/')}/api/tags",
                                         headers={"User-Agent": "LucIA-MDSTM/2026"})
            with urllib.request.urlopen(req, timeout=3.0):
                return True
        except Exception:
            return False

    def _pull_ollama(self, tag: str) -> bool:
        try:
            if shutil.which("ollama"):
                r = subprocess.run(["ollama", "pull", tag], capture_output=True,
                                   text=True, encoding="utf-8", errors="replace", timeout=900)
                return r.returncode == 0
            payload = json.dumps({"name": tag}).encode("utf-8")
            req = urllib.request.Request(f"{OLLAMA_HOST.rstrip('/')}/api/pull", data=payload,
                                         headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=900):
                return True
        except Exception as exc:
            logger.warning("MDSTM pull %s fallo: %s", tag, exc)
            return False

    def _bajar_gguf(self, url: str, destino: Path) -> bool:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "LucIA-MDSTM/2026"})
            with urllib.request.urlopen(req, timeout=120.0) as resp, open(destino, "wb") as fh:
                shutil.copyfileobj(resp, fh)
            return destino.exists() and destino.stat().st_size > 1024
        except Exception as exc:
            logger.warning("MDSTM GGUF fallo: %s", exc)
            try:
                destino.unlink(missing_ok=True)
            except Exception:
                pass
            return False

    def descargar(self, modelo_id: str, via: str = "auto") -> Dict[str, Any]:
        """Descarga REAL del modelo en LC/modelosIAlocal. Retorna reporte factual."""
        meta = next((m for m in CATALOGO if m["id"] == modelo_id), None) or self.buscar(modelo_id)
        if meta is None:
            return {"exito": False, "modelo": modelo_id,
                    "mensaje": f"No conozco '{modelo_id}'. Opciones: {', '.join(m['id'] for m in CATALOGO)}."}
        t0, ok, metodo = time.time(), False, ""
        if via in ("auto", "ollama") and self._pull_ollama(meta["tag"]):
            ok, metodo = True, "ollama"
        if not ok and via in ("auto", "gguf"):
            destino = MODELOS_DIR / f"{meta['tag'].replace(':', '_')}.gguf"
            if destino.exists() and destino.stat().st_size > 1024:
                ok, metodo = True, "gguf-cache"
            elif self._bajar_gguf(str(meta["gguf"]), destino):
                ok, metodo = True, "gguf"
        self._anotar(meta["id"], metodo or "fallido", ok)
        seg = round(time.time() - t0, 1)
        if ok:
            return {"exito": True, "modelo": meta["id"], "metodo": metodo, "segundos": seg,
                    "mensaje": f"He descargado {meta['id']} ({meta['gb']}GB, via {metodo}) en {seg}s. "
                               f"Esta en LC/modelosIAlocal y listo para usar offline."}
        return {"exito": False, "modelo": meta["id"], "metodo": "", "segundos": seg,
                "mensaje": f"No pude descargar {meta['id']}: sin red ni Ollama. "
                           "Instala Ollama (ollama.com) y reintenta."}

    def descargar_recomendados(self, limite: int = 2) -> List[Dict[str, Any]]:
        self.perfilar_hardware()
        return [self.descargar(m["id"]) for m in self.recomendar()[:max(1, limite)]]

    def descargar_en_fondo(self, modelo_id: str = "") -> str:
        """Lanza descarga sin bloquear el turno; LucIA sigue conversando."""
        if self._hilo_fondo and self._hilo_fondo.is_alive():
            return "Ya hay una descarga en curso en segundo plano; espera a que termine."
        objetivo = modelo_id or (self.recomendar()[0]["id"] if self.recomendar() else "")
        if not objetivo:
            return "No hay modelo objetivo para descargar."
        self._hilo_fondo = threading.Thread(target=self.descargar, args=(objetivo,),
                                            daemon=True, name="MDSTM-descarga")
        self._hilo_fondo.start()
        return f"Descargando {objetivo} en segundo plano; te aviso al terminar."

    # ── Lenguaje natural ──────────────────────────────────────
    def es_orden_descarga(self, texto: str) -> bool:
        t = texto.lower()
        return bool(PATRON_DESCARGA.search(texto)) and any(p in t for p in PALABRAS_MODELO)

    def ejecutar_orden(self, texto: str) -> Optional[str]:
        """Si el usuario pide descargar, EJECUTA y devuelve respuesta factual. None si no aplica."""
        if not self.es_orden_descarga(texto):
            return None
        meta = self.buscar(texto)
        if meta is None:
            if re.search(r"recomendad|liger|que\s+quep|autonom", texto, re.IGNORECASE):
                reps = self.descargar_recomendados(limite=1)
                r = reps[0] if reps else {}
                return ("Claro que si: SI puedo descargar modelos. " + str(r.get("mensaje", "sin resultado.")))
            lista = ", ".join(m["id"] for m in self.recomendar())
            return (f"Por supuesto, SI tengo capacidad de descarga ({self.capacidad_real()}). "
                    f"Dime cual de estos: {lista}.")
        rep = self.descargar(meta["id"])
        return ("Claro que si, ya lo estoy haciendo. " if rep["exito"] else "Lo he intentado: ") + rep["mensaje"]

    def estado(self) -> Dict[str, Any]:
        return {"version": __version__, "perfil": self.perfil,
                "capacidad": self.capacidad_real(),
                "recomendados": [m["id"] for m in self.recomendar()],
                "descargados": self.listar_descargados(),
                "ollama_online": self.ollama_online(),
                "carpeta": str(MODELOS_DIR), "catalogo": len(CATALOGO)}

    # ── Mantenimiento ─────────────────────────────────────────
    def verificar_integridad(self) -> List[Dict[str, Any]]:
        """Comprueba que cada GGUF registrado exista y pese mas de 1KB."""
        reporte: List[Dict[str, Any]] = []
        for item in self.listar_descargados():
            arch = MODELOS_DIR / f"{item['id'].replace(':', '_')}.gguf"
            ok = arch.exists() and arch.stat().st_size > 1024
            reporte.append({"id": item["id"], "integro": ok,
                            "bytes": arch.stat().st_size if arch.exists() else 0})
        return reporte

    def eliminar_modelo(self, modelo_id: str) -> Dict[str, Any]:
        """Borra el GGUF local para liberar disco y lo anota en el registro."""
        meta = next((m for m in CATALOGO if m["id"] == modelo_id), None) or self.buscar(modelo_id)
        if meta is None:
            return {"exito": False, "mensaje": f"Modelo desconocido: {modelo_id}."}
        arch = MODELOS_DIR / f"{meta['tag'].replace(':', '_')}.gguf"
        try:
            if arch.exists():
                arch.unlink()
            with self._lock:
                reg = self._leer_registro()
                reg.get("descargas", {}).pop(meta["id"], None)
                REGISTRO.write_text(json.dumps(reg, indent=2, ensure_ascii=False), encoding="utf-8")
            return {"exito": True, "mensaje": f"He eliminado {meta['id']} y liberado {meta['gb']}GB."}
        except Exception as exc:
            return {"exito": False, "mensaje": f"No pude eliminar {meta['id']}: {exc}."}

    def espacio_usado_gb(self) -> float:
        """Suma el peso real de los GGUF presentes en la carpeta."""
        total = 0
        try:
            for f in MODELOS_DIR.glob("*.gguf"):
                total += f.stat().st_size
        except Exception:
            pass
        return round(total / 1e9, 2)

    def informe_para_lucia(self) -> str:
        """Texto factual completo para que LucIA responda con datos reales."""
        est = self.estado()
        perf = est["perfil"]
        lineas = [
            f"Soy LucIA y SI puedo descargar modelos de IA localmente (MDSTM v{__version__}).",
            f"Hardware: {perf['ram_total_gb']}GB RAM, {perf['vram_total_gb']}GB VRAM, "
            f"{perf['disco_libre_gb']}GB libres. Ollama: {'online' if est['ollama_online'] else 'offline'}.",
            f"Recomendados para este PC: {', '.join(est['recomendados'])}.",
            f"Descargados ({self.espacio_usado_gb()}GB): "
            + (", ".join(d["id"] for d in est["descargados"]) or "ninguno aun; pideme 'descarga el modelo X' "
               "y lo descargo de inmediato con ollama pull o GGUF directo."),
        ]
        return " ".join(lineas)

    def ayuda_comandos(self) -> str:
        """Ayuda breve de las ordenes de descarga que LucIA entiende."""
        return ("Puedo descargar modelos YA. Prueba: 'descarga el modelo qwen2.5:0.5b', "
                "'descarga un modelo ligero recomendado', 'que modelos tienes descargados', "
                "'cuanta RAM y VRAM tengo', 'elimina el modelo X'. "
                f"Opciones: {', '.join(m['id'] for m in CATALOGO)}.")


