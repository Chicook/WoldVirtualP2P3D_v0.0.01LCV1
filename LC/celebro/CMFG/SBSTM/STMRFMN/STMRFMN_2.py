# ─── CICLO DE VIDA PROGRAMATICO ──────────────────────────────────────────

def iniciar_lucia() -> OrquestadorSistemaLucIA:
    """Crea el orquestador e inicializa todos los subsistemas (pasos 0-8)."""
    orq = OrquestadorSistemaLucIA()
    if not orq.inicializar_subsistemas():
        raise RuntimeError("[mainLCSTM] Fallo inicializando subsistemas.")
    return orq


def detener_lucia(orq: OrquestadorSistemaLucIA) -> None:
    """Cierre ordenado: refactor -> .md -> pesos -> IPFS -> devopencode."""
    orq.cerrar_sistema()


def turno(orq: OrquestadorSistemaLucIA, prompt: str) -> Dict[str, Any]:
    """Ejecuta un turno de dialogo y devuelve su telemetria basica."""
    t0 = time.perf_counter()
    orq.procesar_turno_dialogo(prompt)
    return {"turno": orq.turno_actual, "segundos": round(time.perf_counter() - t0, 2)}


def estado(orq: OrquestadorSistemaLucIA) -> Dict[str, Any]:
    """Foto del sistema: cadena, neuronas, deriva, turnos y modelos."""
    try:
        valida, _ = orq.servidor_bks.validar_cadena()
    except Exception:
        valida = False
    try:
        mod = orq.cliente_iafree.gestor.obtener_modelo_activo()["id"] if orq.cliente_iafree else "N/A"
    except Exception:
        mod = "N/A"
    return {"sesion": orq.sesion_id, "activa": orq.activa, "turnos": orq.turno_actual,
            "cadena_valida": bool(valida),
            "bloques": len(orq.servidor_bks.cadena) if orq.servidor_bks else 0,
            "neuronas": len(orq.conversor_psn.neuronas) if orq.conversor_psn else 0,
            "modelo_activo": mod,
            "ia_local_lista": orq.ia_local_lista,
            "constructor_activo": bool(orq.gestor_hrctrc and orq.gestor_hrctrc.esta_activa())}