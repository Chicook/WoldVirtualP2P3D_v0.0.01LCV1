# CHG — Papelera de sesión

Todo residuo de la sesión (`__pycache__`, `*.pyc`, temporales `_tmp_*.bin`,
informes intermedios) se deposita aquí durante la ejecución vía
`PURGADOR.depositar_en_chg()` y **se borra al cerrar sesión** desde
`mainLCSTM.cerrar_sistema()` → `PURGADOR.solo_limpiar_pycache()`.

No guardar nada permanente aquí.
