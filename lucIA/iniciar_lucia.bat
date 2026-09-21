@echo off
title LucIA - entorno virtual (venv)
echo ======================================================================
echo [1/2] Activando entorno virtual (venv)...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo Error: No se pudo activar el entorno virtual.
    echo Asegurate de que la carpeta 'venv' exista.
    pause
    exit /b 1
)

echo [2/2] venv OK (rotacion L-C-L, 1 modelo/pregunta). Python en uso:
python -c "import sys; print(sys.executable)"
echo Iniciando LucIA...
python main.py

echo.
echo LucIA se ha cerrado. Desactivando entorno virtual...
deactivate
pause
