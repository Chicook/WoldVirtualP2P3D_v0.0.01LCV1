@echo off
title LucIA - WoldVirtualP2P3D LCV1
echo ======================================================================
echo           Iniciando Sistema LucIA (Modo Total Libertad)
echo ======================================================================
cd /d "%~dp0"

if exist venv\Scripts\activate.bat (
    echo [1/2] Activando entorno virtual local...
    call venv\Scripts\activate.bat
) else (
    echo [1/2] Usando entorno Python del sistema...
)

echo [2/2] Lanzando LucIA...
python lucIA\main.py

echo.
echo Sesion finalizada.
pause
