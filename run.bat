@echo off
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
if "%~1"=="" (
  python -m vkr_terrain.desktop_app
) else (
  python -m vkr_terrain.main %*
)
pause
