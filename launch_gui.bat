@echo off
rem One-click launcher for the Transformer inference GUI
chcp 65001 >nul
cd /d %~dp0

rem NOTE: no if/else ( ) blocks here - the system PATH contains paths with
rem parentheses (e.g. NVIDIA), which breaks cmd parsing inside ( ) blocks.

set ENV=H:\Anaconda\envs\annotated-transformer
set PY=python
if exist "%ENV%\python.exe" set PY=%ENV%\python.exe

rem Put the conda env at the FRONT of PATH (like "conda activate"),
rem otherwise DLL lookups can hit incompatible copies from H:\msys2\mingw64\bin.
set PATH=%ENV%;%ENV%\Library\mingw-w64\bin;%ENV%\Library\usr\bin;%ENV%\Library\bin;%ENV%\Scripts;%PATH%

echo Starting GUI with: %PY%
"%PY%" src\model_gui.py
if errorlevel 1 pause
