@echo off
title Veda - Personal Offline AI Companion
cd /d "%~dp0"
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
rem Use the project's virtual environment if there is one (.venv or vedic-atlas-env, or the folder named in VENV);
rem otherwise the python on PATH (for example an already-activated environment).
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if exist "vedic-atlas-env\Scripts\python.exe" set "PY=vedic-atlas-env\Scripts\python.exe"
if defined VENV if exist "%VENV%\Scripts\python.exe" set "PY=%VENV%\Scripts\python.exe"
"%PY%" -m controller.cli %*
