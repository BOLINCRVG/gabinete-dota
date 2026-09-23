@echo off
setlocal
cd /d "%~dp0"

rem Gabinete Dota (clone IGUAL ao do LoL, vermelho) — porta 8502; LoL fica na 8501.
rem PORTATIL: funciona em qualquer PC com as duas pastas lado a lado.
rem Ordem do python: (1) .venv local desta pasta (SETUP do amigo),
rem                  (2) .venv do repo do LoL vizinho, (3) python do sistema.

set "PYEXE=python"
if exist ".venv\Scripts\python.exe" set "PYEXE=.venv\Scripts\python.exe"
if not exist ".venv\Scripts\python.exe" (
  for /d %%D in ("%~dp0..\gabinetelolweb*") do (
    if exist "%%D\.venv\Scripts\python.exe" set "PYEXE=%%D\.venv\Scripts\python.exe"
  )
)

rem venv copiado de outro PC pode existir mas nao funcionar — testa antes;
rem se falhar, cai pro python do sistema (e, se faltar dependencia, rode o
rem SETUP_AND_RUN_GABINETE_DOTA_WINDOWS.bat uma vez).
"%PYEXE%" -c "import streamlit" >nul 2>&1
if errorlevel 1 set "PYEXE=python"

rem Traz o que o robo publicou no GitHub ANTES de abrir (sincronizar.py).
rem Nunca impede o app de abrir: qualquer problema vira so um aviso aqui.
"%PYEXE%" "sincronizar.py"

"%PYEXE%" -m streamlit run "gabinete_dota_app.py" --server.port 8502

pause
