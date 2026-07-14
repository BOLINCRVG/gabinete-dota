@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [1/4] Criando ambiente virtual...
  python -m venv .venv
  if errorlevel 1 goto :fail
)

echo [2/4] Ativando ambiente virtual...
call ".venv\Scripts\activate.bat"
if errorlevel 1 goto :fail

echo [3/4] Atualizando pip...
python -m pip install --upgrade pip
if errorlevel 1 goto :fail

echo [4/4] Instalando dependencias...
if exist "requirements.txt" (
  python -m pip install -r requirements.txt
) else (
  python -m pip install streamlit pandas numpy requests
)
if errorlevel 1 goto :fail

echo Iniciando app...
python -m streamlit run "GabineteDotaFINAL.py"
goto :eof

:fail
echo Falha ao configurar/rodar o Gabinete Dota.
pause
exit /b 1
