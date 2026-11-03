@echo off
setlocal
pushd "%~dp0" || exit /b 1

if defined LAB_PYTHON goto custom_python
if exist ".venv\Scripts\python.exe" goto existing_python
where py >nul 2>nul
if errorlevel 1 goto missing_python
py -3.12 scripts\setup_environment.py %*
goto finished

:existing_python
".venv\Scripts\python.exe" scripts\setup_environment.py %*
goto finished

:custom_python
"%LAB_PYTHON%" scripts\setup_environment.py %*
goto finished

:missing_python
echo Python launcher not found. Install Python 3.12 with its launcher, reopen the terminal, and try again. 1>&2
echo See README.md, Prerequisites. 1>&2
popd
exit /b 2

:finished
set "LAB_SETUP_EXIT=%ERRORLEVEL%"
popd
exit /b %LAB_SETUP_EXIT%
