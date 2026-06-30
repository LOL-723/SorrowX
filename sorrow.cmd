@echo off
setlocal
set "PROJECT_ROOT=%~dp0"
set "PYTHONPATH=%PROJECT_ROOT%src;%PROJECT_ROOT%src\core;%PYTHONPATH%"
set "PYTHON_EXE=%PROJECT_ROOT%.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  echo Project virtual environment not found: %PROJECT_ROOT%.venv
  exit /b 1
)
"%PYTHON_EXE%" -m cli.main %*
exit /b %ERRORLEVEL%
