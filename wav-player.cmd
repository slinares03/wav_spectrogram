@echo off
rem Windows launcher for the wav_spectrogram player — the counterpart to the
rem bash `wav-player` script, so `wav-player ...` works in PowerShell and cmd.
rem
rem First run bootstraps everything: it creates an isolated virtual environment
rem (.venv) inside the project and installs the dependencies, so a fresh clone
rem starts with a single `.\wav-player --view 3d` and no manual pip step. Later
rem runs reuse that environment and start immediately. All args pass through.
setlocal
rem Directory of this script, with the trailing backslash trimmed.
set "HERE=%~dp0"
set "HERE=%HERE:~0,-1%"
rem The package is this directory; running it as a module needs its parent.
for %%I in ("%HERE%") do set "PKG=%%~nxI"
for %%I in ("%HERE%\..") do set "PARENT=%%~fI"
set "VENV=%HERE%\.venv"
set "VPY=%VENV%\Scripts\python.exe"

rem First run: build the environment and install requirements.
if not exist "%VPY%" (
    echo First run: setting up a local environment in .venv ^(one-time^)...
    python -m venv "%VENV%"
    "%VPY%" -m pip install --quiet --upgrade pip
    "%VPY%" -m pip install --quiet -r "%HERE%\requirements.txt"
    echo Setup complete.
)

rem Guard against a half-finished setup or updated requirements: reinstall if
rem the core deps are missing rather than crashing with an ImportError.
"%VPY%" -c "import numpy, matplotlib" 2>NUL || "%VPY%" -m pip install --quiet -r "%HERE%\requirements.txt"

set "PYTHONPATH=%PARENT%;%PYTHONPATH%"
"%VPY%" -m %PKG% %*
