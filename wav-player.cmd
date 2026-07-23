@echo off
rem Windows launcher for the wav_spectrogram player — the counterpart to the
rem bash `wav-player` script, so `wav-player ...` works in PowerShell and cmd.
rem It runs `python -m wav_spectrogram` with the package's parent directory on
rem the import path, so relative WAV paths still resolve against your own cwd.
rem All arguments are passed straight through.
setlocal
rem Directory of this script, with the trailing backslash trimmed.
set "HERE=%~dp0"
set "HERE=%HERE:~0,-1%"
rem The package is this directory; running it as a module needs its parent.
for %%I in ("%HERE%") do set "PKG=%%~nxI"
for %%I in ("%HERE%\..") do set "PARENT=%%~fI"
set "PYTHONPATH=%PARENT%;%PYTHONPATH%"
python -m %PKG% %*
