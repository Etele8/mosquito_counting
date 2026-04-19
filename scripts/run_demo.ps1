param(
    [string]$InputRoot = "data\images\20250816\cropped",
    [string]$OutputRoot = "data\outputs\portfolio_demo",
    [string]$PythonExe = ".venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

& $PythonExe counter.py --input-root $InputRoot --output-root $OutputRoot
