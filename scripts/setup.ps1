param(
    [string]$VenvPath = ".venv"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $VenvPath)) {
    python -m venv $VenvPath
}

$python = Join-Path $VenvPath "Scripts\python.exe"

& $python -m pip install --upgrade pip
& $python -m pip install -r requirements.txt

Write-Host "Environment ready. Activate with: .\$VenvPath\Scripts\Activate.ps1"
