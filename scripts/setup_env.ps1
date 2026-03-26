param(
    [string]$CondaPrefix = ".conda",
    [string]$PythonVersion = "3.12"
)

$ErrorActionPreference = "Stop"

conda create --prefix $CondaPrefix python=$PythonVersion -y

$pythonExe = Join-Path $CondaPrefix "python.exe"

& $pythonExe -m pip install --upgrade pip
& $pythonExe -m pip install -r requirements.txt

Write-Host "Environment ready at $CondaPrefix"
