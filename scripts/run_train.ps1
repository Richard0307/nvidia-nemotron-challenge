param(
    [string]$Config = "configs/sft_baseline.yaml",
    [switch]$DryRun,
    [int]$MaxSamples = 0,
    [string]$CondaPrefix = ".conda"
)

$ErrorActionPreference = "Stop"
$pythonExe = Join-Path $CondaPrefix "python.exe"

$argsList = @("train.py", "--config", $Config)

if ($DryRun) {
    $argsList += "--dry-run"
}

if ($MaxSamples -gt 0) {
    $argsList += @("--max-samples", "$MaxSamples")
}

& $pythonExe @argsList
