param(
    [string]$Input = "data/train.csv",
    [string]$Output = "data/train_cot.jsonl",
    [int]$MaxSamples = 0,
    [string]$CondaPrefix = ".conda"
)

$ErrorActionPreference = "Stop"
$pythonExe = Join-Path $CondaPrefix "python.exe"

$argsList = @("generate_cot.py", "--input", $Input, "--output", $Output)

if ($MaxSamples -gt 0) {
    $argsList += @("--max-samples", "$MaxSamples")
}

& $pythonExe @argsList
