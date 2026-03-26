param(
    [Parameter(Mandatory = $true)]
    [string]$AdapterDir,
    [string]$Output = "",
    [string]$CondaPrefix = ".conda"
)

$ErrorActionPreference = "Stop"
$pythonExe = Join-Path $CondaPrefix "python.exe"

$argsList = @("package_submission.py", "--adapter-dir", $AdapterDir)

if ($Output -ne "") {
    $argsList += @("--output", $Output)
}

& $pythonExe @argsList
