<#
.SYNOPSIS
    One-command launcher for the vehicle tracking pipeline (Windows).

.EXAMPLE
    .\scripts\start.ps1 -Input "data/outputs/h264/admin_in" -Camera admin_in
#>
param(
    [Parameter(Mandatory = $true)]
    [Alias("Input")]
    [string]$VideoInput,

    [string]$Camera = "admin_in",
    [string]$Output = "data/outputs/full_run",
    [int]$Device = 0,
    [string]$Config = "config/default.yaml",
    [int]$MaxFrames = 0
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot ".venv\python.exe"

if (-not (Test-Path $python)) {
    $python = Join-Path $repoRoot ".venv\Scripts\python.exe"
}
if (-not (Test-Path $python)) {
    throw "Python not found. Create the venv first: python -m venv .venv"
}

$arguments = @(
    (Join-Path $repoRoot "scripts\run_full_pipeline.py"),
    "--input", $VideoInput,
    "--camera", $Camera,
    "--output", $Output,
    "--device", $Device,
    "--config", $Config
)
if ($MaxFrames -gt 0) {
    $arguments += @("--max-frames", $MaxFrames)
}

Write-Host "Running: $python $($arguments -join ' ')" -ForegroundColor Cyan
& $python @arguments
exit $LASTEXITCODE
