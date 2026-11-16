<#
.SYNOPSIS
    Local CI: unit tests + optional end-to-end smoke test + diagnostics.

.DESCRIPTION
    Steps:
      1/4 unit tests
      2/4 end-to-end smoke test (skipped with -SkipSmoke or when the video is missing)
      3/4 rule trigger audit   (runs on the smoke output when available)
      4/5 metrics report       (runs on the smoke output when available)
      5/6 demo pack            (one-click report + gang case + summary)
      6/6 delivery checklist   (files / YAML / doc references / stray dirs)

.EXAMPLE
    .\scripts\ci.ps1
    .\scripts\ci.ps1 -SkipSmoke
#>
param(
    [switch]$SkipSmoke,
    [string]$Video = "data/outputs/h264/admin_in/clip_08_short.h264",
    [int]$MaxFrames = 1500
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

Write-Host "`n[1/6] unit tests" -ForegroundColor Cyan
& $python -m pytest -q
if ($LASTEXITCODE -ne 0) {
    Write-Host "UNIT TESTS FAILED" -ForegroundColor Red
    exit $LASTEXITCODE
}

$smokeRoot = Join-Path $repoRoot "data/outputs/smoke"
$videoPath = Join-Path $repoRoot $Video
if ($SkipSmoke) {
    Write-Host "`n[2/6] smoke test skipped" -ForegroundColor Yellow
} elseif (-not (Test-Path $videoPath)) {
    Write-Host "`n[2/6] smoke test skipped: video not found at $Video" -ForegroundColor Yellow
} else {
    Write-Host "`n[2/6] end-to-end smoke test" -ForegroundColor Cyan
    & $python (Join-Path $repoRoot "scripts/smoke_test.py") --video $Video --max-frames $MaxFrames
    if ($LASTEXITCODE -ne 0) {
        Write-Host "SMOKE TEST FAILED" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

# 3/4 规则触发率审计：规则“写了但从不工作”是这类系统最常见的问题
$vectors = $null
$alertsDir = Join-Path $smokeRoot "alerts"
if (Test-Path $alertsDir) {
    $vectors = Get-ChildItem -Path $alertsDir -Filter "vectors*.csv" -ErrorAction SilentlyContinue |
        Select-Object -First 1
}
if ($vectors) {
    Write-Host "`n[3/6] rule trigger audit" -ForegroundColor Cyan
    & $python (Join-Path $repoRoot "scripts/audit_rules.py") --input $vectors.FullName
    if ($LASTEXITCODE -ne 0) {
        Write-Host "RULE AUDIT FAILED" -ForegroundColor Red
        exit $LASTEXITCODE
    }
} else {
    Write-Host "`n[3/6] rule audit skipped: no vectors under $alertsDir" -ForegroundColor Yellow
}

# 4/4 指标汇总：确保关键数字可复现
if (Test-Path $smokeRoot) {
    Write-Host "`n[4/6] metrics report" -ForegroundColor Cyan
    & $python (Join-Path $repoRoot "scripts/report_metrics.py") --root $smokeRoot --require-graph
    if ($LASTEXITCODE -ne 0) {
        Write-Host "METRICS REPORT FAILED" -ForegroundColor Red
        exit $LASTEXITCODE
    }
} else {
    Write-Host "`n[4/6] metrics report skipped: no output at $smokeRoot" -ForegroundColor Yellow
}

# 5/5 演示包：验证“一键出样例”链路可用
if (Test-Path $smokeRoot) {
    Write-Host "`n[5/6] demo pack" -ForegroundColor Cyan
    & $python (Join-Path $repoRoot "scripts/build_demo_pack.py") --root $smokeRoot --demo-dir (Join-Path $smokeRoot "demo")
    if ($LASTEXITCODE -ne 0) {
        Write-Host "DEMO PACK FAILED" -ForegroundColor Red
        exit $LASTEXITCODE
    }
} else {
    Write-Host "`n[5/6] demo pack skipped: no output at $smokeRoot" -ForegroundColor Yellow
}
# 6/6 交付清单：文件齐备、YAML 可解析、文档引用有效、无遗留临时目录
Write-Host "`n[6/6] delivery checklist" -ForegroundColor Cyan
& $python (Join-Path $repoRoot "scripts/delivery_check.py")
if ($LASTEXITCODE -ne 0) {
    Write-Host "DELIVERY CHECK FAILED" -ForegroundColor Red
    exit $LASTEXITCODE
}
Write-Host "`nCI PASSED" -ForegroundColor Green
exit 0
