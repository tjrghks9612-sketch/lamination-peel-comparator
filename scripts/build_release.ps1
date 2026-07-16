param(
    [switch]$SkipTests
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $root

if (-not $SkipTests) {
    uv run --extra dev pytest
}

uv run --extra dev pyinstaller --clean --noconfirm LaminationPeelComparator.spec

& (Join-Path $PSScriptRoot 'audit_dist.ps1') -DistPath (Join-Path $root 'dist')
& (Join-Path $PSScriptRoot 'smoke_app.ps1')

Write-Host 'Release build and verification completed.' -ForegroundColor Green
