[CmdletBinding()]
param(
    [string]$DistPath = ""
)

$ErrorActionPreference = "Stop"
$limitBytes = 100MB
$projectRoot = Split-Path -Parent $PSScriptRoot

if ([string]::IsNullOrWhiteSpace($DistPath)) {
    $DistPath = Join-Path $projectRoot "dist"
}

if (-not (Test-Path -LiteralPath $DistPath -PathType Container)) {
    throw "Distribution directory does not exist: $DistPath"
}

$resolvedDist = (Resolve-Path -LiteralPath $DistPath).Path
$files = @(Get-ChildItem -LiteralPath $resolvedDist -Recurse -File | Sort-Object FullName)

if ($files.Count -eq 0) {
    throw "Distribution directory contains no files: $resolvedDist"
}

$relativePrefix = $resolvedDist.TrimEnd("\", "/") + [System.IO.Path]::DirectorySeparatorChar
$rows = foreach ($file in $files) {
    $relativePath = $file.FullName.Substring($relativePrefix.Length)
    [PSCustomObject]@{
        File = $relativePath
        Bytes = $file.Length
        MiB = [Math]::Round($file.Length / 1MB, 3)
        Status = if ($file.Length -ge $limitBytes) { "FAIL >= 100 MiB" } else { "OK" }
    }
}

$rows | Format-Table -AutoSize | Out-String -Width 4096 | Write-Host

$totalBytes = ($files | Measure-Object -Property Length -Sum).Sum
Write-Host ("Files: {0}; total: {1:N3} MiB; per-file limit: 100 MiB" -f $files.Count, ($totalBytes / 1MB))

$oversized = @($files | Where-Object Length -GE $limitBytes)
if ($oversized.Count -gt 0) {
    Write-Error ("Distribution audit failed: {0} file(s) are at least 100 MiB." -f $oversized.Count)
    exit 1
}

Write-Host "Distribution audit passed: every individual file is below 100 MiB."
exit 0
