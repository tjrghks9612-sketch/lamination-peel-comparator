[CmdletBinding()]
param(
    [string]$ExecutablePath = "",
    [ValidateRange(1, 120)]
    [int]$StartupSeconds = 8
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot

if ([string]::IsNullOrWhiteSpace($ExecutablePath)) {
    $defaultExecutable = Join-Path $projectRoot "dist\LaminationPeelComparator\LaminationPeelComparator.exe"
    if (Test-Path -LiteralPath $defaultExecutable -PathType Leaf) {
        $ExecutablePath = $defaultExecutable
    }
    else {
        $distRoot = Join-Path $projectRoot "dist"
        $candidates = @(
            Get-ChildItem -LiteralPath $distRoot -Recurse -File -Filter "LaminationPeelComparator.exe" -ErrorAction SilentlyContinue
        )
        if ($candidates.Count -ne 1) {
            throw "Expected one LaminationPeelComparator.exe below $distRoot, found $($candidates.Count). Pass -ExecutablePath explicitly."
        }
        $ExecutablePath = $candidates[0].FullName
    }
}

if (-not (Test-Path -LiteralPath $ExecutablePath -PathType Leaf)) {
    throw "Application executable does not exist: $ExecutablePath"
}

$resolvedExecutable = (Resolve-Path -LiteralPath $ExecutablePath).Path
$workingDirectory = Split-Path -Parent $resolvedExecutable
$process = $null

try {
    $selfTest = Start-Process `
        -FilePath $resolvedExecutable `
        -ArgumentList "--self-test" `
        -WorkingDirectory $workingDirectory `
        -WindowStyle Hidden `
        -Wait `
        -PassThru
    if ($selfTest.ExitCode -ne 0) {
        throw "Packaged numerical self-test failed with code $($selfTest.ExitCode)."
    }
    Write-Host "Packaged numerical self-test passed."

    $process = Start-Process `
        -FilePath $resolvedExecutable `
        -WorkingDirectory $workingDirectory `
        -WindowStyle Hidden `
        -PassThru

    if ($process.WaitForExit($StartupSeconds * 1000)) {
        throw "Smoke test failed: the GUI exited during startup with code $($process.ExitCode)."
    }

    Write-Host "Smoke test passed: application stayed alive for $StartupSeconds second(s)."
}
finally {
    if ($null -ne $process -and -not $process.HasExited) {
        Stop-Process -Id $process.Id -Force
        [void]$process.WaitForExit(5000)
    }
}
