param(
    [string]$Top = "lr_train_accel",
    [string]$VitisSettings = "D:\AMD\2025.2\Vitis\settings64.bat",
    [string]$Part = "xc7z020clg400-1",
    [string]$ClockPeriod = "10",
    [switch]$CsimOnly
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot

if (-not (Test-Path $VitisSettings)) {
    throw "Vitis settings script not found: $VitisSettings"
}

$env:TOP_NAME = $Top
$env:PART_NAME = $Part
$env:CLOCK_PERIOD = $ClockPeriod
if ($CsimOnly) {
    $env:HLS_CSIM_ONLY = "1"
} else {
    Remove-Item Env:\HLS_CSIM_ONLY -ErrorAction SilentlyContinue
}

Write-Host "=== Building $Top for $Part, clock period ${ClockPeriod}ns ==="
cmd /c "call `"$VitisSettings`" && vitis-run.bat --mode hls --tcl `"$PSScriptRoot\run_hls.tcl`""
if ($LASTEXITCODE -ne 0) {
    throw "HLS build failed for $Top"
}

Write-Host "HLS build completed under $(Join-Path $projectRoot 'build\hls')."
if (-not $CsimOnly) {
    Write-Host "Exported IP archive under $(Join-Path $projectRoot 'build\ip')."
}
