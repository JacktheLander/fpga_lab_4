param(
    [string]$Top = "lr_train_accel",
    [string]$VitisSettings = "D:\AMD\2025.2\Vitis\settings64.bat",
    [string]$Part = "xc7z020clg400-1",
    [string]$ClockPeriod = "10",
    [string]$RequiredVersion = "2025.2",
    [switch]$CsimOnly
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot

if (-not (Test-Path $VitisSettings)) {
    throw "Vitis settings script not found: $VitisSettings"
}
foreach ($requiredPath in @(
    (Join-Path $PSScriptRoot "run_hls.tcl"),
    (Join-Path $PSScriptRoot "src\lr_train_accel.cpp"),
    (Join-Path $PSScriptRoot "tb\tb_lr_train_accel.cpp"),
    (Join-Path $PSScriptRoot "include\lr_config.h")
)) {
    if (-not (Test-Path $requiredPath)) {
        throw "Required HLS file not found: $requiredPath"
    }
}

$versionOutput = cmd /c "call `"$VitisSettings`" >nul && vitis-run.bat --version"
if ($LASTEXITCODE -ne 0) {
    throw "Unable to run vitis-run.bat --version through $VitisSettings"
}
$versionText = $versionOutput -join "`n"
if ($versionText -notmatch [regex]::Escape($RequiredVersion)) {
    throw "Incompatible Vitis version. Required $RequiredVersion. Output: $versionText"
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
    $ipArchive = Join-Path $projectRoot "build\ip\$Top.zip"
    if (-not (Test-Path $ipArchive)) {
        throw "HLS completed but expected IP archive was not produced: $ipArchive"
    }
    Write-Host "Exported IP archive: $ipArchive"
}
