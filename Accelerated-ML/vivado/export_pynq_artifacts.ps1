param(
    [string]$VivadoSettings = "D:\AMD\2025.2\Vivado\settings64.bat",
    [string]$ProjectXpr = "",
    [string]$ArtifactName = "lr_train_accel",
    [string]$ArtifactDir = "",
    [string]$RequiredVersion = "2025.2"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot

if (-not (Test-Path $VivadoSettings)) {
    throw "Vivado settings script not found: $VivadoSettings"
}
$versionOutput = cmd /c "call `"$VivadoSettings`" >nul && vivado.bat -version"
$versionText = $versionOutput -join "`n"
if ($LASTEXITCODE -ne 0 -and $versionText -notmatch [regex]::Escape($RequiredVersion)) {
    throw "Unable to verify Vivado version through $VivadoSettings. Output: $versionText"
}
if ($versionText -notmatch [regex]::Escape($RequiredVersion)) {
    throw "Incompatible Vivado version. Required $RequiredVersion. Output: $versionText"
}

if ($ProjectXpr -eq "") {
    $ProjectXpr = Join-Path $projectRoot "build\vivado\lr_train_accel\lr_train_accel.xpr"
}
if ($ArtifactDir -eq "") {
    $ArtifactDir = Join-Path $projectRoot "artifacts\pynq"
}
if (-not (Test-Path $ProjectXpr)) {
    throw "Vivado project not found: $ProjectXpr. Run vivado\build_vivado_project.ps1 first."
}

$env:LR_PROJECT_XPR = $ProjectXpr
$env:LR_ARTIFACT_NAME = $ArtifactName
$env:LR_ARTIFACT_DIR = $ArtifactDir
$env:REQUIRED_VIVADO_VERSION = $RequiredVersion

cmd /c "call `"$VivadoSettings`" && vivado.bat -mode batch -source `"$PSScriptRoot\implement_lr_overlay.tcl`""
if ($LASTEXITCODE -ne 0) {
    throw "Vivado implementation/export failed"
}

$bitPath = Join-Path $ArtifactDir "$ArtifactName.bit"
$hwhPath = Join-Path $ArtifactDir "$ArtifactName.hwh"
if (-not (Test-Path $bitPath)) {
    throw "Vivado implementation completed but bitstream was not produced: $bitPath"
}
if (-not (Test-Path $hwhPath)) {
    throw "Vivado implementation completed but HWH was not produced: $hwhPath"
}
Write-Host "PYNQ artifacts exported to $ArtifactDir."
