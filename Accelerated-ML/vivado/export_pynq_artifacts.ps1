param(
    [string]$VivadoSettings = "D:\AMD\2025.2\Vivado\settings64.bat",
    [string]$ProjectXpr = "",
    [string]$ArtifactName = "lr_train_accel",
    [string]$ArtifactDir = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot

if (-not (Test-Path $VivadoSettings)) {
    throw "Vivado settings script not found: $VivadoSettings"
}

if ($ProjectXpr -eq "") {
    $ProjectXpr = Join-Path $projectRoot "build\vivado\lr_train_accel\lr_train_accel.xpr"
}
if ($ArtifactDir -eq "") {
    $ArtifactDir = Join-Path $projectRoot "artifacts\pynq"
}

$env:LR_PROJECT_XPR = $ProjectXpr
$env:LR_ARTIFACT_NAME = $ArtifactName
$env:LR_ARTIFACT_DIR = $ArtifactDir

cmd /c "call `"$VivadoSettings`" && vivado.bat -mode batch -source `"$PSScriptRoot\implement_lr_overlay.tcl`""
if ($LASTEXITCODE -ne 0) {
    throw "Vivado implementation/export failed"
}

Write-Host "PYNQ artifacts exported to $ArtifactDir."
