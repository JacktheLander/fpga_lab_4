param(
    [string]$VivadoSettings = "D:\AMD\2025.2\Vivado\settings64.bat",
    [string]$Part = "xc7z020clg400-1",
    [string]$Kernel = "lr_train_accel",
    [string]$RequiredVersion = "2025.2"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot

if (-not (Test-Path $VivadoSettings)) {
    throw "Vivado settings script not found: $VivadoSettings"
}
$ipDir = Join-Path $projectRoot "build\hls\$Kernel\solution1\impl\ip"
if (-not (Test-Path $ipDir)) {
    throw "Exported HLS IP directory not found: $ipDir. Run hls\build_hls.ps1 first."
}
$versionOutput = cmd /c "call `"$VivadoSettings`" >nul && vivado.bat -version"
$versionText = $versionOutput -join "`n"
if ($LASTEXITCODE -ne 0 -and $versionText -notmatch [regex]::Escape($RequiredVersion)) {
    throw "Unable to verify Vivado version through $VivadoSettings. Output: $versionText"
}
if ($versionText -notmatch [regex]::Escape($RequiredVersion)) {
    throw "Incompatible Vivado version. Required $RequiredVersion. Output: $versionText"
}

$env:PYNQ_PART = $Part
$env:LR_KERNEL = $Kernel
$env:REQUIRED_VIVADO_VERSION = $RequiredVersion

cmd /c "call `"$VivadoSettings`" && vivado.bat -mode batch -source `"$PSScriptRoot\create_lr_burst_bd.tcl`""
if ($LASTEXITCODE -ne 0) {
    throw "Vivado block-design generation failed"
}

$xpr = Join-Path $projectRoot "build\vivado\lr_train_accel\lr_train_accel.xpr"
if (-not (Test-Path $xpr)) {
    throw "Vivado project generation completed but .xpr was not produced: $xpr"
}
Write-Host "Vivado project generated under $(Join-Path $projectRoot 'build\vivado\lr_train_accel')."
