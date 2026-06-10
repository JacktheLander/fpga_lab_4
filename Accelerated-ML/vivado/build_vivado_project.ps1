param(
    [string]$VivadoSettings = "D:\AMD\2025.2\Vivado\settings64.bat",
    [string]$Part = "xc7z020clg400-1",
    [string]$Kernel = "lr_train_accel"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $VivadoSettings)) {
    throw "Vivado settings script not found: $VivadoSettings"
}

$env:PYNQ_PART = $Part
$env:LR_KERNEL = $Kernel

cmd /c "call `"$VivadoSettings`" && vivado.bat -mode batch -source `"$PSScriptRoot\create_lr_burst_bd.tcl`""
if ($LASTEXITCODE -ne 0) {
    throw "Vivado block-design generation failed"
}

Write-Host "Vivado project generated under $(Join-Path (Split-Path -Parent $PSScriptRoot) 'build\vivado\lr_train_accel')."
