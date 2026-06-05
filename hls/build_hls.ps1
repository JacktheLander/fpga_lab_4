param(
    [string[]]$Top = @(
        "spmv_burst_s1",
        "spmv_burst_s2",
        "spmv_burst_s3",
        "spmv_burst_s4",
        "spmv_stream_s1",
        "spmv_stream_s2",
        "spmv_stream_s3",
        "spmv_stream_s4",
        "spmv_burst_s4_rows16",
        "spmv_burst_s4_rows25",
        "spmv_burst_s4_rows50",
        "spmv_stream_s4_rows16",
        "spmv_stream_s4_rows25",
        "spmv_stream_s4_rows50"
    ),
    [string]$VitisSettings = "D:\AMD\2025.2\Vitis\settings64.bat",
    [string]$Part = "xc7k70tfbg484-1"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot

foreach ($name in $Top) {
    if ($name -like "spmv_burst*") {
        $env:SRC_FILE = Join-Path $PSScriptRoot "burst\$name.cpp"
        $env:TB_FILE = Join-Path $PSScriptRoot "tb\spmv_burst_tb.cpp"
    } else {
        $env:SRC_FILE = Join-Path $PSScriptRoot "stream\$name.cpp"
        $env:TB_FILE = Join-Path $PSScriptRoot "tb\spmv_stream_tb.cpp"
    }
    $env:TOP_NAME = $name
    $env:PART_NAME = $Part

    Write-Host "=== Building $name for $Part ==="
    cmd /c "call `"$VitisSettings`" && vitis-run.bat --mode hls --tcl `"$PSScriptRoot\run_hls.tcl`""
    if ($LASTEXITCODE -ne 0) {
        throw "HLS failed for $name"
    }
}

Write-Host "HLS builds completed. Results are under $(Join-Path $repo 'build\hls')."
