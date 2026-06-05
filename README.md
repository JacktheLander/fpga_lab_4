# ECEN 529 Lab 4: FPGA Sparse Matrix-Vector Multiplication

This repository contains the Lab 4 sparse matrix-vector multiplication project for Vitis HLS, Vivado, and PYNQ.

The final implementation includes:

- Four AXI4-Burst SPMV HLS kernels using optimization strategies 1-4.
- Four AXI4-Stream SPMV HLS kernels using the same strategies.
- Batch-size variants for the 50 x 50 experiment.
- Vitis HLS test benches for C simulation.
- PYNQ host programs for burst, stream, and batch experiments.
- Vivado block-design TCL scripts for demonstration builds.
- An answers-only PDF report with explanations and results.

The original lab PDFs are not tracked in this repository.

## Repository Structure

```text
hls/
  include/                 Shared HLS implementation headers
  burst/                   AXI4-Burst top functions
  stream/                  AXI4-Stream top functions
  tb/                      Vitis HLS C simulation test benches
  build_hls.ps1            PowerShell wrapper to build all HLS kernels
  run_hls.tcl              Vitis HLS build script

pynq/
  spmv_utils.py            CSR generation, batching, timing, register helpers
  spmv_burst_host.py       PYNQ host program for AXI4-Burst kernels
  spmv_stream_host.py      PYNQ host program for AXI4-Stream kernels with DMA
  spmv_batch_experiment.py PYNQ host program for local-array batch sweep

vivado/
  create_spmv_burst_bd.tcl  Vivado block design for Zynq PS + burst SPMV IP
  create_spmv_stream_bd.tcl Vivado block design for Zynq PS + AXI DMA + stream SPMV IP

scripts/
  generate_lab4_answers_only.py Generates the final answers-only report PDF
  generate_lab4_report.py       Older full report generator with code listings

reports/
  ECEN529_Lab4_answers_only.pdf Final answers-only report
  hls_summary.csv               Parsed Vitis HLS synthesis summary
```

Generated Vitis/Vivado outputs are written under `build/` and are intentionally ignored by Git.

## Prerequisites

For HLS synthesis and IP export:

- Windows PowerShell
- AMD Vitis/Vivado 2025.2 or compatible version
- Xilinx/AMD device support for the part you target

For the PYNQ runtime programs:

- A PYNQ board with a compatible bitstream and `.hwh`
- Python on the board with `pynq` and `numpy`

For report generation:

- Python 3
- `numpy`
- `matplotlib`
- `pypdf` only if you want to inspect generated PDF text

## Build the HLS Kernels

From the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File hls\build_hls.ps1
```

This runs C simulation, HLS synthesis, and IP export for:

- `spmv_burst_s1`
- `spmv_burst_s2`
- `spmv_burst_s3`
- `spmv_burst_s4`
- `spmv_stream_s1`
- `spmv_stream_s2`
- `spmv_stream_s3`
- `spmv_stream_s4`
- `spmv_burst_s4_rows16`
- `spmv_burst_s4_rows25`
- `spmv_burst_s4_rows50`
- `spmv_stream_s4_rows16`
- `spmv_stream_s4_rows25`
- `spmv_stream_s4_rows50`

By default the script targets the installed part used during development:

```text
xc7k70tfbg484-1
```

For a PYNQ-Z2/PYNQ-Z1-style Zynq-7020 build, run with the PYNQ part installed:

```powershell
powershell -ExecutionPolicy Bypass -File hls\build_hls.ps1 -Part xc7z020clg400-1
```

To build one kernel:

```powershell
powershell -ExecutionPolicy Bypass -File hls\build_hls.ps1 -Top spmv_burst_s4
```

Expected generated output:

```text
build/hls/<kernel>/solution1/
build/ip/<kernel>.zip
```

## Build the Vivado Project Structure for Demonstration

The Vivado scripts create the block-design project structure. They assume HLS IP has already been exported.

### AXI4-Burst Demonstration Design

```powershell
$env:PYNQ_PART = "xc7z020clg400-1"
$env:SPMV_BURST_KERNEL = "spmv_burst_s4"
cmd /c "call D:\AMD\2025.2\Vivado\settings64.bat && vivado.bat -mode batch -source vivado\create_spmv_burst_bd.tcl"
```

This creates:

```text
build/vivado/spmv_burst/
```

The design contains:

- Zynq Processing System
- HLS SPMV burst IP
- AXI-Lite control path
- AXI HP DDR path for the kernel `m_axi` ports

After the project is created, open it in Vivado, inspect the block diagram, run validation, and generate the bitstream.

### AXI4-Stream Demonstration Design

```powershell
$env:PYNQ_PART = "xc7z020clg400-1"
$env:SPMV_STREAM_KERNEL = "spmv_stream_s4"
cmd /c "call D:\AMD\2025.2\Vivado\settings64.bat && vivado.bat -mode batch -source vivado\create_spmv_stream_bd.tcl"
```

This creates:

```text
build/vivado/spmv_stream/
```

The design contains:

- Zynq Processing System
- AXI DMA
- HLS SPMV stream IP
- AXI-Lite control path
- AXI HP DDR path for DMA transfers
- AXI4-Stream connections between DMA and the kernel

After generating the bitstream, copy the `.bit` and `.hwh` files to the PYNQ board with the matching host script.

## Run on PYNQ

Copy the generated `.bit`, `.hwh`, and the `pynq/` Python files to the same folder on the PYNQ board.

### Burst Host Program

```bash
python3 spmv_burst_host.py --bitfile spmv_burst.bit --size 1000 --density 0.001
```

The script prints:

- Software runtime
- NNZ and operation count
- Pass/fail status for each kernel present in the overlay
- Setup time
- Kernel time
- Total time
- Throughput in operations per second

### Stream Host Program

```bash
python3 spmv_stream_host.py --bitfile spmv_stream.bit --dma axi_dma_0 --size 1000 --density 0.001
```

The stream input order is:

```text
rowPtr, columnIndex, values, x
```

The kernel writes `y` to the output stream and asserts `TLAST` on the final output word.

### Batch Experiment

```bash
python3 spmv_batch_experiment.py --bitfile spmv_burst_batch.bit --size 50 --density 0.01
```

The batch experiment compares local kernel array capacities:

- 16 rows / 32 NNZ
- 25 rows / 64 NNZ
- 50 rows / 128 NNZ

The batching algorithm splits complete CSR rows, rebases each local `rowPtr` to zero, keeps `columnIndex` global for the full `x` vector, and copies local `y` results back into the global output.

## Regenerate the Answers-Only PDF

After HLS synthesis has generated `reports/hls_summary.csv`, run:

```powershell
python scripts\generate_lab4_answers_only.py
```

Output:

```text
reports/ECEN529_Lab4_answers_only.pdf
```

## Development Notes

During development, all HLS C simulations passed for the burst, stream, and batch kernels. The local Vitis/Vivado installation did not include Zynq-7000 part support, so final PYNQ bitstream generation and board runtime timing must be performed on a lab machine with the PYNQ part installed and a connected board.
