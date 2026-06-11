# Accelerated-ML Final Project

This project implements the ECEN 529 final-project option for logistic regression training with gradient descent. The accelerator trains a small dense logistic-regression model on the FPGA fabric and exposes the design as a Vitis HLS IP block for Vivado/PYNQ integration.

The implementation follows the handout constraints:

- Machine-learning algorithm: logistic regression training by gradient descent.
- Hardware interface: AXI4-Burst memory transfers for `X`, `y`, and `weights`.
- Verification: Vitis HLS C simulation testbench checks the accelerator against both true-sigmoid and hardware-compatible approximate software references.
- System integration: Vivado block-design Tcl integrates the HLS IP with the Zynq Processing System.
- Deployment: PYNQ Python host benchmarks software-only training and hardware-accelerated training.

## Project Layout

```text
hls/
  include/lr_config.h          Shared problem-size constants
  src/lr_train_accel.cpp       Vitis HLS top function
  tb/tb_lr_train_accel.cpp     C simulation testbench
  run_hls.tcl                  Vitis HLS project recipe
  build_hls.ps1                PowerShell build wrapper

vivado/
  create_lr_burst_bd.tcl       Zynq PS + AXI burst accelerator block design
  build_vivado_project.ps1     Vivado project creation wrapper
  implement_lr_overlay.tcl     Synthesis, implementation, bit/hwh export
  export_pynq_artifacts.ps1    PowerShell implementation/export wrapper

pynq/
  lr_pynq_host.py              PYNQ runtime and software baseline

scripts/
  run_host_smoke_test.py       Local Python validation without board hardware
  preprocess_marketing_dataset.py Converts an external dataset CSV for accelerator use

data/
  marketing_campaign_representative.csv Larger representative marketing fixture
  marketing_campaign_sample.csv         Small CSV fixture for host-code testing
```

Generated outputs are intentionally kept out of Git:

```text
build/hls/lr_train_accel/
build/ip/lr_train_accel.zip
build/vivado/lr_train_accel/
artifacts/pynq/lr_train_accel.bit
artifacts/pynq/lr_train_accel.hwh
```

## How the HLS Project Works

`lr_train_accel` accepts a fixed window of up to 30 samples and 30 features, matching the example limit in the project handout. The matrix is flattened using a fixed stride of 30:

```text
X[i, j] -> x[i * 30 + j]
```

The first feature should be a bias column of `1.0`. The PYNQ host adds this bias column automatically.

The kernel:

1. Bursts `X`, `y`, and initial `weights` from DDR into local arrays.
2. Runs up to 200 gradient-descent iterations.
3. Computes the logistic prediction with a bounded piecewise-linear sigmoid approximation to avoid an expensive exponent unit in the FPGA fabric.
4. Accumulates gradients across all samples.
5. Writes the trained weights back to DDR.

The testbench and Python host keep this hardware approximation separate from the true logistic-regression reference:

- true software reference: `1.0 / (1.0 + exp(-z))`
- approximate software reference: same bounded piecewise-linear sigmoid used by the HLS kernel
- hardware output: compared directly against the approximate software reference

Build and export the HLS IP:

```powershell
cd D:\ECEN529\Lab4\Accelerated-ML
powershell -ExecutionPolicy Bypass -File hls\build_hls.ps1
```

For a quick C-simulation-only check:

```powershell
powershell -ExecutionPolicy Bypass -File hls\build_hls.ps1 -CsimOnly
```

The default target part is `xc7z020clg400-1`, which matches PYNQ-Z1/PYNQ-Z2 style boards. Override it if your lab board uses a different part:

```powershell
powershell -ExecutionPolicy Bypass -File hls\build_hls.ps1 -Part <your-part-name>
```

## How the Vivado Project Works

The Vivado Tcl creates a block design containing:

- Zynq Processing System 7
- The exported `lr_train_accel` HLS IP
- AXI-Lite control from `M_AXI_GP0`
- AXI HP DDR access from the accelerator `m_axi` ports
- External `FIXED_IO` and `DDR` ports for the PYNQ board

Create the Vivado project after HLS IP export:

```powershell
powershell -ExecutionPolicy Bypass -File vivado\build_vivado_project.ps1
```

Run synthesis, implementation, and export `.bit`/`.hwh` for PYNQ:

```powershell
powershell -ExecutionPolicy Bypass -File vivado\export_pynq_artifacts.ps1
```

The exported board files are written to:

```text
artifacts/pynq/lr_train_accel.bit
artifacts/pynq/lr_train_accel.hwh
```

## How the PYNQ Host Works

`pynq/lr_pynq_host.py` can run with a full dataset CSV, the committed representative sample, or an explicit synthetic demo. CSV files may use a label column named `label`, `target`, `response`, `converted`, `y`, or `outcome`; otherwise the final column is treated as the label. Numeric columns are selected, train/eval split is created, training statistics are used for normalization, a bias column is added, and the feature set is capped to 29 data columns plus the bias column.

Software-only validation on the workstation:

```powershell
python scripts\run_host_smoke_test.py
python pynq\lr_pynq_host.py --software-only --mode sample
```

Preprocess an external full Marketing Campaigns dataset:

```powershell
python scripts\preprocess_marketing_dataset.py path\to\marketing_campaign.csv --output data\marketing_campaign_preprocessed.csv
python pynq\lr_pynq_host.py --software-only --mode dataset --dataset data\marketing_campaign_preprocessed.csv
```

Run on the PYNQ board after copying the script, `.bit`, `.hwh`, and dataset to the same board directory:

```bash
python3 lr_pynq_host.py --overlay lr_train_accel.bit --mode sample --sample-dataset marketing_campaign_representative.csv
```

For a board or workstation run using deterministic synthetic data, request it explicitly:

```bash
python3 lr_pynq_host.py --software-only --mode synthetic
```

The host prints:

- sample count, feature count, and iteration count
- true-sigmoid software training time and accuracy
- approximate software training time and accuracy
- accelerator setup time and kernel time
- accelerator accuracy
- max and mean absolute differences for hardware-vs-approximate and approximate-vs-true weights
- pass/fail status

## Board Run Checklist

1. Copy these files to one directory on the PYNQ board:

   ```text
   artifacts/pynq/lr_train_accel.bit
   artifacts/pynq/lr_train_accel.hwh
   pynq/lr_pynq_host.py
   data/marketing_campaign_representative.csv
   ```

2. SSH into the board or open a Jupyter terminal.

3. Run:

   ```bash
   python3 lr_pynq_host.py --overlay lr_train_accel.bit --mode sample --sample-dataset marketing_campaign_representative.csv
   ```

4. Confirm `pass=True` for hardware-vs-approximate weights, then review the true-sigmoid comparison metrics separately.

## Validation Status

Local validation results are recorded in `reports/validation.md`. See `BUILD.md` for exact reproduction commands and `IMPLEMENTATION_EVIDENCE.md` for concise grading evidence. Board runtime validation requires a PYNQ board with matching `.bit` and `.hwh` files in the same directory as the host script.
