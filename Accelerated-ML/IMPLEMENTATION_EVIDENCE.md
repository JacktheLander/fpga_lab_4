# Implementation Evidence

## HLS Kernel

- File: `hls/src/lr_train_accel.cpp`
- Top function: `lr_train_accel`
- Algorithm: dense logistic-regression training by gradient descent.
- Interface: AXI burst-style `m_axi` memory ports for `x`, `y`, and `weights`; AXI-Lite control for pointers, dimensions, learning rate, iteration count, and return.
- Compile-time bounds: `LR_MAX_SAMPLES=30`, `LR_MAX_FEATURES=30`, `LR_MAX_ITERS=200` in `hls/include/lr_config.h`.
- Defensive runtime behavior:
  - `n_samples <= 0` or `n_features <= 0`: writes deterministic zero weights and returns.
  - `n_samples`, `n_features`, and `n_iterations` above compile-time limits are clamped.
  - negative iteration counts are treated as zero iterations.
- HLS pragmas include AXI interface pragmas, pipelined load/store/update loops, and partitioning of small local arrays.
- The synthesizable kernel uses a bounded piecewise-linear sigmoid approximation to avoid an exponent unit.

## Verification

- File: `hls/tb/tb_lr_train_accel.cpp`
- References implemented in code:
  - true logistic sigmoid: `1.0 / (1.0 + exp(-z))`
  - approximate hardware-compatible sigmoid
  - HLS kernel output
- Test cases:
  - tiny synthetic sanity case
  - linearly separable synthetic case
  - zero-initial-weights edge case
  - near-max 30 sample x 30 feature shape
  - marketing-campaign representative sample
  - invalid runtime argument case
- The testbench reports max/mean absolute weight differences and classification accuracy, and returns nonzero on failed thresholds.

## Dataset and Preprocessing

- Representative sample: `data/marketing_campaign_representative.csv`
- Smaller fixture retained: `data/marketing_campaign_sample.csv`
- Preprocessing script: `scripts/preprocess_marketing_dataset.py`
- Host dataset logic:
  - selects numeric features
  - extracts label from common label-column names or the final column
  - creates train/eval split
  - normalizes features using train statistics
  - adds a bias column
  - clips/truncates to accelerator limits with warnings

The full external Marketing Campaigns dataset is not committed. Use the preprocessing script to convert a downloaded full CSV into accelerator-compatible form.

## Vivado Automation

- `vivado/create_lr_burst_bd.tcl`: creates Zynq PS, HLS IP, AXI-Lite control path, AXI interconnect, HP0 DDR path, address assignment, validation, and wrapper.
- `vivado/build_vivado_project.ps1`: checks Vivado version and HLS IP presence, then runs block-design creation.
- `vivado/implement_lr_overlay.tcl`: runs synthesis, implementation, bitstream generation, and `.bit`/`.hwh` copy.
- `vivado/export_pynq_artifacts.ps1`: wrapper with Vivado version check, project existence check, and output artifact checks.

## Python Host

- File: `pynq/lr_pynq_host.py`
- Modes:
  - `--mode auto`: prefer provided full dataset, then committed sample, then synthetic fallback.
  - `--mode dataset`: use full or preprocessed CSV path.
  - `--mode sample`: use committed representative sample.
  - `--mode synthetic`: deterministic demo data.
- Software-only mode runs on a normal workstation without PYNQ.
- PYNQ path validates `.bit` and `.hwh` files, allocates buffers, flushes inputs, invalidates output weights, and compares hardware against the approximate software reference.

## Known Limitations

- The HLS kernel uses an approximate sigmoid for hardware compatibility; the true sigmoid is used only in software references and comparisons.
- The accelerator trains at most 30 samples and 30 total features, including bias.
- Physical board execution and video demonstration are not included here.
- No hardware runtime numbers should be claimed until the PYNQ host has been run on the target board.

## Reproduce

See `BUILD.md` for exact commands. At minimum, run:

```powershell
python scripts\run_host_smoke_test.py
powershell -ExecutionPolicy Bypass -File hls\build_hls.ps1 -CsimOnly
```
