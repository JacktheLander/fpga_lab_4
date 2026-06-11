# Build and Validation Guide

Run all commands from:

```powershell
cd D:\ECEN529\Lab4\Accelerated-ML
```

## Software-Only Validation

Host smoke test using synthetic and representative dataset paths:

```powershell
python scripts\run_host_smoke_test.py
```

Dataset-backed host validation without PYNQ:

```powershell
python pynq\lr_pynq_host.py --software-only --mode sample
```

Explicit synthetic demo mode:

```powershell
python pynq\lr_pynq_host.py --software-only --mode synthetic --sample-count 30 --feature-count 7
```

Preprocess a full Marketing Campaigns dataset CSV:

```powershell
python scripts\preprocess_marketing_dataset.py path\to\marketing_campaign.csv --output data\marketing_campaign_preprocessed.csv
python pynq\lr_pynq_host.py --software-only --mode dataset --dataset data\marketing_campaign_preprocessed.csv
```

## Vitis HLS

Run C simulation only:

```powershell
powershell -ExecutionPolicy Bypass -File hls\build_hls.ps1 -CsimOnly
```

Run C simulation, synthesis, and IP export:

```powershell
powershell -ExecutionPolicy Bypass -File hls\build_hls.ps1
```

Expected IP export after synthesis:

```text
build/ip/lr_train_accel.zip
build/hls/lr_train_accel/solution1/impl/ip/component.xml
```

## Vivado

Create and validate the block-design project:

```powershell
powershell -ExecutionPolicy Bypass -File vivado\build_vivado_project.ps1
```

Run synthesis, implementation, bitstream generation, and PYNQ artifact export:

```powershell
powershell -ExecutionPolicy Bypass -File vivado\export_pynq_artifacts.ps1
```

Expected generated artifacts:

```text
artifacts/pynq/lr_train_accel.bit
artifacts/pynq/lr_train_accel.hwh
```

These generated outputs are ignored by Git.

## PYNQ Board Validation

Copy the bitstream, HWH, host script, and dataset to one folder on the board:

```text
lr_train_accel.bit
lr_train_accel.hwh
lr_pynq_host.py
marketing_campaign_representative.csv
```

Run:

```bash
python3 lr_pynq_host.py --overlay lr_train_accel.bit --mode sample --sample-dataset marketing_campaign_representative.csv
```

The host validates hardware weights against the approximate software model and reports true-sigmoid software results separately.
