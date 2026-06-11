# Validation Log

Validation performed on Wednesday, June 10, 2026 with AMD Vitis/Vivado 2025.2.

## Commands Run

```powershell
python scripts\run_host_smoke_test.py
python pynq\lr_pynq_host.py --software-only --mode sample
python pynq\lr_pynq_host.py --software-only --mode dataset --dataset data\marketing_campaign_preprocessed.csv
powershell -ExecutionPolicy Bypass -File hls\build_hls.ps1
powershell -ExecutionPolicy Bypass -File vivado\build_vivado_project.ps1
powershell -ExecutionPolicy Bypass -File vivado\export_pynq_artifacts.ps1
```

## Python Software Validation

| Check | Result |
| --- | --- |
| Host smoke test | Passed: synthetic true accuracy 1.0000, synthetic approximate accuracy 1.0000, synthetic max true-vs-approx weight difference 6.651163e-01 |
| Host smoke test, representative CSV | Passed: sample true accuracy 1.0000, sample approximate accuracy 1.0000, sample max true-vs-approx weight difference 3.547387e-01 |
| Host sample mode | Passed: `data\marketing_campaign_representative.csv`, 28 train / 12 eval samples, 7 features, true accuracy 1.0000, approximate accuracy 1.0000 |
| Host preprocessed dataset mode | Passed: `data\marketing_campaign_preprocessed.csv`, 28 train / 12 eval samples, 7 features, true accuracy 1.0000, approximate accuracy 1.0000 |

The sample and preprocessed dataset host runs reported an approximate-vs-true max absolute weight difference of 4.749095e-01 and mean absolute weight difference of 1.023968e-01 for the configured run.

## HLS C Simulation

The HLS testbench ran multiple cases and compared the HLS kernel against the approximate software model while also reporting the true-sigmoid reference accuracy.

| Case | Shape | Iterations | Learning rate | HLS vs approximate max / mean abs diff | Accuracy: HLS / approximate / true |
| --- | --- | ---: | ---: | --- | --- |
| Tiny synthetic sanity | 4 samples x 2 features | 40 | 0.6000 | 0.00000000 / 0.00000000 | 1.0000 / 1.0000 / 1.0000 |
| Linearly separable | 12 samples x 3 features | 80 | 0.8000 | 0.00000000 / 0.00000000 | 1.0000 / 1.0000 / 1.0000 |
| Zero initial edge | 6 samples x 3 features | 1 | 0.4000 | 0.00000000 / 0.00000000 | 0.8333 / 0.8333 / 0.8333 |
| Near max shape | 30 samples x 30 features | 40 | 0.2500 | 0.00000000 / 0.00000000 | 0.9333 / 0.9333 / 1.0000 |
| Marketing sample | 30 samples x 7 features | 80 | 0.3500 | 0.00000000 / 0.00000000 | 1.0000 / 1.0000 / 1.0000 |
| Invalid zero samples | 0 samples x 3 features | N/A | N/A | N/A | Passed deterministic zero-weight behavior |

CSim completed with `0` errors and printed `PASSED: all logistic-regression HLS tests`.

## Synthesis And Implementation

| Check | Result |
| --- | --- |
| Vitis HLS synthesis | Passed for `xc7z020-clg400-1`; target clock 10.00 ns, estimated clock 7.300 ns, estimated Fmax 136.99 MHz |
| HLS utilization estimate | BRAM_18K 3%, DSP 2%, FF 7%, LUT 15%, URAM 0% |
| Vitis HLS IP export | Passed: `build/ip/lr_train_accel.zip` generated |
| Vivado block design validation | Passed: Zynq PS, HLS IP, AXI-Lite control, AXI interconnect, HP0 DDR path |
| Vivado implementation and bitstream | Passed: routed WNS 8.706 ns, TNS 0.000 ns, 0 failing setup endpoints; WHS 0.022 ns, THS 0.000 ns, 0 failing hold endpoints |
| PYNQ artifact export | Passed locally: `artifacts/pynq/lr_train_accel.bit` and `artifacts/pynq/lr_train_accel.hwh` were produced by the export script |

The local workstation validated software references, HLS C simulation, HLS synthesis/export, Vivado block design generation, implementation timing, and PYNQ artifact export. Physical PYNQ board runtime was not executed in this session because no live board session was available to the agent.
