# Validation Log

Validation performed on Wednesday, June 10, 2026 with AMD Vitis/Vivado 2025.2.

## Checks Run

- Python host smoke test:

  ```powershell
  python scripts\run_host_smoke_test.py
  ```

- Vitis HLS C simulation, synthesis, and IP export:

  ```powershell
  powershell -ExecutionPolicy Bypass -File hls\build_hls.ps1
  ```

- Vivado block-design generation:

  ```powershell
  powershell -ExecutionPolicy Bypass -File vivado\build_vivado_project.ps1
  ```

- Vivado implementation and PYNQ artifact export:

  ```powershell
  powershell -ExecutionPolicy Bypass -File vivado\export_pynq_artifacts.ps1
  ```

## Results

| Check | Result |
| --- | --- |
| Python host smoke test | Passed: synthetic dataset, 30 samples, 3 features, 0.9667 software accuracy |
| Python CSV software path | Passed: sample CSV, 12 samples, 5 features, 1.0000 software accuracy |
| Vitis HLS C simulation | Passed: C model matched reference, max absolute weight difference 0.000000 |
| Vitis HLS synthesis | Passed for `xc7z020clg400-1`; estimated clock 7.300 ns, estimated Fmax 136.99 MHz |
| Vitis HLS IP export | Passed: `build/ip/lr_train_accel.zip` generated |
| Vivado block design validation | Passed: Zynq PS, HLS IP, AXI-Lite control, AXI interconnect, HP0 DDR path |
| Vivado implementation and bitstream | Passed: routed WNS 6.861 ns, TNS 0.000 ns, 0 failing setup endpoints |
| PYNQ artifact export | Passed: `artifacts/pynq/lr_train_accel.bit` and `artifacts/pynq/lr_train_accel.hwh` generated |

The local workstation validated synthesis, implementation, timing, and artifact export. Board runtime was not executed in this session because no live PYNQ board session was available to the agent.
