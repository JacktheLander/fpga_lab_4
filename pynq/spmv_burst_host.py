import argparse
import time

import numpy as np
from pynq import Overlay, allocate

from spmv_utils import csr_spmv, make_sparse_matrix, throughput_ops, write_arg, start_and_wait


KERNELS = {
    "s1": "spmv_burst_s1_0",
    "s2": "spmv_burst_s2_0",
    "s3": "spmv_burst_s3_0",
    "s4": "spmv_burst_s4_0",
}


def run_kernel(ip, csr, x, max_rows=1000, max_nnz=1000, max_size=1000):
    row_ptr = allocate(shape=(max_rows + 1,), dtype=np.int32)
    column_index = allocate(shape=(max_nnz,), dtype=np.int32)
    values = allocate(shape=(max_nnz,), dtype=np.int32)
    x_buf = allocate(shape=(max_size,), dtype=np.int32)
    y_buf = allocate(shape=(max_rows,), dtype=np.int32)

    setup_start = time.perf_counter()
    row_ptr[:] = 0
    column_index[:] = 0
    values[:] = 0
    x_buf[:] = 0
    y_buf[:] = 0
    row_ptr[: csr.num_rows + 1] = csr.row_ptr
    column_index[: csr.nnz] = csr.column_index
    values[: csr.nnz] = csr.values
    x_buf[: csr.size] = x

    write_arg(ip, "rowPtr", row_ptr.physical_address)
    write_arg(ip, "columnIndex", column_index.physical_address)
    write_arg(ip, "values", values.physical_address)
    write_arg(ip, "x", x_buf.physical_address)
    write_arg(ip, "y", y_buf.physical_address)
    write_arg(ip, "num_rows", csr.num_rows)
    write_arg(ip, "nnz", csr.nnz)
    write_arg(ip, "size", csr.size)
    setup_elapsed = time.perf_counter() - setup_start

    kernel_start = time.perf_counter()
    start_and_wait(ip)
    kernel_elapsed = time.perf_counter() - kernel_start
    result = np.asarray(y_buf[: csr.num_rows], dtype=np.int32).copy()
    total_elapsed = setup_elapsed + kernel_elapsed

    row_ptr.freebuffer()
    column_index.freebuffer()
    values.freebuffer()
    x_buf.freebuffer()
    y_buf.freebuffer()
    return result, setup_elapsed, kernel_elapsed, total_elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bitfile", default="spmv_burst.bit")
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--density", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=529)
    args = parser.parse_args()

    overlay = Overlay(args.bitfile)
    csr = make_sparse_matrix(size=args.size, density=args.density, seed=args.seed)
    x = np.arange(args.size, dtype=np.int32) % 17 - 8

    sw_start = time.perf_counter()
    expected = csr_spmv(csr, x)
    sw_elapsed = time.perf_counter() - sw_start
    print(f"software_time_s={sw_elapsed:.6e}")
    print(f"nnz={csr.nnz} operations={2 * csr.nnz}")

    for label, ip_name in KERNELS.items():
        try:
            ip = getattr(overlay, ip_name)
        except AttributeError:
            print(f"{label}: skipped, IP {ip_name!r} is not present in the overlay")
            continue
        result, setup_t, kernel_t, total_t = run_kernel(ip, csr, x)
        ok = np.array_equal(result, expected)
        print(
            f"{label}: pass={ok} setup_s={setup_t:.6e} kernel_s={kernel_t:.6e} "
            f"total_s={total_t:.6e} throughput_ops_s={throughput_ops(csr.nnz, total_t):.3f}"
        )


if __name__ == "__main__":
    main()
