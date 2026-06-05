import argparse
import time

import numpy as np
from pynq import Overlay, allocate

from spmv_utils import (
    csr_spmv,
    make_sparse_matrix,
    split_csr_batches,
    throughput_ops,
    write_arg,
    start_and_wait,
)


BATCH_KERNELS = {
    16: ("spmv_burst_s4_rows16_0", 16, 32),
    25: ("spmv_burst_s4_rows25_0", 25, 64),
    50: ("spmv_burst_s4_rows50_0", 50, 128),
}


def run_batched_burst(ip, csr, x, max_rows, max_nnz, max_size=50):
    row_ptr = allocate(shape=(max_rows + 1,), dtype=np.int32)
    column_index = allocate(shape=(max_nnz,), dtype=np.int32)
    values = allocate(shape=(max_nnz,), dtype=np.int32)
    x_buf = allocate(shape=(max_size,), dtype=np.int32)
    y_buf = allocate(shape=(max_rows,), dtype=np.int32)
    result = np.zeros((csr.num_rows,), dtype=np.int32)

    start = time.perf_counter()
    x_buf[:] = 0
    x_buf[: csr.size] = x
    batches = split_csr_batches(csr, max_rows=max_rows, max_nnz=max_nnz)

    write_arg(ip, "rowPtr", row_ptr.physical_address)
    write_arg(ip, "columnIndex", column_index.physical_address)
    write_arg(ip, "values", values.physical_address)
    write_arg(ip, "x", x_buf.physical_address)
    write_arg(ip, "y", y_buf.physical_address)
    write_arg(ip, "size", csr.size)

    for row_start, local_row_ptr, local_columns, local_values in batches:
        rows = local_row_ptr.shape[0] - 1
        nnz = local_values.shape[0]
        row_ptr[:] = 0
        column_index[:] = 0
        values[:] = 0
        y_buf[:] = 0
        row_ptr[: rows + 1] = local_row_ptr
        column_index[:nnz] = local_columns
        values[:nnz] = local_values

        write_arg(ip, "num_rows", rows)
        write_arg(ip, "nnz", nnz)
        start_and_wait(ip)
        result[row_start : row_start + rows] = np.asarray(y_buf[:rows], dtype=np.int32)

    elapsed = time.perf_counter() - start
    row_ptr.freebuffer()
    column_index.freebuffer()
    values.freebuffer()
    x_buf.freebuffer()
    y_buf.freebuffer()
    return result, elapsed, len(batches)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bitfile", default="spmv_burst_batch.bit")
    parser.add_argument("--size", type=int, default=50)
    parser.add_argument("--density", type=float, default=0.01)
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

    for local_rows, (ip_name, max_rows, max_nnz) in BATCH_KERNELS.items():
        try:
            ip = getattr(overlay, ip_name)
        except AttributeError:
            print(f"local_rows={local_rows}: skipped, IP {ip_name!r} is not present")
            continue
        result, elapsed, batch_count = run_batched_burst(ip, csr, x, max_rows, max_nnz)
        ok = np.array_equal(result, expected)
        print(
            f"local_rows={local_rows}: pass={ok} batches={batch_count} "
            f"total_s={elapsed:.6e} throughput_ops_s={throughput_ops(csr.nnz, elapsed):.3f}"
        )


if __name__ == "__main__":
    main()
