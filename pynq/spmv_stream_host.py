import argparse
import time

import numpy as np
from pynq import Overlay, allocate

from spmv_utils import csr_spmv, make_sparse_matrix, throughput_ops, write_arg


KERNELS = {
    "s1": "spmv_stream_s1_0",
    "s2": "spmv_stream_s2_0",
    "s3": "spmv_stream_s3_0",
    "s4": "spmv_stream_s4_0",
}


def pack_stream_input(csr, x):
    return np.concatenate(
        [
            csr.row_ptr.astype(np.int32),
            csr.column_index.astype(np.int32),
            csr.values.astype(np.int32),
            x.astype(np.int32),
        ]
    )


def run_kernel(ip, dma, csr, x):
    input_words = pack_stream_input(csr, x)
    in_buf = allocate(shape=(input_words.shape[0],), dtype=np.int32)
    out_buf = allocate(shape=(csr.num_rows,), dtype=np.int32)

    setup_start = time.perf_counter()
    np.copyto(in_buf, input_words)
    out_buf[:] = 0
    write_arg(ip, "num_rows", csr.num_rows)
    write_arg(ip, "nnz", csr.nnz)
    write_arg(ip, "size", csr.size)
    setup_elapsed = time.perf_counter() - setup_start

    kernel_start = time.perf_counter()
    dma.recvchannel.transfer(out_buf)
    ip.write(0x00, 0x01)
    dma.sendchannel.transfer(in_buf)
    dma.sendchannel.wait()
    dma.recvchannel.wait()
    kernel_elapsed = time.perf_counter() - kernel_start

    result = np.asarray(out_buf, dtype=np.int32).copy()
    total_elapsed = setup_elapsed + kernel_elapsed
    in_buf.freebuffer()
    out_buf.freebuffer()
    return result, setup_elapsed, kernel_elapsed, total_elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bitfile", default="spmv_stream.bit")
    parser.add_argument("--dma", default="axi_dma_0")
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--density", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=529)
    args = parser.parse_args()

    overlay = Overlay(args.bitfile)
    dma = getattr(overlay, args.dma)
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
        result, setup_t, kernel_t, total_t = run_kernel(ip, dma, csr, x)
        ok = np.array_equal(result, expected)
        print(
            f"{label}: pass={ok} setup_s={setup_t:.6e} kernel_s={kernel_t:.6e} "
            f"total_s={total_t:.6e} throughput_ops_s={throughput_ops(csr.nnz, total_t):.3f}"
        )


if __name__ == "__main__":
    main()
