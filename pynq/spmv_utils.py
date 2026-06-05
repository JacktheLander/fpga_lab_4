import time
from dataclasses import dataclass

import numpy as np


@dataclass
class CsrMatrix:
    row_ptr: np.ndarray
    column_index: np.ndarray
    values: np.ndarray
    dense: np.ndarray

    @property
    def size(self):
        return self.dense.shape[0]

    @property
    def num_rows(self):
        return self.dense.shape[0]

    @property
    def nnz(self):
        return int(self.values.shape[0])


def make_sparse_matrix(size=1000, density=0.001, seed=529, value_low=-8, value_high=9):
    rng = np.random.default_rng(seed)
    mask = rng.random((size, size)) < density
    if not mask.any():
        mask[rng.integers(0, size), rng.integers(0, size)] = True

    values = rng.integers(value_low, value_high, size=(size, size), dtype=np.int32)
    values[values == 0] = 1
    dense = np.where(mask, values, 0).astype(np.int32)
    return dense_to_csr(dense)


def dense_to_csr(dense):
    dense = np.asarray(dense, dtype=np.int32)
    row_ptr = [0]
    column_index = []
    values = []
    for row in dense:
        cols = np.nonzero(row)[0]
        column_index.extend(cols.astype(np.int32).tolist())
        values.extend(row[cols].astype(np.int32).tolist())
        row_ptr.append(len(values))

    return CsrMatrix(
        row_ptr=np.asarray(row_ptr, dtype=np.int32),
        column_index=np.asarray(column_index, dtype=np.int32),
        values=np.asarray(values, dtype=np.int32),
        dense=dense,
    )


def csr_spmv(csr, x):
    y = np.zeros((csr.num_rows,), dtype=np.int32)
    for i in range(csr.num_rows):
        acc = np.int32(0)
        for k in range(csr.row_ptr[i], csr.row_ptr[i + 1]):
            acc += csr.values[k] * x[csr.column_index[k]]
        y[i] = acc
    return y


def time_call(fn, repeats=10):
    best = None
    last = None
    for _ in range(repeats):
        start = time.perf_counter()
        last = fn()
        elapsed = time.perf_counter() - start
        if best is None or elapsed < best:
            best = elapsed
    return best, last


def split_csr_batches(csr, max_rows, max_nnz):
    batches = []
    start_row = 0
    while start_row < csr.num_rows:
        end_row = start_row
        start_nnz = int(csr.row_ptr[start_row])
        end_nnz = start_nnz

        while end_row < csr.num_rows and (end_row - start_row) < max_rows:
            next_nnz = int(csr.row_ptr[end_row + 1])
            if (next_nnz - start_nnz) > max_nnz:
                if end_row == start_row:
                    raise ValueError(
                        f"Row {start_row} has {next_nnz - start_nnz} entries, "
                        f"which exceeds max_nnz={max_nnz}"
                    )
                break
            end_row += 1
            end_nnz = next_nnz

        local_row_ptr = csr.row_ptr[start_row : end_row + 1].copy()
        local_row_ptr -= local_row_ptr[0]
        local_columns = csr.column_index[start_nnz:end_nnz].copy()
        local_values = csr.values[start_nnz:end_nnz].copy()
        batches.append((start_row, local_row_ptr, local_columns, local_values))
        start_row = end_row

    return batches


def throughput_ops(nnz, seconds):
    return (2.0 * nnz) / seconds if seconds > 0 else float("inf")


def _register_offset(ip, arg_name):
    lname = arg_name.lower()
    candidates = []
    for name, meta in getattr(ip, "registers", {}).items():
        low = name.lower()
        if low == lname or low.startswith(lname + "_") or low.startswith(lname + "["):
            offset = meta["address_offset"]
            if isinstance(offset, str):
                offset = int(offset, 0)
            candidates.append((name, int(offset)))
    if not candidates:
        available = ", ".join(getattr(ip, "registers", {}).keys())
        raise KeyError(f"Register {arg_name!r} not found. Available: {available}")
    candidates.sort(key=lambda item: item[1])
    return candidates[0][1]


def write_arg(ip, arg_name, value):
    ip.write(_register_offset(ip, arg_name), int(value))


def start_and_wait(ip):
    ip.write(0x00, 0x01)
    while (ip.read(0x00) & 0x02) == 0:
        pass
