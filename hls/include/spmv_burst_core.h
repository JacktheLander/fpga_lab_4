#ifndef LAB4_SPMV_BURST_CORE_H
#define LAB4_SPMV_BURST_CORE_H

#include "spmv_common.h"

#ifndef SPMV_BURST_FN
#error "SPMV_BURST_FN must be defined before including spmv_burst_core.h"
#endif

extern "C" void SPMV_BURST_FN(
    const int *rowPtr,
    const int *columnIndex,
    const DTYPE *values,
    const DTYPE *x,
    DTYPE *y,
    int num_rows,
    int nnz,
    int size) {
#pragma HLS INTERFACE m_axi port = rowPtr offset = slave bundle = gmem0 depth = SPMV_MAX_ROWS + 1 max_read_burst_length = 64
#pragma HLS INTERFACE m_axi port = columnIndex offset = slave bundle = gmem1 depth = SPMV_MAX_NNZ max_read_burst_length = 64
#pragma HLS INTERFACE m_axi port = values offset = slave bundle = gmem2 depth = SPMV_MAX_NNZ max_read_burst_length = 64
#pragma HLS INTERFACE m_axi port = x offset = slave bundle = gmem3 depth = SPMV_MAX_SIZE max_read_burst_length = 64
#pragma HLS INTERFACE m_axi port = y offset = slave bundle = gmem4 depth = SPMV_MAX_ROWS max_write_burst_length = 64
#pragma HLS INTERFACE s_axilite port = rowPtr bundle = control
#pragma HLS INTERFACE s_axilite port = columnIndex bundle = control
#pragma HLS INTERFACE s_axilite port = values bundle = control
#pragma HLS INTERFACE s_axilite port = x bundle = control
#pragma HLS INTERFACE s_axilite port = y bundle = control
#pragma HLS INTERFACE s_axilite port = num_rows bundle = control
#pragma HLS INTERFACE s_axilite port = nnz bundle = control
#pragma HLS INTERFACE s_axilite port = size bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    int rows = clamp_to_capacity(num_rows, SPMV_MAX_ROWS);
    int entries = clamp_to_capacity(nnz, SPMV_MAX_NNZ);
    int x_size = clamp_to_capacity(size, SPMV_MAX_SIZE);

    int localRowPtr[SPMV_MAX_ROWS + 1];
    int localColumnIndex[SPMV_MAX_NNZ];
    DTYPE localValues[SPMV_MAX_NNZ];
    DTYPE localX[SPMV_MAX_SIZE];
    DTYPE localY[SPMV_MAX_ROWS];

#if SPMV_STRATEGY == 4
#pragma HLS ARRAY_PARTITION variable = localColumnIndex cyclic factor = SPMV_UNROLL dim = 1
#pragma HLS ARRAY_PARTITION variable = localValues cyclic factor = SPMV_UNROLL dim = 1
#endif

COPY_ROW_PTR:
    for (int i = 0; i <= rows; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 2 max = SPMV_MAX_ROWS + 1
        int ptr = rowPtr[i];
        localRowPtr[i] = ptr < entries ? ptr : entries;
    }

COPY_COLS_VALUES:
    for (int i = 0; i < entries; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = SPMV_MAX_NNZ
        localColumnIndex[i] = columnIndex[i];
        localValues[i] = values[i];
    }

COPY_X:
    for (int i = 0; i < x_size; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = SPMV_MAX_SIZE
        localX[i] = x[i];
    }

#if SPMV_STRATEGY == 1
    spmv_strategy1_compute(localRowPtr, localColumnIndex, localValues, localX, localY, rows);
#elif SPMV_STRATEGY == 2
    spmv_strategy2_compute(localRowPtr, localColumnIndex, localValues, localX, localY, rows);
#elif SPMV_STRATEGY == 3
    spmv_strategy3_compute(localRowPtr, localColumnIndex, localValues, localX, localY, rows);
#elif SPMV_STRATEGY == 4
    spmv_strategy4_compute(localRowPtr, localColumnIndex, localValues, localX, localY, rows);
#else
#error "Unsupported SPMV_STRATEGY"
#endif

WRITE_Y:
    for (int i = 0; i < rows; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = SPMV_MAX_ROWS
        y[i] = localY[i];
    }
}

#endif
