#ifndef LAB4_SPMV_COMMON_H
#define LAB4_SPMV_COMMON_H

#include <string.h>

#ifndef SPMV_MAX_SIZE
#define SPMV_MAX_SIZE 1000
#endif

#ifndef SPMV_MAX_ROWS
#define SPMV_MAX_ROWS 1000
#endif

#ifndef SPMV_MAX_NNZ
#define SPMV_MAX_NNZ 1000
#endif

#ifndef SPMV_UNROLL
#define SPMV_UNROLL 4
#endif

typedef int DTYPE;

static int clamp_to_capacity(int value, int capacity) {
    return value < capacity ? value : capacity;
}

static void spmv_strategy1_compute(
    const int rowPtr[SPMV_MAX_ROWS + 1],
    const int columnIndex[SPMV_MAX_NNZ],
    const DTYPE values[SPMV_MAX_NNZ],
    const DTYPE x[SPMV_MAX_SIZE],
    DTYPE y[SPMV_MAX_ROWS],
    int num_rows) {
L1_S1:
    for (int i = 0; i < num_rows; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = SPMV_MAX_ROWS
        DTYPE acc = 0;
        int row_start = rowPtr[i];
        int row_end = rowPtr[i + 1];
    L2_S1:
        for (int k = row_start; k < row_end; k++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 0 max = SPMV_MAX_NNZ
            acc += values[k] * x[columnIndex[k]];
        }
        y[i] = acc;
    }
}

static void spmv_strategy2_compute(
    const int rowPtr[SPMV_MAX_ROWS + 1],
    const int columnIndex[SPMV_MAX_NNZ],
    const DTYPE values[SPMV_MAX_NNZ],
    const DTYPE x[SPMV_MAX_SIZE],
    DTYPE y[SPMV_MAX_ROWS],
    int num_rows) {
    DTYPE products[SPMV_MAX_NNZ];
#pragma HLS BIND_STORAGE variable = products type = ram_2p impl = bram

L1_S2:
    for (int i = 0; i < num_rows; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = SPMV_MAX_ROWS
        int row_start = rowPtr[i];
        int row_end = rowPtr[i + 1];
    MUL_S2:
        for (int k = row_start; k < row_end; k++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 0 max = SPMV_MAX_NNZ
            products[k] = values[k] * x[columnIndex[k]];
        }

        DTYPE acc = 0;
    REDUCE_S2:
        for (int k = row_start; k < row_end; k++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 0 max = SPMV_MAX_NNZ
            acc += products[k];
        }
        y[i] = acc;
    }
}

static void spmv_strategy3_compute(
    const int rowPtr[SPMV_MAX_ROWS + 1],
    const int columnIndex[SPMV_MAX_NNZ],
    const DTYPE values[SPMV_MAX_NNZ],
    const DTYPE x[SPMV_MAX_SIZE],
    DTYPE y[SPMV_MAX_ROWS],
    int num_rows) {
L1_S3:
    for (int i = 0; i < num_rows; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = SPMV_MAX_ROWS
        DTYPE lanes[SPMV_UNROLL];
#pragma HLS ARRAY_PARTITION variable = lanes complete dim = 1
    INIT_S3:
        for (int u = 0; u < SPMV_UNROLL; u++) {
#pragma HLS UNROLL
            lanes[u] = 0;
        }

        int row_start = rowPtr[i];
        int row_end = rowPtr[i + 1];
    L2_S3:
        for (int k = row_start; k < row_end; k += SPMV_UNROLL) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 0 max = SPMV_MAX_NNZ
        LANE_S3:
            for (int u = 0; u < SPMV_UNROLL; u++) {
#pragma HLS UNROLL
                int idx = k + u;
                if (idx < row_end) {
                    lanes[u] += values[idx] * x[columnIndex[idx]];
                }
            }
        }

        DTYPE acc = 0;
    REDUCE_S3:
        for (int u = 0; u < SPMV_UNROLL; u++) {
#pragma HLS UNROLL
            acc += lanes[u];
        }
        y[i] = acc;
    }
}

static void spmv_strategy4_compute(
    const int rowPtr[SPMV_MAX_ROWS + 1],
    const int columnIndex[SPMV_MAX_NNZ],
    const DTYPE values[SPMV_MAX_NNZ],
    const DTYPE x[SPMV_MAX_SIZE],
    DTYPE y[SPMV_MAX_ROWS],
    int num_rows) {
L1_S4:
    for (int i = 0; i < num_rows; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = SPMV_MAX_ROWS
        int row_start = rowPtr[i];
        int row_end = rowPtr[i + 1];
        DTYPE acc = 0;

    L2_S4:
        for (int k = row_start; k < row_end; k += SPMV_UNROLL) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 0 max = SPMV_MAX_NNZ
            DTYPE partial[SPMV_UNROLL];
#pragma HLS ARRAY_PARTITION variable = partial complete dim = 1

        INIT_S4:
            for (int u = 0; u < SPMV_UNROLL; u++) {
#pragma HLS UNROLL
                partial[u] = 0;
            }

        LANE_S4:
            for (int u = 0; u < SPMV_UNROLL; u++) {
#pragma HLS UNROLL
                int idx = k + u;
                if (idx < row_end) {
                    partial[u] = values[idx] * x[columnIndex[idx]];
                }
            }

        REDUCE_S4:
            for (int u = 0; u < SPMV_UNROLL; u++) {
#pragma HLS UNROLL
                acc += partial[u];
            }
        }
        y[i] = acc;
    }
}

#endif
