#ifndef LAB4_SPMV_STREAM_CORE_H
#define LAB4_SPMV_STREAM_CORE_H

#include "spmv_common.h"
#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "hls_stream.h"

typedef ap_axiu<32, 0, 0, 0> axis_word_t;

#ifndef SPMV_STREAM_FN
#error "SPMV_STREAM_FN must be defined before including spmv_stream_core.h"
#endif

static int axis_to_int(axis_word_t word) {
    ap_int<32> signed_word = word.data;
    return (int)signed_word;
}

static axis_word_t int_to_axis(int value, bool last) {
    axis_word_t word;
    word.data = (ap_uint<32>)value;
    word.keep = -1;
    word.strb = -1;
    word.last = last ? 1 : 0;
    return word;
}

extern "C" void SPMV_STREAM_FN(
    hls::stream<axis_word_t> &in_stream,
    hls::stream<axis_word_t> &out_stream,
    int num_rows,
    int nnz,
    int size) {
#pragma HLS INTERFACE axis port = in_stream
#pragma HLS INTERFACE axis port = out_stream
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

READ_ROW_PTR:
    for (int i = 0; i <= rows; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 2 max = SPMV_MAX_ROWS + 1
        int ptr = axis_to_int(in_stream.read());
        localRowPtr[i] = ptr < entries ? ptr : entries;
    }

READ_COLS:
    for (int i = 0; i < entries; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = SPMV_MAX_NNZ
        localColumnIndex[i] = axis_to_int(in_stream.read());
    }

READ_VALUES:
    for (int i = 0; i < entries; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = SPMV_MAX_NNZ
        localValues[i] = axis_to_int(in_stream.read());
    }

READ_X:
    for (int i = 0; i < x_size; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = SPMV_MAX_SIZE
        localX[i] = axis_to_int(in_stream.read());
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

WRITE_STREAM:
    for (int i = 0; i < rows; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = SPMV_MAX_ROWS
        out_stream.write(int_to_axis(localY[i], i == rows - 1));
    }
}

#endif
