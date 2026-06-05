#include <iostream>
#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "hls_stream.h"

#ifndef TOP_FUNC
#define TOP_FUNC spmv_stream_s1
#endif

typedef int DTYPE;
typedef ap_axiu<32, 0, 0, 0> axis_word_t;

extern "C" void TOP_FUNC(
    hls::stream<axis_word_t> &in_stream,
    hls::stream<axis_word_t> &out_stream,
    int num_rows,
    int nnz,
    int size);

static axis_word_t int_to_axis(int value, bool last) {
    axis_word_t word;
    word.data = (ap_uint<32>)value;
    word.keep = -1;
    word.strb = -1;
    word.last = last ? 1 : 0;
    return word;
}

static int axis_to_int(axis_word_t word) {
    ap_int<32> signed_word = word.data;
    return (int)signed_word;
}

static void reference_spmv(
    const int *rowPtr,
    const int *columnIndex,
    const DTYPE *values,
    const DTYPE *x,
    DTYPE *y,
    int num_rows) {
    for (int i = 0; i < num_rows; i++) {
        DTYPE acc = 0;
        for (int k = rowPtr[i]; k < rowPtr[i + 1]; k++) {
            acc += values[k] * x[columnIndex[k]];
        }
        y[i] = acc;
    }
}

int main() {
    const int size = 8;
    const int num_rows = 8;
    const int nnz = 13;
    int rowPtr[num_rows + 1] = {0, 2, 3, 5, 7, 8, 10, 12, 13};
    int columnIndex[nnz] = {0, 4, 1, 2, 6, 3, 7, 0, 2, 5, 1, 6, 4};
    DTYPE values[nnz] = {3, -1, 7, 2, 4, -3, 5, 6, 1, -2, 9, 3, 8};
    DTYPE x[size] = {2, -1, 4, 5, 3, -2, 1, 6};
    DTYPE y_hw[num_rows] = {0};
    DTYPE y_sw[num_rows] = {0};
    hls::stream<axis_word_t> in_stream;
    hls::stream<axis_word_t> out_stream;

    reference_spmv(rowPtr, columnIndex, values, x, y_sw, num_rows);

    for (int i = 0; i <= num_rows; i++) {
        in_stream.write(int_to_axis(rowPtr[i], false));
    }
    for (int i = 0; i < nnz; i++) {
        in_stream.write(int_to_axis(columnIndex[i], false));
    }
    for (int i = 0; i < nnz; i++) {
        in_stream.write(int_to_axis(values[i], false));
    }
    for (int i = 0; i < size; i++) {
        in_stream.write(int_to_axis(x[i], i == size - 1));
    }

    TOP_FUNC(in_stream, out_stream, num_rows, nnz, size);

    bool pass = true;
    for (int i = 0; i < num_rows; i++) {
        if (out_stream.empty()) {
            std::cout << "Missing output at row " << i << std::endl;
            pass = false;
            break;
        }
        axis_word_t word = out_stream.read();
        y_hw[i] = axis_to_int(word);
        if (y_hw[i] != y_sw[i]) {
            pass = false;
            std::cout << "Mismatch at row " << i << ": hw=" << y_hw[i]
                      << " sw=" << y_sw[i] << std::endl;
        }
        if ((i == num_rows - 1) && word.last != 1) {
            std::cout << "Missing TLAST on final output" << std::endl;
            pass = false;
        }
    }

    if (pass) {
        std::cout << "PASS" << std::endl;
        return 0;
    }
    std::cout << "FAIL" << std::endl;
    return 1;
}
