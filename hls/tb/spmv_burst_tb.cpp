#include <iostream>

#ifndef TOP_FUNC
#define TOP_FUNC spmv_burst_s1
#endif

typedef int DTYPE;

extern "C" void TOP_FUNC(
    const int *rowPtr,
    const int *columnIndex,
    const DTYPE *values,
    const DTYPE *x,
    DTYPE *y,
    int num_rows,
    int nnz,
    int size);

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

    reference_spmv(rowPtr, columnIndex, values, x, y_sw, num_rows);
    TOP_FUNC(rowPtr, columnIndex, values, x, y_hw, num_rows, nnz, size);

    bool pass = true;
    for (int i = 0; i < num_rows; i++) {
        if (y_hw[i] != y_sw[i]) {
            pass = false;
            std::cout << "Mismatch at row " << i << ": hw=" << y_hw[i]
                      << " sw=" << y_sw[i] << std::endl;
        }
    }

    if (pass) {
        std::cout << "PASS" << std::endl;
        return 0;
    }
    std::cout << "FAIL" << std::endl;
    return 1;
}
