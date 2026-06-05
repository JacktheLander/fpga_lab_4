#define SPMV_BURST_FN spmv_burst_s4_rows50
#define SPMV_STRATEGY 4
#define SPMV_MAX_ROWS 50
#define SPMV_MAX_NNZ 128
#define SPMV_MAX_SIZE 50
#define SPMV_UNROLL 4
#include "../include/spmv_burst_core.h"
