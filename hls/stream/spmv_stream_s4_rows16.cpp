#define SPMV_STREAM_FN spmv_stream_s4_rows16
#define SPMV_STRATEGY 4
#define SPMV_MAX_ROWS 16
#define SPMV_MAX_NNZ 32
#define SPMV_MAX_SIZE 50
#define SPMV_UNROLL 4
#include "../include/spmv_stream_core.h"
