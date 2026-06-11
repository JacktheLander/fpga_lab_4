#include "lr_config.h"

static float lr_sigmoid(float z) {
    if (z >= 4.0f) {
        return 1.0f;
    }
    if (z <= -4.0f) {
        return 0.0f;
    }
    return 0.5f + 0.125f * z;
}

extern "C" {

void lr_train_accel(
    const float *x,
    const float *y,
    float *weights,
    int n_samples,
    int n_features,
    float learning_rate,
    int n_iterations) {
#pragma HLS INTERFACE m_axi port=x offset=slave bundle=gmem0 depth=900 max_read_burst_length=64 num_read_outstanding=4
#pragma HLS INTERFACE m_axi port=y offset=slave bundle=gmem1 depth=30 max_read_burst_length=32 num_read_outstanding=2
#pragma HLS INTERFACE m_axi port=weights offset=slave bundle=gmem2 depth=30 max_read_burst_length=32 max_write_burst_length=32 num_read_outstanding=2 num_write_outstanding=2
#pragma HLS INTERFACE s_axilite port=x bundle=control
#pragma HLS INTERFACE s_axilite port=y bundle=control
#pragma HLS INTERFACE s_axilite port=weights bundle=control
#pragma HLS INTERFACE s_axilite port=n_samples bundle=control
#pragma HLS INTERFACE s_axilite port=n_features bundle=control
#pragma HLS INTERFACE s_axilite port=learning_rate bundle=control
#pragma HLS INTERFACE s_axilite port=n_iterations bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    float x_local[LR_MAX_SAMPLES][LR_MAX_FEATURES];
    float y_local[LR_MAX_SAMPLES];
    float w_local[LR_MAX_FEATURES];
    float gradients[LR_MAX_FEATURES];
#pragma HLS ARRAY_PARTITION variable=w_local complete dim=1
#pragma HLS ARRAY_PARTITION variable=y_local cyclic factor=6 dim=1
#pragma HLS ARRAY_PARTITION variable=gradients cyclic factor=6 dim=1

    if (n_samples <= 0 || n_features <= 0) {
    ZERO_INVALID_WEIGHTS:
        for (int j = 0; j < LR_MAX_FEATURES; ++j) {
#pragma HLS PIPELINE II=1
            weights[j] = 0.0f;
        }
        return;
    }

    int samples = n_samples;
    int features = n_features;
    int iterations = n_iterations;
    if (samples > LR_MAX_SAMPLES) {
        samples = LR_MAX_SAMPLES;
    }
    if (features > LR_MAX_FEATURES) {
        features = LR_MAX_FEATURES;
    }
    if (iterations < 0) {
        iterations = 0;
    }
    if (iterations > LR_MAX_ITERS) {
        iterations = LR_MAX_ITERS;
    }
    const float step_scale = (samples > 0) ? (learning_rate / static_cast<float>(samples)) : 0.0f;

LOAD_X_ROWS:
    for (int i = 0; i < LR_MAX_SAMPLES; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=30 max=30
    LOAD_X_COLS:
        for (int j = 0; j < LR_MAX_FEATURES; ++j) {
#pragma HLS PIPELINE II=1
            float value = 0.0f;
            if (i < samples && j < features) {
                value = x[i * LR_MAX_FEATURES + j];
            }
            x_local[i][j] = value;
        }
    }

LOAD_LABELS:
    for (int i = 0; i < LR_MAX_SAMPLES; ++i) {
#pragma HLS PIPELINE II=1
        y_local[i] = (i < samples) ? y[i] : 0.0f;
    }

LOAD_WEIGHTS:
    for (int j = 0; j < LR_MAX_FEATURES; ++j) {
#pragma HLS PIPELINE II=1
        w_local[j] = (j < features) ? weights[j] : 0.0f;
    }

TRAIN_ITERS:
    for (int iter = 0; iter < LR_MAX_ITERS; ++iter) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=200
        if (iter < iterations) {
        CLEAR_GRAD:
            for (int j = 0; j < LR_MAX_FEATURES; ++j) {
#pragma HLS PIPELINE II=1
                gradients[j] = 0.0f;
            }

        SAMPLE_LOOP:
            for (int i = 0; i < LR_MAX_SAMPLES; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=30
                if (i < samples) {
                    float dot = 0.0f;
                DOT_LOOP:
                    for (int j = 0; j < LR_MAX_FEATURES; ++j) {
#pragma HLS PIPELINE II=6
                        if (j < features) {
                            dot += x_local[i][j] * w_local[j];
                        }
                    }

                    float error = y_local[i] - lr_sigmoid(dot);
                GRAD_LOOP:
                    for (int j = 0; j < LR_MAX_FEATURES; ++j) {
#pragma HLS PIPELINE II=6
                        if (j < features) {
                            gradients[j] += x_local[i][j] * error;
                        }
                    }
                }
            }

        UPDATE_WEIGHTS:
            for (int j = 0; j < LR_MAX_FEATURES; ++j) {
#pragma HLS PIPELINE II=1
                if (j < features) {
                    w_local[j] += step_scale * gradients[j];
                }
            }
        }
    }

STORE_WEIGHTS:
    for (int j = 0; j < LR_MAX_FEATURES; ++j) {
#pragma HLS PIPELINE II=1
        weights[j] = w_local[j];
    }
}

}
