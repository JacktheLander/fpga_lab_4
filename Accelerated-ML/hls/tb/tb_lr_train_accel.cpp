#include "lr_config.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

extern "C" void lr_train_accel(
    const float *x,
    const float *y,
    float *weights,
    int n_samples,
    int n_features,
    float learning_rate,
    int n_iterations);

static float sigmoid_ref(float z) {
    if (z >= 4.0f) {
        return 1.0f;
    }
    if (z <= -4.0f) {
        return 0.0f;
    }
    return 0.5f + 0.125f * z;
}

static void train_reference(
    const float *x,
    const float *y,
    float *weights,
    int n_samples,
    int n_features,
    float learning_rate,
    int n_iterations) {
    for (int iter = 0; iter < n_iterations; ++iter) {
        float gradients[LR_MAX_FEATURES] = {0.0f};
        for (int i = 0; i < n_samples; ++i) {
            float dot = 0.0f;
            for (int j = 0; j < n_features; ++j) {
                dot += x[i * LR_MAX_FEATURES + j] * weights[j];
            }
            const float error = y[i] - sigmoid_ref(dot);
            for (int j = 0; j < n_features; ++j) {
                gradients[j] += x[i * LR_MAX_FEATURES + j] * error;
            }
        }
        for (int j = 0; j < n_features; ++j) {
            weights[j] += learning_rate * gradients[j] / static_cast<float>(n_samples);
        }
    }
}

static int count_correct(const float *x, const float *y, const float *weights, int n_samples, int n_features) {
    int correct = 0;
    for (int i = 0; i < n_samples; ++i) {
        float dot = 0.0f;
        for (int j = 0; j < n_features; ++j) {
            dot += x[i * LR_MAX_FEATURES + j] * weights[j];
        }
        const int pred = sigmoid_ref(dot) >= 0.5f ? 1 : 0;
        const int label = y[i] >= 0.5f ? 1 : 0;
        if (pred == label) {
            ++correct;
        }
    }
    return correct;
}

int main() {
    const int n_samples = 12;
    const int n_features = 3;
    const float learning_rate = 0.8f;
    const int n_iterations = 80;

    float x[LR_MAX_SAMPLES * LR_MAX_FEATURES] = {0.0f};
    float y[LR_MAX_SAMPLES] = {0.0f};
    float weights_hw[LR_MAX_FEATURES] = {0.0f};
    float weights_ref[LR_MAX_FEATURES] = {0.0f};

    const float raw[n_samples][2] = {
        {-2.0f, -1.0f}, {-1.5f, 0.5f}, {-1.0f, -1.0f}, {-0.5f, 1.5f},
        {0.2f, -0.5f}, {0.5f, 1.5f}, {1.0f, -1.0f}, {1.2f, 0.2f},
        {1.5f, -0.4f}, {1.8f, 1.0f}, {2.0f, -1.2f}, {2.2f, 0.8f},
    };

    for (int i = 0; i < n_samples; ++i) {
        const float score = -0.15f + 1.35f * raw[i][0] - 0.85f * raw[i][1];
        x[i * LR_MAX_FEATURES + 0] = 1.0f;
        x[i * LR_MAX_FEATURES + 1] = raw[i][0];
        x[i * LR_MAX_FEATURES + 2] = raw[i][1];
        y[i] = score >= 0.0f ? 1.0f : 0.0f;
    }

    lr_train_accel(x, y, weights_hw, n_samples, n_features, learning_rate, n_iterations);
    train_reference(x, y, weights_ref, n_samples, n_features, learning_rate, n_iterations);

    float max_abs_diff = 0.0f;
    for (int j = 0; j < n_features; ++j) {
        const float diff = std::fabs(weights_hw[j] - weights_ref[j]);
        if (diff > max_abs_diff) {
            max_abs_diff = diff;
        }
        std::printf("weight[%d] hw=%f ref=%f diff=%f\n", j, weights_hw[j], weights_ref[j], diff);
    }

    const int correct = count_correct(x, y, weights_hw, n_samples, n_features);
    std::printf("accuracy=%d/%d max_abs_diff=%f\n", correct, n_samples, max_abs_diff);

    if (max_abs_diff > 1.0e-4f) {
        std::fprintf(stderr, "FAILED: hardware C model differs from reference\n");
        return EXIT_FAILURE;
    }
    if (correct < 10) {
        std::fprintf(stderr, "FAILED: trained model accuracy is too low\n");
        return EXIT_FAILURE;
    }

    std::printf("PASSED\n");
    return EXIT_SUCCESS;
}
