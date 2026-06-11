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

enum SigmoidKind {
    SIGMOID_APPROX = 0,
    SIGMOID_TRUE = 1
};

struct WeightDiff {
    float max_abs;
    float mean_abs;
};

struct TestConfig {
    const char *name;
    int n_samples;
    int n_features;
    int n_iterations;
    float learning_rate;
    float min_accuracy;
    float max_hw_approx_diff;
    float max_hw_approx_mean_diff;
    float max_approx_true_diff;
};

static float sigmoid_approx(float z) {
    if (z >= 4.0f) {
        return 1.0f;
    }
    if (z <= -4.0f) {
        return 0.0f;
    }
    return 0.5f + 0.125f * z;
}

static float sigmoid_true(float z) {
    return 1.0f / (1.0f + std::exp(-z));
}

static float sigmoid_eval(float z, SigmoidKind kind) {
    return kind == SIGMOID_TRUE ? sigmoid_true(z) : sigmoid_approx(z);
}

static void clear_arrays(float *x, float *y, float *weights) {
    for (int i = 0; i < LR_MAX_SAMPLES * LR_MAX_FEATURES; ++i) {
        x[i] = 0.0f;
    }
    for (int i = 0; i < LR_MAX_SAMPLES; ++i) {
        y[i] = 0.0f;
    }
    for (int j = 0; j < LR_MAX_FEATURES; ++j) {
        weights[j] = 0.0f;
    }
}

static void set_x(float *x, int row, int col, float value) {
    x[row * LR_MAX_FEATURES + col] = value;
}

static void train_reference(
    const float *x,
    const float *y,
    float *weights,
    int n_samples,
    int n_features,
    float learning_rate,
    int n_iterations,
    SigmoidKind sigmoid_kind) {
    int samples = n_samples;
    int features = n_features;
    int iterations = n_iterations;
    if (samples <= 0 || features <= 0) {
        for (int j = 0; j < LR_MAX_FEATURES; ++j) {
            weights[j] = 0.0f;
        }
        return;
    }
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

    const float step_scale = learning_rate / static_cast<float>(samples);
    for (int iter = 0; iter < iterations; ++iter) {
        float gradients[LR_MAX_FEATURES] = {0.0f};
        for (int i = 0; i < samples; ++i) {
            float dot = 0.0f;
            for (int j = 0; j < features; ++j) {
                dot += x[i * LR_MAX_FEATURES + j] * weights[j];
            }
            const float error = y[i] - sigmoid_eval(dot, sigmoid_kind);
            for (int j = 0; j < features; ++j) {
                gradients[j] += x[i * LR_MAX_FEATURES + j] * error;
            }
        }
        for (int j = 0; j < features; ++j) {
            weights[j] += step_scale * gradients[j];
        }
    }
}

static float classification_accuracy(
    const float *x,
    const float *y,
    const float *weights,
    int n_samples,
    int n_features,
    SigmoidKind sigmoid_kind) {
    if (n_samples <= 0 || n_features <= 0) {
        return 0.0f;
    }
    int correct = 0;
    for (int i = 0; i < n_samples; ++i) {
        float dot = 0.0f;
        for (int j = 0; j < n_features; ++j) {
            dot += x[i * LR_MAX_FEATURES + j] * weights[j];
        }
        const int pred = sigmoid_eval(dot, sigmoid_kind) >= 0.5f ? 1 : 0;
        const int label = y[i] >= 0.5f ? 1 : 0;
        if (pred == label) {
            ++correct;
        }
    }
    return static_cast<float>(correct) / static_cast<float>(n_samples);
}

static WeightDiff compare_weights(const float *a, const float *b, int n_features) {
    WeightDiff diff = {0.0f, 0.0f};
    if (n_features <= 0) {
        return diff;
    }
    for (int j = 0; j < n_features; ++j) {
        const float abs_diff = std::fabs(a[j] - b[j]);
        if (abs_diff > diff.max_abs) {
            diff.max_abs = abs_diff;
        }
        diff.mean_abs += abs_diff;
    }
    diff.mean_abs /= static_cast<float>(n_features);
    return diff;
}

static void make_tiny_case(float *x, float *y) {
    set_x(x, 0, 0, 1.0f); set_x(x, 0, 1, -1.0f); y[0] = 0.0f;
    set_x(x, 1, 0, 1.0f); set_x(x, 1, 1, -0.2f); y[1] = 0.0f;
    set_x(x, 2, 0, 1.0f); set_x(x, 2, 1, 0.3f); y[2] = 1.0f;
    set_x(x, 3, 0, 1.0f); set_x(x, 3, 1, 1.0f); y[3] = 1.0f;
}

static void make_separable_case(float *x, float *y) {
    const float raw[12][2] = {
        {-2.0f, -1.0f}, {-1.5f, 0.5f}, {-1.0f, -1.0f}, {-0.5f, 1.5f},
        {0.2f, -0.5f}, {0.5f, 1.5f}, {1.0f, -1.0f}, {1.2f, 0.2f},
        {1.5f, -0.4f}, {1.8f, 1.0f}, {2.0f, -1.2f}, {2.2f, 0.8f},
    };
    for (int i = 0; i < 12; ++i) {
        const float score = -0.15f + 1.35f * raw[i][0] - 0.85f * raw[i][1];
        set_x(x, i, 0, 1.0f);
        set_x(x, i, 1, raw[i][0]);
        set_x(x, i, 2, raw[i][1]);
        y[i] = score >= 0.0f ? 1.0f : 0.0f;
    }
}

static void make_zero_initial_edge_case(float *x, float *y) {
    const float raw[6][2] = {
        {-0.3f, 0.9f}, {-0.2f, -0.4f}, {0.0f, 0.0f},
        {0.2f, 0.3f}, {0.4f, -0.1f}, {0.5f, 0.6f},
    };
    for (int i = 0; i < 6; ++i) {
        set_x(x, i, 0, 1.0f);
        set_x(x, i, 1, raw[i][0]);
        set_x(x, i, 2, raw[i][1]);
        y[i] = i >= 3 ? 1.0f : 0.0f;
    }
}

static void make_near_max_case(float *x, float *y) {
    for (int i = 0; i < LR_MAX_SAMPLES; ++i) {
        set_x(x, i, 0, 1.0f);
        for (int j = 1; j < LR_MAX_FEATURES; ++j) {
            const int pattern = ((i + 3) * (j + 5) + 7 * j) % 17;
            set_x(x, i, j, (static_cast<float>(pattern) - 8.0f) / 8.0f);
        }
        const float score =
            -0.10f +
            1.10f * x[i * LR_MAX_FEATURES + 1] -
            0.80f * x[i * LR_MAX_FEATURES + 2] +
            0.55f * x[i * LR_MAX_FEATURES + 3] +
            0.30f * x[i * LR_MAX_FEATURES + 4];
        y[i] = score >= 0.0f ? 1.0f : 0.0f;
    }
}

static void make_marketing_sample_case(float *x, float *y) {
    const int rows = 30;
    const int raw_features = 6;
    const float raw[rows][raw_features] = {
        {23, 42000, 7, 1, 1, 0}, {31, 61000, 4, 2, 2, 0}, {35, 72000, 5, 3, 3, 1},
        {42, 88000, 2, 5, 5, 1}, {27, 45000, 8, 1, 1, 0}, {51, 110000, 1, 6, 7, 1},
        {46, 98000, 3, 5, 5, 1}, {29, 52000, 7, 2, 2, 0}, {39, 77000, 4, 4, 4, 1},
        {33, 58000, 6, 2, 2, 0}, {55, 125000, 1, 7, 8, 1}, {24, 39000, 9, 1, 0, 0},
        {48, 101000, 2, 5, 6, 1}, {37, 69000, 5, 3, 3, 0}, {44, 91000, 3, 4, 5, 1},
        {28, 48000, 8, 1, 1, 0}, {52, 118000, 2, 6, 7, 1}, {30, 56000, 6, 2, 2, 0},
        {41, 83000, 4, 4, 4, 1}, {26, 43000, 9, 1, 1, 0}, {57, 132000, 1, 8, 9, 1},
        {34, 64000, 5, 3, 3, 0}, {45, 95000, 3, 5, 6, 1}, {32, 59000, 6, 2, 2, 0},
        {49, 104000, 2, 6, 6, 1}, {36, 71000, 5, 3, 4, 1}, {25, 41000, 8, 1, 1, 0},
        {53, 121000, 2, 7, 8, 1}, {38, 75000, 4, 4, 4, 1}, {29, 50000, 7, 2, 2, 0},
    };
    const float labels[rows] = {
        0, 0, 1, 1, 0, 1, 1, 0, 1, 0,
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
        1, 0, 1, 0, 1, 1, 0, 1, 1, 0,
    };

    float mean[raw_features] = {0.0f};
    float stddev[raw_features] = {0.0f};
    for (int j = 0; j < raw_features; ++j) {
        for (int i = 0; i < rows; ++i) {
            mean[j] += raw[i][j];
        }
        mean[j] /= static_cast<float>(rows);
        for (int i = 0; i < rows; ++i) {
            const float centered = raw[i][j] - mean[j];
            stddev[j] += centered * centered;
        }
        stddev[j] = std::sqrt(stddev[j] / static_cast<float>(rows));
        if (stddev[j] < 1.0e-6f) {
            stddev[j] = 1.0f;
        }
    }

    for (int i = 0; i < rows; ++i) {
        set_x(x, i, 0, 1.0f);
        for (int j = 0; j < raw_features; ++j) {
            set_x(x, i, j + 1, (raw[i][j] - mean[j]) / stddev[j]);
        }
        y[i] = labels[i];
    }
}

static bool run_case(const TestConfig &cfg, void (*make_case)(float *, float *)) {
    float x[LR_MAX_SAMPLES * LR_MAX_FEATURES];
    float y[LR_MAX_SAMPLES];
    float initial_weights[LR_MAX_FEATURES];
    float weights_hw[LR_MAX_FEATURES];
    float weights_approx[LR_MAX_FEATURES];
    float weights_true[LR_MAX_FEATURES];
    clear_arrays(x, y, initial_weights);
    make_case(x, y);

    for (int j = 0; j < LR_MAX_FEATURES; ++j) {
        weights_hw[j] = initial_weights[j];
        weights_approx[j] = initial_weights[j];
        weights_true[j] = initial_weights[j];
    }

    lr_train_accel(
        x, y, weights_hw, cfg.n_samples, cfg.n_features,
        cfg.learning_rate, cfg.n_iterations);
    train_reference(
        x, y, weights_approx, cfg.n_samples, cfg.n_features,
        cfg.learning_rate, cfg.n_iterations, SIGMOID_APPROX);
    train_reference(
        x, y, weights_true, cfg.n_samples, cfg.n_features,
        cfg.learning_rate, cfg.n_iterations, SIGMOID_TRUE);

    const WeightDiff hw_vs_approx = compare_weights(weights_hw, weights_approx, cfg.n_features);
    const WeightDiff approx_vs_true = compare_weights(weights_approx, weights_true, cfg.n_features);
    const float hw_acc = classification_accuracy(x, y, weights_hw, cfg.n_samples, cfg.n_features, SIGMOID_APPROX);
    const float approx_acc = classification_accuracy(x, y, weights_approx, cfg.n_samples, cfg.n_features, SIGMOID_APPROX);
    const float true_acc = classification_accuracy(x, y, weights_true, cfg.n_samples, cfg.n_features, SIGMOID_TRUE);

    std::printf(
        "CASE %-28s samples=%d features=%d iterations=%d lr=%.4f\n",
        cfg.name, cfg.n_samples, cfg.n_features, cfg.n_iterations, cfg.learning_rate);
    std::printf(
        "  hw_vs_approx: max_abs=%.8f mean_abs=%.8f\n",
        hw_vs_approx.max_abs, hw_vs_approx.mean_abs);
    std::printf(
        "  approx_vs_true: max_abs=%.8f mean_abs=%.8f\n",
        approx_vs_true.max_abs, approx_vs_true.mean_abs);
    std::printf(
        "  accuracy: hw_approx=%.4f approx_sw=%.4f true_sw=%.4f\n",
        hw_acc, approx_acc, true_acc);

    bool pass = true;
    if (hw_vs_approx.max_abs > cfg.max_hw_approx_diff ||
        hw_vs_approx.mean_abs > cfg.max_hw_approx_mean_diff) {
        std::fprintf(stderr, "FAILED %s: HLS kernel diverged from approximate software model\n", cfg.name);
        pass = false;
    }
    if (approx_vs_true.max_abs > cfg.max_approx_true_diff) {
        std::fprintf(stderr, "FAILED %s: approximate model drift exceeded threshold\n", cfg.name);
        pass = false;
    }
    if (hw_acc < cfg.min_accuracy || approx_acc < cfg.min_accuracy || true_acc < cfg.min_accuracy) {
        std::fprintf(stderr, "FAILED %s: classification accuracy below threshold\n", cfg.name);
        pass = false;
    }
    return pass;
}

static bool run_invalid_argument_case() {
    float x[LR_MAX_SAMPLES * LR_MAX_FEATURES];
    float y[LR_MAX_SAMPLES];
    float weights[LR_MAX_FEATURES];
    clear_arrays(x, y, weights);
    for (int j = 0; j < LR_MAX_FEATURES; ++j) {
        weights[j] = 123.0f;
    }
    lr_train_accel(x, y, weights, 0, 3, 0.5f, 10);

    bool pass = true;
    for (int j = 0; j < LR_MAX_FEATURES; ++j) {
        if (std::fabs(weights[j]) > 1.0e-6f) {
            pass = false;
        }
    }
    std::printf("CASE %-28s samples=0 features=3 expected_zero_weights pass=%s\n",
                "invalid zero samples", pass ? "true" : "false");
    if (!pass) {
        std::fprintf(stderr, "FAILED invalid zero samples: weights were not cleared\n");
    }
    return pass;
}

int main() {
    const TestConfig tiny = {
        "tiny synthetic sanity", 4, 2, 40, 0.6f, 0.75f, 1.0e-4f, 1.0e-5f, 2.5f};
    const TestConfig separable = {
        "linearly separable", 12, 3, 80, 0.8f, 0.85f, 1.0e-4f, 1.0e-5f, 4.0f};
    const TestConfig zero_initial = {
        "zero initial edge", 6, 3, 1, 0.4f, 0.50f, 1.0e-4f, 1.0e-5f, 1.0f};
    const TestConfig near_max = {
        "near max shape", 30, 30, 40, 0.25f, 0.65f, 1.0e-4f, 1.0e-5f, 8.0f};
    const TestConfig marketing = {
        "marketing sample", 30, 7, 80, 0.35f, 0.70f, 1.0e-4f, 1.0e-5f, 5.0f};

    bool pass = true;
    pass = run_case(tiny, make_tiny_case) && pass;
    pass = run_case(separable, make_separable_case) && pass;
    pass = run_case(zero_initial, make_zero_initial_edge_case) && pass;
    pass = run_case(near_max, make_near_max_case) && pass;
    pass = run_case(marketing, make_marketing_sample_case) && pass;
    pass = run_invalid_argument_case() && pass;

    if (!pass) {
        std::fprintf(stderr, "FAILED: one or more logistic-regression HLS tests failed\n");
        return EXIT_FAILURE;
    }

    std::printf("PASSED: all logistic-regression HLS tests\n");
    return EXIT_SUCCESS;
}
