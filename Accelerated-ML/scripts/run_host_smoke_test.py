import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pynq"))

from lr_pynq_host import (
    MAX_FEATURES,
    MAX_SAMPLES,
    accuracy,
    default_sample_dataset,
    generate_synthetic_dataset,
    load_raw_dataset,
    pack_fixed_window,
    train_logistic,
    weight_metrics,
)


def validate_dataset(dataset, learning_rate=0.35, iterations=40):
    x = dataset["train_x"]
    y = dataset["train_y"]
    true_weights = train_logistic(x, y, learning_rate, iterations, "true")
    approx_weights = train_logistic(x, y, learning_rate, iterations, "approx")
    diff = weight_metrics(approx_weights, true_weights)
    x_flat, y_fixed, w_fixed = pack_fixed_window(x, y, approx_weights)

    assert x.shape[0] <= MAX_SAMPLES
    assert x.shape[1] <= MAX_FEATURES
    assert x_flat.shape == (MAX_SAMPLES * MAX_FEATURES,)
    assert y_fixed.shape == (MAX_SAMPLES,)
    assert w_fixed.shape == (MAX_FEATURES,)
    assert np.isfinite(true_weights).all()
    assert np.isfinite(approx_weights).all()
    assert diff["max_abs"] >= 0.0
    true_acc = accuracy(dataset["eval_x"], dataset["eval_y"], true_weights, "true")
    approx_acc = accuracy(dataset["eval_x"], dataset["eval_y"], approx_weights, "approx")
    assert true_acc >= 0.5
    assert approx_acc >= 0.5
    return true_acc, approx_acc, diff


def main():
    synthetic = generate_synthetic_dataset(sample_count=MAX_SAMPLES, feature_count=3)
    synth_true_acc, synth_approx_acc, synth_diff = validate_dataset(synthetic, learning_rate=0.8, iterations=80)

    sample = load_raw_dataset(default_sample_dataset(), sample_count=MAX_SAMPLES, feature_count=7)
    sample_true_acc, sample_approx_acc, sample_diff = validate_dataset(sample)

    print(
        "PASSED host smoke test: "
        f"synthetic_true_acc={synth_true_acc:.4f} synthetic_approx_acc={synth_approx_acc:.4f} "
        f"synthetic_max_diff={synth_diff['max_abs']:.6e}"
    )
    print(
        "PASSED dataset smoke test: "
        f"sample_true_acc={sample_true_acc:.4f} sample_approx_acc={sample_approx_acc:.4f} "
        f"sample_max_diff={sample_diff['max_abs']:.6e}"
    )


if __name__ == "__main__":
    main()
