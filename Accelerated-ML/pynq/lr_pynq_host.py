import argparse
import csv
import time
from pathlib import Path

import numpy as np


MAX_SAMPLES = 30
MAX_FEATURES = 30
MAX_ITERS = 200


def _parse_float(value):
    text = str(value).strip()
    if text == "":
        return None
    lowered = text.lower()
    if lowered in {"true", "yes", "y", "success", "converted"}:
        return 1.0
    if lowered in {"false", "no", "n", "failure", "not converted"}:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return None


def generate_synthetic_dataset(samples=30):
    rng = np.random.default_rng(529)
    raw = rng.normal(size=(samples, 2)).astype(np.float32)
    logits = -0.15 + 1.35 * raw[:, 0] - 0.85 * raw[:, 1]
    y = (logits >= 0.0).astype(np.float32)
    return _add_bias_and_normalize(raw, y)


def _add_bias_and_normalize(features, labels):
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError("features must be a 2D array")
    features = features[:MAX_SAMPLES, : MAX_FEATURES - 1]
    labels = labels[: features.shape[0]]
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1.0e-6] = 1.0
    normalized = (features - mean) / std
    bias = np.ones((normalized.shape[0], 1), dtype=np.float32)
    return np.concatenate([bias, normalized], axis=1).astype(np.float32), labels.astype(np.float32)


def load_csv_dataset(path):
    csv_path = Path(path)
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"no rows found in {csv_path}")

    label_candidates = ["label", "target", "response", "converted", "y", "outcome"]
    lower_to_name = {name.lower(): name for name in rows[0].keys()}
    label_name = None
    for candidate in label_candidates:
        if candidate in lower_to_name:
            label_name = lower_to_name[candidate]
            break
    if label_name is None:
        label_name = list(rows[0].keys())[-1]

    feature_names = [name for name in rows[0].keys() if name != label_name]
    category_maps = {name: {} for name in feature_names}
    feature_rows = []
    labels = []

    for row in rows[:MAX_SAMPLES]:
        label_value = _parse_float(row[label_name])
        if label_value is None:
            continue
        labels.append(1.0 if label_value > 0 else 0.0)
        encoded = []
        for name in feature_names[: MAX_FEATURES - 1]:
            value = _parse_float(row[name])
            if value is None:
                mapping = category_maps[name]
                key = str(row[name]).strip().lower()
                if key not in mapping:
                    mapping[key] = float(len(mapping))
                value = mapping[key]
            encoded.append(value)
        feature_rows.append(encoded)

    if not feature_rows:
        raise ValueError(f"no usable numeric/categorical rows found in {csv_path}")

    return _add_bias_and_normalize(np.asarray(feature_rows, dtype=np.float32), np.asarray(labels, dtype=np.float32))


def load_or_generate_dataset(csv_path=None):
    if csv_path:
        return load_csv_dataset(csv_path)
    return generate_synthetic_dataset(MAX_SAMPLES)


def sigmoid(z):
    z = np.asarray(z, dtype=np.float32)
    return np.clip(0.5 + 0.125 * z, 0.0, 1.0).astype(np.float32)


def train_software(x, y, learning_rate, iterations):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    weights = np.zeros((x.shape[1],), dtype=np.float32)
    iterations = max(1, min(int(iterations), MAX_ITERS))
    for _ in range(iterations):
        preds = sigmoid(x @ weights).astype(np.float32)
        error = y - preds
        gradient = (x.T @ error) / np.float32(x.shape[0])
        weights += np.float32(learning_rate) * gradient.astype(np.float32)
    return weights


def predict(x, weights):
    return (sigmoid(np.asarray(x, dtype=np.float32) @ np.asarray(weights, dtype=np.float32)) >= 0.5).astype(np.float32)


def accuracy(x, y, weights):
    return float(np.mean(predict(x, weights) == y))


def pack_fixed_window(x, y, weights):
    x_buf = np.zeros((MAX_SAMPLES, MAX_FEATURES), dtype=np.float32)
    y_buf = np.zeros((MAX_SAMPLES,), dtype=np.float32)
    w_buf = np.zeros((MAX_FEATURES,), dtype=np.float32)
    n_samples, n_features = x.shape
    x_buf[:n_samples, :n_features] = x
    y_buf[:n_samples] = y
    w_buf[: weights.shape[0]] = weights
    return x_buf.reshape(-1), y_buf, w_buf


def _register_offset(ip, arg_name):
    lname = arg_name.lower()
    candidates = []
    for name, meta in getattr(ip, "registers", {}).items():
        low = name.lower()
        if low == lname or low.startswith(lname + "_") or low.startswith(lname + "["):
            offset = meta["address_offset"]
            if isinstance(offset, str):
                offset = int(offset, 0)
            candidates.append((name, int(offset)))
    if not candidates:
        available = ", ".join(getattr(ip, "registers", {}).keys())
        raise KeyError(f"Register {arg_name!r} not found. Available: {available}")
    candidates.sort(key=lambda item: item[1])
    return candidates[0][1]


def write_arg(ip, arg_name, value):
    ip.write(_register_offset(ip, arg_name), int(value))


def write_float_arg(ip, arg_name, value):
    bits = np.asarray([value], dtype=np.float32).view(np.uint32)[0]
    ip.write(_register_offset(ip, arg_name), int(bits))


def start_and_wait(ip):
    ip.write(0x00, 0x01)
    while (ip.read(0x00) & 0x02) == 0:
        pass


def run_accelerator(bitfile, ip_name, x, y, learning_rate, iterations):
    from pynq import Overlay, allocate

    overlay = Overlay(bitfile)
    ip = getattr(overlay, ip_name)
    initial_weights = np.zeros((x.shape[1],), dtype=np.float32)
    x_flat, y_fixed, w_fixed = pack_fixed_window(x, y, initial_weights)

    setup_start = time.perf_counter()
    x_dev = allocate(shape=x_flat.shape, dtype=np.float32)
    y_dev = allocate(shape=y_fixed.shape, dtype=np.float32)
    w_dev = allocate(shape=w_fixed.shape, dtype=np.float32)
    np.copyto(x_dev, x_flat)
    np.copyto(y_dev, y_fixed)
    np.copyto(w_dev, w_fixed)

    write_arg(ip, "x", x_dev.physical_address)
    write_arg(ip, "y", y_dev.physical_address)
    write_arg(ip, "weights", w_dev.physical_address)
    write_arg(ip, "n_samples", x.shape[0])
    write_arg(ip, "n_features", x.shape[1])
    write_float_arg(ip, "learning_rate", learning_rate)
    write_arg(ip, "n_iterations", iterations)
    setup_elapsed = time.perf_counter() - setup_start

    kernel_start = time.perf_counter()
    start_and_wait(ip)
    kernel_elapsed = time.perf_counter() - kernel_start

    result_start = time.perf_counter()
    weights = np.asarray(w_dev[: x.shape[1]], dtype=np.float32).copy()
    setup_elapsed += time.perf_counter() - result_start

    x_dev.freebuffer()
    y_dev.freebuffer()
    w_dev.freebuffer()
    return weights, setup_elapsed, kernel_elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bitfile", default="lr_train_accel.bit")
    parser.add_argument("--ip-name", default="lr_train_accel_0")
    parser.add_argument("--csv")
    parser.add_argument("--learning-rate", type=float, default=0.8)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--software-only", action="store_true")
    args = parser.parse_args()

    x, y = load_or_generate_dataset(args.csv)
    iterations = max(1, min(args.iterations, MAX_ITERS))

    sw_start = time.perf_counter()
    sw_weights = train_software(x, y, args.learning_rate, iterations)
    sw_elapsed = time.perf_counter() - sw_start
    sw_acc = accuracy(x, y, sw_weights)

    print(f"samples={x.shape[0]} features={x.shape[1]} iterations={iterations}")
    print(f"software_time_s={sw_elapsed:.6e} software_accuracy={sw_acc:.4f}")
    print("software_weights=" + ",".join(f"{v:.6f}" for v in sw_weights))

    if args.software_only:
        return

    hw_weights, setup_elapsed, kernel_elapsed = run_accelerator(
        args.bitfile, args.ip_name, x, y, args.learning_rate, iterations
    )
    hw_acc = accuracy(x, y, hw_weights)
    max_abs_diff = float(np.max(np.abs(hw_weights - sw_weights)))

    print(f"accelerator_setup_s={setup_elapsed:.6e}")
    print(f"accelerator_kernel_s={kernel_elapsed:.6e}")
    print(f"accelerator_accuracy={hw_acc:.4f}")
    print(f"max_abs_weight_diff={max_abs_diff:.6e}")
    print("accelerator_weights=" + ",".join(f"{v:.6f}" for v in hw_weights))
    print(f"pass={max_abs_diff < 1.0e-3 and abs(hw_acc - sw_acc) < 1.0e-6}")


if __name__ == "__main__":
    main()
