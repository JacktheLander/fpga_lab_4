import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np


MAX_SAMPLES = 30
MAX_FEATURES = 30
MAX_ITERS = 200
LABEL_CANDIDATES = ("label", "target", "response", "converted", "y", "outcome")


def warn(message):
    print(f"WARNING: {message}", file=sys.stderr)


def _parse_float(value):
    text = str(value).strip()
    if text == "":
        return None
    lowered = text.lower()
    if lowered in {"true", "yes", "y", "success", "converted", "purchase", "purchased"}:
        return 1.0
    if lowered in {"false", "no", "n", "failure", "not converted", "no purchase"}:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return None


def sigmoid_true(z):
    z = np.asarray(z, dtype=np.float32)
    clipped = np.clip(z, -40.0, 40.0)
    return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32)


def sigmoid_approx(z):
    z = np.asarray(z, dtype=np.float32)
    return np.clip(0.5 + 0.125 * z, 0.0, 1.0).astype(np.float32)


def _sigmoid(z, model):
    if model == "true":
        return sigmoid_true(z)
    if model == "approx":
        return sigmoid_approx(z)
    raise ValueError(f"unknown model {model!r}")


def clamp_iterations(iterations):
    if iterations < 0:
        warn(f"iteration count {iterations} is invalid; using 0")
        return 0
    if iterations > MAX_ITERS:
        warn(f"iteration count {iterations} exceeds LR_MAX_ITERS={MAX_ITERS}; truncating")
        return MAX_ITERS
    return int(iterations)


def validate_limits(sample_count, feature_count):
    if sample_count <= 0:
        raise ValueError("sample count must be positive")
    if feature_count <= 1:
        raise ValueError("feature count must include bias plus at least one data feature")
    if sample_count > MAX_SAMPLES:
        raise ValueError(f"sample count {sample_count} exceeds LR_MAX_SAMPLES={MAX_SAMPLES}")
    if feature_count > MAX_FEATURES:
        raise ValueError(f"feature count {feature_count} exceeds LR_MAX_FEATURES={MAX_FEATURES}")


def train_logistic(x, y, learning_rate, iterations, model):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    weights = np.zeros((x.shape[1],), dtype=np.float32)
    iterations = clamp_iterations(int(iterations))
    if x.shape[0] == 0 or x.shape[1] == 0:
        return weights
    step_scale = np.float32(learning_rate) / np.float32(x.shape[0])
    for _ in range(iterations):
        preds = _sigmoid(x @ weights, model)
        error = y - preds
        gradients = x.T @ error
        weights += step_scale * gradients.astype(np.float32)
    return weights.astype(np.float32)


def predict(x, weights, model):
    scores = np.asarray(x, dtype=np.float32) @ np.asarray(weights, dtype=np.float32)
    return (_sigmoid(scores, model) >= 0.5).astype(np.float32)


def accuracy(x, y, weights, model):
    if len(y) == 0:
        return 0.0
    return float(np.mean(predict(x, weights, model) == np.asarray(y, dtype=np.float32)))


def weight_metrics(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    diff = np.abs(a - b)
    return {
        "max_abs": float(np.max(diff)) if diff.size else 0.0,
        "mean_abs": float(np.mean(diff)) if diff.size else 0.0,
    }


def _find_label_column(fieldnames):
    lower_to_name = {name.lower(): name for name in fieldnames}
    for candidate in LABEL_CANDIDATES:
        if candidate in lower_to_name:
            return lower_to_name[candidate]
    return fieldnames[-1]


def _read_csv_rows(path):
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"dataset not found: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    if not rows:
        raise ValueError(f"no rows found in {csv_path}")
    if not fieldnames:
        raise ValueError(f"no header row found in {csv_path}")
    return rows, fieldnames


def _select_numeric_columns(rows, fieldnames, label_name, max_data_features):
    numeric_columns = []
    for name in fieldnames:
        if name == label_name or name.lower() == "split":
            continue
        values = [_parse_float(row.get(name, "")) for row in rows]
        usable = [value for value in values if value is not None]
        if len(usable) == len(rows):
            numeric_columns.append(name)
    if not numeric_columns:
        raise ValueError("no fully numeric feature columns found")
    if len(numeric_columns) > max_data_features:
        warn(
            f"dataset has {len(numeric_columns)} numeric features; "
            f"using first {max_data_features} to fit LR_MAX_FEATURES"
        )
        numeric_columns = numeric_columns[:max_data_features]
    return numeric_columns


def _split_indices(count, test_fraction, seed):
    rng = np.random.default_rng(seed)
    indices = np.arange(count)
    rng.shuffle(indices)
    eval_count = max(1, int(round(count * test_fraction))) if count > 1 else 0
    if eval_count >= count:
        eval_count = count - 1
    eval_idx = np.sort(indices[:eval_count])
    train_idx = np.sort(indices[eval_count:])
    return train_idx, eval_idx


def _normalize_with_bias(train_raw, eval_raw):
    train_raw = np.asarray(train_raw, dtype=np.float32)
    eval_raw = np.asarray(eval_raw, dtype=np.float32)
    mean = train_raw.mean(axis=0)
    std = train_raw.std(axis=0)
    std[std < 1.0e-6] = 1.0
    train_norm = (train_raw - mean) / std
    eval_norm = (eval_raw - mean) / std if eval_raw.size else eval_raw.reshape(0, train_raw.shape[1])
    train_bias = np.ones((train_norm.shape[0], 1), dtype=np.float32)
    eval_bias = np.ones((eval_norm.shape[0], 1), dtype=np.float32)
    return (
        np.concatenate([train_bias, train_norm], axis=1).astype(np.float32),
        np.concatenate([eval_bias, eval_norm], axis=1).astype(np.float32),
        mean.astype(np.float32),
        std.astype(np.float32),
    )


def load_raw_dataset(path, sample_count, feature_count, test_fraction=0.3, seed=529):
    rows, fieldnames = _read_csv_rows(path)
    label_name = _find_label_column(fieldnames)
    feature_names = _select_numeric_columns(rows, fieldnames, label_name, feature_count - 1)

    raw_features = []
    labels = []
    for row in rows:
        label_value = _parse_float(row.get(label_name, ""))
        if label_value is None:
            continue
        feature_values = [_parse_float(row[name]) for name in feature_names]
        if any(value is None for value in feature_values):
            continue
        raw_features.append(feature_values)
        labels.append(1.0 if label_value > 0 else 0.0)
    if len(raw_features) < 2:
        raise ValueError("dataset must contain at least two usable labeled rows")

    raw_features = np.asarray(raw_features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    train_idx, eval_idx = _split_indices(len(labels), test_fraction, seed)
    train_raw = raw_features[train_idx]
    eval_raw = raw_features[eval_idx]
    train_y = labels[train_idx]
    eval_y = labels[eval_idx]

    if train_raw.shape[0] > sample_count:
        warn(
            f"training split has {train_raw.shape[0]} rows; "
            f"using first {sample_count} for the accelerator"
        )
        train_raw = train_raw[:sample_count]
        train_y = train_y[:sample_count]

    train_x, eval_x, mean, std = _normalize_with_bias(train_raw, eval_raw)
    return {
        "source": str(path),
        "mode": "dataset",
        "feature_names": ["bias"] + feature_names,
        "train_x": train_x,
        "train_y": train_y.astype(np.float32),
        "eval_x": eval_x,
        "eval_y": eval_y.astype(np.float32),
        "mean": mean,
        "std": std,
    }


def load_preprocessed_csv(path, sample_count, feature_count, seed=529):
    rows, fieldnames = _read_csv_rows(path)
    label_name = _find_label_column(fieldnames)
    split_name = next((name for name in fieldnames if name.lower() == "split"), None)
    feature_names = [name for name in fieldnames if name not in {label_name, split_name}]
    if len(feature_names) > feature_count:
        warn(f"preprocessed file has {len(feature_names)} features; truncating to {feature_count}")
        feature_names = feature_names[:feature_count]

    features = []
    labels = []
    splits = []
    for row in rows:
        label_value = _parse_float(row.get(label_name, ""))
        values = [_parse_float(row.get(name, "")) for name in feature_names]
        if label_value is None or any(value is None for value in values):
            continue
        features.append(values)
        labels.append(1.0 if label_value > 0 else 0.0)
        splits.append(str(row.get(split_name, "")).strip().lower() if split_name else "")
    if len(features) < 2:
        raise ValueError("preprocessed dataset must contain at least two usable rows")

    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    if split_name:
        train_mask = np.asarray([split != "eval" and split != "test" for split in splits])
        eval_mask = ~train_mask
        if not train_mask.any() or not eval_mask.any():
            train_idx, eval_idx = _split_indices(len(labels), 0.3, seed)
            train_x = features[train_idx]
            eval_x = features[eval_idx]
            train_y = labels[train_idx]
            eval_y = labels[eval_idx]
        else:
            train_x = features[train_mask]
            eval_x = features[eval_mask]
            train_y = labels[train_mask]
            eval_y = labels[eval_mask]
    else:
        train_idx, eval_idx = _split_indices(len(labels), 0.3, seed)
        train_x = features[train_idx]
        eval_x = features[eval_idx]
        train_y = labels[train_idx]
        eval_y = labels[eval_idx]

    if train_x.shape[0] > sample_count:
        warn(f"preprocessed training split has {train_x.shape[0]} rows; truncating to {sample_count}")
        train_x = train_x[:sample_count]
        train_y = train_y[:sample_count]
    return {
        "source": str(path),
        "mode": "preprocessed",
        "feature_names": feature_names,
        "train_x": train_x.astype(np.float32),
        "train_y": train_y.astype(np.float32),
        "eval_x": eval_x.astype(np.float32),
        "eval_y": eval_y.astype(np.float32),
        "mean": None,
        "std": None,
    }


def generate_synthetic_dataset(sample_count, feature_count, seed=529):
    data_features = feature_count - 1
    total_rows = max(sample_count + 8, sample_count)
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(total_rows, data_features)).astype(np.float32)
    coeffs = np.zeros((data_features,), dtype=np.float32)
    if data_features > 0:
        coeffs[0] = 1.35
    if data_features > 1:
        coeffs[1] = -0.85
    if data_features > 2:
        coeffs[2] = 0.45
    logits = -0.15 + raw @ coeffs
    labels = (logits >= 0.0).astype(np.float32)
    train_raw = raw[:sample_count]
    eval_raw = raw[sample_count:]
    train_y = labels[:sample_count]
    eval_y = labels[sample_count:]
    train_x, eval_x, mean, std = _normalize_with_bias(train_raw, eval_raw)
    return {
        "source": "synthetic",
        "mode": "synthetic",
        "feature_names": ["bias"] + [f"synthetic_{i}" for i in range(data_features)],
        "train_x": train_x,
        "train_y": train_y,
        "eval_x": eval_x,
        "eval_y": eval_y,
        "mean": mean,
        "std": std,
    }


def default_sample_dataset():
    root = Path(__file__).resolve().parents[1]
    representative = root / "data" / "marketing_campaign_representative.csv"
    if representative.exists():
        return representative
    return root / "data" / "marketing_campaign_sample.csv"


def prepare_dataset(args):
    validate_limits(args.sample_count, args.feature_count)
    mode = args.mode
    if mode == "auto":
        if args.dataset:
            mode = "dataset"
        elif Path(args.sample_dataset).exists():
            mode = "sample"
        else:
            mode = "synthetic"

    if mode == "synthetic":
        warn("using explicit synthetic dataset mode")
        return generate_synthetic_dataset(args.sample_count, args.feature_count, args.seed)

    path = Path(args.dataset if mode == "dataset" else args.sample_dataset)
    if not path.exists():
        raise FileNotFoundError(f"{mode} dataset not found: {path}")

    if path.name.endswith("_preprocessed.csv"):
        dataset = load_preprocessed_csv(path, args.sample_count, args.feature_count, args.seed)
    else:
        dataset = load_raw_dataset(path, args.sample_count, args.feature_count, args.test_fraction, args.seed)
    if mode == "sample":
        dataset["mode"] = "sample"
    return dataset


def pack_fixed_window(x, y, weights):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    if x.shape[0] > MAX_SAMPLES:
        raise ValueError(f"training rows {x.shape[0]} exceed LR_MAX_SAMPLES={MAX_SAMPLES}")
    if x.shape[1] > MAX_FEATURES:
        raise ValueError(f"features {x.shape[1]} exceed LR_MAX_FEATURES={MAX_FEATURES}")
    x_buf = np.zeros((MAX_SAMPLES, MAX_FEATURES), dtype=np.float32)
    y_buf = np.zeros((MAX_SAMPLES,), dtype=np.float32)
    w_buf = np.zeros((MAX_FEATURES,), dtype=np.float32)
    x_buf[: x.shape[0], : x.shape[1]] = x
    y_buf[: y.shape[0]] = y
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


def validate_overlay_files(overlay_path, hwh_path=None):
    bit_path = Path(overlay_path)
    if not bit_path.exists():
        raise FileNotFoundError(f"overlay bitfile not found: {bit_path}")
    if hwh_path is None:
        hwh_path = bit_path.with_suffix(".hwh")
    hwh = Path(hwh_path)
    if not hwh.exists():
        raise FileNotFoundError(f"HWH handoff file not found: {hwh}")
    return bit_path, hwh


def _flush_if_supported(buffer):
    flush = getattr(buffer, "flush", None)
    if callable(flush):
        flush()


def _invalidate_if_supported(buffer):
    invalidate = getattr(buffer, "invalidate", None)
    if callable(invalidate):
        invalidate()


def run_accelerator(overlay_path, hwh_path, ip_name, x, y, learning_rate, iterations):
    validate_overlay_files(overlay_path, hwh_path)
    try:
        from pynq import Overlay, allocate
    except ImportError as exc:
        raise RuntimeError("PYNQ is not available; rerun with --software-only on this machine") from exc

    overlay = Overlay(str(overlay_path))
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
    _flush_if_supported(x_dev)
    _flush_if_supported(y_dev)
    _flush_if_supported(w_dev)

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
    _invalidate_if_supported(w_dev)
    weights = np.asarray(w_dev[: x.shape[1]], dtype=np.float32).copy()
    readback_elapsed = time.perf_counter() - result_start

    x_dev.freebuffer()
    y_dev.freebuffer()
    w_dev.freebuffer()
    return weights, setup_elapsed, kernel_elapsed, readback_elapsed


def _timed_train(x, y, learning_rate, iterations, model):
    start = time.perf_counter()
    weights = train_logistic(x, y, learning_rate, iterations, model)
    elapsed = time.perf_counter() - start
    return weights, elapsed


def print_summary(dataset, args, true_weights, true_time, approx_weights, approx_time, hw_result):
    train_x = dataset["train_x"]
    train_y = dataset["train_y"]
    eval_x = dataset["eval_x"]
    eval_y = dataset["eval_y"]
    eval_target_x = eval_x if eval_x.shape[0] else train_x
    eval_target_y = eval_y if eval_y.shape[0] else train_y

    true_acc = accuracy(eval_target_x, eval_target_y, true_weights, "true")
    approx_acc = accuracy(eval_target_x, eval_target_y, approx_weights, "approx")
    approx_vs_true = weight_metrics(approx_weights, true_weights)

    print("=== Dataset ===")
    print(f"mode={dataset['mode']} source={dataset['source']}")
    print(f"train_samples={train_x.shape[0]} eval_samples={eval_target_x.shape[0]} features={train_x.shape[1]}")
    print("features=" + ",".join(dataset["feature_names"][: train_x.shape[1]]))

    print("=== Software References ===")
    print(f"true_sigmoid_time_s={true_time:.6e}")
    print(f"approx_sigmoid_time_s={approx_time:.6e}")
    print(f"true_sigmoid_accuracy={true_acc:.4f}")
    print(f"approx_sigmoid_accuracy={approx_acc:.4f}")
    print(f"approx_vs_true_max_abs_weight_diff={approx_vs_true['max_abs']:.6e}")
    print(f"approx_vs_true_mean_abs_weight_diff={approx_vs_true['mean_abs']:.6e}")

    if hw_result is None:
        print("=== Hardware ===")
        print("hardware_run=false")
        return

    hw_weights, setup_elapsed, kernel_elapsed, readback_elapsed = hw_result
    hw_acc = accuracy(eval_target_x, eval_target_y, hw_weights, "approx")
    hw_vs_approx = weight_metrics(hw_weights, approx_weights)
    hw_vs_true = weight_metrics(hw_weights, true_weights)
    speedup_true = true_time / kernel_elapsed if kernel_elapsed > 0 else float("inf")
    speedup_approx = approx_time / kernel_elapsed if kernel_elapsed > 0 else float("inf")

    print("=== Hardware ===")
    print("hardware_run=true")
    print(f"accelerator_setup_s={setup_elapsed:.6e}")
    print(f"accelerator_kernel_s={kernel_elapsed:.6e}")
    print(f"accelerator_readback_s={readback_elapsed:.6e}")
    print(f"speedup_vs_true_software_kernel_only={speedup_true:.6f}")
    print(f"speedup_vs_approx_software_kernel_only={speedup_approx:.6f}")
    print(f"hardware_accuracy={hw_acc:.4f}")
    print(f"hw_vs_approx_max_abs_weight_diff={hw_vs_approx['max_abs']:.6e}")
    print(f"hw_vs_approx_mean_abs_weight_diff={hw_vs_approx['mean_abs']:.6e}")
    print(f"hw_vs_true_max_abs_weight_diff={hw_vs_true['max_abs']:.6e}")
    print(f"hw_vs_true_mean_abs_weight_diff={hw_vs_true['mean_abs']:.6e}")
    print(f"pass={hw_vs_approx['max_abs'] < 1.0e-3 and hw_vs_approx['mean_abs'] < 1.0e-4}")


def build_arg_parser():
    sample_default = default_sample_dataset()
    parser = argparse.ArgumentParser(description="Logistic-regression PYNQ host and software validator")
    parser.add_argument("--overlay", "--bitfile", dest="overlay", default="lr_train_accel.bit")
    parser.add_argument("--hwh")
    parser.add_argument("--ip-name", default="lr_train_accel_0")
    parser.add_argument("--dataset", help="Path to the full/raw Marketing Campaigns dataset CSV")
    parser.add_argument("--sample-dataset", default=str(sample_default), help="Path to the committed representative sample CSV")
    parser.add_argument("--mode", choices=("auto", "dataset", "sample", "synthetic"), default="auto")
    parser.add_argument("--sample-count", type=int, default=MAX_SAMPLES)
    parser.add_argument("--feature-count", type=int, default=MAX_FEATURES)
    parser.add_argument("--learning-rate", type=float, default=0.35)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=529)
    parser.add_argument("--true-validation", choices=("on", "off"), default="on")
    parser.add_argument("--software-only", action="store_true")
    return parser


def main():
    args = build_arg_parser().parse_args()
    iterations = clamp_iterations(args.iterations)
    dataset = prepare_dataset(args)
    train_x = dataset["train_x"]
    train_y = dataset["train_y"]
    validate_limits(train_x.shape[0], train_x.shape[1])

    if args.true_validation == "on":
        true_weights, true_time = _timed_train(train_x, train_y, args.learning_rate, iterations, "true")
    else:
        true_weights = np.zeros((train_x.shape[1],), dtype=np.float32)
        true_time = 0.0
        warn("true sigmoid validation disabled by --true-validation=off")
    approx_weights, approx_time = _timed_train(train_x, train_y, args.learning_rate, iterations, "approx")

    hw_result = None
    if not args.software_only:
        hw_result = run_accelerator(
            Path(args.overlay), Path(args.hwh) if args.hwh else None,
            args.ip_name, train_x, train_y, args.learning_rate, iterations)

    print_summary(dataset, args, true_weights, true_time, approx_weights, approx_time, hw_result)


if __name__ == "__main__":
    main()
