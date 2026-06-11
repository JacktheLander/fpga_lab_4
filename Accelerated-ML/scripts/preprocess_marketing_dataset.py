import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pynq"))

from lr_pynq_host import MAX_FEATURES, MAX_SAMPLES, load_raw_dataset


def write_preprocessed_csv(dataset, output_path):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    feature_names = dataset["feature_names"]
    fieldnames = ["split", "label"] + feature_names

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for split_name, x, y in (
            ("train", dataset["train_x"], dataset["train_y"]),
            ("eval", dataset["eval_x"], dataset["eval_y"]),
        ):
            for row_idx in range(x.shape[0]):
                row = {"split": split_name, "label": int(y[row_idx])}
                for col_idx, name in enumerate(feature_names[: x.shape[1]]):
                    row[name] = f"{float(x[row_idx, col_idx]):.8g}"
                writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess a Marketing Campaigns Logistic Regression CSV for the HLS accelerator"
    )
    parser.add_argument("input_csv", help="Path to the full/raw dataset CSV")
    parser.add_argument("--output", default=str(ROOT / "data" / "marketing_campaign_preprocessed.csv"))
    parser.add_argument("--sample-count", type=int, default=MAX_SAMPLES)
    parser.add_argument("--feature-count", type=int, default=MAX_FEATURES)
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=529)
    args = parser.parse_args()

    dataset = load_raw_dataset(
        args.input_csv,
        sample_count=args.sample_count,
        feature_count=args.feature_count,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    write_preprocessed_csv(dataset, args.output)
    print(f"wrote {args.output}")
    print(f"train_samples={dataset['train_x'].shape[0]} eval_samples={dataset['eval_x'].shape[0]}")
    print("features=" + ",".join(dataset["feature_names"]))


if __name__ == "__main__":
    main()
