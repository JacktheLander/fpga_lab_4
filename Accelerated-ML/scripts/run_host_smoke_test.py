import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pynq"))

from lr_pynq_host import accuracy, load_or_generate_dataset, pack_fixed_window, train_software


def main():
    x, y = load_or_generate_dataset()
    weights = train_software(x, y, learning_rate=0.8, iterations=80)
    acc = accuracy(x, y, weights)
    x_flat, y_fixed, w_fixed = pack_fixed_window(x, y, weights)

    assert x.shape[0] <= 30
    assert x.shape[1] <= 30
    assert x_flat.shape == (900,)
    assert y_fixed.shape == (30,)
    assert w_fixed.shape == (30,)
    assert np.isfinite(weights).all()
    assert acc >= 0.9

    print(f"PASSED host smoke test: samples={x.shape[0]} features={x.shape[1]} accuracy={acc:.4f}")


if __name__ == "__main__":
    main()
