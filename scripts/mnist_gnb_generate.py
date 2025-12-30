from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

from scripts._bootstrap import add_src_to_path
add_src_to_path()

import numpy as np

from ml_classifiers.data.mnist import load_mnist_openml, split_mnist
from ml_classifiers.models.gaussian_nb import fit_gnb_smoothing, generate_gnb
from ml_classifiers.utils import FigurePaths, set_seed
from ml_classifiers.viz.plotting import save_digit_grid


def main() -> None:
    set_seed(42)
    figs = FigurePaths()

    X, y = load_mnist_openml(cache_dir="openml_cache", as_frame=False)
    X_train, X_test, y_train, y_test = split_mnist(X, y, test_size=0.1, random_state=42)

    # Use the typical best from your notebook sweep.
    # If you want it fully dynamic, run mnist_gnb_sweep.py and pass it in.
    alpha_best = 0.1

    params = fit_gnb_smoothing(X_train, y_train, alpha=alpha_best)

    Xgen = []
    ygen = []
    for k in range(10):
        xk = generate_gnb(params, label=k, clip_min=0.0, clip_max=255.0)
        Xgen.append(xk)
        ygen.append(k)

    save_digit_grid(
        Xgen,
        ygen,
        figs.path("mnist_gnb_generated_digits.png"),
        n_row=3,
        n_col=4,
        fsize=(14, 8),
    )

    print("Saved GNB generated digit grid to outputs/figures/mnist_gnb_generated_digits.png")


if __name__ == "__main__":
    main()
