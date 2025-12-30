from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

from scripts._bootstrap import add_src_to_path
add_src_to_path()

import numpy as np

from ml_classifiers.data.mnist import load_mnist_openml, split_mnist
from ml_classifiers.models.bernoulli_nb import fit_bnb_smoothing, generate_bnb
from ml_classifiers.utils import FigurePaths, set_seed
from ml_classifiers.viz.plotting import save_digit_grid


def main() -> None:
    set_seed(42)
    figs = FigurePaths()

    X, y = load_mnist_openml(cache_dir="openml_cache", as_frame=False)
    X_train, X_test, y_train, y_test = split_mnist(X, y, test_size=0.1, random_state=42)

    alpha = 1e-8
    params = fit_bnb_smoothing(X_train, y_train, alpha=alpha)

    Xgen = []
    ygen = []
    for k in range(10):
        xk = generate_bnb(params, label=k)
        # scale to 0/255 so it looks like an image instead of faint dust
        Xgen.append(xk * 255.0)
        ygen.append(k)

    save_digit_grid(
        Xgen,
        ygen,
        figs.path("mnist_bnb_generated_digits.png"),
        n_row=3,
        n_col=4,
        fsize=(14, 8),
    )

    print("Saved BNB generated digit grid to outputs/figures/mnist_bnb_generated_digits.png")


if __name__ == "__main__":
    main()
