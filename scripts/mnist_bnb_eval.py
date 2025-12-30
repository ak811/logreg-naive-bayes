from __future__ import annotations

from scripts._bootstrap import add_src_to_path
add_src_to_path()

import numpy as np

from ml_classifiers.data.mnist import load_mnist_openml, split_mnist
from ml_classifiers.models.bernoulli_nb import fit_bnb_smoothing, predict_bnb
from ml_classifiers.utils import accuracy, set_seed


def main() -> None:
    set_seed(42)

    X, y = load_mnist_openml(cache_dir="openml_cache", as_frame=False)
    X_train, X_test, y_train, y_test = split_mnist(X, y, test_size=0.1, random_state=42)

    alpha = 1e-8
    params = fit_bnb_smoothing(X_train, y_train, alpha=alpha)

    _, yhat_tr = predict_bnb(X_train, params)
    _, yhat_te = predict_bnb(X_test, params)

    print("MNIST | Bernoulli NB (smoothed)")
    print("alpha:", alpha)
    print("Train accuracy:", accuracy(yhat_tr, y_train))
    print("Test  accuracy:", accuracy(yhat_te, y_test))

    # Feature params for pixel 0 and pixel 99
    print("\nPixel 0 Phi:", params.Phi[0])
    print("Pixel 99 Phi:", params.Phi[99])


if __name__ == "__main__":
    main()
