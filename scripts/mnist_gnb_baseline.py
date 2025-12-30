from __future__ import annotations

from scripts._bootstrap import add_src_to_path
add_src_to_path()

import numpy as np

from ml_classifiers.data.mnist import load_mnist_openml, split_mnist
from ml_classifiers.models.gaussian_nb import fit_gnb_smoothing, predict_gnb
from ml_classifiers.utils import accuracy, set_seed


def main() -> None:
    set_seed(42)

    X, y = load_mnist_openml(cache_dir="openml_cache", as_frame=False)
    X_train, X_test, y_train, y_test = split_mnist(X, y, test_size=0.1, random_state=42)

    alpha = 1e-7
    params = fit_gnb_smoothing(X_train, y_train, alpha=alpha)

    _, yhat_tr = predict_gnb(X_train, params)
    _, yhat_te = predict_gnb(X_test, params)

    print("MNIST | Gaussian NB (smoothed)")
    print("alpha:", alpha)
    print("Train accuracy:", accuracy(yhat_tr, y_train))
    print("Test  accuracy:", accuracy(yhat_te, y_test))

    # Print feature params for pixel 0 and pixel 99 (100th pixel feature)
    print("\nPixel 0 Mu:", params.Mu[0])
    print("Pixel 0 Var:", params.Var[0])
    print("\nPixel 99 Mu:", params.Mu[99])
    print("Pixel 99 Var:", params.Var[99])


if __name__ == "__main__":
    main()
