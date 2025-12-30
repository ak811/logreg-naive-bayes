from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

from scripts._bootstrap import add_src_to_path
add_src_to_path()

import numpy as np

from ml_classifiers.data.mnist import load_mnist_openml, split_mnist
from ml_classifiers.models.gaussian_nb import fit_gnb_smoothing, predict_gnb
from ml_classifiers.utils import FigurePaths, accuracy, set_seed
from ml_classifiers.viz.plotting import plot_accuracy_sweep


def main() -> None:
    set_seed(42)
    figs = FigurePaths()

    X, y = load_mnist_openml(cache_dir="openml_cache", as_frame=False)
    X_train, X_test, y_train, y_test = split_mnist(X, y, test_size=0.1, random_state=42)

    alphas = np.logspace(-7, 1, 17)
    acc_train = []
    acc_test = []

    for a in alphas:
        params = fit_gnb_smoothing(X_train, y_train, alpha=float(a))
        _, yhat_tr = predict_gnb(X_train, params)
        _, yhat_te = predict_gnb(X_test, params)

        acc_train.append(accuracy(yhat_tr, y_train))
        acc_test.append(accuracy(yhat_te, y_test))

    acc_train = np.array(acc_train)
    acc_test = np.array(acc_test)

    best_idx = int(np.argmax(acc_test))
    best_alpha = float(alphas[best_idx])

    print("MNIST | Gaussian NB smoothing sweep")
    print("alphas:", alphas)
    print("Train accuracies:", acc_train)
    print("Test  accuracies:", acc_test)
    print("Best alpha:", best_alpha, "Best test accuracy:", float(acc_test[best_idx]))

    plot_accuracy_sweep(
        alphas,
        acc_train,
        acc_test,
        figs.path("mnist_gnb_accuracy_sweep.png"),
        title="MNIST: Gaussian NB smoothing sweep",
    )


if __name__ == "__main__":
    main()
