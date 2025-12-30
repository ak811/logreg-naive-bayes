from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

from scripts._bootstrap import add_src_to_path
add_src_to_path()

import numpy as np

from ml_classifiers.data.iris import load_iris_data
from ml_classifiers.models.multiclass_logreg import loss_softmax, grad_softmax, gradient_descent
from ml_classifiers.utils import FigurePaths, accuracy, set_seed
from ml_classifiers.viz.plotting import plot_loss_curve


def main() -> None:
    set_seed(42)
    figs = FigurePaths()

    X, y = load_iris_data()
    K = int(y.max()) + 1
    d = X.shape[1]

    W0 = np.zeros((K, d), dtype=float)

    alpha = 1e-4
    niter = 5000

    f = lambda W: loss_softmax(W, X, y)
    g = lambda W: grad_softmax(W, X, y)

    res = gradient_descent(f, g, W0, alpha=alpha, niter=niter)
    W_final = res.W_list[-1]
    loss_final = res.f_vals[-1]

    yhat = np.argmax(X @ W_final.T, axis=1)
    acc = accuracy(yhat, y)

    print("Iris | Softmax LR (GD) | no bias")
    print("alpha:", alpha, "niter:", niter)
    print("Final loss:", float(loss_final))
    print("Final accuracy:", acc)
    print("W_final:\n", W_final)

    plot_loss_curve(
        res.f_vals,
        figs.path("iris_gd_loss.png"),
        semilogy=True,
        title="Iris: Softmax Logistic Regression (GD, no bias)",
        label="GD Loss",
    )


if __name__ == "__main__":
    main()
