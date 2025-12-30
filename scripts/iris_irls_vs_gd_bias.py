from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

from scripts._bootstrap import add_src_to_path
add_src_to_path()

import numpy as np

from ml_classifiers.data.iris import load_iris_data, add_intercept
from ml_classifiers.models.multiclass_logreg import loss_softmax, grad_softmax, gradient_descent
from ml_classifiers.models.irls_logreg import loss_km1, grad_km1, hessian_km1, newton_method, predict_km1
from ml_classifiers.utils import FigurePaths, accuracy, set_seed
from ml_classifiers.viz.plotting import plot_compare_curves


def main() -> None:
    set_seed(42)
    figs = FigurePaths()

    X, y = load_iris_data()
    Xb = add_intercept(X)
    K = int(y.max()) + 1
    d = Xb.shape[1]

    # GD (full K)
    W0_gd = np.zeros((K, d), dtype=float)
    alpha = 1e-4
    niter_gd = 5000
    f_gd = lambda W: loss_softmax(W, Xb, y)
    g_gd = lambda W: grad_softmax(W, Xb, y)
    gd_res = gradient_descent(f_gd, g_gd, W0_gd, alpha=alpha, niter=niter_gd)

    # IRLS / Newton (K-1)
    W0_n = np.zeros((K - 1, d), dtype=float)
    niter_newton = 30
    f_n = lambda W: loss_km1(W, Xb, y)
    g_n = lambda W: grad_km1(W, Xb, y)
    h_n = lambda W: hessian_km1(W, Xb)
    newton_res = newton_method(f_n, g_n, h_n, W0_n, niter=niter_newton, ridge=1e-9)

    yhat_irls = predict_km1(newton_res.W_list[-1], Xb)
    acc_irls = accuracy(yhat_irls, y)

    print("Iris | IRLS/Newton (with bias via intercept feature)")
    print("Newton iterations:", niter_newton)
    print("Final loss:", float(newton_res.f_vals[-1]))
    print("Final accuracy:", acc_irls)

    nplot = 50
    plot_compare_curves(
        curves=[gd_res.f_vals[:nplot], newton_res.f_vals[: min(nplot, len(newton_res.f_vals))]],
        labels=["Gradient Descent", "IRLS/Newton"],
        path=figs.path("iris_irls_vs_gd_bias.png"),
        semilogy=True,
        title="Iris: IRLS/Newton vs Gradient Descent (with bias)",
        ylabel="Loss",
    )


if __name__ == "__main__":
    main()
