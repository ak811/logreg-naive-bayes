import numpy as np

from ml_classifiers.data.iris import load_iris_data
from ml_classifiers.models.multiclass_logreg import grad_softmax
from ml_classifiers.models.irls_logreg import grad_km1, hessian_km1


def test_grad_shapes():
    X, y = load_iris_data()
    K = int(y.max()) + 1
    d = X.shape[1]

    W = np.zeros((K, d))
    g = grad_softmax(W, X, y)
    assert g.shape == (K, d)

    Wm1 = np.zeros((K - 1, d))
    g2 = grad_km1(Wm1, X, y)
    assert g2.shape == (K - 1, d)

    H = hessian_km1(Wm1, X)
    assert H.shape == ((K - 1) * d, (K - 1) * d)
