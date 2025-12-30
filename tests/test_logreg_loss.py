import numpy as np

from ml_classifiers.data.iris import load_iris_data
from ml_classifiers.models.multiclass_logreg import loss_softmax


def test_iris_zero_weights_loss_is_logK():
    X, y = load_iris_data()
    K = int(y.max()) + 1
    d = X.shape[1]
    W0 = np.zeros((K, d))
    loss = loss_softmax(W0, X, y)
    assert np.isfinite(loss)
    # log(3) ≈ 1.0986122886681098
    assert abs(loss - np.log(K)) < 1e-6
