import numpy as np

from ml_classifiers.data.iris import load_iris_data
from ml_classifiers.models.gaussian_nb import fit_gnb, predict_gnb
from ml_classifiers.models.bernoulli_nb import fit_bnb_smoothing, predict_bnb


def test_nb_predict_no_nan_inf():
    X, y = load_iris_data()

    gparams = fit_gnb(X, y)
    logp, yhat = predict_gnb(X, gparams)
    assert np.all(np.isfinite(logp))
    assert yhat.shape == y.shape

    bparams = fit_bnb_smoothing(X, y, alpha=1e-8)
    logp2, yhat2 = predict_bnb(X, bparams)
    assert np.all(np.isfinite(logp2))
    assert yhat2.shape == y.shape
