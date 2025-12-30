from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris


def load_iris_data() -> tuple[np.ndarray, np.ndarray]:
    X, y = load_iris(return_X_y=True)
    return X.astype(float), y.astype(int)


def add_intercept(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    ones = np.ones((X.shape[0], 1), dtype=float)
    return np.hstack([ones, X])
