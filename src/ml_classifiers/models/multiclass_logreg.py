from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np
import scipy.special


def loss_softmax(W: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """
    Multiclass logistic regression (softmax) cross-entropy loss:
    L(W) = - (1/n) * sum_i log p(y_i | x_i)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    W = np.asarray(W, dtype=float)

    logits = X @ W.T  # (n, K)
    log_probs = scipy.special.log_softmax(logits, axis=1)
    n = X.shape[0]
    return float(-np.mean(log_probs[np.arange(n), y]))


def grad_softmax(W: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Gradient of softmax cross-entropy:
    ∇W L = (1/n) * (P - Y)^T X
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    W = np.asarray(W, dtype=float)

    logits = X @ W.T  # (n, K)
    P = scipy.special.softmax(logits, axis=1)  # (n, K)
    n, d = X.shape
    K = W.shape[0]

    Y = np.zeros((n, K), dtype=float)
    Y[np.arange(n), y] = 1.0

    grad = (P - Y).T @ X
    grad /= n
    return grad


def predict_softmax(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    W = np.asarray(W, dtype=float)
    logits = X @ W.T
    return np.argmax(logits, axis=1).astype(int)


@dataclass
class GDResult:
    W_list: List[np.ndarray]
    f_vals: np.ndarray  # (niter,)


def gradient_descent(
    f: Callable[[np.ndarray], float],
    gradf: Callable[[np.ndarray], np.ndarray],
    W0: np.ndarray,
    alpha: float,
    niter: int = 100,
) -> GDResult:
    W = np.asarray(W0, dtype=float).copy()
    fvals = np.zeros((niter,), dtype=float)
    W_list: List[np.ndarray] = []

    for i in range(niter):
        W_list.append(W.copy())
        fvals[i] = f(W)
        W = W - alpha * gradf(W)

    return GDResult(W_list=W_list, f_vals=fvals)
