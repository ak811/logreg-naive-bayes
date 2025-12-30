from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import numpy as np
import scipy.linalg
import scipy.special


def _logits_km1(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Build full K logits using (K-1) weight rows and a fixed 0 logit for the last class.
    If W is (K-1, d), returns logits (n, K) where last column is zeros.
    """
    X = np.asarray(X, dtype=float)
    W = np.asarray(W, dtype=float)
    n = X.shape[0]
    Km1 = W.shape[0]
    logits_km1 = X @ W.T  # (n, K-1)
    zeros = np.zeros((n, 1), dtype=float)
    return np.hstack([logits_km1, zeros])  # (n, K)


def loss_km1(W: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """
    Cross-entropy loss under K-1 parameterization.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    logits = _logits_km1(W, X)
    log_probs = scipy.special.log_softmax(logits, axis=1)
    n = X.shape[0]
    return float(-np.mean(log_probs[np.arange(n), y]))


def grad_km1(W: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Gradient for K-1 parameterization:
      grad = (1/n) * (P - Y)_{[:, :K-1]}^T X
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    W = np.asarray(W, dtype=float)

    logits = _logits_km1(W, X)              # (n, K)
    P = scipy.special.softmax(logits, axis=1)  # (n, K)
    n, d = X.shape
    K = P.shape[1]
    Km1 = K - 1

    Y = np.zeros((n, K), dtype=float)
    Y[np.arange(n), y] = 1.0

    E = (P - Y)[:, :Km1]  # (n, K-1)
    grad = E.T @ X
    grad /= n
    return grad


def hessian_km1(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Hessian for K-1 parameterization:
      H = (1/n) * sum_i [ (R_i) ⊗ (x_i x_i^T) ]
    where R_i = diag(p_i[:K-1]) - p_i[:K-1] p_i[:K-1]^T
    """
    X = np.asarray(X, dtype=float)
    W = np.asarray(W, dtype=float)

    n, d = X.shape
    logits = _logits_km1(W, X)
    P = scipy.special.softmax(logits, axis=1)  # (n, K)
    Km1 = W.shape[0]

    H = np.zeros((Km1 * d, Km1 * d), dtype=float)

    for i in range(n):
        p = P[i, :Km1]  # (K-1,)
        R = np.diag(p) - np.outer(p, p)  # (K-1, K-1)
        xxT = np.outer(X[i], X[i])       # (d, d)
        H += np.kron(R, xxT)

    H /= n
    return H


def predict_km1(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    logits = _logits_km1(W, X)
    return np.argmax(logits, axis=1).astype(int)


@dataclass
class NewtonResult:
    W_list: List[np.ndarray]
    f_vals: np.ndarray  # (niter,)


def newton_method(
    f: Callable[[np.ndarray], float],
    gradf: Callable[[np.ndarray], np.ndarray],
    hessf: Callable[[np.ndarray], np.ndarray],
    W0: np.ndarray,
    niter: int = 20,
    ridge: float = 1e-9,
) -> NewtonResult:
    """
    Newton / IRLS iterations:
      W_{k+1} = W_k - reshape( H^{-1} vec(grad), (K-1, d) )

    ridge: small diagonal regularization to stabilize solves.
    """
    W = np.asarray(W0, dtype=float).copy()
    Km1, d = W.shape

    fvals = np.zeros((niter,), dtype=float)
    W_list: List[np.ndarray] = []

    for i in range(niter):
        W_list.append(W.copy())
        fvals[i] = f(W)

        g = gradf(W).reshape(-1, 1)  # ((K-1)*d, 1)
        H = hessf(W)                 # ((K-1)*d, (K-1)*d)

        # Stabilize if needed
        if ridge > 0:
            H = H + ridge * np.eye(H.shape[0], dtype=float)

        # Solve H delta = g
        try:
            delta = scipy.linalg.solve(H, g, assume_a="sym")
        except Exception:
            # Fallback to least squares if something goes sideways
            delta = scipy.linalg.lstsq(H, g)[0]

        W = W - delta.reshape((Km1, d))

    return NewtonResult(W_list=W_list, f_vals=fvals)
