from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import scipy.special


@dataclass(frozen=True)
class GNBParams:
    Mu: np.ndarray   # (d, K)
    Var: np.ndarray  # (d, K)
    Pi: np.ndarray   # (K,)


def fit_gnb(X: np.ndarray, y: np.ndarray) -> GNBParams:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    n, d = X.shape
    K = int(y.max()) + 1

    Mu = np.zeros((d, K), dtype=float)
    Var = np.zeros((d, K), dtype=float)
    Pi = np.zeros((K,), dtype=float)

    for k in range(K):
        idx = (y == k)
        Xk = X[idx]
        Pi[k] = Xk.shape[0] / n
        Mu[:, k] = np.mean(Xk, axis=0)
        Var[:, k] = np.mean((Xk - Mu[:, k]) ** 2, axis=0)

    return GNBParams(Mu=Mu, Var=Var, Pi=Pi)


def fit_gnb_smoothing(X: np.ndarray, y: np.ndarray, alpha: float) -> GNBParams:
    """
    Applies variance smoothing as:
      Var_smoothed = Var + (alpha / d) * max_{j,k} Var[j,k]
    """
    params = fit_gnb(X, y)
    Mu, Var, Pi = params.Mu, params.Var, params.Pi

    d = Var.shape[0]
    vmax = float(np.max(Var))
    eps = (alpha / d) * vmax
    Var_s = Var + eps

    return GNBParams(Mu=Mu, Var=Var_s, Pi=Pi)


def predict_gnb(X: np.ndarray, params: GNBParams) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict using log probabilities for numerical stability.
    Returns (log_posterior, yhat).
    log_posterior shape: (n, K)
    """
    X = np.asarray(X, dtype=float)
    Mu = np.asarray(params.Mu, dtype=float)   # (d, K)
    Var = np.asarray(params.Var, dtype=float) # (d, K)
    Pi = np.asarray(params.Pi, dtype=float)   # (K,)

    n, d = X.shape
    K = Mu.shape[1]

    # Avoid log(0) and division by 0
    Var = np.maximum(Var, 1e-12)
    Pi = np.maximum(Pi, 1e-12)

    log_joint = np.zeros((n, K), dtype=float)

    # Vectorized per-class computation
    # For each k: log N(x | mu_k, var_k) + log pi_k
    const = -0.5 * d * np.log(2.0 * np.pi)
    for k in range(K):
        mu_k = Mu[:, k]
        var_k = Var[:, k]
        log_det = -0.5 * np.sum(np.log(var_k))
        quad = -0.5 * np.sum(((X - mu_k) ** 2) / var_k, axis=1)
        log_joint[:, k] = const + log_det + quad + np.log(Pi[k])

    log_norm = scipy.special.logsumexp(log_joint, axis=1, keepdims=True)
    log_post = log_joint - log_norm
    yhat = np.argmax(log_post, axis=1).astype(int)
    return log_post, yhat


def generate_gnb(params: GNBParams, label: int, clip_min: float = 0.0, clip_max: float | None = None) -> np.ndarray:
    Mu = np.asarray(params.Mu, dtype=float)
    Var = np.asarray(params.Var, dtype=float)
    label = int(label)

    x = np.random.normal(loc=Mu[:, label], scale=np.sqrt(np.maximum(Var[:, label], 1e-12)))
    if clip_min is not None:
        x = np.maximum(x, clip_min)
    if clip_max is not None:
        x = np.minimum(x, clip_max)
    return x
