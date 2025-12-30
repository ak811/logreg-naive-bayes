from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.special


@dataclass(frozen=True)
class BNBParams:
    Phi: np.ndarray  # (d, K)  feature conditional prob: P(x_j != 0 | y=k)
    Pi: np.ndarray   # (K,)


def fit_bnb_smoothing(X: np.ndarray, y: np.ndarray, alpha: float) -> BNBParams:
    """
    Lidstone / Laplace-style smoothing:
      Phi[j,k] = (count_nonzero_feature_in_class + alpha) / (N_k + alpha*d)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    Xb = (X != 0).astype(int)

    n, d = X.shape
    K = int(y.max()) + 1

    Phi = np.zeros((d, K), dtype=float)
    Pi = np.zeros((K,), dtype=float)

    for k in range(K):
        idx = (y == k)
        Xk = Xb[idx]  # (N_k, d)
        Nk = Xk.shape[0]
        Pi[k] = Nk / n
        counts = np.sum(Xk, axis=0)  # (d,)
        Phi[:, k] = (counts + alpha) / (Nk + alpha * d)

    return BNBParams(Phi=Phi, Pi=Pi)


def predict_bnb(X: np.ndarray, params: BNBParams) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict using log probabilities:
      log p(y=k|x) ∝ log pi_k + sum_j [ x_j log Phi_jk + (1-x_j) log(1-Phi_jk) ]
    Returns (log_posterior, yhat)
    """
    X = np.asarray(X, dtype=float)
    Xb = (X != 0).astype(int)

    Phi = np.asarray(params.Phi, dtype=float)
    Pi = np.asarray(params.Pi, dtype=float)

    # clip to avoid log(0)
    Phi = np.clip(Phi, 1e-12, 1 - 1e-12)
    Pi = np.clip(Pi, 1e-12, 1.0)

    n, d = Xb.shape
    K = Phi.shape[1]

    # log_joint = log Pi + X log Phi + (1-X) log(1-Phi)
    log_Phi = np.log(Phi)           # (d, K)
    log_1mPhi = np.log(1.0 - Phi)   # (d, K)

    log_joint = np.log(Pi)[None, :] + (Xb @ log_Phi) + ((1 - Xb) @ log_1mPhi)  # (n, K)

    log_norm = scipy.special.logsumexp(log_joint, axis=1, keepdims=True)
    log_post = log_joint - log_norm
    yhat = np.argmax(log_post, axis=1).astype(int)
    return log_post, yhat


def generate_bnb(params: BNBParams, label: int) -> np.ndarray:
    """
    Sample a binary feature vector x ~ Bernoulli(Phi[:, label]).
    """
    Phi = np.asarray(params.Phi, dtype=float)
    Phi = np.clip(Phi, 1e-12, 1 - 1e-12)
    label = int(label)

    p = Phi[:, label]
    return (np.random.rand(p.shape[0]) < p).astype(float)
