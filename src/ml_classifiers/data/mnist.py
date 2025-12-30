from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def load_mnist_openml(
    cache_dir: str | Path = "openml_cache",
    as_frame: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads MNIST (mnist_784) from OpenML via scikit-learn.
    Uses local caching to avoid repeated downloads.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    mnist = fetch_openml(
        "mnist_784",
        as_frame=as_frame,
        data_home=str(cache_dir),
    )
    X = mnist.data.astype(float)
    y = mnist.target
    y = to_int_labels(y)
    return X, y


def to_int_labels(y: np.ndarray) -> np.ndarray:
    """
    OpenML often returns labels as strings. Convert to int.
    """
    y = np.asarray(y)
    if y.dtype.kind in {"U", "S", "O"}:
        y = y.astype(int)
    return y.astype(int)


def split_mnist(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
