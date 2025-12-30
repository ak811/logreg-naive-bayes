from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def one_hot(y: np.ndarray, K: Optional[int] = None) -> np.ndarray:
    """
    One-hot encode labels y of shape (n,).
    Returns array of shape (n, K).
    """
    y = np.asarray(y).astype(int)
    if K is None:
        K = int(y.max()) + 1
    oh = np.zeros((y.shape[0], K), dtype=float)
    oh[np.arange(y.shape[0]), y] = 1.0
    return oh


def accuracy(yhat: np.ndarray, y: np.ndarray) -> float:
    yhat = np.asarray(yhat).astype(int)
    y = np.asarray(y).astype(int)
    return float(np.mean(yhat == y))


@dataclass(frozen=True)
class FigurePaths:
    root: Path = Path("outputs/figures")

    def __post_init__(self):
        ensure_dir(self.root)

    def path(self, name: str) -> Path:
        if not name.lower().endswith(".png"):
            name = f"{name}.png"
        return self.root / name
