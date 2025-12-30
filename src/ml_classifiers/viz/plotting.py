from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import matplotlib.pyplot as plt

from ml_classifiers.utils import ensure_dir


def save_figure(fig: plt.Figure, path: str | Path, dpi: int = 200, close: bool = True) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)
    return path


def plot_loss_curve(
    fvals: np.ndarray,
    path: str | Path,
    semilogy: bool = True,
    title: str = "",
    xlabel: str = "Iteration",
    ylabel: str = "Loss",
    label: str = "Loss",
) -> Path:
    fvals = np.asarray(fvals).reshape(-1)
    fig = plt.figure()
    if semilogy:
        plt.semilogy(fvals, label=label)
    else:
        plt.plot(fvals, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.legend()
    return save_figure(fig, path)


def plot_compare_curves(
    curves: Sequence[np.ndarray],
    labels: Sequence[str],
    path: str | Path,
    semilogy: bool = False,
    title: str = "",
    xlabel: str = "Iteration",
    ylabel: str = "Loss",
) -> Path:
    fig = plt.figure()
    for c, lab in zip(curves, labels):
        c = np.asarray(c).reshape(-1)
        if semilogy:
            plt.semilogy(c, label=lab)
        else:
            plt.plot(c, label=lab)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.legend()
    return save_figure(fig, path)


def plot_accuracy_sweep(
    alphas: np.ndarray,
    acc_train: np.ndarray,
    acc_test: np.ndarray,
    path: str | Path,
    title: str = "Accuracy vs smoothing",
) -> Path:
    alphas = np.asarray(alphas).reshape(-1)
    acc_train = np.asarray(acc_train).reshape(-1)
    acc_test = np.asarray(acc_test).reshape(-1)

    fig = plt.figure()
    plt.semilogx(alphas, acc_train, label="Train Accuracy")
    plt.semilogx(alphas, acc_test, label="Test Accuracy")
    plt.xlabel("alpha")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    return save_figure(fig, path)


def save_digit_grid(
    X_list: Sequence[np.ndarray],
    y_list: Sequence[int],
    path: str | Path,
    n_row: int = 3,
    n_col: int = 4,
    fsize: tuple[int, int] = (12, 10),
    cmap: str = "gray",
) -> Path:
    """
    Saves a grid montage of digit images stored as flattened vectors.
    """
    Xs = [np.asarray(x).reshape(-1) for x in X_list]
    ys = [int(v) for v in y_list]
    n = len(Xs)

    dim = int(np.sqrt(Xs[0].size))
    n_col_ = min(n_col, n)
    n_row_ = min(n_row, (n - 1) // n_col_ + 1)

    fig, axs = plt.subplots(n_row_, n_col_, figsize=fsize)
    axs = np.array(axs).reshape(-1)

    for i in range(n_row_ * n_col_):
        ax = axs[i]
        if i < n:
            ax.set_title(f"Label: {ys[i]}")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(Xs[i].reshape((dim, dim)), cmap=cmap)
        else:
            ax.set_visible(False)

    return save_figure(fig, path)
