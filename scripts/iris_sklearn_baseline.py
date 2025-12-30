from __future__ import annotations

from scripts._bootstrap import add_src_to_path
add_src_to_path()

import numpy as np
import sklearn.linear_model

from ml_classifiers.data.iris import load_iris_data, add_intercept
from ml_classifiers.models.multiclass_logreg import loss_softmax
from ml_classifiers.utils import accuracy


def main() -> None:
    X, y = load_iris_data()

    # no intercept
    clf = sklearn.linear_model.LogisticRegression(penalty=None, fit_intercept=False, solver="lbfgs", max_iter=500)
    clf.fit(X, y)
    W = clf.coef_
    yhat = clf.predict(X)
    print("sklearn softmax LR | no intercept")
    print("Accuracy:", accuracy(yhat, y))
    print("Loss:", loss_softmax(W, X, y))
    print("W:\n", W)

    # with intercept: combine into single matrix [b | W]
    clf_b = sklearn.linear_model.LogisticRegression(penalty=None, fit_intercept=True, solver="lbfgs", max_iter=500)
    clf_b.fit(X, y)
    Wb = np.hstack([clf_b.intercept_.reshape(-1, 1), clf_b.coef_])
    Xb = add_intercept(X)
    yhat_b = clf_b.predict(X)
    print("\nsklearn softmax LR | with intercept")
    print("Accuracy:", accuracy(yhat_b, y))
    print("Loss:", loss_softmax(Wb, Xb, y))
    print("Wb:\n", Wb)


if __name__ == "__main__":
    main()
