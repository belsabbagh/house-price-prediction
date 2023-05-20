import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder


def mse(y_pred, y):
    return sum([(i - j) ** 2 for i, j in zip(y, y_pred)]) / len(y)


def format_score(scores):
    return "%.3f +/- %.3f" % (np.mean(scores), np.std(scores))


def score_test(y_pred: np.ndarray, y_test):
    return {"mse": mse(y_pred, y_test)}


def train_test_iter(X, y, n_splits):
    kf = KFold(n_splits=n_splits, random_state=1, shuffle=True)
    for train_index, test_index in kf.split(X):
        yield (
            X.iloc[train_index],
            X.iloc[test_index],
            y.iloc[train_index],
            y.iloc[test_index],
        )


def cross_validate(n_splits, model, X, y, verbose=False):
    scores = {"mse": []}
    for X_train, X_test, y_train, y_test in train_test_iter(X, y, n_splits):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results = score_test(y_pred, y_test)
        scores = {k: scores[k] + [results[k]] for k in scores}
        if verbose:
            print(f"mse: {results['mse']}")
    return scores


def load_dataset(preprocess=None):
    if preprocess is None:
        preprocess = lambda x: x
    df = pd.read_csv("data/house-price-data.csv")
    X, y = preprocess(df)
    return X, y


def base_test(model, preprocess=None):
    X, y = load_dataset(preprocess)
    scores = cross_validate(5, model, X, y)
    for k, v in scores.items():
        print(f"{k}: {format_score(v)}")
