import numpy as np
import pandas as pd


def add_constant(x: np.ndarray):
    return np.append(np.ones((len(x), 1)), x, axis=1)


def least_squares(X: pd.DataFrame, y):
    """
    Return the best fit line for the given data.
    It works by finding the slope of the line of best fit and then using the slope to find the y-intercept.

    DISCLAIMER: I know it's very sensitive to outliers but it's so much faster than gradient descent.
    """
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)


def found_min(cost, prev_cost, threshold, max_iter, i):
    diff = abs(prev_cost - cost) if prev_cost is not None else 0
    return diff < threshold and diff != 0 or (max_iter is not None and i >= max_iter)


def mse(y_pred, y):
    return sum([(i - j) ** 2 for i, j in zip(y, y_pred)]) / len(y)


def gradient(n, x, y, y_pred):
    """
    Derivative of the mean squared error cost function.
    The derivative of the MSE cost function is the negative of the average of the product of the input and the error.
    """
    return -(1 / n) * sum(x * (y - y_pred))


def update_weights(rate, n, X, y, y_pred, weights):
    """Update weights using the derivative of the cost function."""
    return [w - rate * gradient(n, X.T[i], y, y_pred) for i, w in enumerate(weights)]


def gradient_descent(
    X: np.ndarray,
    y,
    rate=0.05,
    threshold=1e-9,
    max_iter=None,
    pred=None,
    cost_fn=mse,
    verbose=False,
):
    pred = pred or (lambda x, w: x.dot(w))
    weights = least_squares(X, y)
    n, _ = X.shape
    i, prev_cost, cost = 1, None, None
    while not found_min(cost, prev_cost, threshold, max_iter, i):
        prev_cost = cost if cost is not None else 0
        y_pred = pred(X, weights)
        cost = cost_fn(y_pred, y)
        weights = update_weights(rate, n, X, y, y_pred, weights)
        if verbose:
            print(f"{i}: {dict(cost=cost)}")
        i += 1
    if verbose:
        print(f"Gradient Descent: It took {i} iterations to converge.")
    return np.array(weights)
