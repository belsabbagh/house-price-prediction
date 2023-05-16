"""
Module for models.
"""
import numpy as np
from .fitting_functions import add_constant, gradient_descent, least_squares


class UnfittedModelException(Exception):
    """Exception raised when prediction is attempted before fitting."""

    def __init__(self):
        super().__init__("Model is not fitted yet. Call fit() method first.")


class LinearRegression:
    """Linear regression model."""

    weights: list[float | int] | None = None

    def __init__(self, add_intercept=True):
        self.add_intercept = add_intercept

    @staticmethod
    def __predict(x, weights):
        if isinstance(x, (int, float)):
            return weights[1] * x + weights[0]
        return weights[0] + np.dot(x, weights[1:])

    def _pre_fit(self, x, y):
        if len(x) != len(y):
            raise ValueError("Length of x and y must be the same.")

        return add_constant(x) if self.add_intercept else x, y

    def fit(self, x, y):
        x, y = self._pre_fit(x, y)
        self.weights = least_squares(x, y)

    def predict(self, x):
        if self.weights is None:
            raise UnfittedModelException()
        return self.__predict(x, self.weights)

    def __repr__(self) -> str:
        return f"LinearRegression(b={self.weights[0]}, W={self.weights[1:]})"


class GDLinearRegression(LinearRegression):
    rate: float | int = 0.05
    threshold: float | int = 1e-15
    max_iter: int | None = None

    def __init__(
        self, learning_rate=0.05, threshold=1e-15, max_iter=None, verbose=False
    ):
        super().__init__()
        if learning_rate <= 0:
            raise ValueError("Learning rate must be greater than 0.")
        if threshold <= 0:
            raise ValueError("Threshold must be greater than 0.")
        self.rate = learning_rate
        self.threshold = threshold
        self.max_iter = max_iter
        self.verbose = verbose

    def fit(self, x, y):
        x, y = self._pre_fit(x, y)
        self.weights = gradient_descent(
            x,
            y,
            self.rate,
            self.threshold,
            max_iter=self.max_iter,
            verbose=self.verbose,
        )


class LogisticRegression(GDLinearRegression):
    """Logistic regression model."""

    def __init__(self, learning_rate=0.05, threshold=1e-6, max_iter=None):
        super().__init__(learning_rate, threshold, max_iter)

    @staticmethod
    def __predict(x, weights):
        return 1 / (1 + np.exp(x.dot(weights[1:]) + weights[0]))

    @staticmethod
    def __pred(x, w):
        return np.array(
            [1 if i > 0.5 else 0 for i in LogisticRegression.__predict(x[:, 1:], w)]
        )

    @staticmethod
    def __cost(y_pred, y):
        """Logistic Regression cost function"""
        return sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)) / -len(y)

    def fit(self, X, y):
        X, y = self._pre_fit(X, y)
        self.weights = gradient_descent(
            X,
            y,
            self.rate,
            self.threshold,
            pred=self.__pred,
            cost_fn=self.__cost,
            max_iter=self.max_iter,
        )

    def predict(self, X):
        return 1 if self.__predict(X, self.weights) > 0.5 else 0
