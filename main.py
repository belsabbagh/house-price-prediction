from src.linear_regression import GDLinearRegression, LinearRegression
from sklearn.linear_model import LinearRegression as SklearnLinearRegression

from src.tester import base_test


def get_price(y):
    return y * y.std() + y.mean()


if __name__ == "__main__":
    models = {
        "SklearnLinearRegression": SklearnLinearRegression(),
        "GDLinearRegression": GDLinearRegression(learning_rate=0.5, threshold=1e-4, verbose=True),
    }
    for name, model in models.items():
        print(f"Testing {name}")
        base_test(model)
