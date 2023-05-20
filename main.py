from src.linear_regression import GDLinearRegression, LinearRegression
from sklearn.linear_model import LinearRegression as SklearnLinearRegression

from src.preprocessing import city_price_encoding
from src.tester import base_test


def get_price(y):
    return y * y.std() + y.mean()


if __name__ == "__main__":
    models = {
        "GDLinearRegression": GDLinearRegression(learning_rate=0.05, threshold=1e-9),
        "LinearRegression": LinearRegression(),
        "SklearnLinearRegression": SklearnLinearRegression(),
    }
    for name, model in models.items():
        print(f"Testing {name}")
        base_test(model, city_price_encoding)
