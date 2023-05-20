import numpy as np
from scipy import stats


def city_price_encoding(df):
    df = df[(np.abs(stats.zscore(df["price"])) < 2.8)]
    df2 = df[["city", "price"]]
    average_prices = {city: p["price"].mean() for city, p in df2.groupby("city")}
    df.insert(len(df.columns),"city_price", df["city"].map(average_prices))
    X, y = df.loc[:, df.columns != "price"], df["price"]
    X = X.drop(["date", "street", "statezip", "country", "city"], axis=1, inplace=False)
    X = (X - X.mean()) / X.std()
    y = (y - y.mean()) / y.std()
    return X, y
