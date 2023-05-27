import numpy as np
import pandas as pd
from scipy import stats


def city_price_encoding(df):
    df = df[(np.abs(stats.zscore(df["price"])) < 2.8)]
    df2 = df[["city", "price"]]
    average_prices = {city: p["price"].mean() for city, p in df2.groupby("city")}
    df.insert(len(df.columns), "city_price", df["city"].map(average_prices))
    X, y = df.loc[:, df.columns != "price"], df["price"]
    X = X.drop(["date", "street", "statezip", "country", "city"], axis=1, inplace=False)
    X = (X - X.min()) / (X.max() - X.min())
    y = (y - y.min()) / (y.max() - y.min())
    return X, y


def frequency_encoding(df):
    price_distribution = np.abs(stats.zscore(df["price"]))
    df = df[(price_distribution < 2.9) & (price_distribution > -2.9)]
    city_counts = df["city"].map(df["city"].value_counts())
    df.insert(len(df.columns), "city_counts", city_counts)
    df = df[city_counts > 100]
    X, y = df.loc[:, df.columns != "price"], df["price"]
    X = X.drop(["date", "street", "statezip", "country", "city"], axis=1, inplace=False)
    X = (X - X.min()) / (X.max() - X.min())
    y = (y - y.min()) / (y.max() - y.min())
    return X, y


def one_hot_encoding(df):
    X, y = df.loc[:, df.columns != "price"], df["price"]
    X = pd.get_dummies(X, columns=["city"])
    X = X.drop(["date", "street", "statezip", "country"], axis=1, inplace=False)
    X = (X - X.min()) / (X.max() - X.min())
    y = (y - y.min()) / (y.max() - y.min())
    return X, y
