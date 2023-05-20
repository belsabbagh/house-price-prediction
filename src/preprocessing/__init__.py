def city_price_encoding(df):
    df2 = df[['city', 'price']]
    average_prices = {city: prices['price'].mean() for city, prices in df2.groupby('city')}
    df['city'] = df['city'].map(average_prices)
    X, y = df.loc[:, df.columns != "price"], df["price"]
    X = X.drop(["date", "street", "statezip", "country"], axis=1, inplace=False)
    X = (X - X.mean()) / X.std()
    y = (y - y.mean()) / y.std()
    return X, y
