import pandas as pd
from hmmlearn import hmm
import numpy as np
from sklearn.model_selection import train_test_split


df = pd.read_csv("data/house-price-data.csv")
df["city"] = df["city"].astype("category")
enc_data = pd.get_dummies(df["city"])
X, y = df.loc[:, df.columns != "price"], df["price"]
X = X.drop(["date", "street", "city", "statezip", "country"], axis=1, inplace=False)
X = (X - X.mean()) / X.std()
y = (y - y.mean()) / y.std()
X = X.join(enc_data)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Create an HMM model
model = hmm.GaussianHMM(n_components=3, covariance_type="diag")

# Train the HMM model on the dataset
model.fit(X_train)

# Predict the most likely hidden state sequence for the training data
# hidden_states_train = model.predict(X_train)

# Predict the most likely hidden state sequence for the test data
hidden_states_test = model.predict(X_test)

# Generate predictions for each data point using the corresponding hidden state
predictions_train = []
for i, state in enumerate(3):
    prediction = model.means_[state]  # Assuming the means represent the predictions
    predictions_train.append(prediction)

predictions_test = []
for i, state in enumerate(hidden_states_test):
    prediction = model.means_[state]  # Assuming the means represent the predictions
    predictions_test.append(prediction)

# Print the predictions
print("Training predictions:", predictions_train)
print("Testing predictions:", predictions_test)
