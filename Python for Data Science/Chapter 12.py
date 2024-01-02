import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing


def load_california_housing_data():
    dataset = fetch_california_housing()
    X = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
    y = pd.DataFrame(data=dataset.target, columns=["target"])
    return X, y


X, y = load_california_housing_data()
print(f"X:{X.shape}, y:{y.shape}")


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X)
scaled_X = scaler.transform(X)


from sklearn.linear_model import LinearRegression

linear_regression = LinearRegression()
linear_regression.fit(scaled_X, y)
print(linear_regression.coef_.round(5))


values = [[1.21315, 32., 3.31767135, 1.07731985, 898., 2.1424809, 37.82, -122.27]]
obs = pd.DataFrame(values, columns=X.columns)

scaled_obs = scaler.transform(obs)

pred = linear_regression.predict(scaled_obs)
value = pred[0][0] * 100_000
print(f"Estimated median house value: {value: .2f} USD")

print(linear_regression.score(scaled_X, y))
