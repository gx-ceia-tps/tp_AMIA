import numpy as np
from sklearn.datasets import load_iris, fetch_openml


def get_iris_dataset():
    data = load_iris()
    X_full = data.data
    y_full = np.array([data.target_names[y] for y in data.target.reshape(-1, 1)])
    return X_full, y_full


def get_penguins():
    # get data
    df, tgt = fetch_openml(name="penguins", return_X_y=True, as_frame=True, parser='auto')

    # drop non-numeric columns
    df.drop(columns=["island", "sex"], inplace=True)

    # drop rows with missing values
    mask = df.isna().sum(axis=1) == 0
    df = df[mask]
    tgt = tgt[mask]

    return df.values, tgt.to_numpy().reshape(-1, 1)
