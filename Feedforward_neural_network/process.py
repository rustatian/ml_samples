import numpy as np
import pandas as pd


def get_data():
    df = pd.read_csv('ecommerce_data.csv')
    data = df.as_matrix()

    X = data[:, :-1]
    Y = data[:, -1]

    # Data normalization
    X[:, 1] = (X[:, 1] - X[:, 1].mean()) / np.std(X[:, 1])
    X[:, 2] = (X[:, 2] - X[:, 2].mean()) / np.std(X[:, 2])


    # Working with categorical values

    N, D = X.shape

    # Add dimensions for one-hot encoding
    X2 = np.zeros((N, D+3))
    X2[:, 0:(D-1)] = X[:, 0:(D-1)]

    for n in range(N):
        t = int(X[n, D-1])
        X2[n, t+D-1] = 1
    return X2, Y


def get_binary():
    X, Y = get_data()
    X2 = X[Y <= 1]
    Y2 = Y[Y <= 1]
    return X2, Y2

get_data()
