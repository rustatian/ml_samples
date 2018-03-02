import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def get_transformed_data():
    print("Reading in and transforming data...")
    df = pd.read_csv('train.csv')
    data = df.as_matrix().astype(np.float32)
    np.random.shuffle(data)

    X = data[:, 1:]
    mu = X.mean(axis=0)
    X = X - mu  # center the data
    pca = PCA()
    Z = pca.fit_transform(X)
    Y = data[:, 0]

    return Z, Y, pca, mu


def get_normalized_data():
    print("Reading in and transforming data...")
    df = pd.read_csv('train.csv')
    data = df.as_matrix().astype(np.float32)
    np.random.shuffle(data)
    X = data[:, 1:]
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    np.place(std, std == 0, 1)
    X = (X - mu) / std  # normalize the data
    Y = data[:, 0]
    return X, Y


def forward(X, W, b):
    a = X.dot(W) + b
    expa = np.exp(a)
    y = expa / expa.sum(axis=1, keepdims=True)
    return y


def predict(p_y):
    return np.argmax(p_y, axis=1)


def error_rate(p_y, t):
    prediction = predict(p_y)
    return np.mean(prediction != t)


def cost(p_y, t):
    tot = t * np.log(p_y)
    return -tot.sum()


def gradW(t, y, X):
    return X.T.dot(t - y)


def gradb(t, y):
    return (t - y).sum(axis=0)


def y2indicator(y):
    N = len(y)
    ind = np.zeros((N, 10))
    for i in range(N):
        ind[i, int(y[i])] = 1
    return ind
