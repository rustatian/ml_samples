import numpy as np
import pandas as pd


def get_data(limit=None):
    print("Reading in and transforming data...")
    df = pd.read_csv('train.csv')
    data = df.as_matrix()
    np.random.shuffle(data)
    X = data[:, 1:] / 255.0
    Y = data[:, 0]
    if limit is not None:
        X, Y = X[:limit], Y[:limit]
    return X, Y


def get_test_data(limit=None):
    print("Reading in and transforming test data...")
    df = pd.read_csv('test.csv')
    data = df.as_matrix()
    np.random.shuffle(data)
    X = data[:, 1:] / 255.0
    Y = data[:, 0]
    if limit is not None:
        X, Y = X[:limit], Y[:limit]
    return X, Y


def get_xor():
    X = np.zeros((200, 2))
    X[:50] = np.random.random((50, 2)) / 2 + 0.5
    X[50:100] = np.random.random((50, 2)) / 2
    X[100:150] = np.random.random((50, 2)) / 2 + np.array([[0, 0.5]])
    X[150:] = np.random.random((50, 2)) / 2 + np.array([[0, 0.5]])
    Y = np.array(([0] * 100 + [1] * 100))
    return X, Y


def get_donut():
    N = 200
    Rinner = 5
    Routher = 20

    R1 = np.random.randn(N//2) + Rinner
    theta = 2 * np.pi * np.random.random(N//2)
    x_inner = np.concatenate([[R1 * np.cos(theta), R1 * np.sin(theta)]]).T

    R2 = np.random.randn(N//2) + Routher
    theta = 2 * np.pi * np.random.random(N//2)
    x_outher = np.concatenate([[R2 * np.cos(theta), R2 * np.sin(theta)]]).T

    X = np.concatenate([x_inner, x_outher])
    Y = np.array([0] * (N//2) + [1] * (N//2))

    return X, Y




