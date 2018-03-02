import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


def get_data():
    df = pd.read_csv('ecommerce_data.csv')
    data = df.as_matrix()

    # Input data
    X = data[:, :-1]
    # Output data
    Y = data[:, -1]

    # Normalize data (X - mu)/sigma
    X1data = X[:, 1]
    X1mean = X[:, 1].mean()  # mu
    X1stdev = X[:, 1].std()  # sigma
    X[:, 1] = (X1data - X1mean) / X1stdev

    X2data = X[:, 2]
    X2mean = X[:, 2].mean()  # mu
    X2stdev = X[:, 2].std()  # sigma
    X[:, 2] = (X2data - X2mean) / X2stdev

    N, D = X.shape
    X2 = np.zeros((N, D+3))

    # copy all data and exclude time column, prepare for one-hot encoding
    X2[:, 0:(D - 1)] = X[:, 0:(D - 1)]

    for n in range(N):
        # data in column is 0, 1, 2, 3
        # 0 - [1,0,0,0]
        # 1 - [0,1,0,0]
        # 2 - [0,0,1,0]
        # 3 - [0,0,0,1]

        t = int(X[n, (D - 1)])  # Get value from column
        X2[n, t+D-1] = 1  # Set value into one-hot encoding column

    return X2, Y


def get_binary_data():
    X, Y = get_data()
    X2 = X[Y <= 1]
    Y2 = Y[Y <= 1]
    return X2, Y2


X, Y = get_binary_data()
X, Y = shuffle(X, Y)

Xtrain = X[:-100]
Ytrain = Y[:-100]

Xtest = X[-100:]
Ytest = Y[-100:]


D = X.shape[1]
w = np.random.randn(D)
b = 0

# a = -Wx - b
def sigmoid(a):
    return 1 / (1 + np.exp(-a))


def forward(X, w, b):
    return sigmoid(X.dot(w) + b)


def cross_entropy(T, pY):
    return -np.mean(T*np.log(pY) + (1 - T)*np.log(1 - pY))

training_costs = []
test_consts = []
learning_rate = 0.001

for i in range(10000):
    pYtrain = forward(Xtrain, w, b)
    pYtest = forward(Xtest, w, b)
    ctrain = cross_entropy(Ytrain, pYtrain)
    ctest = cross_entropy(Ytest, pYtest)

    training_costs.append(ctrain)
    training_costs.append(ctest)

    w -= learning_rate*Xtrain.T.dot(pYtrain - Ytrain)
    b -= learning_rate*(pYtrain - Ytrain).sum()

    if i % 100 == 0:
        print("i: {ti}, ctest: {ct}, ctrain: {ctr}".format(ti=i, ct=ctest, ctr=ctrain))
