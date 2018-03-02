import numpy as np

X = np.array([[1, 2]], dtype=np.float64)  #1x2
W1 = np.array([[1, 1], [1, 0]], dtype=np.float64)  #2x2
W2 = np.array([[0, 1], [1, 1]], dtype=np.float64)  #2x2


N = 1
D = 2

def softmax(a):
    expA = np.exp(a)
    return expA / expA.sum(axis=1, keepdims=True)


def sigmoid(a):
    return 1 / (1 + np.exp(-a))


def forward(X, W1, W2):
    Z = np.tanh(X.dot(W1))
    A = softmax(Z.dot(W2))
    return A


P_Y_given_X = forward(X, W1, W2)


print(P_Y_given_X)
