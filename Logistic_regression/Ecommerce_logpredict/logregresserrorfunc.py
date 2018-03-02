import numpy as np
import matplotlib.pyplot as plt

N = 100
D = 2

X = np.random.rand(N, D)

X[:50, :] = X[:50, :] - 2*np.ones((50, D))
X[50:, :] = X[50:, :] + 2*np.ones((50, D))

T = np.array([0]*50 + [1]*50)

ones = np.array([[1]*N]).T
Xb = np.concatenate((ones, X), axis=1)
w = np.random.rand(D + 1)

Z = Xb.dot(w)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

Y = sigmoid(Z)


def cross_entropy(T, Y):
    E = 0
    for i in range(N):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1 - Y[i])
    return E

print(cross_entropy(T, Y))

# Discriminant line
plt.scatter(X[:, 0], X[:, 1], c=T, alpha=0.5, s=100)
x_axis = np.linspace(-6, 6, 100)
y_axis = -x_axis
plt.plot(x_axis, y_axis)
plt.show()

# GD

learning_rate = 0.1
for i in range(10000):
    if i % 10 == 0:
        print(cross_entropy(T, Y))

    w += learning_rate * Xb.T.dot(T - Y)
    Y = sigmoid(Xb.dot(w))

print("Final w: ", w)




