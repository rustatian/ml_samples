import numpy as np
import matplotlib.pyplot as plt

N = 10
D = 3

X = np.zeros((N, D))

# bias
X[:, 0] = 1
X[:5, 1] = 1
X[:5, 2] = 1
print(X)

# output 0 for the first half and 1 for the second
Y = np.array([0]*5 + [1]*5)
print(Y)

# w = np.linalg.solve(X.T.dot(X), X.T.dot(Y))

costs = []

w = np.random.randn(D) / np.sqrt(D)
learning_rate = 0.001

for t in range(1000):
    Yhat = X.dot(w)
    delta = Yhat - Y
    w = w - learning_rate*X.T.dot(delta)
    mse = delta.dot(delta) / N
    costs.append(mse)

plt.plot(costs)
plt.show()

plt.plot(Yhat, label='Yhat')
plt.plot(Y, label='predictions')
plt.legend()
plt.show()
