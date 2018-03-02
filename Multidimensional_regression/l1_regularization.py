import numpy as np
import matplotlib.pyplot as plt

N = 50
D = 50

X = (np.random.random((N, D)) - 0.5)*10
true_w = np.array([1, 0.5, -0.5] + [0]*(D - 3))

Y = X.dot(true_w) + np.random.randn(N)*0.5

costs = []
w = np.random.randn(D) / np.sqrt(D)
learning_rate = 0.001

l1 = 10.0

for t in range(500):
    Yhat = X.dot(w)
    delta = Yhat - Y
    w = w - learning_rate*(X.T.dot(delta) +l1*np.sign(w))
    mse = delta.dot(delta) / N
    costs.append(mse)

print("W: ", w)
plt.plot(costs)
plt.show()

plt.plot(true_w, label='true_w')
plt.plot(w, label='w')
plt.legend()
plt.show()
