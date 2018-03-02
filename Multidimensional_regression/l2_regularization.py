import numpy as np
import matplotlib.pyplot as plt

N = 50
X = np.linspace(0, 10, N)
Y = 0.5*X + np.random.randn(N)

Y[-1] += 30
Y[-2] += 30

plt.scatter(X, Y)
plt.show()

# Solve for best weights X.T + b (1)
X = np.vstack([np.ones(N), X]).T
w_ml = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
Yhat_ml = X.dot(w_ml)

plt.scatter(X[:, 1], Y)
plt.plot(X[:, 1], Yhat_ml)
plt.show()

l2 = 1500.0
# w_map = np.random.randn(50) / np.sqrt(50)
# learning_rate = 0.001
# costs = []
#
#
# for t in range(1000):
#     Yhat = X.dot(w_map)
#     delta = Yhat - Y
#     w = w_map - learning_rate*X.T.dot(delta)
#     mse = delta.dot(delta) / N
#     costs.append(mse)

w_map = np.linalg.solve(l2*np.eye(2) + X.T.dot(X), X.T.dot(Y))

Yhat_map = X.dot(w_map)

plt.scatter(X[:, 1], Y)
plt.plot(X[:, 1], Yhat_ml, label='maximum likelihood')
plt.plot(X[:, 1], Yhat_map, label='l2')
plt.show()
