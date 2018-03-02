import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_excel('mlr02.xls')

print(data.head)
# bias
data['ones'] = 1

# X = data.as_matrix()
X = data[['X2', 'X3', 'ones']]
Y = data['X1']

# plt.scatter(X[:, 1], X[:, 0])
# plt.show()


def get_r_squared(x, y):
    w = np.linalg.solve(np.dot(x.T, x), np.dot(x.T, y))
    yhat = np.dot(x, w)
    d1 = y - yhat
    d2 = y - y.mean()
    yi = 1
    r2 = yi - np.dot(d1, d1) / d2.dot(d2)
    return r2

print("r2 : ", get_r_squared(X, Y))

print()
