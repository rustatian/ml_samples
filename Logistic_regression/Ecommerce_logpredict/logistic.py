import numpy as np
import matplotlib.pyplot as plt

N = 200
D = 2

X = np.random.randn(N, D)

# Bias term
ones = np.array([[1]*N]).T

# Concatenate with input X
Xb = np.concatenate((ones, X), axis=1)

# Randomly initialize w
w = np.random.randn(D)
z = X.dot(w)


def sigmoid(z):
    return 1/(1 + np.exp(-z))


def hypertan(z):
    return (np.exp(z) - np.exp(-z))/(np.exp(z) + np.exp(-z))


print(sigmoid(z))
plt.plot(sigmoid(z))
plt.show()

print(hypertan(z))
plt.plot(hypertan(z))
plt.show()

