import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor


N = 200
X = np.linspace(0, 10, N).reshape(N, 1)
Y = np.sin(X)

Ntrain = 20
idx = np.random.choice(N, Ntrain)

Xtrain = X[idx]
Ytrain = Y[idx]

knn = KNeighborsRegressor(n_neighbors=2, weights='distance')
knn.fit(Xtrain, Ytrain)
Yknn = knn.predict(X)

dt = DecisionTreeRegressor()
dt.fit(Xtrain, Ytrain)
Ydt = dt.predict(X)

plt.scatter(Xtrain, Ytrain)
plt.plot(X, Y)
plt.plot(X, Yknn, label='knn')
plt.plot(X, Ydt, label='dt')
plt.legend()
plt.show()




