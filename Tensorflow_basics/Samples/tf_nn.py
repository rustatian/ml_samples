import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import better_exceptions


def get_normalized_data():
    print("Reading in and transforming data...")
    df = pd.read_csv('train.csv')
    data = df.as_matrix().astype(np.float32)
    np.random.shuffle(data)
    X = data[:, 1:]
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    np.place(std, std == 0, 1)
    X = (X - mu) / std  # normalize the data
    Y = data[:, 0]
    return X, Y


def y2indicator(y):
    N = len(y)
    ind = np.zeros((N, 10))
    for i in range(N):
        ind[i, int(y[i])] = 1
    return ind


def error_rate(p, t):
    return np.mean(p != t)


def relu(a):
    return a * (a > 0)


def main():
    X, Y = get_normalized_data()
    max_iter = 20
    print_period = 10
    lr = 0.00004
    reg = 0.01

    Xtrain = X[:-1000, ]
    Ytrain = Y[:-1000]
    Xtest = X[-1000:, ]
    Ytest = Y[-1000:]
    Ytrain_ind = y2indicator(Ytrain)
    Ytest_ind = y2indicator(Ytest)

    N, D = Xtrain.shape
    batch_sz = 500
    n_batches = N / batch_sz

    M = 300
    K = 10
    W1_init = np.random.randn(D, M) / 28
    b1_init = np.zeros(M)
    W2_init = np.random.randn(M, K) / np.sqrt(M)
    b2_init = np.zeros(K)

    X = tf.placeholder(dtype=tf.float32, shape=(None, D), name='X')
    T = tf.placeholder(dtype=tf.float32, shape=(None, K), name='T')
    W1 = tf.Variable(W1_init.astype(np.float32))
    b1 = tf.Variable(b1_init.astype(np.float32))
    W2 = tf.Variable(W2_init.astype(np.float32))
    b2 = tf.Variable(b2_init.astype(np.float32))

    Z = tf.nn.relu(tf.matmul(X, W1) + b1)
    Yish = tf.matmul(Z, W2) + b2

    cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=T, logits=Yish))
    train = tf.train.GradientDescentOptimizer(lr).minimize(cost)

    prediction = tf.argmax(Y, axis=1)

    LL = []
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(max_iter):
            for j in range(int(n_batches)):
                Xbatch = Xtrain[j * batch_sz:(j * batch_sz + batch_sz), ]
                Ybatch = Ytrain_ind[j * batch_sz:(j * batch_sz + batch_sz), ]
                sess.run(train, feed_dict={X: Xbatch, T: Ybatch})

                if j % print_period == 0:
                    cost_val = sess.run(cost, feed_dict={X: Xtest, T: Ytest_ind})
                    err = error_rate(prediction, Ytest)
                    print("Iteration: {iter}, Cost: {cst}, Error: {error}".format(iter=i, cst=cost_val,
                                                                                  error=cost_val / err))
                    LL.append(cost_val)

    plt.plot(LL)
    plt.show()


if __name__ == '__main__':
    main()
