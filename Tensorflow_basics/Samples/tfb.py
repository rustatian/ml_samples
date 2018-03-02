import numpy as np
import tensorflow as tf


# Placeholder for variable in tensorflow
A = tf.placeholder(dtype=tf.float32, shape=(5, 5), name='A')

v = tf.placeholder(tf.float32)

# Matrix multiplication
w = tf.matmul(A, v)

# Session in tensorflow
with tf.Session() as sess:
    output = sess.run(w, feed_dict={A: np.random.rand(5, 5), v: np.random.randn(5, 1)})
    print(output)

shape = (2, 2)

# Variable (updatable) in tensorflow
x = tf.Variable(tf.random_normal(shape))
t = tf.Variable(0)

# You must initialize all of the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print('----------')
    print(x.eval())

# Cost function in tensorflow
u = tf.Variable(20.0)
cost = u*u + u + 1

# train
train_op = tf.train.GradientDescentOptimizer(0.03).minimize(cost)

# Don't forget to initialize variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(200):
        sess.run(train_op)
        print("Cost: {cst}, u:{yu}".format(cst=cost.eval(), yu=u.eval()))



