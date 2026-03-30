import numpy as np
import tensorflow as tf


# Direct tensor operations (TF2 eager mode — no Session needed)
A = tf.constant(np.random.rand(5, 5), dtype=tf.float32)
v = tf.constant(np.random.randn(5, 1), dtype=tf.float32)

# Matrix multiplication
w = tf.matmul(A, v)
print(w.numpy())

# Variables are directly usable in eager mode
shape = (2, 2)
x = tf.Variable(tf.random.normal(shape))
print('----------')
print(x.numpy())

# Gradient descent with GradientTape
u = tf.Variable(20.0)

for i in range(200):
    with tf.GradientTape() as tape:
        cost = u * u + u + 1
    grad = tape.gradient(cost, u)
    u.assign_sub(0.03 * grad)
    print("Cost: {cst}, u:{yu}".format(cst=cost.numpy(), yu=u.numpy()))
