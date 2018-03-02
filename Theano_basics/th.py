import theano.tensor as T
import numpy as np
import theano
from Theano_basics.util import y2indicator, get_normalized_data


# Types
c = T.scalar('c')
v = T.vector('v')
A = T.matrix('A')

# Multiplication
w = A.dot(v)

# Multiply matrix and vectors
m_v = theano.function(inputs=[A, v], outputs=w)

a_val = np.array([[1, 2], [3, 4]])
v_val = np.array([5, 6])
w_val = m_v(a_val, v_val)
print(w_val)

# Updatable variable (in theano called shared variable)
x = theano.shared(20.0, 'x')

# Cost formulae
cost = x.dot(x) + x + 1

# How to update (Gradient), where cost - function to take the gradient of, x - variable to gradient with respect to
# can be list
x_update = x - 0.3 * T.grad(cost, x)

# Theano train function
# 1 - The shared variable to update (x), 2 - the update expression (x_update)
train = theano.function(inputs=[], outputs=cost, updates=[(x, x_update)])

for i in range(250):
    cost_val = train()
    print(cost_val)

# Print the optimal value for x
print(x.get_value())
print(theano.printing.debugprint(train))





