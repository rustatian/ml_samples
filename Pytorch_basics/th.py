import torch
import numpy as np


# Tensor types (analogous to Theano's scalar/vector/matrix)
c = torch.tensor(3.0)
v = torch.tensor([5.0, 6.0])
A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

# Matrix-vector multiplication
w = A @ v
print(w)

# Direct tensor operations with random data
A_rand = torch.randn(5, 5)
v_rand = torch.randn(5, 1)
w_rand = A_rand @ v_rand
print(w_rand)

# Gradient descent — minimize cost = x^2 + x + 1
# Optimal: x = -0.5 (where derivative 2x + 1 = 0)
x = torch.tensor(20.0, requires_grad=True)
optimizer = torch.optim.SGD([x], lr=0.3)

for i in range(250):
    optimizer.zero_grad()
    cost = x * x + x + 1
    cost.backward()
    optimizer.step()
    print(cost.item())

# Print the optimal value for x
print("Optimal x:", x.item())
