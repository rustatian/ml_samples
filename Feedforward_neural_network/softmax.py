import numpy as np

a = np.random.randn(5)
expa = np.exp(a)

answer = expa / expa.sum()
print(answer.sum())  # Sum must be equal 1 (probability)

# 100 by 5 array

A = np.random.randn(100, 5)
expA = np.exp(A)
Answer = expA / expA.sum(axis=1, keepdims=True)

print(Answer.sum(axis=1))
