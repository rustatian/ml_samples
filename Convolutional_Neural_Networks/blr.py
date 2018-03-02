import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# Custom Convolution
def conv2d(X, W):
    n1, n2 = X.shape
    m1, m2 = W.shape
    y = np.zeros((n1 + m1 - 1, n2 + m2 - 1))
    for i in range(n1 + m1 - 1):
        for ii in range(m1):
            for j in range(m2 + n2 - 1):
                for jj in range(m2):
                    if i >= ii and j >= jj and i - ii < n1 and j - jj < n2:
                        y[i, j] = W[ii, jj] * X[i - ii, j - jj]

    return y

img = mpimg.imread('lena.png')

bw = img.mean(axis=2)
W = np.zeros((20, 20))

for i in range(20):
    for j in range(20):
        dist = (i - 9.5)**2 + (j - 9.5)**2
        W[i, j] = np.exp(-dist / 50)

out = conv2d(bw, W)# convolve2d(in1=bw, in2=W)


plt.imshow(out, cmap='gray')
plt.show()

# Conv2d on all 3 color channels
out3 = np.zeros(img.shape)

for i in range(3):
    out3[:, :, i] = convolve2d(in1=img[:, :, i], in2=W, mode='same')

plt.imshow(out3)
plt.show()






