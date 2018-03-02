import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from scipy.signal import convolve2d

img = mpimg.imread('lena.png')
bw = img.mean(axis=2)

Hx = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]],
    dtype=np.float32)
Hy = Hx.T

Gx = convolve2d(bw, Hx)

plt.imshow(Gx, cmap='gray')
plt.show()

Gy = convolve2d(bw, Hy)

plt.imshow(Gy, cmap='gray')
plt.show()

G = np.sqrt(Gx*Gx + Gy*Gy)
theta = np.arctan2(Gx, Gy)

plt.imshow(G, cmap='gray')
plt.show()

plt.imshow(theta, cmap='gray')
plt.show()

#  https://en.wikipedia.org/wiki/Edge_detection
#  https://en.wikipedia.org/wiki/Sobel_operator
