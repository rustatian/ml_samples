# If you right-click on the file and go to "Get Info", you can see:
# sampling rate = 16000 Hz
# bits per sample = 16
# The first is quantization in time
# The second is quantization in amplitude
# We also do this for images!
# 2^16 = 65536 is how many different sound levels we have
# 2^8 * 2^8 * 2^8 = 2^24 is how many different colors we can represent

import matplotlib.pyplot as plt
import numpy as np
import wave
import sys
from scipy.signal import fftconvolve
from scipy.io.wavfile import write


spf = wave.open('helloworld.wav', 'r')

signal = spf.readframes(-1)
signal = np.fromstring(signal, 'Int16')
print("Numpy signal shape:", signal.shape)

# plt.plot(signal)
# plt.show()

delta = np.array([1., 0., 0., ])
noecho = fftconvolve(signal, delta)

print("Noecho shape:", noecho.shape)

# plt.plot(noecho)
# plt.show()

noecho = noecho.astype(np.int16)
write('noecho.wav', 16000, noecho)

filt = np.zeros(16000)
filt[0] = 1
filt[4000] = 0.6
filt[8000] = 0.3
filt[12000] = 0.2
filt[15999] = 0.1

out = fftconvolve(signal, filt)

out = out.astype(np.int16)
write('out.wav', 16000, out)




