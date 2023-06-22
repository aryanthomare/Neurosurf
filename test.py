import numpy as np
from numpy.fft import fft, ifft

# Set the seed for reproducibility (optional)
np.random.seed(42)

# Create a NumPy array with 100 random values
random_array = np.random.random(100)

fourier = fft(random_array)

sr = 5
N = len(fourier)
n = np.arange(N)
ts = 1.0/sr
T = N/sr
freq = n/T # array of all frequencies

fft_magnitudes = np.abs(fourier) # array of amplitudes

inv = ifft(fft_magnitudes)
print(inv.shape)