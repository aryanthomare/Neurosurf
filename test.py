import numpy as np
import pywt
import matplotlib.pyplot as plt

# Create a 5 Hz wave
duration = 1.0  # Duration of the wave in seconds
sampling_rate = 1000  # Number of samples per second
t = np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)
frequency = 5  # Frequency of the wave in Hz
amplitude = 1  # Amplitude of the wave
wave = amplitude * np.sin(2 * np.pi * frequency * t)

# Set wavelet transform parameters
wavelet = 'db4'  # Choose the discrete wavelet (here, we use Daubechies-4)
level = 1  # Number of decomposition levels (scales)

# Apply the wavelet transform
coeffs = pywt.wavedec(wave, wavelet, level=level)

# Calculate the amplitudes from the wavelet coefficients
amplitudes = [np.abs(c) for c in coeffs]

# Plot the amplitude-frequency graph
plt.figure(figsize=(10, 6))
for i, amp in enumerate(amplitudes):
    freq = np.arange(len(amp)) * (sampling_rate / len(amp))
    plt.plot(freq, amp, label='Scale {}'.format(i+1))

# Add vertical line at x = 5
plt.axvline(x=5, color='r', linestyle='--', label='5 Hz')

plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Amplitude-Frequency Graph')
plt.legend()
plt.grid(True)
plt.show()