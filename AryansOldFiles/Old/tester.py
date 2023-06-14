import numpy as np
import matplotlib.pyplot as plt
import time
# # Generate some sample sensor values
# sensor_values = np.random.rand(1000)

# # Compute the FFT
# fft_values = np.fft.fft(sensor_values)

# # Compute the magnitudes of the FFT values
# fft_magnitudes = np.abs(fft_values)

# # Generate the frequencies corresponding to the FFT values
# freqs = np.fft.fftfreq(len(sensor_values))
# print(fft_values)
# # Plot the magnitudes of the FFT values
# plt.plot(freqs, fft_magnitudes)
# plt.xlabel('Frequency')
# plt.ylabel('Magnitude')
# plt.title('FFT Amplitudes')
# plt.grid(True)
# plt.show()
import csv  

# header = ['name', 'area', 'country_code2', 'country_code3']
# data = ['Afghanistan', 652090, 'AF', 'AFG']

# with open('countries.csv', 'w', encoding='UTF8') as f:
#     writer = csv.writer(f)

#     # write the header
#     writer.writerow(header)

#     # write the data
#     writer.writerow(data)

new = np.array([1,
                2,
                3])
x = np.rot90(new, k=1, axes=(0,1))
print(x)