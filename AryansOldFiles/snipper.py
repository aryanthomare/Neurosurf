import numpy as np
import math
import pylsl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from typing import List
import csv
from scipy.fft import rfft,rfftfreq,irfft
import matplotlib.ticker as ticker
 # how many seconds of data to show

channels = ['TP9', 'AF7', 'AF8', 'TP10', 'Right AUX']
counter = 0
figure_width = 12  # width of the figure in inches
figure_height = 6
offset = 0




def press(event):
    global counter,offset
    print('press', event.key)

    if event.key == 'left':
        if offset > 0:
            offset -= 10
        ax[0].set_title(channels[abs(counter) % 5])

        ax[0].clear()  # Clear the current plot
        ax[0].plot(lis[:,-1][offset:offset+256],lis[:,abs(counter) % 5][offset:offset+256])
        plt.draw()

    if event.key == 'right':
        if offset < lis.shape[0] - 256:
            offset += 10
        ax[0].set_title(channels[abs(counter) % 5])

        ax[0].clear()  # Clear the current plot
        ax[0].plot(lis[:,-1][offset:offset+256],lis[:,abs(counter) % 5][offset:offset+256])
        plt.draw()


    if event.key == 'up':

        counter += 1
        ax[0].clear()  # Clear the current plot
        ax[0].set_title(channels[abs(counter) % 5])

        ax[0].plot(lis[:,-1][offset:offset+256],lis[:,abs(counter) % 5][offset:offset+256])
        plt.draw()
    if event.key == 'down':
        counter -= 1
        ax[0].clear()  # Clear the current plot
        ax[0].set_title(channels[abs(counter) % 5])

        ax[0].plot(lis[:,-1][offset:offset+256],lis[:,abs(counter) % 5][offset:offset+256])
        plt.draw()

def trim_file(file):
    #open csv file and save rows in a list
    with open(file, 'r') as file:
        reader = csv.reader(file)
        lines = np.array(list(reader),dtype=float)

        lines = lines[ lines[:, -1]>= 0]
        return lines


def sort_sensor_data(timestamps, sensor_data):
        # Get the indices that would sort the timestamps array
        sorted_indices = np.argsort(timestamps, axis=0)

        # Sort the timestamps and sensor data arrays based on the sorted indices
        sorted_timestamps = timestamps[sorted_indices]
        sorted_sensor_data = sensor_data[sorted_indices]

        return sorted_sensor_data




lis = trim_file('test.csv')
lis = sort_sensor_data(lis[:,-1],lis)

fig, ax = plt.subplots(1,2,figsize=(figure_width, figure_height))
ax[0].set_title(channels[abs(counter) % 5])
fig.canvas.mpl_connect('key_press_event', press)
ax[0].plot(lis[:,-1][offset:offset+256],lis[:,abs(counter) % 5][offset:offset+256])

plt.show()