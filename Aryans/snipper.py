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
from matplotlib.widgets import Button

channels = ['TP9', 'AF7', 'AF8', 'TP10', 'Right AUX']
counter = 0
figure_width = 12  # width of the figure in inches
figure_height = 6
offset = 0



def get_powers(PSD,freq):

    """Calculating the frequency spectrum powers
    Delta: 1-4 hz
    Theta: 4-8 hz
    Alpha: 8-13 hz
    Beta: 13-30 hz
    Gamma: 30-80 hz
    """
    delta = np.sum(np.real(PSD * ((freq < 4) & (freq >= 0.5)).astype(int)))  #1-4 hz
    theta = np.sum(np.real(PSD *((freq < 7) & (freq >= 4)).astype(int))) #4-8 hz
    alpha = np.sum(np.real(PSD *((freq < 13) & (freq >= 7)).astype(int))) #8-13 hz
    beta =  np.sum(np.real(PSD *((freq < 30) & (freq >= 13)).astype(int))) #13-30 hz
    gamma = np.sum(np.real(PSD *((freq <= 50) & (freq >= 30)).astype(int))) #30-80 hz

    final = [delta,theta,alpha,beta,gamma]
    return final

def message_writer(file,message):
    # #print(f"quote update {message}")
    with open(file, 'a', encoding='UTF8',newline='') as f:

        writer = csv.writer(f)
        writer.writerow(message)


def click_update_graph():
    ax[1].clear()  # Clear the current plot

    ax[0].clear() 
    ax[0].set_title(channels[abs(counter) % 5])
 # Clear the current plot
    ax[0].plot(lis[:,-1][offset:offset+256],lis[:,abs(counter) % 5][offset:offset+256])
    fourier = rfft(lis[:,abs(counter) % 5][offset:offset+256],256)
    freq = rfftfreq(256, d=1/256)
    #L = np.arange(1,np.floor(256/2),dtype='int')
    PSD = fourier * np.conj(fourier) / 256


    ax[1].plot(np.real(freq[L][0:50]), np.real(PSD[L][0:50]))        
        
    plt.draw()


def press(event):
    global counter,offset
    if event.key == 'left':
        if offset > 0:
            offset -= 10
        click_update_graph()
        


    if event.key == 'right':
        if offset < lis.shape[0] - 256:
            offset += 10
        click_update_graph()

    if event.key == 'up':
        counter += 1
        click_update_graph()
        
        
    if event.key == 'down':
        counter -= 1
        click_update_graph()


    if event.key == 'b':
        print("Saved to Blink")
        fourier = rfft(lis[:,abs(counter) % 5][offset:offset+256],256)
        freq = rfftfreq(256, d=1/256)
        L = np.arange(1,np.floor(256/2),dtype='int')
        PSD = fourier * np.conj(fourier) / 256
        pows = get_powers(PSD,freq)
        message_writer(f'Neurosurf\\Aryans\\Exported_Values\\blinks\\blinks{channels[abs(counter) % 5]}.csv',pows)
    
    if event.key == 'n':
        print("Saved to Normal")
        fourier = rfft(lis[:,abs(counter) % 5][offset:offset+256],256)
        freq = rfftfreq(256, d=1/256)
        L = np.arange(1,np.floor(256/2),dtype='int')
        PSD = fourier * np.conj(fourier) / 256
        pows = get_powers(PSD,freq)
        message_writer(f'Neurosurf\\Aryans\\Exported_Values\\normal\\normal{channels[abs(counter) % 5]}.csv',pows)


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

print

filename = 't3.csv'
lis = trim_file('Neurosurf\\Aryans\\DataFiles\\' + filename)
lis = sort_sensor_data(lis[:,-1],lis)

fig, ax = plt.subplots(1,2,figsize=(figure_width, figure_height))


ax[0].set_title(channels[abs(counter) % 5])
fig.canvas.mpl_connect('key_press_event', press)
# Set up the button subplot


fourier = rfft(lis[:,abs(counter) % 5][offset:offset+256],256)
freq = rfftfreq(256, d=1/256)
L = np.arange(1,np.floor(256/2),dtype='int')
PSD = fourier * np.conj(fourier) / 256
ax[1].plot(np.real(freq[L][0:50]), np.real(PSD[L][0:50]))        
        

ax[0].plot(lis[:,-1][offset:offset+256],lis[:,abs(counter) % 5][offset:offset+256])
#ax[1].set_xlim(freq[L[0]],50)

plt.show()