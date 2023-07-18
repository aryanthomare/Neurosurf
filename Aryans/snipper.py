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
from matplotlib.gridspec import GridSpec

channels = ['TP9', 'AF7', 'AF8', 'TP10', 'Right AUX']
counter = 0
figure_width = 12  # width of the figure in inches
figure_height = 6
offset = 0
offsets = [1,2,5,10,50,100,256]
offset_counter = 0
lines = [-1,-1]
categories = ['Delta', 'Theta', 'Alpha', 'Beta','Gamma']

print("""
    Press the left and right arrow keys to move the graph
    Press the up and down arrow keys to change the channel
    Press b to save the current graph to the blink csv file
    Press n to save the current graph to the calm csv file
    Press m to save the current graph to the focus csv file
    Press z to add a start line
    Press x to add an end line
    Press c to remove all lines
      """)

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
    with open(file, 'a', encoding='UTF8',newline='') as f:

        writer = csv.writer(f)
        writer.writerow(message)


def click_update_graph():
    
    ax1.cla()
    ax2.cla()
    ax3.cla()

    ax2.clear()  # Clear the current plot

    ax1.clear() 
    ax1.set_title(f'Channel: {channels[abs(counter) % 5]}, Offset Size: {offsets[offset_counter]}')





 # Clear the current plot
    ax1.plot(lis[:,-1][offset:offset+256],lis[:,abs(counter) % 5][offset:offset+256])
    fourier = rfft(lis[:,abs(counter) % 5][offset:offset+256],256)
    freq = rfftfreq(256, d=1/256)
    #L = np.arange(1,np.floor(256/2),dtype='int')
    PSD = fourier * np.conj(fourier) / 256


    ax2.plot(np.real(freq[L][0:50]), np.real(PSD[L][0:50]))        
    
    ax3.clear()
    ax3.bar(categories,get_powers(PSD,freq))

    ax1.axvline(x = lis[:,-1][offset+128], color = 'r', label = 'axvline - full height')
    if lines[0] != -1 and lis[:,-1][offset] <= lis[:,-1][lines[0]] <= lis[:,-1][offset+256]:
        ax1.axvline(x = lis[:,-1][lines[0]], color = 'g', label = 'axvline - full height')
    if lines[1] != -1 and lis[:,-1][offset] <= lis[:,-1][lines[1]] <= lis[:,-1][offset+256]:
        ax1.axvline(x = lis[:,-1][lines[1]], color = 'r', label = 'axvline - full height')

    plt.draw()


def press(event):
    global counter,offset,lines,offset_counter,offsets
    if event.key == 'left':
        if offset > 0:
            offset -=   offsets[offset_counter]




    if event.key == 'right':
        if offset < lis.shape[0] - 256:
            offset += offsets[offset_counter]

    if event.key == 'up':
        counter += 1

        
        
    if event.key == 'down':
        counter -= 1



    if event.key == 'b':
        if lines == [-1,-1]:
            fourier = rfft(lis[:,abs(counter) % 5][offset:offset+256],256)
            freq = rfftfreq(256, d=1/256)
            L = np.arange(1,np.floor(256/2),dtype='int')
            PSD = fourier * np.conj(fourier) / 256
            pows = get_powers(PSD,freq)
            message_writer(f'Neurosurf\\Aryans\\Exported_Values\\blinks\\blinks{channels[abs(counter) % 5]}.csv',pows)
        else:
            if lines[0] != -1 and lines[1] != -1 and lines[1] - lines[0] >= 256:
                for x in range(abs(lines[1] - lines[0])-256):
                    fourier = rfft(lis[:,abs(counter) % 5][lines[0]+x:lines[0]+x+256],256)
                    freq = rfftfreq(256, d=1/256)
                    L = np.arange(1,np.floor(256/2),dtype='int')
                    PSD = fourier * np.conj(fourier) / 256
                    pows = get_powers(PSD,freq)
                    message_writer(f'Neurosurf\\Aryans\\Exported_Values\\blinks\\blinks{channels[abs(counter) % 5]}.csv',pows)




    if event.key == 'n':
        if lines == [-1,-1]:
            fourier = rfft(lis[:,abs(counter) % 5][offset:offset+256],256)
            freq = rfftfreq(256, d=1/256)
            L = np.arange(1,np.floor(256/2),dtype='int')
            PSD = fourier * np.conj(fourier) / 256
            pows = get_powers(PSD,freq)
            message_writer(f'Neurosurf\\Aryans\\Exported_Values\\normal\\normal{channels[abs(counter) % 5]}.csv',pows)
        else:
            if lines[0] != -1 and lines[1] != -1 and lines[1] - lines[0] >= 256:
                for x in range(abs(lines[1] - lines[0])-256):
                    fourier = rfft(lis[:,abs(counter) % 5][lines[0]+x:lines[0]+x+256],256)
                    freq = rfftfreq(256, d=1/256)
                    L = np.arange(1,np.floor(256/2),dtype='int')
                    PSD = fourier * np.conj(fourier) / 256
                    pows = get_powers(PSD,freq)
                    message_writer(f'Neurosurf\\Aryans\\Exported_Values\\normal\\n{channels[abs(counter) % 5]}.csv',pows)

                    #message_writer(f'Neurosurf\\Aryans\\Exported_Values\\normal\\normal{channels[abs(counter) % 5]}.csv',pows)




    if event.key == 'z':
        lines[0] = offset+128
    if event.key == 'x':
         
        lines[1] = offset+128


    if event.key == 'c':
        lines = [-1,-1]



    if event.key == '+':
        if offset_counter < len(offsets)-1:
            offset_counter += 1


    if event.key == '-':
        if offset_counter > 0:
            offset_counter -= 1
    click_update_graph()

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



filename = 'calm3.csv'
lis = trim_file('Neurosurf\\Aryans\\DataFiles\\' + filename)
lis = sort_sensor_data(lis[:,-1],lis)
fig = plt.figure(figsize=(10, 6))


#fig, ax = plt.subplots(1,3,figsize=(figure_width, figure_height))
grid = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[2, 1])
ax1 = fig.add_subplot(grid[0, :])
ax2 = fig.add_subplot(grid[1, 0])
ax3 = fig.add_subplot(grid[1, 1])


fig.canvas.mpl_connect('key_press_event', press)

fourier = rfft(lis[:,abs(counter) % 5][offset:offset+256],256)
freq = rfftfreq(256, d=1/256)
L = np.arange(1,np.floor(256/2),dtype='int')
PSD = fourier * np.conj(fourier) / 256

click_update_graph()

plt.show()