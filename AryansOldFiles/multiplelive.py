from pylsl import StreamInlet, resolve_stream
import matplotlib.pyplot as plt
import numpy as np
from time import time
import time as ttt
import csv
import winsound
import matplotlib.animation as animation


times = []
val = []



# first resolve an EEG stream on the lab network
print("looking for an EEG stream...")
streams = resolve_stream('type', 'Accelerometer')

# create a new inlet to read from the stream
inlet = StreamInlet(streams[0])
start = time()

#print(start)


fig, ax = plt.subplots()
line1, = ax.plot([], [], label='Line 1')
line2, = ax.plot([], [], label='Line 2')
line3, = ax.plot([], [], label='Line 3')

# Initialize the data
x_data = []
y_data1 = []
y_data2 = []
y_data3 = []

# Set the y-axis limits
ax.set_ylim(-1, 1)

ax.set_xlim(0, 20)

def update_plot(frame):
    sample, timestamp = inlet.pull_sample()
    global x_data, y_data1, y_data2, y_data3  # Declare the variables as global


    # Update the data
    x_data.append(frame)
    y_data1.append(sample[0])
    y_data2.append(sample[1])
    y_data3.append(sample[2])
    
    # Keep only the 200 most recent values
    x_data = x_data[-20:]
    y_data1 = y_data1[-20:]
    y_data2 = y_data2[-20:]
    y_data3 = y_data3[-20:]
    
    line1.set_data(x_data, y_data1)
    line2.set_data(x_data, y_data2)
    line3.set_data(x_data, y_data3)
    ax.relim()
    ax.autoscale_view()
    return line1, line2, line3


# Create the animation
animation = animation.FuncAnimation(fig, update_plot, frames=range(100), interval=100, blit=True)
ax.legend()
# Show the plot
plt.show()