from pylsl import StreamInlet, resolve_stream
import matplotlib.pyplot as plt
import numpy as np
from time import time
import time as ttt
import csv
import winsound
from matplotlib.animation import FuncAnimation


times = []
val = []



# first resolve an EEG stream on the lab network
print("looking for an EEG stream...")
streams = resolve_stream('type', 'Accelerometer')

# create a new inlet to read from the stream
inlet = StreamInlet(streams[0])
start = time()

#print(start)


sensor = [0 for x in range(500)]

ttt.sleep(5)

fig = plt.figure(1)
plt.title("tp9")


def func():
    # get a new sample (you can also omit the timestamp part if you're not
    # interested in it)
    sample, timestamp = inlet.pull_sample()
    print(sample)
    #print('INLET',inlet.pull_sample())
    #print(timestamp - start,sample)

    times.append((time()-start))
    #val.append(sample[1])
    sensor.append(sample[0])

    print(timestamp)


    plt.plot(sensor[-100:])



    # af7 = plt.figure(3)
    # plt.title("af7")

    # plt.plot(times,af7sensor)
    # af8 = plt.figure(4)
    # plt.title("af8")

    # plt.plot(times,af8sensor)

ani = FuncAnimation(fig, func, interval=1000)
plt.show()

