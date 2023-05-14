from pylsl import StreamInlet, resolve_stream
import matplotlib.pyplot as plt
# import numpy as np
from time import time
import time as ttt
import csv
times = []
val = []
import winsound

res = input('Test: ')

def main(res):
    # first resolve an EEG stream on the lab network
    print("looking for an EEG stream...")
    streams = resolve_stream('type', 'PPG')
    # create a new inlet to read from the stream

    inlet = StreamInlet(streams[0])
    start = time()
    #print(start)
    run = True

    tp9sensor = []
    af8sensor = []
    af7sensor = []


    ttt.sleep(10)
    while run:
        ttt.sleep(0.1)
        # get a new sample (you can also omit the timestamp part if you're not
        # interested in it)
        sample, timestamp = inlet.pull_sample()
        #print('INLET',inlet.pull_sample())
        #print(timestamp - start,sample)

        times.append((time()-start))
        #val.append(sample[1])
        tp9sensor.append(sample[0])

        af7sensor.append(sample[1])
        af8sensor.append(sample[2])
        print(timestamp)
        if len(af7sensor) > 200:
            run = False
            # winsound.Beep(2000,1000)
            print("!!!")
        print(len(af7sensor))

    tp9 = plt.figure(1)
    plt.title("tp9")
    plt.plot(times,tp9sensor)



    af7 = plt.figure(3)
    plt.title("af7")

    plt.plot(times,af7sensor)
    af8 = plt.figure(4)
    plt.title("af8")

    plt.plot(times,af8sensor)


    plt.show()
if __name__ == '__main__':
    main(res)
