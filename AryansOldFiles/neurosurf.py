import numpy as np
import math
import pylsl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from typing import List
import pandas as pd
from numpy.fft import fft, ifft


plot_duration = 1  # how many seconds of data to show
figure_width = 12  # width of the figure in inches
figure_height = 6

view_size=200
plt.style.use('dark_background')

class Inlet:
    """Base class to represent a plottable inlet"""
    def __init__(self, info: pylsl.StreamInfo):
        # create an inlet and connect it to the outlet we found earlier.
        # max_buflen is set so data older the plot_duration is discarded
        # automatically and we only pull data new enough to show it

        # Also, perform online clock synchronization so all streams are in the
        # same time domain as the local lsl_clock()
        # (see https://labstreaminglayer.readthedocs.io/projects/liblsl/ref/enums.html#_CPPv414proc_clocksync)
        # and dejitter timestamps
        self.inlet = pylsl.StreamInlet(info, max_buflen=plot_duration,
                                       processing_flags=pylsl.proc_clocksync | pylsl.proc_dejitter)
        # store the name and channel count
        self.name = info.name()
        self.channel_count = info.channel_count()

    def pull_and_plot(self):
        """Pull data from the inlet and add it to the plot.
        :param plot_time: lowest timestamp that's still visible in the plot
        :param plt: the plot the data should be shown on
        """
        # We don't know what to do with a generic inlet, so we skip it.
        pass

class TestInlet():
    def __init__(self):
    # create an inlet and connect it to the outlet we found earlier.
    # max_buflen is set so data older the plot_duration is discarded
    # automatically and we only pull data new enough to show it

    # Also, perform online clock synchronization so all streams are in the
    # same time domain as the local lsl_clock()
    # (see https://labstreaminglayer.readthedocs.io/projects/liblsl/ref/enums.html#_CPPv414proc_clocksync)
    # and dejitter timestamps

        self.name = "Test"
        self.channel_count = 3
        self.all_data = np.zeros((1, self.channel_count))

        self.fig, self.ax = plt.subplots(self.channel_count,2,figsize=(figure_width, figure_height))
        self.lines = []
        for i in range(self.channel_count):
            line, = self.ax[i][0].plot([], [])
            self.lines.append(line)

        self.all_ts = np.zeros(1)

        #self.ax.set_ylim(-1, 1)  # Set the y-range
    def sort_sensor_data(self,timestamps, sensor_data):
        # Get the indices that would sort the timestamps array
        sorted_indices = np.argsort(timestamps, axis=0)

        # Sort the timestamps and sensor data arrays based on the sorted indices
        sorted_timestamps = timestamps[sorted_indices]
        sorted_sensor_data = sensor_data[sorted_indices]

        return sorted_timestamps, sorted_sensor_data
        
    def pull_and_plot(self,*fargs):
        n = np.random.randint(1, 6)  # Generate a random number between 1 and 5
        array = np.random.rand(n, 3)  # Create a random array of size n by 3 with values between 0 and 1
        times = np.zeros(n)
        for x in range(n):
            times[x] = time.time()


        self.all_data = np.concatenate((self.all_data, array), axis=0)
        self.all_ts = np.concatenate((self.all_ts, times), axis=0)
        self.last_viewsize_values=self.all_data[-view_size:, :]
        self.last_viewsize_timestamps = np.array([x for x in range(0,self.last_viewsize_values.shape[0])])

        self.last_viewsize_timestamps, self.last_viewsize_values = self.sort_sensor_data(self.last_viewsize_timestamps, self.last_viewsize_values)
        for i in range(0,self.channel_count):

            self.vals=self.last_viewsize_values[:,i]                
            self.lines[i].set_data(self.last_viewsize_timestamps, self.vals)
            self.ax[i][0].relim()
            self.ax[i][0].autoscale_view()
        plt.tight_layout()



class DataInlet(Inlet):
    
    """A DataInlet represents an inlet with continuous, multi-channel data that
    should be plotted as multiple lines."""
    dtypes = [[], np.float32, np.float64, None, np.int32, np.int16, np.int8, np.int64]

    def __init__(self, info: pylsl.StreamInfo):
        super().__init__(info)
        self.normal_rate = info.nominal_srate()

        self.all_data = np.zeros((1, info.channel_count()))
        self.fig, self.ax = plt.subplots(info.channel_count(),2,figsize=(figure_width, figure_height))
        self.lines = []
        for j in range(2):
            for i in range(self.channel_count):
                line, = self.ax[i][j].plot([], [])
                self.lines.append(line)

        self.all_ts = np.zeros(1)


        self.starttime = time.time()



    def sort_sensor_data(self,timestamps, sensor_data):
        # Get the indices that would sort the timestamps array
        sorted_indices = np.argsort(timestamps, axis=0)

        # Sort the timestamps and sensor data arrays based on the sorted indices
        sorted_timestamps = timestamps[sorted_indices]
        sorted_sensor_data = sensor_data[sorted_indices]

        return sorted_timestamps, sorted_sensor_data

    def pull_and_plot(self,*fargs):
        vals, ts = self.inlet.pull_chunk()
        if self.starttime-time.time() > 1:
            self.starttime = time.time()

        
        if ts:
            new = np.array(vals)
            times = np.array(ts)

            self.all_data = np.concatenate((self.all_data, new), axis=0)
            self.all_ts = np.concatenate((self.all_ts, times), axis=0)

            self.last_viewsize_values=self.all_data[-view_size:, :]
            self.last_viewsize_timestamps = self.all_ts[-view_size:]

            self.last_viewsize_timestamps, self.last_viewsize_values = self.sort_sensor_data(self.last_viewsize_timestamps, self.last_viewsize_values)





            #print(self.last_viewsize_timestamps[1]-self.last_viewsize_timestamps[0])
            for i in range(0,self.channel_count):
                self.vals=self.last_viewsize_values[:,i]               
                self.lines[i].set_data(self.last_viewsize_timestamps, self.vals)
                #print(self.last_viewsize_timestamps[1]-self.last_viewsize_timestamps[0])

                self.ax[i][0].set_ylim(np.amin(self.all_data), np.amax(self.all_data))  # Set the y-range
                self.ax[i][0].relim()
                self.ax[i][0].autoscale_view()

                            

                fourier = fft(self.last_viewsize_values)
                sr  = self.normal_rate

                N = len(fourier)
                n = np.arange(fourier)
                ts = 1.0/sr
                T = N/sr
                freq = n/T
                self.lines[i+3].set_data(freq, np.abs(fourier))
                self.ax[i][1].relim()
                self.ax[i][1].autoscale_view()


def main():
    inlets: List[Inlet] = []
    print("looking for streams")
    streams = pylsl.resolve_streams()
    for info in streams:
        if info.nominal_srate() != pylsl.IRREGULAR_RATE \
                and info.channel_format() != pylsl.cf_string:
            if info.type() == "EEG":
                print('Adding data inlet: ' + info.name())
                inlets.append(DataInlet(info))
        else:
            
            print('Don\'t know what to do with stream ' + info.name())
    if not inlets:
        inlets.append(TestInlet())


    plt.ion()  # Enable interactive mode
    
    while True:
        for inlet in inlets:
            inlet.pull_and_plot()

            plt.draw()
            plt.pause(0.1)

if __name__ == '__main__':
    main()