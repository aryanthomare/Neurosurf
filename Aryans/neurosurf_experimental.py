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


figure_width = 12  # width of the figure in inches
figure_height = 6

view_size=256

record = True



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
        self.inlet = pylsl.StreamInlet(info,
                                       processing_flags=pylsl.proc_clocksync | pylsl.proc_dejitter)
        # store the name and channel count
        self.name = info.name()
        self.channel_count = info.channel_count()


    def message_writer(self,message):
        # #print(f"quote update {message}")
        with open('self.filename', 'w', encoding='UTF8') as f:

            writer = csv.writer(f)
            writer.writerow(message)

    def pull_and_plot(self):
        """Pull data from the inlet and add it to the plot.
        :param plot_time: lowest timestamp that's still visible in the plot
        :param plt: the plot the data should be shown on
        """
        # We don't know what to do with a generic inlet, so we skip it.
        pass
    def normalize(self,data):
        max_val = np.max(data)
        normalized_data = data/max_val

        return np.real(normalized_data)
    

class DataInlet(Inlet):
    
    """A DataInlet represents an inlet with continuous, multi-channel data that
    should be plotted as multiple lines."""
    dtypes = [[], np.float32, np.float64, None, np.int32, np.int16, np.int8, np.int64]
    def __init__(self, info: pylsl.StreamInfo,record):
        super().__init__(info)

        self.normal_rate = 895#info.nominal_srate()
        print(self.normal_rate)
        self.channel_count = info.channel_count()
        self.all_data = np.zeros((1, info.channel_count()))
        self.fig, self.ax = plt.subplots(info.channel_count(),2,figsize=(figure_width, figure_height))
        self.lines = []

        self.categories = ['Delta', 'Theta', 'Alpha', 'Beta','Gamma']

        self.name = info.type()
        for i in range(self.channel_count):
            line, = self.ax[i][0].plot([], [])
            self.lines.append(line)
            self.bars = self.ax[i][1].bar(self.categories, np.ones(len(self.categories)))


        self.record = record
        self.all_ts = np.zeros(1)
        self.filename = f"{self.name}_File_{time.time()}.csv"
        self.starttime = time.time()
        self.tick_spacing = 10

        for i in range(self.channel_count):
            self.ax[i][0].set_title(f"Fourier Channel {i}")
            self.ax[i][0].grid(True)
            self.ax[i][0].set_xlabel('Freq (Hz)', fontsize=12, fontweight='bold')
            self.ax[i][0].set_ylabel('Amplitude', fontsize=12, fontweight='bold')
            self.ax[i][0].xaxis.set_major_locator(ticker.MultipleLocator(self.tick_spacing))    





    def message_writer(self,message):
        # #print(f"quote update {message}")
        with open(self.filename, 'a', encoding='UTF8',newline='') as f:

            writer = csv.writer(f)
            writer.writerow(message)

    def filter_data(self, target_frequency,tol):

        idx = ((self.freq <= target_frequency-tol) | (self.freq >= target_frequency+tol)).astype(int)
        # maxfilter = self.fourier > 50
        # idx = idx * maxfilter        
        self.fourier = self.fourier * idx
        self.vals = irfft(self.fourier,view_size)
        
    def get_powers(self):

        """Calculating the frequency spectrum powers
        Delta: 1-4 hz
        Theta: 4-8 hz
        Alpha: 8-13 hz
        Beta: 13-30 hz
        Gamma: 30-80 hz
        """
        delta = np.sum(np.real(self.PSD * ((self.freq < 4) & (self.freq >= 0.5)).astype(int)))  #1-4 hz
        theta = np.sum(np.real(self.PSD *((self.freq < 7) & (self.freq >= 4)).astype(int))) #4-8 hz
        alpha = np.sum(np.real(self.PSD *((self.freq < 13) & (self.freq >= 7)).astype(int))) #8-13 hz
        beta =  np.sum(np.real(self.PSD *((self.freq < 30) & (self.freq >= 13)).astype(int))) #13-30 hz
        gamma = np.sum(np.real(self.PSD *((self.freq <= 50) & (self.freq >= 30)).astype(int))) #30-80 hz

        final = (self.normalize([delta,theta,alpha,beta,gamma]))
        return final



    def sort_sensor_data(self,timestamps, sensor_data):
        # Get the indices that would sort the timestamps array
        sorted_indices = np.argsort(timestamps, axis=0)

        # Sort the timestamps and sensor data arrays based on the sorted indices
        sorted_timestamps = timestamps[sorted_indices]
        sorted_sensor_data = sensor_data[sorted_indices]

        return sorted_timestamps, sorted_sensor_data

    def pull_and_plot(self,*fargs):
        vals, ts = self.inlet.pull_chunk()        
        for i in range(len(ts)):
            ts[i] = ts[i]-self.starttime
        
        if ts:



            new = np.array(vals)
            times = np.array(ts)




            self.all_data = np.concatenate((self.all_data, new), axis=0)
            self.all_ts = np.concatenate((self.all_ts, times), axis=0)

            self.last_viewsize_values=self.all_data[-view_size:, :]
            self.last_viewsize_timestamps = self.all_ts[-view_size:]
            self.last_viewsize_timestamps, self.last_viewsize_values = self.sort_sensor_data(self.last_viewsize_timestamps, self.last_viewsize_values)

            for i in range(0,self.channel_count):
                self.vals=self.last_viewsize_values[:,i]               
                self.fourier = rfft(self.vals,view_size)
                self.freq = rfftfreq(view_size, d=1/self.normal_rate)
                self.L = np.arange(1,np.floor(view_size/2),dtype='int')

                self.PSD = self.fourier * np.conj(self.fourier) / view_size


                


                self.lines[i].set_data(np.real(self.freq[self.L]), np.real(self.PSD[self.L]))
                self.ax[i][0].set_xlim(self.freq[self.L[0]],50)
                self.ax[i][0].relim()
                self.ax[i][0].autoscale_view()


                self.ax[i][1].clear()
                self.ax[i][1].bar(self.categories, self.get_powers())

            if self.record:

                combined_array = np.hstack((new, times[:, np.newaxis]))
                columns = []
                for i in range(combined_array.shape[0]):
                    column = combined_array[i,:]
                    columns.append(column)
                for each in columns:
                    self.message_writer(each)
    




def main():
    inlets: List[Inlet] = []
    print("looking for streams")
    streams = pylsl.resolve_streams()
    for info in streams:
        if info.nominal_srate() != pylsl.IRREGULAR_RATE \
                and info.channel_format() != pylsl.cf_string:
            if info.type() == "EEG":
                print('Adding data inlet: ' + info.name())
                inlets.append(DataInlet(info,record))
        else:
            
            print('Don\'t know what to do with stream ' + info.name())
    if not inlets:
        inlets.append(TestInlet(record))


    plt.ion()  # Enable interactive mode
    
    while True:
        for inlet in inlets:
            inlet.pull_and_plot()
            plt.draw()
            plt.pause(0.1)

if __name__ == '__main__':
    main()