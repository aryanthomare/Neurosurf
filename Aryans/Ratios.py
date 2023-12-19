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

record = False



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

        self.normal_rate = info.nominal_srate()
        print(self.normal_rate)
        self.channel_count = info.channel_count()
        self.all_data = np.zeros((1, info.channel_count()))
        self.fig, self.ax = plt.subplots(1,1,figsize=(figure_width, figure_height))
        self.lines = []

        self.categories = ['Delta', 'Theta', 'Alpha', 'Beta','Gamma']

        self.name = info.type()
            
        line = self.ax.plot([], [])
        self.lines.append(line)
        self.freq = rfftfreq(256, d=1/256)
        self.record = record
        self.all_ts = np.zeros(1)
        self.filename = f"{self.name}_File_{time.time()}.csv"
        self.starttime = time.time()
        self.all_ratios = np.zeros(1)


    def pt_filter(self,data, window):
        num_rows = data.shape[0] - window + 1
        num_columns = data.shape[1]

        total = np.zeros((num_rows, num_columns))

        for i in range(num_columns):
            row = np.convolve(data[:, i], np.ones(window), 'valid') / window
            total[:, i] = row

        return total


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

    def remove_upper(self):
        idx = (self.freq <= 55).astype(int)
        print(idx)
        self.fourier = self.fourier * idx
        self.vals = irfft(self.fourier,view_size)

    def twenty_point_avg(self):
        avg = np.zeros(self.vals.shape)
        for i in range(0,self.vals.shape[0]-20):
            avg[i] = np.mean(self.vals[i:i+20])
        return avg


    def get_powers(self,PSD,freq):

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

        t=delta+theta+alpha+beta+gamma
        final = [delta,theta,alpha,beta,gamma]/t
        return final

    def get_ratios(self,lis):
        d_g = lis[0]/lis[4]
        t_g = lis[1]/lis[4]
        a_g = lis[2]/lis[4]
        b_g = lis[3]/lis[4]
        return d_g

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
            

            #WHICH ONE TO USE
            self.vals=self.last_viewsize_values[:,0]


            fourier = rfft(self.vals,256)
            PSD = fourier * np.conj(fourier) / 256
            powers = self.get_powers(PSD,self.freq)
            self.all_ratios = np.concatenate((self.all_ratios, [powers[0]/powers[4]]), axis=0)

            #ratios = self.get_ratios(powers)
        
   
            print(self.all_data.shape,self.all_ratios.shape)
            
            #self.lines[0][0].set_data(self.last_viewsize_timestamps, self.all_ratios[-view_size:])




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
        print('No EEG streams found. Exiting...')


    plt.ion()  # Enable interactive mode
    
    while True:
        for inlet in inlets:
            inlet.pull_and_plot()
            plt.draw()
            plt.pause(0.1)

if __name__ == '__main__':
    main()