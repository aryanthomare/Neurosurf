import numpy as np
import math
import pylsl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from typing import List
import pandas as pd
from numpy.fft import fft, ifft
import csv
import scipy.signal as signal

plot_duration = 1  # how many seconds of data to show
figure_width = 12  # width of the figure in inches
figure_height = 6

view_size=200

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
        self.inlet = pylsl.StreamInlet(info, max_buflen=plot_duration,
                                       processing_flags=pylsl.proc_clocksync | pylsl.proc_dejitter)
        # store the name and channel count
        self.name = info.name()
        self.channel_count = info.channel_count()


    def message_writer(self,message):
        # print(f"quote update {message}")
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

class TestInlet(Inlet):
    def __init__(self,record):
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
        self.start_time = time.time()
        self.fig, self.ax = plt.subplots(self.channel_count,2,figsize=(figure_width, figure_height))
        self.lines = []
        for j in range(2):
            for i in range(self.channel_count):
                line, = self.ax[i][j].plot([], [])
                self.lines.append(line)
        self.start = time.time()
        self.all_ts = np.zeros(1)
        self.rate = 7
        self.flag = True
        self.time =0
        self.filename = f"{self.name}_File_{time.time()}.csv"
        self.record = record
    
    
        #self.ax.set_ylim(-1, 1)  # Set the y-range


    
    def message_writer(self,message):
        # print(f"quote update {message}")
        with open(self.filename, 'a', encoding='UTF8',newline='') as f:

            writer = csv.writer(f)
            writer.writerow(message)


    def sort_sensor_data(self,timestamps, sensor_data):
        # Get the indices that would sort the timestamps array
        sorted_indices = np.argsort(timestamps, axis=0)

        # Sort the timestamps and sensor data arrays based on the sorted indices
        sorted_timestamps = timestamps[sorted_indices]
        sorted_sensor_data = sensor_data[sorted_indices]

        return sorted_timestamps, sorted_sensor_data
    
        

    def filter_data(array, frequency, fft_vals):
        # Get the number of samples and sampling rate
        N = len(array)
        sr = 1.0

        # Compute the frequency axis
        freq = np.fft.fftfreq(N, 1.0/sr)

        # Find the index corresponding to the target frequency
        index = np.argmin(np.abs(freq - frequency))

        # Set the Fourier coefficient at the target frequency to zero
        fft_vals[index] = 0

        # Perform inverse FFT to obtain the filtered array
        filtered_array = np.real(np.fft.ifft(fft_vals))

        return filtered_array


        
    def pull_and_plot(self,*fargs):
        array = np.zeros((1, 3))  # Create a random array of size n by 3 with values between 0 and 1



        for x in range(self.channel_count):
            value = np.sin(3*(np.pi * self.time)/1.5)+np.sin(1*(np.pi * self.time)/1.5)
            print(value)
            self.time += 1 / self.rate
            array[0][x]=value
        




        times = np.array([time.time()-self.start_time])



        if self.record:
            write=array[0]
            write.flatten()
            message = np.concatenate((write, times), axis=0)
            self.message_writer(message)




        if self.flag:
            if time.time()-self.start_time > 5:
                self.rate = self.all_data.shape[0]/5
                self.flag = False



        self.all_data = np.concatenate((self.all_data, array), axis=0)
        self.all_ts = np.concatenate((self.all_ts, times), axis=0)
        self.last_viewsize_values=self.all_data[-view_size:, :]
        self.last_viewsize_timestamps = self.all_ts[-view_size:]

        self.last_viewsize_timestamps, self.last_viewsize_values = self.sort_sensor_data(self.last_viewsize_timestamps, self.last_viewsize_values)
        for i in range(0,self.channel_count):
            self.vals=self.last_viewsize_values[:,i]               

            fourier = fft(self.vals)
            sr  = self.rate
            N = len(fourier)
            n = np.arange(N)
            ts = 1.0/sr
            T = N/sr
            freq = n/T
            fft_magnitudes = np.abs(fourier)
            max_magnitude = np.max(fft_magnitudes)
            normalized_fft = fft_magnitudes / max_magnitude


            self.lines[i+self.channel_count].set_data(freq, normalized_fft)
            self.ax[i][1].relim()
            self.ax[i][1].set_xlim(0,freq[int(np.ceil((freq.size-1)/2))])
            self.ax[i][1].autoscale_view()
            self.ax[i][1].set_title(f"Fourier Channel {i}")
            self.ax[i][1].grid(True)
            self.ax[i][1].set_xlabel('Freq (Hz)', fontsize=12, fontweight='bold')
            self.ax[i][1].set_ylabel('Amplitude', fontsize=12, fontweight='bold')


            self.vals=self.last_viewsize_values[:,i]                
            self.lines[i].set_data(self.last_viewsize_timestamps, self.vals)
            self.ax[i][0].relim()
            self.ax[i][0].autoscale_view()
            self.ax[i][0].set_title(f"DATA PLOTS Channel {i}")
            self.ax[i][0].grid(True)
            self.ax[i][0].set_xlabel('Time (S)', fontsize=12, fontweight='bold')
            self.ax[i][0].set_ylabel('Value', fontsize=12, fontweight='bold')





        plt.tight_layout()



class DataInlet(Inlet):
    
    """A DataInlet represents an inlet with continuous, multi-channel data that
    should be plotted as multiple lines."""
    dtypes = [[], np.float32, np.float64, None, np.int32, np.int16, np.int8, np.int64]
    def __init__(self, info: pylsl.StreamInfo,record):
        super().__init__(info)
        self.normal_rate = info.nominal_srate()
        self.channel_count = info.channel_count()
        self.all_data = np.zeros((1, info.channel_count()))
        self.fig, self.ax = plt.subplots(info.channel_count(),2,figsize=(figure_width, figure_height))
        self.lines = []
        self.name = info.type()
        for j in range(2):
            for i in range(self.channel_count):
                line, = self.ax[i][j].plot([], [])
                self.lines.append(line)
        self.record = record
        self.all_ts = np.zeros(1)
        self.filename = f"{self.name}_File_{time.time()}.csv"
        self.starttime = time.time()
        
    def message_writer(self,message):
        # print(f"quote update {message}")
        with open(self.filename, 'a', encoding='UTF8',newline='') as f:

            writer = csv.writer(f)
            writer.writerow(message)

    def sort_sensor_data(self,timestamps, sensor_data):
        # Get the indices that would sort the timestamps array
        sorted_indices = np.argsort(timestamps, axis=0)

        # Sort the timestamps and sensor data arrays based on the sorted indices
        sorted_timestamps = timestamps[sorted_indices]
        sorted_sensor_data = sensor_data[sorted_indices]

        return sorted_timestamps, sorted_sensor_data

    def pull_and_plot(self,*fargs):
        vals, ts = self.inlet.pull_chunk()
        #print("!!!: ",vals,ts)
        
        for x in range(len(ts)):
            ts[x] = ts[x]-self.starttime
        
        if ts:



            new = np.array(vals)
            times = np.array(ts)

            #self.writer = np.concatenate((new, times), axis=1)

            if self.record:
                combined_array = np.hstack((new, times[:, np.newaxis]))

                #print(combined_array)



                columns = []
                for i in range(combined_array.shape[0]):
                    column = combined_array[i,:]
                    columns.append(column)

                for each in columns:
                    self.message_writer(each)


            self.all_data = np.concatenate((self.all_data, new), axis=0)
            self.all_ts = np.concatenate((self.all_ts, times), axis=0)

            self.last_viewsize_values=self.all_data[-view_size:, :]
            self.last_viewsize_timestamps = self.all_ts[-view_size:]

            self.last_viewsize_timestamps, self.last_viewsize_values = self.sort_sensor_data(self.last_viewsize_timestamps, self.last_viewsize_values)

            for i in range(0,self.channel_count):
                self.vals=self.last_viewsize_values[:,i]               


                fourier = fft(self.vals)

                sr  = self.normal_rate
                N = len(fourier)
                n = np.arange(N)
                ts = 1.0/sr
                T = N/sr
                freq = n/T

                fft_magnitudes = np.abs(fourier)
                max_magnitude = np.max(fft_magnitudes)
                normalized_fft = fft_magnitudes / max_magnitude

                print(np.ceil((freq.size-1)/2))
                # filter_frequency = 60  # Specify the frequency you want to filter out
                # filter_index = np.abs(freq - filter_frequency).argmin()
                # normalized_fft[filter_index] = 0

                sig_filtered+ = np.real(np.fft.ifft(normalized_fft))

                self.lines[i+self.channel_count].set_data(freq, normalized_fft)
                self.ax[i][1].relim()
                self.ax[i][1].set_xlim(0,freq[int(np.ceil((freq.size-1)/2))])
                self.ax[i][1].autoscale_view()
                self.ax[i][1].set_title(f"Fourier Channel {i}")
                self.ax[i][1].grid(True)
                self.ax[i][1].set_xlabel('Freq (Hz)', fontsize=12, fontweight='bold')
                self.ax[i][1].set_ylabel('Amplitude', fontsize=12, fontweight='bold')











                # self.lines[i].set_data(self.last_viewsize_timestamps, self.vals)
                self.lines[i].set_data(self.last_viewsize_timestamps, self.vals)
                #print(self.last_viewsize_timestamps[1]-self.last_viewsize_timestamps[0])

                self.ax[i][0].set_ylim(np.amin(self.all_data), np.amax(self.all_data))  # Set the y-range
                self.ax[i][0].relim()
                self.ax[i][0].autoscale_view()
                self.ax[i][0].set_title(f"DATA PLOTS Channel {i}")
                self.ax[i][1].grid(True)
                self.ax[i][1].set_xlabel('Freq (Hz)', fontsize=12, fontweight='bold')
                self.ax[i][1].set_ylabel('Amplitude', fontsize=12, fontweight='bold')
                            



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