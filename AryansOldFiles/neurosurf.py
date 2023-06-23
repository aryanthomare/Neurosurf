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

class TestInlet(Inlet):
    """Class to represent a Test plottable inlet"""

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

        #Create Subplots on Graph
        self.fig, self.ax = plt.subplots(self.channel_count,2,figsize=(figure_width, figure_height))

        #list of lines graphs on screen
        self.lines = []

        #Loops through both columns of graphs and creates their graph and line
        for j in range(2):
            for i in range(self.channel_count):
                line, = self.ax[i][j].plot([], [])
                self.lines.append(line)

        self.start = time.time()
        self.all_ts = np.zeros(1)

        #inital guess of SR
        self.rate = 7
        self.flag = True
        self.time =0
        
        #File name to record to
        self.filename = f"{self.name}_File_{time.time()}.csv"
        self.record = record
    
    
        #self.ax.set_ylim(-1, 1)  # Set the y-range


    
    def message_writer(self,message):
        """Function to write Messages to the Class's File Name"""

        with open(self.filename, 'a', encoding='UTF8',newline='') as f:

            writer = csv.writer(f)
            writer.writerow(message)


    def sort_sensor_data(self,timestamps, sensor_data):
        """Sorts Sensor Data by their Timestamp array"""

        # Get the indices that would sort the timestamps array
        sorted_indices = np.argsort(timestamps, axis=0)

        # Sort the timestamps and sensor data arrays based on the sorted indices
        sorted_timestamps = timestamps[sorted_indices]
        sorted_sensor_data = sensor_data[sorted_indices]

        return sorted_timestamps, sorted_sensor_data
    
    def filter_data(self, target_frequency,tol):
        """Filters Data from vals and removes certan freqency with a tolerence"""


        #get indicies of items near the target frequency
        idx = ((self.freq <= target_frequency-tol) | (self.freq >= target_frequency+tol)).astype(int)

        #Filter out freqencies below a certain threshold
        # maxfilter = self.fourier > 50
        # idx = idx * maxfilter
        self.fourier = self.fourier * idx

        #Inverting Fourier to get real valued data
        self.vals = irfft(self.fourier,view_size)
        self.PSD = self.fourier * np.conj(self.fourier) / view_size




    def pull_and_plot(self):
        """Main Function where new data is added to graphs"""

        # Create an array of size 1 by number of channels with values of -
        array = np.zeros((1, self.channel_count))

        #adding new data from sin function to values
        for i in range(self.channel_count):
            value = 10*np.sin(i*2*np.pi * self.time)
            array[0][i]=value

        self.time += 1 / self.rate

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

            if self.all_data.shape[0] > view_size:
                self.dt = 1.0/self.rate
                self.fourier = rfft(self.vals)
                self.freq = rfftfreq(view_size, d=1/self.rate)
                self.L = np.arange(1,np.floor(view_size/2),dtype='int')
                #print("FSHAPE: ",self.fourier.shape,self.freq.shape)


                #self.filter_data(2,0.5)
                self.PSD = self.fourier * np.conj(self.fourier) / view_size


            # max_magnitude = np.max(PSD)
            # normalized_fft = PSD / max_magnitude
                self.filter_data(2,0.5)

                self.lines[i+self.channel_count].set_data(self.freq[self.L], self.PSD[self.L])
                self.ax[i][1].set_xlim(self.freq[self.L[0]],self.freq[self.L[-1]])
            
            #print("Shape of TS & vals: ",self.last_viewsize_timestamps.shape,self.vals.shape)
            self.lines[i].set_data(self.last_viewsize_timestamps, self.vals)
            self.ax[i][0].relim()
            self.ax[i][0].autoscale_view()
            self.ax[i][0].set_title(f"DATA PLOTS Channel {i}")
            self.ax[i][0].grid(True)
            self.ax[i][0].set_xlabel('Time (S)', fontsize=12, fontweight='bold')
            self.ax[i][0].set_ylabel('Value', fontsize=12, fontweight='bold')
            self.ax[i][0].set_ylim(-10,10)


            
            self.ax[i][1].relim()
            self.ax[i][1].autoscale_view()
            self.ax[i][1].set_title(f"Fourier Channel {i} (unfiltered)")
            self.ax[i][1].grid(True)
            self.ax[i][1].set_xlabel('Freq (Hz)', fontsize=12, fontweight='bold')
            self.ax[i][1].set_ylabel('Amplitude', fontsize=12, fontweight='bold')



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
        self.tick_spacing = 10
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
        
        for i in range(len(ts)):
            ts[i] = ts[i]-self.starttime
        
        if ts:



            new = np.array(vals)
            times = np.array(ts)


            if self.record:
                combined_array = np.hstack((new, times[:, np.newaxis]))
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

                dt = 1.0/self.normal_rate
                self.fourier = rfft(self.vals,view_size)
                self.freq = rfftfreq(view_size, d=1/self.normal_rate)
                self.L = np.arange(1,np.floor(view_size/2),dtype='int')
                # max_magnitude = np.max(PSD)
                # normalized_fft = PSD / 
                if self.all_data.shape[0] > 201:
                    print(self.all_data.shape[0])
                    self.filter_data(60,2)
                    self.filter_data(66,2)
                    self.filter_data(120,2)

                self.PSD = self.fourier * np.conj(self.fourier) / view_size

                self.lines[i+self.channel_count].set_data(self.freq[self.L], self.PSD[self.L])
                self.ax[i][1].set_xlim(self.freq[self.L[0]],self.freq[self.L[-1]])

                #self.lines[i+self.channel_count].set_data(freq, normalized_fft)
                self.ax[i][1].relim()
  
                self.ax[i][1].autoscale_view()
                self.ax[i][1].set_title(f"Fourier Channel {i}")
                self.ax[i][1].grid(True)
                self.ax[i][1].set_xlabel('Freq (Hz)', fontsize=12, fontweight='bold')
                self.ax[i][1].set_ylabel('Amplitude', fontsize=12, fontweight='bold')
                self.ax[i][1].xaxis.set_major_locator(ticker.MultipleLocator(self.tick_spacing))












                # self.lines[i].set_data(self.last_viewsize_timestamps, self.vals)
                self.lines[i].set_data(self.last_viewsize_timestamps, self.vals)
                #print(self.last_viewsize_timestamps[1]-self.last_viewsize_timestamps[0])

                self.ax[i][0].set_ylim(np.amin(self.all_data), np.amax(self.all_data))  # Set the y-range
                self.ax[i][0].relim()
                self.ax[i][0].autoscale_view()
                self.ax[i][0].set_title(f"DATA PLOTS Channel {i}")
                self.ax[i][1].grid(True)
                self.ax[i][1].set_xlabel('Time (s)', fontsize=12, fontweight='bold')
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