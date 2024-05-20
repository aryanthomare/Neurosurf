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
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib



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
        # self.model = tf.keras.models.load_model('Aryans\\test_model_v1.model')
        self.clf2 = joblib.load("model_af7.pkl")

        self.normal_rate = 256#info.nominal_srate()
        print(self.normal_rate)
        self.outputs = ['blink','normal']
        self.channel_count = info.channel_count()
        self.all_data = np.zeros((1, info.channel_count()))
        self.lines = []

        self.categories = ['Delta', 'Theta', 'Alpha', 'Beta','Gamma']

        self.name = info.type()


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

        t=delta+theta+alpha+beta+gamma
        final = [delta,theta,alpha,beta,gamma]/t
        return final

    def get_rolling_avg(self,v):
        return np.mean(np.absolute(v[-256:]))

    def get_blink(self):
        return self.clf2.predict(self.get_powers)
    
    def check_sensors(self):
        f = True
        for i in range(0,self.channel_count):
            v=self.last_viewsize_values[:,i]
            avg = self.get_rolling_avg(v)
            if avg > 400:
                f = False
                print(avg)
                print(f"BAD SENSOR: {i}")
        return f



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


            self.vals=self.last_viewsize_values[:,0]   

            if self.check_sensors():
                print(self.get_blink())
                self.fourier = rfft(self.vals,view_size)
                self.freq = rfftfreq(view_size, d=1/self.normal_rate)
                self.L = np.arange(1,np.floor(view_size/2),dtype='int')

                self.PSD = self.fourier * np.conj(self.fourier) / view_size


                # print(self.get_blink())





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
        print("UNDETECTED")

 
    
    while True:
        for inlet in inlets:
            inlet.pull_and_plot()
  

if __name__ == '__main__':
    main()