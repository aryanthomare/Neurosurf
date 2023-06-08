import numpy as np
import math
import pylsl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from typing import List
import seaborn as sns
import matplotlib.pyplot as plt

plot_duration = 1  # how many seconds of data to show
figure_width = 12  # width of the figure in inches
figure_height = 6



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



class DataInlet(Inlet):
    """A DataInlet represents an inlet with continuous, multi-channel data that
    should be plotted as multiple lines."""
    dtypes = [[], np.float32, np.float64, None, np.int32, np.int16, np.int8, np.int64]

    def __init__(self, info: pylsl.StreamInfo):
        super().__init__(info)
        self.all_data = np.zeros((1, info.channel_count()))
        self.last_20_values = np.zeros((1, info.channel_count()))
        self.last_20_timestamps = np.zeros(1)

        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], lw=2)
        self.animation = None

    def sort_by_timestamp(sensor_values, timestamps):
        sorted_indices = np.argsort(timestamps[:, 0])[::-1]  # Sort indices in descending order
        sorted_sensor_values = sensor_values[sorted_indices]
        sorted_timestamps = timestamps[sorted_indices]
        return sorted_sensor_values, sorted_timestamps


    def update_plot(self, frame):
        self.pull()
        self.line.set_data(self.last_20_timestamps, self.last_20_values)
        self.ax.relim()
        self.ax.autoscale_view()
        return self.line,

    def plot(self):
        self.animation = FuncAnimation(self.fig, self.update_plot, interval=200)
        plt.show()
        
    def pull(self):
        vals, ts = self.inlet.pull_chunk()
        if ts:
            new = np.array(vals)
            self.all_data = np.concatenate((self.all_data, new), axis=0)
            ts = np.array(ts)  # Convert timestamps to numpy array
            self.last_20_values = self.all_data[-20:, :]
            self.last_20_timestamps = ts[-20:, 0]

    def pull_and_plot(self):
        self.pull()
        self.ax.plot(self.last_20_timestamps, self.last_20_values)
        self.ax.relim()
        self.ax.autoscale_view()
        plt.draw()
def main():
    time.sleep(2)
    inlets: List[Inlet] = []
    print("looking for streams")  
    streams = pylsl.resolve_streams()
    for info in streams:
        if info.nominal_srate() != pylsl.IRREGULAR_RATE \
                and info.channel_format() != pylsl.cf_string:
            if info.type() == "Accelerometer":
                print('Adding data inlet: ' + info.name())
                inlets.append(DataInlet(info))
        else:
            print('Dont know what to do with stream ' + info.name())

    if len(inlets) != 0:
        for inlet in inlets:
            inlet.plot()
            
if __name__ == '__main__':
    main()
