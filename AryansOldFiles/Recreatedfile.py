import numpy as np
import math
import pylsl
import matplotlib.pyplot as plt

from typing import List
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
        bufsize = (2 * math.ceil(info.nominal_srate() * plot_duration), info.channel_count())
        self.buffer = np.empty(bufsize, dtype=self.dtypes[info.channel_format()])
        self.fig, self.ax = plt.subplots(figsize=(figure_width, figure_height))
        self.lines = []
        for i in range(self.channel_count):
            line, = self.ax.plot([], [])
            self.lines.append(line)
        self.ax.set_ylim(-1, 1)  # Set the y-range

    def pull_and_plot(self):
        vals, ts = self.inlet.pull_chunk(max_samples=self.buffer.shape[0], dest_obj=self.buffer)
        if ts:
            ts = np.array(ts)  # Convert timestamps to numpy array
            ts_length = min(len(ts), self.buffer.shape[0])  # Get the minimum length of ts and buffer
            for i in range(self.channel_count):
                y_values = self.buffer[:ts_length, i]
                self.lines[i].set_data(ts[:ts_length], y_values)
            self.ax.relim()
            self.ax.autoscale_view()

def main():
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
            print('Don\'t know what to do with stream ' + info.name())


    plt.ion()  # Enable interactive mode
    while True:
        for inlet in inlets:
            #print(inlet.name)
            print(inlet.buffer.shape[0])
            inlet.pull_and_plot()
            print(inlet.buffer)

            plt.pause(0.001)
            plt.draw()
            
if __name__ == '__main__':
    main()
