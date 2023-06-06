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

    def sort_by_timestamp(sensor_values, timestamps):
        sorted_indices = np.argsort(timestamps[:, 0])[::-1]  # Sort indices in descending order

        sorted_sensor_values = sensor_values[sorted_indices]
        sorted_timestamps = timestamps[sorted_indices]

        return sorted_sensor_values, sorted_timestamps
        
    def pull_test_data(self):
        n = np.random.randint(1, 6)  # Generate a random number between 1 and 5
        array = np.random.rand(n, 3)  # Create a random array of size n by 3 with values between 0 and 1

        self.all_data = np.concatenate((self.all_data, array), axis=0)

        print(self.all_data.shape)

class DataInlet(Inlet):
    
    """A DataInlet represents an inlet with continuous, multi-channel data that
    should be plotted as multiple lines."""
    dtypes = [[], np.float32, np.float64, None, np.int32, np.int16, np.int8, np.int64]

    def __init__(self, info: pylsl.StreamInfo):
        super().__init__(info)
        self.all_data = np.zeros((1, info.channel_count()))
        #bufsize = (2 * math.ceil(info.nominal_srate() * plot_duration), info.channel_count())
        #self.buffer = np.empty(bufsize, dtype=self.dtypes[info.channel_format()])
        self.fig, self.ax = plt.subplots(figsize=(figure_width, figure_height))
        self.lines = []
        self.ax.set_ylim(-1, 1)  # Set the y-range

    def sort_by_timestamp(sensor_values, timestamps):
        sorted_indices = np.argsort(timestamps[:, 0])[::-1]  # Sort indices in descending order

        sorted_sensor_values = sensor_values[sorted_indices]
        sorted_timestamps = timestamps[sorted_indices]

        return sorted_sensor_values, sorted_timestamps
    
    def pull_test_data(self):
        n = np.random.randint(1, 6)  # Generate a random number between 1 and 5
        array = np.random.rand(n, 3)  # Create a random array of size n by 3 with values between 0 and 1

        self.all_data = np.concatenate((self.all_data, array), axis=0)

        print(self.all_data)


    def pull_and_plot(self):
        vals, ts = self.inlet.pull_chunk()
        #print(vals)



        if ts:

            new = np.array(vals)

            print(self.all_data.shape,new.shape)

            self.all_data = np.concatenate((self.all_data, new), axis=0)
            ts = np.array(ts)  # Convert timestamps to numpy array

            last_200_values,last_200_timestamps = self.sort_by_timestamp(self.all_data[:, -200:], self.ts[:, -200:])

            

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
    if len(inlets) != 0:
        while True:
            for inlet in inlets:
                print(inlet.all_data)
                #print(inlet.buffer.shape[0])
                inlet.pull_and_plot()
                #print(inlet.buffer)
    else:
        inlets.append(TestInlet())
        while True:
            for inlet in inlets:
                inlet.pull_test_data()

            
if __name__ == '__main__':
    main()
