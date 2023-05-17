from pylsl import StreamInlet, resolve_stream
import matplotlib.pyplot as plt
import numpy as np
from time import time
import time as ttt
import csv
import winsound
import matplotlib.animation as animation

import pylsl
print(pylsl.__file__)
pylsl.__file__
# Resolve a stream with the name "MyStream"
streams = resolve_stream()

# Print information about the resolved stream(s)
for stream in streams:
    print("")
    print("Stream Name:", stream.name())
    print("Stream Type:", stream.type())
    print("Stream Source ID:", stream.source_id())
    print("Number of Channels:", stream.channel_count())
print(pylsl.__file__)
pylsl.__file__