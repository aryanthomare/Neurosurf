import numpy as np
import math
import pylsl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from typing import List
import seaborn as sns
import pandas as pd

all_data = np.zeros((20, 3))
print(all_data,"\n")
last_200_values=all_data[:, -200:]
print(last_200_values,"\n")

