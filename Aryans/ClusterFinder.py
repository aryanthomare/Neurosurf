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

def open_file(file):
    #open csv file and save rows in a list
    with open(file, 'r') as file:
        reader = csv.reader(file)
        lines = np.array(list(reader),dtype=float)

        lines = lines[ lines[:, -1]>= 0]
        return lines


normal = open_file('Neurosurf\\Aryans\\Exported_Values\\normal\\normalTP9.csv')
focus = open_file('Neurosurf\\Aryans\\Exported_Values\\concentrated\\conTP9.csv')
blink = open_file('Neurosurf\\Aryans\\Exported_Values\\blink\\blinkTP9.csv')

print(normal.shape)