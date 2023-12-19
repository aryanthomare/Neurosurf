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
import os
def open_file(file):
    #open csv file and save rows in a list
    with open(file, 'r') as file:
        reader = csv.reader(file)
        lines = np.array(list(reader),dtype=float)

        lines = lines[ lines[:, -1]>= 0]
        return lines

def distance(p1,p2):
    return math.sqrt(np.sum(np.square(np.subtract(p1,p2))))




#print(os.listdir('C:\\Users\\aryan\\OneDrive\\Desktop\\PROJECT\\Neurosurf\\Aryans\\Exported_Values\\normal\\'))
normal = open_file('C:\\Users\\aryan\\OneDrive\\Desktop\\PROJECT\\Neurosurf\\Aryans\\Exported_Values\\normal\\normalTP9.csv')
focus = open_file('C:\\Users\\aryan\\OneDrive\\Desktop\\PROJECT\\Neurosurf\\Aryans\\Exported_Values\\concentrated\\conTP9.csv')
blink = open_file('C:\\Users\\aryan\\OneDrive\\Desktop\\PROJECT\\Neurosurf\\Aryans\\Exported_Values\\blinks\\blinksTP9.csv')


normal_mean = np.mean(normal,axis=0)
focus_mean = np.mean(focus,axis=0)
blink_mean = np.mean(blink,axis=0)



rights = [0,0,0]
wrongs = [0,0,0]
for each in normal:
    dn = distance(each,normal_mean)
    df = distance(each,focus_mean)
    db = distance(each,blink_mean)

    if dn < df and dn < db:
        rights[0] += 1
    else:
        wrongs[0] += 1

for each in focus:
    dn = distance(each,normal_mean)
    df = distance(each,focus_mean)
    db = distance(each,blink_mean)

    if df < dn and df < db:
        rights[1] += 1
    else:
        wrongs[1] += 1

for each in blink:
    dn = distance(each,normal_mean)
    df = distance(each,focus_mean)
    db = distance(each,blink_mean)

    if db < dn and db < df:
        rights[2] += 1
    else:
        wrongs[2] += 1
print(normal_mean,focus_mean,blink_mean)
print(rights/np.add(rights,wrongs))