import numpy as np
import pandas as pd

arr = np.asarray([x for x in range(100)])
numbers_series = pd.Series(arr)
window_size = 20
def twenty_point_avg(lis):
     windows = numbers_series.rolling(window_size)
     moving_averages = windows.mean()
     moving_averages_list = moving_averages.tolist()
     final_list = moving_averages_list[window_size - 1:]
     return final_list

print(twenty_point_avg(arr))