import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sklearn as sk

# tuples are (minutes, number of times)
# this data show how long you've been late and based on this belated behaviour how many times you
# missed the train (too_late) and how many times you catch the train (in_time)
in_time = [(0, 22), (1, 19), (2, 17), (3, 18),
           (4, 16), (5, 15), (6, 9), (7, 7),
           (8, 4), (9, 3), (10, 3), (11, 2)]
too_late = [(6, 6), (7, 9), (8, 12), (9, 17),
            (10, 18), (11, 15), (12,16), (13, 7),
            (14, 8), (15, 5)]

X, Y = zip(*in_time)
X2, Y2 = zip(*too_late)

# visualization of the data distribution for in_time and too too_late

# plt.bar(X, Y, alpha=0.7, width=0.8, label='in_time', color='red')
# plt.bar(X2, Y2, alpha=0.7, width=0.8, label='too_late', color='blue')
# plt.xlabel('minutes')
# plt.ylabel('frequency')
# plt.legend()
# plt.show()

in_time_dict = dict(in_time)
too_late_dict = dict(too_late)

# create a function to give us the chance (Probability) of catching the train
def catchTrain(min):
    # number of time with this much (min) latency you catch the train
    s = in_time_dict.get(min, 0)
    if s == 0:
        return 0
    else:
        # number of times this with much (min) latency you miss the train
        m = too_late_dict.get(min, 0)
        catching_chance = s/(s+m)
        return catching_chance

for min in range(-1, 13):
    catchP = catchTrain(min)
    print(f'the chance of getting the train is: {catchP}')
