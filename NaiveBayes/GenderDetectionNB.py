import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from collections import Counter


genderes = ['male', 'female']
persons = []

# datasets include the name, last name, height, and gender of some people.
with open('person_data.txt') as f:
    for line in f:
        persons.append(line.strip().split())
firstNames = {}
heights = {}
for gender in genderes:
    firstNames[gender] = [x[0] for x in persons if x[4]==gender]
    heights[gender] = [x[2] for x in persons if x[4]==gender]
    heights[gender] = np.array(heights[gender], np.int32)


class Feature:

    def __init__(self, data, name=None, bin_width=None):
        self.name = name
        self.bin_width = bin_width
        if bin_width:
            self.min, self.max = min(data), max(data)
            bins = np.arange((self.min // bin_width) * bin_width,
                                (self.max // bin_width) * bin_width,
                                bin_width)
            freq, bins = np.histogram(data, bins)
            self.freq_dict = dict(zip(bins, freq))
            self.freq_sum = sum(freq)
        else:
            self.freq_dict = dict(Counter(data))
            self.freq_sum = sum(self.freq_dict.values())



    def frequency(self, value):
        if self.bin_width:
            value = (value // self.bin_width) * self.bin_width
        if value in self.freq_dict:
            return self.freq_dict[value]
        else:
            return 0


fts = {}
for gender in genderes:
    fts[gender] = Feature(heights[gender], name=gender, bin_width=5)
    print(gender, fts[gender].freq_dict)

plt.bar(fts['male'].freq_dict.keys(), fts['male'].freq_dict.values(), alpha=0.5, width=2, color='red', label='male')
plt.bar(fts['female'].freq_dict.keys(), fts['female'].freq_dict.values(), alpha=0.5, width=2, color='blue', label='female')
plt.legend()
plt.show()




















#
# if __name__ == '__main__':
#     main()
