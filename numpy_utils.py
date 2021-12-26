import random
import numpy as np
import os
import itertools as it


def arrays(arr):
    # complete this function
    # use numpy.array
    reversed_array = numpy.zeros(shape=len(arr), dtype='float')
    for i in range(len(arr)):
        reversed_array[i] = float(arr[-(i+1)])

    return reversed_array

def array_propreties(arr):
    print(f"arr_dims={arr.ndim}, arr_size={arr.size()}, arr_shape={arr.shape()}")

def main():
    np_arr = np.empty(shape=(random.randint(1, 5), 3))
    array_propreties(np_arr)





if __name__ == '__main__':
    main()
