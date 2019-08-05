import numpy as np


def centered_mean(x,y):
    return (x+y)/2.


def shiftg(x):
    b=len(x)
    return np.concatenate([x[np.arange(1, b, 1)], np.array([x[0]])])


def shiftd(x):
    b=len(x)
    return x[np.arange(-1, b-1, 1)]
