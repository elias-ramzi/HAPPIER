import numpy as np


def around(val, decimals=4):
    return np.around(val, decimals=decimals)


def percentage(val, decimals=2):
    return around(val*100, decimals=decimals)
