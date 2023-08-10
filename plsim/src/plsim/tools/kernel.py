import numpy as np

def epanechnikov_kernel(x):
    return np.where(np.abs(x) <= 1, 3/4 * (1 - x**2), 0)

def uniform_kernel(x):
    return np.where(np.abs(x) <= 1, 1, 0)

def triangle_kernel(x):
    return np.where(np.abs(x) <= 1, 1 - np.abs(x), 0)

