import numpy as np
from scipy.linalg import block_diag

# Define kernel functions
def epanechnikov_kernel(x):
    return np.where(np.abs(x) <= 1, 3/4 * (1 - x**2), 0)

def uniform_kernel(x):
    return np.where(np.abs(x) <= 1, 1, 0)

def triangle_kernel(x):
    return np.where(np.abs(x) <= 1, 1 - np.abs(x), 0)

# Local polynomial estimation
def local_polynomial_regression(data, kernel_type, bandwidth, degree, x0):
    x = data[:, :-1]
    y = data[:, -1]
    n = len(y)
    p = x.shape[1]
    
    if kernel_type == 'epa':
        kernel = epanechnikov_kernel
    elif kernel_type == 'unif':
        kernel = uniform_kernel
    elif kernel_type == 'triangle':
        kernel = triangle_kernel
    else:
        raise ValueError('Unsupported kernel type')
    
    X = np.ones((n, 1))
    for i in range(1, degree + 1):
        for j in range(p):
            X = np.hstack((X, (x[:, j] - x0[j])**i))
    
    W = block_diag(*[kernel(np.linalg.norm((x[i] - x0) / bandwidth)) for i in range(n)])
    beta_hat = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y
    
    return beta_hat[-1]
