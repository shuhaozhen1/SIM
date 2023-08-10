import numpy as np

# Define kernel functions
def epanechnikov_kernel(x):
    return np.where(np.abs(x) <= 1, 3/4 * (1 - x**2), 0)

def uniform_kernel(x):
    return np.where(np.abs(x) <= 1, 1, 0)

def triangle_kernel(x):
    return np.where(np.abs(x) <= 1, 1 - np.abs(x), 0)

# Local polynomial estimation
def local_polynomial_regression(data, kernel_type, bandwidth, degree, x0):
    x = data[:, 0]
    y = data[:, 1]
    n = len(x)
    
    if kernel_type == 'epa':
        kernel = epanechnikov_kernel
    elif kernel_type == 'unif':
        kernel = uniform_kernel
    elif kernel_type == 'triangle':
        kernel = triangle_kernel
    else:
        raise ValueError('Unsupported kernel type')
    
    beta_hat = np.zeros((degree + 1,len(x0)))
    for i in range(len(x0)):
        X = np.fliplr(np.vander(x-x0[i], degree + 1))
        W = np.diag(kernel((x0[i] - x) / bandwidth))
        beta_hat[:,i] = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y
    
    return beta_hat

# Loss function of partial linear single index model
def loss_plsim(data, kernel_type, bandwidth, degree):
    