
import numpy as np

# Define kernel functions
def epanechnikov_kernel(x):
    return np.where(np.abs(x) <= 1, 3/4 * (1 - x**2), 0)

def uniform_kernel(x):
    return np.where(np.abs(x) <= 1, 1, 0)

def triangle_kernel(x):
    return np.where(np.abs(x) <= 1, 1 - np.abs(x), 0)


# Epanechnikov Kernel
def epanechnikov_kernel_2d(x, y):
    z = epanechnikov_kernel(x) * epanechnikov_kernel(y)
    return z

# Uniform Kernel
def uniform_kernel_2d(x, y): 
    z = uniform_kernel(x) *  uniform_kernel(y) 
    return z

# Triangle Kernel
def triangle_kernel_2d(x, y): 
    z = triangle_kernel(x) *  triangle_kernel(y) 
    return z

# Define the local polynomial regression function
def local_polynomial_regression_2d(data, kernel_type, bandwidth1, bandwidth2, degree, xy0):
    x1 = data[:, 0]
    x2 = data[:, 1]
    y = data[:, 2]
    
    if kernel_type == 'epa':
        kernel = epanechnikov_kernel_2d
    elif kernel_type == 'unif':
        kernel = uniform_kernel_2d
    elif kernel_type == 'triangle':
        kernel = triangle_kernel_2d
    else:
        raise ValueError('Unsupported kernel type')
    
    # Compute the polynomial coefficients for each x0 value
    beta_hat = []

    for xi1, xi2 in xy0:
        # Construct the weight and design matrices
        W = np.diag(kernel(np.abs(x1-xi1)/bandwidth1, np.abs(x2-xi2)/bandwidth2)/bandwidth1/bandwidth2)
        X1_design = np.vander(x1 - xi1, degree + 1, increasing=True)
        X2_design = np.vander(x2 - xi2, degree + 1, increasing=True)[:, 1:]
        X = np.column_stack((X1_design,X2_design))
        
        beta_hat_i = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y
        beta_hat.append(beta_hat_i)
    
    # Reshape
    beta_hat = np.matrix(beta_hat)
    
    # return beta_hat_reshaped
    return beta_hat



def generate_data(n, m, beta, theta):
    """
    Generates n sets of data for the partially linear single index model with functional data.

    Parameters:
    n (int): Number of realizations.
    m (int): Average number of time points for each realization.
    beta (ndarray): Array of shape (p,) containing the unknown parameters beta.
    theta (ndarray): Array of shape (q,) containing the unknown parameters theta.

    Returns:
    Tuple containing the following arrays:
        - time_points: List of length n containing arrays of shape (m_i,) containing the time points for each realization i.
        - X_samples: List of length n containing arrays of shape (m_i, p) containing the predictor variables X(T_{ij}) for each realization i.
        - Z_samples: List of length n containing arrays of shape (m_i, q) containing the predictor variables Z(T_{ij}) for each realization i.
        - Y_samples: List of length n containing arrays of shape (m_i,) containing the response variable Y(T_{ij}) for each realization i.
    """
    # Define functions for X and Z
    def X1(t):
        return t ** 2

    def X2(t):
        return t ** 3

    def Z1(t):
        return np.cos(2 * np.pi * t)

    def Z2(t):
        return np.sin(2 * np.pi * t)

    # Define function for mu
    def mu(x, y):
        return x ** 2 + y ** 2

    # Define function for epsilon
    def epsilon(t):
        return t

    # Generate data for each realization
    time_points = []
    X_samples = []
    Z_samples = []
    Y_samples = []
    for i in range(n):
        # Generate m_i sets of time points (T_{ij}) uniformly on [0,1] for each realization i
        m_i = np.random.randint(m - 2, m + 3)
        time_points_i = np.random.uniform(size=m_i)
        time_points.append(time_points_i)

        # Generate m_i sets of predictor variables X(T_{ij}) and Z(T_{ij}) using known functions
        X1_i = np.random.uniform(1) * X1(time_points_i)
        X2_i = np.random.uniform(1) * X2(time_points_i)
        Z1_i = np.random.normal(loc=1, scale=1, size=1) * Z1(time_points_i)
        Z2_i = np.random.normal(loc=1, scale=1, size=1) * Z2(time_points_i)
        X_i = np.column_stack((X1_i, X2_i))
        Z_i = np.column_stack((Z1_i, Z2_i))
        X_samples.append(X_i)
        Z_samples.append(Z_i)

        # Calculate U and mu
        U_i = np.dot(X_i, beta)
        mu_i = mu(time_points_i, U_i)

        # Calculate Y
        thetaZi = np.dot(Z_i, theta)
        e_i = np.random.normal(loc=0, scale=0.1, size=m_i) * epsilon(time_points_i)
        Y_i = mu_i + thetaZi + e_i
        Y_samples.append(Y_i)

    return time_points, X_samples, Z_samples, Y_samples



time_points, X_samples, Z_samples, Y_samples = generate_data(3, 10, np.array([1, 2]), np.array([4, 5]))
print("time_points:", time_points)
print("X_samples:", X_samples)
print("Z_samples:", Z_samples)
print("Y_samples:", len(Y_samples))