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

def loss_plsim(data, kernel_type, bandwidth1, bandwidth2, degree, beta, theta):
    # data should be in the form of (x,z,y), where x is the non-parametric one, z is the linear one, and y is the response
    # each raw is the observation
    t = np.array(data['t'])
    x = data['x']
    z = data['z']
    y = data['y']

    # transformed model: y- \theta^T Z = eta(\beta^T X) + \epsilon
    y_transformed = y - theta.T @ z.T
    x_transformed = beta.T @ x.T
    data_transformed = np.column_stack((t, x_transformed,y_transformed))
    tx = np.column_stack((t, x_transformed))

    # estimate eta
    eta_hat_d = local_polynomial_regression_2d(data_transformed, kernel_type, bandwidth1, bandwidth2, degree, xy0=tx)
    eta_hat = eta_hat_d[:,0]

    loss = np.sum((y - eta_hat - theta.T @ z.T) ** 2)

    return loss
