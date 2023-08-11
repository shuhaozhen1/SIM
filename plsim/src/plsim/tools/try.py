
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
    n = len(y)
    
    if kernel_type == 'epa':
        kernel = epanechnikov_kernel
    elif kernel_type == 'unif':
        kernel = uniform_kernel
    elif kernel_type == 'triangle':
        kernel = triangle_kernel
    else:
        raise ValueError('Unsupported kernel type')
    
    # Compute the polynomial coefficients for each x0 value
    beta_hat = []
    for xi in x0:
        # Construct the weight and design matrices
        W = np.diag(kernel(np.abs((x - xi) / bandwidth)) / bandwidth)
        X = np.vander(x - xi, degree + 1, increasing=True)
        
        beta_hat_i = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y
        beta_hat.append(beta_hat_i)
    
    return np.array(beta_hat)



# Loss function of partial linear single index model: y = \eta(\beta^T x) + \theta^T Z + \epsilon 
def loss_plsim(data, kernel_type, bandwidth, degree, beta, theta):
    # data should be in the form of (x,z,y), where x is the non-parametric one, z is the linear one, and y is the response
    # each raw is the observation
    x = data['x']
    z = data['z']
    y = data['y']

    # transformed model: y- \theta^T Z = eta(\beta^T X) + \epsilon
    y_transformed = y - theta.T @ z.T
    x_transformed = beta.T @ x.T
    data_transformed = np.column_stack((x_transformed,y_transformed))

    # estimate eta
    eta_hat_d = local_polynomial_regression(data_transformed, kernel_type, bandwidth, degree, x0=x_transformed)
    eta_hat = eta_hat_d[:,0]

    loss = np.sum((y - eta_hat - theta.T @ z.T) ** 2)

    return loss


# define the data
data = {
    'x': np.random.rand(100, 3),
    'z': np.random.rand(100, 2),
    'y': np.random.rand(100, 1)
}

# define the kernel type, bandwidth, and degree
kernel_type = 'epa'
bandwidth = 0.5
degree = 2

# define the beta and theta parameters
beta = np.array([1, 2, 3])
theta = np.array([4, 5])

# calculate the loss
loss = loss_plsim(data, kernel_type, bandwidth, degree, beta, theta)
print(loss)
