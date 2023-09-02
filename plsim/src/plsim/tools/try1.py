import numpy as np

# Define kernel functions
def epanechnikov_kernel(x):
    return np.where(np.abs(x) <= 1, 3/4 * (1 - x**2), 0)

def uniform_kernel(x):
    return np.where(np.abs(x) <= 1, 1, 0)

def triangle_kernel(x):
    return np.where(np.abs(x) <= 1, 1 - np.abs(x), 0)


# Define the 2D kernel functions
def epanechnikov_kernel_2d(x, y):
    z = epanechnikov_kernel(x) * epanechnikov_kernel(y)
    return z

def uniform_kernel_2d(x, y): 
    z = uniform_kernel(x) * uniform_kernel(y) 
    return z

def triangle_kernel_2d(x, y): 
    z = triangle_kernel(x) * triangle_kernel(y) 
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



n=100
# Generate some sample data
x1 = np.random.uniform(0,1,n)
x2 = np.random.uniform(0,1,n)
y = x1**2 + x2**3 + np.random.randn(n)*0.1
data = np.column_stack((x1,x2,y))

# Set the parameters for local polynomial regression
kernel_type = 'epa'
bandwidth1 = 0.3
bandwidth2 = 0.3
degree = 1
xy0 = np.array([(xi,yi) for xi in np.linspace(0,1,50) for yi in np.linspace(0,1,50)])

# Perform local polynomial regression
beta_hat = local_polynomial_regression_2d(data, kernel_type, bandwidth1, bandwidth2, degree, xy0)

print(beta_hat.shape)

