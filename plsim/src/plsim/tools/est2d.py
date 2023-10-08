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
    
    # # Reshape
    beta_hat = np.array(beta_hat)
    
    # return beta_hat_reshaped
    return beta_hat






def loss_plsim_2d(data, kernel_type, bandwidth1, bandwidth2, degree, beta, theta):
    # data should be in the form of (x,z,y), where x is the non-parametric one, z is the linear one, and y is the response
    # each raw is the observation
    t = np.concatenate(data['T'])
    x = np.concatenate(data['X'])
    z = np.concatenate(data['Z'])
    y = np.concatenate(data['Y'])

    # transformed model: y- \theta^T Z = eta(\beta^T X) + \epsilon
    y_transformed = y - np.dot(z, theta)
    u_transformed = np.dot(x, beta) 
    data_transformed = np.column_stack((t, u_transformed,y_transformed))
    tx = np.column_stack((t, u_transformed))

    # estimate eta
    eta_hat_d = local_polynomial_regression_2d(data_transformed, kernel_type, bandwidth1, bandwidth2, degree, xy0=tx)
    eta_hat = eta_hat_d[:,0]

    loss = np.sum((y - eta_hat - np.dot(z, theta)) ** 2)

    return loss


# Profile least sequare estimation
from scipy.optimize import minimize

def optimize_plsim_h_2d(data, kernel_type, degree, bandwidth1,bandwidth2):
    x = np.concatenate(data['X'])
    z = np.concatenate(data['Z'])

    px = x.shape[1]
    pz = z.shape[1]

    # define the objective function
    def objective(params):
        try:
            beta = params[:px]
            theta = params[px:]
            loss = loss_plsim_2d(data, kernel_type, bandwidth1, bandwidth2, degree, beta, theta)
        except:
            # return a large value if an error occurs
            loss = np.inf
        return loss
    
    # define the constraints
    cons = ({'type': 'eq', 'fun': lambda params: np.linalg.norm(params[:px]) - 1},
        {'type': 'ineq', 'fun': lambda params: params[0]})

    # define the constraint
    # cons = ({'type': 'eq', 'fun': lambda params: np.linalg.norm(params[:px]) - 1})
    
    # define the initial values for beta and theta
    beta_init = np.ones(px)/ np.sqrt(px)
    theta_init = np.ones(pz)

    params_init = np.concatenate((beta_init, theta_init))

    # minimize the objective function
    res = minimize(objective, params_init, method='SLSQP', constraints=cons)

    # extract the optimal values for beta and theta
    beta_opt = res.x[:px]
    theta_opt = res.x[px:]
    
    # return beta_opt, theta_opt, bandwidth1, bandwidth2
    return beta_opt, theta_opt


