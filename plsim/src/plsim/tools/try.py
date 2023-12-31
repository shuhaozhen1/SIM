import numpy as np

# Define kernel functions
def epanechnikov_kernel(x):
    return np.where(np.abs(x) <= 1, 3/4 * (1 - x**2), 0)

def uniform_kernel(x):
    return np.where(np.abs(x) <= 1, 1, 0)

def triangle_kernel(x):
    return np.where(np.abs(x) <= 1, 1 - np.abs(x), 0)

# Local polynomial estimation
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

# Profile least sequare estimation
from scipy.optimize import minimize
from math import sqrt

def optimize_plsim(data, kernel_type, degree):
    # define the objective function
    def objective(params):
        try:
            beta = params[:data['x'].shape[1]]
            theta = params[data['x'].shape[1]:-1]
            bandwidth = params[-1]
            loss = loss_plsim(data, kernel_type, bandwidth, degree, beta, theta)
        except:
            # return a large value if an error occurs
            loss = np.inf
        return loss
    
    # define the constraints
    cons = ({'type': 'eq', 'fun': lambda params: np.linalg.norm(params[:data['x'].shape[1]]) - 1},
        {'type': 'ineq', 'fun': lambda params: params[0]},
        {'type': 'ineq', 'fun': lambda params: params[-1]})

    # define the constraint
    # cons = ({'type': 'eq', 'fun': lambda params: np.linalg.norm(params[:data['x'].shape[1]]) - 1})
    
    # define the initial values for beta and theta
    beta_init = np.ones(data['x'].shape[1])/ sqrt(data['x'].shape[1])
    theta_init = np.ones(data['z'].shape[1])
    bandwidth_init = 0.4 * ( max(beta_init.T @ data['x'].T) - min(beta_init.T @ data['x'].T) )
    params_init = np.concatenate((beta_init, theta_init, bandwidth_init), axis=None)

    # minimize the objective function
    res = minimize(objective, params_init, method='SLSQP', constraints=cons)

    # extract the optimal values for beta and theta
    beta_opt = res.x[:data['x'].shape[1]]
    theta_opt = res.x[data['x'].shape[1]:-1]
    bandwidth_opt = res.x[-1]
    
    return beta_opt, theta_opt, bandwidth_opt




# generate the data
n = 200
x= np.zeros((n,2))

x[:,0] = np.random.uniform(0,1,size=n)
x[:,1] = np.random.uniform(0,2,size=n)
z = np.random.uniform(-1,1,size=(n,2))
error = 0.1*np.random.randn(n)
y = (0.6*x[:,0] + 0.8*x[:,1])**2 + 0.5*z[:,0] + 0.8*z[:,1] + error
data = {'x': x, 'z': z, 'y': y}

# define the kernel type, bandwidth, and degree
kernel_type = 'epa'
degree = 1

# find the optimal values for beta and theta
import time

start_time = time.time()
beta_opt, theta_opt, bandwidth_opt = optimize_plsim(data, kernel_type, degree)
end_time = time.time()

print('Execution time:', end_time - start_time)
print(optimize_plsim(data, kernel_type, degree))
print('Optimal beta:', beta_opt)
print(np.sum((beta_opt)**2))
print('Optimal theta:', theta_opt)
print('Optimal bandwidth:', bandwidth_opt)

