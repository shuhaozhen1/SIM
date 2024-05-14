import numpy as np

# Define kernel function
def quartic_kernel(x):
    return np.where(np.abs(x) <= 1, 15/16 * (1 - x**2)**2, 0)



# Local polynomial estimation
def local_polynomial_regression(data, bandwidth, degree, x0, kernel_type='quartic'):
    x = data[:, 0]
    y = data[:, 1]
    
    if kernel_type == 'quartic':
        kernel = quartic_kernel
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


# # 2 dimensional Kernels

# # Epanechnikov Kernel
# def epanechnikov_kernel_2d(x, y):
#     z = epanechnikov_kernel(x) * epanechnikov_kernel(y)
#     return z

# # Uniform Kernel
# def uniform_kernel_2d(x, y): 
#     z = uniform_kernel(x) *  uniform_kernel(y) 
#     return z

# # Triangle Kernel
# def triangle_kernel_2d(x, y): 
#     z = triangle_kernel(x) *  triangle_kernel(y) 
#     return z



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

# def optimize_plsim(data, kernel_type, degree, domain_check=False, lower=None, upper=None):
#     # define the objective function
#     def objective(params):
#         if domain_check == False:
#             try:
#                 beta = params[:data['x'].shape[1]]
#                 theta = params[data['x'].shape[1]:-1]
#                 bandwidth = params[-1]
#                 loss = loss_plsim(data, kernel_type, bandwidth, degree, beta, theta)
#             except:
#                 # return a large value if an error occurs
#                 loss = np.inf
#         else:
#             try:
#                 beta = params[:data['x'].shape[1]]
#                 theta = params[data['x'].shape[1]:-1]
#                 bandwidth = params[-1]
#                 x_to_try = np.linspace(lower,upper,10)
#                 # transformed model: y- \theta^T Z = eta(\beta^T X) + \epsilon
#                 x = data['x']
#                 z = data['z']
#                 y = data['y']
#                 y_transformed = y - theta.T @ z.T
#                 x_transformed = beta.T @ x.T
#                 data_transformed = np.column_stack((x_transformed,y_transformed))
#                 est = local_polynomial_regression(data_transformed, kernel_type, bandwidth, degree,x0=x_to_try)
#                 loss = loss_plsim(data, kernel_type, bandwidth, degree, beta, theta)
#             except:
#                 # return a large value if an error occurs
#                 loss = np.inf
#         return loss
    
#     # define the constraints
#     cons = ({'type': 'eq', 'fun': lambda params: np.linalg.norm(params[:data['x'].shape[1]]) - 1},
#         {'type': 'ineq', 'fun': lambda params: params[0]},
#         {'type': 'ineq', 'fun': lambda params: params[-1]})

#     # define the constraint
#     # cons = ({'type': 'eq', 'fun': lambda params: np.linalg.norm(params[:data['x'].shape[1]]) - 1})
    
#     # define the initial values for beta and theta
#     beta_init = np.ones(data['x'].shape[1])/ np.sqrt(data['x'].shape[1])
#     theta_init = np.ones(data['z'].shape[1])
#     bandwidth_init = 0.4 * ( max(beta_init.T @ data['x'].T) - min(beta_init.T @ data['x'].T) )
#     params_init = np.concatenate((beta_init, theta_init, bandwidth_init), axis=None)

#     # minimize the objective function
#     res = minimize(objective, params_init, method='SLSQP', constraints=cons)

#     # extract the optimal values for beta and theta
#     beta_opt = res.x[:data['x'].shape[1]]
#     theta_opt = res.x[data['x'].shape[1]:-1]
#     bandwidth_opt = res.x[-1]
    
#     return beta_opt, theta_opt, bandwidth_opt


def optimize_plsim_h(data, kernel_type, degree, bandwidth):
    # define the objective function
    def objective(params):
        try:
            beta = params[:data['x'].shape[1]]
            theta = params[data['x'].shape[1]:]
            loss = loss_plsim(data, kernel_type, bandwidth, degree, beta, theta)
        except:
            # return a large value if an error occurs
            loss = np.inf
        return loss
    
    # define the constraints
    cons = ({'type': 'eq', 'fun': lambda params: np.linalg.norm(params[:data['x'].shape[1]]) - 1},
        {'type': 'ineq', 'fun': lambda params: params[0]})

    # define the constraint
    # cons = ({'type': 'eq', 'fun': lambda params: np.linalg.norm(params[:data['x'].shape[1]]) - 1})
    
    # define the initial values for beta and theta
    beta_init = np.ones(data['x'].shape[1])/ np.sqrt(data['x'].shape[1])
    theta_init = np.ones(data['z'].shape[1])

    params_init = np.concatenate((beta_init, theta_init), axis=None)

    # minimize the objective function
    res = minimize(objective, params_init, method='SLSQP', constraints=cons)

    # extract the optimal values for beta and theta
    beta_opt = res.x[:data['x'].shape[1]]
    theta_opt = res.x[data['x'].shape[1]:]
    
    return beta_opt, theta_opt, bandwidth