import numpy as np

# Define kernel functions
def quartic_kernel(x):
    return np.where(np.abs(x) <= 1, 15/16 * (1 - x**2)**2, 0)

# quartic Kernel
def quartic_kernel_2d(x, y): 
    z = quartic_kernel(x) * quartic_kernel(y)
    return z


# Define the local polynomial regression function
def local_p_2d(data, bandwidth1, bandwidth2, xy0, mi = None, kernel_type='quartic', degree = 1):
    """
    This function calculates the local polynomial regression in 2D.

    Parameters:
    data (dic): A dictionary containing: 'x1' for the data in first argument, 'x2' for the second, 'y' for the response. Names are strictly required to be set as 'x1','x2','y'.
    bandwidth1 (float): The bandwidth for the first dimension.
    bandwidth2 (float): The bandwidth for the second dimension.
    xy0 (tuple): A tuple representing the point at which to estimate the function.
    mi (float, optional): An optional parameter representing the number of observations in each sample i
    kernel_type (str, optional): The type of kernel to use. Defaults to 'quartic'.
    degree (int): The degree of the polynomial. Defaults to '1'.

    Returns:
    float: The estimated function value at the point xy0.
    """
    # Your function implementation goes here
    if mi.any() == None:
        mi = np.array([len(vector) for vector in data['y']])
    
    if kernel_type == 'quartic':
        kernel = quartic_kernel_2d
    else:
        raise ValueError('Unsupported kernel type')


    weights = np.repeat(mi, mi)
    
    # Compute the polynomial coefficients for each x0 value
    beta_hat = []

    x1 = data['x1']
    x2 = data['x2']
    y = data['y']

    for xi1, xi2 in xy0:
        # Construct the weight and design matrices
        W = np.diag(kernel(np.abs(x1-xi1)/bandwidth1, np.abs(x2-xi2)/bandwidth2)
                    /bandwidth1/bandwidth2/weights)
        X1_design = np.vander(x1 - xi1, degree + 1, increasing=True)
        X2_design = np.vander(x2 - xi2, degree + 1, increasing=True)[:, 1:]
        X = np.column_stack((X1_design,X2_design))
    
        beta_hat_i = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y
        beta_hat.append(beta_hat_i)
    
    # # Reshape
    beta_hat = np.array(beta_hat)
    
    # return beta_hat_reshaped
    return beta_hat


def loss_plsim_2d(data, bandwidth1, bandwidth2, beta, theta, kernel_type='quartic',  degree=1):
    # data should be in the form of (x,z,y), where x is the non-parametric one, z is the linear one, and y is the response
    # each raw is the observation
    t = np.concatenate(data['t'])
    x = np.concatenate(data['x'])
    z = np.concatenate(data['z'])
    y = np.concatenate(data['y'])

    mi = np.array([len(vector) for vector in data['y']])

    # transformed model: y- \theta^T Z = eta(\beta^T X) + \epsilon
    y_transformed = y - np.dot(z, theta)
    u_transformed = np.dot(x, beta) 

    localp_data = {'x1': t, 'x2': u_transformed, 'y': y_transformed}
    tx = np.column_stack((t, u_transformed))

    # estimate eta
    eta_hat_d = local_p_2d(localp_data, bandwidth1, bandwidth2, xy0=tx, mi=mi, kernel_type= kernel_type, degree= degree)
    eta_hat = eta_hat_d[:,0]

    loss = np.sum((y - eta_hat - np.dot(z, theta)) ** 2)

    return loss

# ### k-fold index sets generator
# def k_fold_indices(n, k):
#     num_folds = n // k
#     remainder = n % k
#     indices = list(range(n))
#     result = []
#     current = 0
#     for i in range(num_folds):
#         fold_size = k + 1 if i < remainder else k
#         start, stop = current, current + fold_size
#         test_indices = indices[start:stop]
#         train_indices = indices[:start] + indices[stop:]
#         result.append((train_indices, test_indices))
#         current = stop
#     return result

# def loss_plsim_2d_kfold(data, kernel_type, bandwidth1, 
#                         bandwidth2, degree, beta, theta, k=1):
#     # data should be in the form of (x,z,y), where x is the non-parametric one, z is the linear one, and y is the response
#     # each raw is the observation

#     n = len(data['T'])

#     losses = []

#     if k == 1: 
#         for i in range(n):
#             t_test = data['T'][i]
#             x_test = data['X'][i]
#             z_test = data['Z'][i]
#             y_test = data['Y'][i]
        
#             t_list = data['T'][:i] + data['T'][i+1:]

#             t = np.concatenate((data['T'][:i] + data['T'][i+1:]))
#             x = np.concatenate((data['X'][:i] + data['X'][i+1:]))
#             z = np.concatenate((data['Z'][:i] + data['Z'][i+1:]))
#             y = np.concatenate((data['Y'][:i] + data['Y'][i+1:]))

#             mi = [len(subarray) for subarray in t_list]

#             y_transformed = y - np.dot(z, theta)
#             u_transformed = np.dot(x, beta) 
#             data_transformed = np.column_stack((t, u_transformed,y_transformed))

#             test_y_transformed = y_test - np.dot(z_test, theta)
#             test_u_transformed = np.dot(x_test, beta)
#             test_tx = np.column_stack((t_test, test_u_transformed))
            
#             # transformed model: y- \theta^T Z = eta(\beta^T X) + \epsilon

#             # estimate eta
#             eta_hat_d = local_polynomial_regression_2d(data_transformed, 
#                                                    kernel_type=kernel_type, bandwidth1=bandwidth1, 
#                                                    bandwidth2=bandwidth2, degree= degree, xy0=test_tx, mi=mi)
#             eta_hat = eta_hat_d[:,0]
#             lossi = np.mean(np.square(test_y_transformed - eta_hat))

#             losses.append(lossi)

#         return(np.mean(losses))
    
#     kfold_index = k_fold_indices(n,k)
    
#     if k > 1: 
#         for i in range(len(kfold_index)):
#             t_test = np.concatenate([data['T'][k] for k in kfold_index[i][1]])
#             x_test = np.concatenate([data['X'][k] for k in kfold_index[i][1]])
#             z_test = np.concatenate([data['Z'][k] for k in kfold_index[i][1]])
#             y_test = np.concatenate([data['Y'][k] for k in kfold_index[i][1]])

#             t_list = [data['T'][k] for k in kfold_index[i][0]]

#             mi = [len(subarray) for subarray in t_list]
            
#             t = np.concatenate([data['T'][k] for k in kfold_index[i][0]])
#             x = np.concatenate([data['X'][k] for k in kfold_index[i][0]])
#             z = np.concatenate([data['Z'][k] for k in kfold_index[i][0]])
#             y = np.concatenate([data['Y'][k] for k in kfold_index[i][0]])

#             y_transformed = y - np.dot(z, theta)
#             u_transformed = np.dot(x, beta) 
#             data_transformed = np.column_stack((t, u_transformed,y_transformed))

#             test_y_transformed = y_test - np.dot(z_test, theta)
#             test_u_transformed = np.dot(x_test, beta)
#             test_tx = np.column_stack((t_test, test_u_transformed))
            
#             # transformed model: y- \theta^T Z = eta(\beta^T X) + \epsilon

#             # estimate eta
#             eta_hat_d = local_polynomial_regression_2d(data_transformed, 
#                                                    kernel_type=kernel_type, bandwidth1=bandwidth1, 
#                                                    bandwidth2=bandwidth2, degree= degree, xy0=test_tx, mi=mi)
#             eta_hat = eta_hat_d[:,0]
#             lossi = np.mean(np.square(test_y_transformed - eta_hat))

#             losses.append(lossi)

#         return(np.mean(losses))
    



# Profile least sequare estimation
from scipy.optimize import minimize

def optimize_plsim_h_2d(data,   bandwidth1,bandwidth2, kernel_type= 'quartic', degree=1):
    x = np.concatenate(data['x'])
    z = np.concatenate(data['z'])

    px = x.shape[1]
    pz = z.shape[1]

    # define the objective function
    def objective(params):
        try:
            beta = params[:px]
            theta = params[px:]
            loss = loss_plsim_2d(data, bandwidth1, bandwidth2, beta, theta,kernel_type=kernel_type, degree=degree)
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

# def optimize_plsim_2d(data, kernel_type, degree=1, k=1):
#     x = np.concatenate(data['X'])
#     z = np.concatenate(data['Z'])

#     n = len(data['T'])

#     px = x.shape[1]
#     pz = z.shape[1]

#     # define the objective function
#     def objective(params):
#         try:
#             bandwidth1 = params[0]
#             bandwidth2 = params[1]
#             beta = params[2:(px+2)]
#             theta = params[(px+2):]
#             loss = loss_plsim_2d_kfold(data, kernel_type, bandwidth1, bandwidth2, 
#                                        degree=degree, beta=beta, theta=theta, k =k)
#         except:
#             # return a large value if an error occurs
#             loss = np.inf
#         return loss
    
#     # define the constraints
#     cons = ({'type': 'eq', 'fun': lambda params: np.linalg.norm(params[2:(px+2)]) - 1},
#         {'type': 'ineq', 'fun': lambda params: params[2]},
#         {'type': 'ineq', 'fun': lambda params: params[0]-(5/n)},
#         {'type': 'ineq', 'fun': lambda params: params[1]-(5/n)})

#     # define the constraint
#     # cons = ({'type': 'eq', 'fun': lambda params: np.linalg.norm(params[:px]) - 1})
    
#     # define the initial values for beta and theta
#     b1_init = 0.5
#     b2_init = 0.5
#     beta_init = np.ones(px)/ np.sqrt(px)
#     theta_init = np.ones(pz)

#     params_init = np.concatenate(([b1_init, b2_init], beta_init, theta_init))

#     # minimize the objective function
#     res = minimize(objective, params_init, method='SLSQP', constraints=cons)

#     # extract the optimal values for beta and theta
#     bandwidth1_opt = res.x[0]
#     bandwidth2_opt = res.x[1]
#     beta_opt = res.x[2:(px+2)]
#     theta_opt = res.x[(px+2):]
    
#     # return beta_opt, theta_opt, bandwidth1, bandwidth2
#     return bandwidth1_opt, bandwidth2_opt, beta_opt, theta_opt

from datagenerate import *

beta_0 = np.array([1/3, -2/3, 2/3])
theta_0 = np.array([2,1])
data = generate_data(200,5)
beta_opt, theta_opt = optimize_plsim_h_2d(data, bandwidth1= 0.5, bandwidth2=0.5)
loss = loss_plsim_2d(data, 0.15,  0.3, beta_0, theta_0)
print(beta_opt)
print(theta_opt)
# print(beta_opt, theta_opt)
