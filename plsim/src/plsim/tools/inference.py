from est2d import optimize_plsim_h_2d, local_polynomial_regression_2d
import numpy as np

# Inference for non-parametric part \eta

def center_eta(data, kernel_type, degree, bandwidth1, bandwidth2):
    # Estimation
    
    beta_hat, theta_hat = optimize_plsim_h_2d(data, kernel_type, degree, bandwidth1, bandwidth2)
    
    # Plug-in for the centralization of the repsonse for non-parametric part \eta
    t = np.concatenate(data['T'])
    x = np.concatenate(data['X'])
    z = np.concatenate(data['Z'])
    y = np.concatenate(data['Y'])

    # transformed model: y- \theta^T Z = eta(\beta^T X) + \epsilon
    y_transformed = y - np.dot(z, theta_hat)
    u_transformed = np.dot(x, beta_hat) 
    data_transformed = np.column_stack((t, u_transformed, y_transformed))
    tx = np.column_stack((t, u_transformed))

    # estimate eta
    eta_hat_d = local_polynomial_regression_2d(data_transformed, kernel_type, bandwidth1, bandwidth2, degree, xy0=tx)
    eta_hat = eta_hat_d[:,0]

    # residual
    e_est = y_transformed - eta_hat

    # Get the lengths of the subarrays in the original list
    lengths = [len(subarray) for subarray in data['T']]

    # Calculate the indices where the splits should happen
    indices = np.cumsum(lengths)[:-1]

    # Use numpy.split to split the array into a list of the same structure
    e = np.split(e_est, indices)
    u = np.split(u_transformed, indices)

    center_data = {'T':data['T'], 'U':u,'E':e, 'Data_transformed':data_transformed}

    return center_data

# Bootstrap for critical value and standard deviation

def bootstrap_inference(data, kernel_type, degree, bandwidth1, bandwidth2, b_time, points, quantile=0.95):
    center_data = center_eta(data, kernel_type, degree, bandwidth1, bandwidth2)
    n = len(center_data['T'])
    lengths = [len(subarray) for subarray in center_data['T']]

    # Bootstrap
    bs_res = []
    t = np.concatenate(center_data['T'])
    u = np.concatenate(center_data['U'])
    e = np.concatenate(center_data['E'])

    for i in range(b_time):
    # Generate n standard normal variables
        std_norm_vars = np.random.standard_normal(n)

# Repeat each standard normal variable according to the lengths and flatten the list
        y_new = np.multiply(np.concatenate([np.repeat(var, length) for var, length in zip(std_norm_vars, lengths)]),
                            e)
        
        data_new = np.column_stack((t,u,y_new))
        est_new = local_polynomial_regression_2d(data_new, kernel_type, 
                                              bandwidth1,bandwidth2, degree,points)
        est_new_i = np.sqrt(n) * est_new[:,0]
        bs_res.append(est_new_i)
    
    bs_res = np.array(bs_res)

    # estimation of standard deviation
    est_var = np.mean(np.square(bs_res), axis=0)
    est_sd = np.sqrt(est_var)

    # obtain the critical value
    standardized_bs = bs_res / est_sd
    row_suprema = np.max(np.abs(standardized_bs), axis=1)
    critical = np.quantile(row_suprema, quantile)

    eta_hat_d = local_polynomial_regression_2d(center_data['Data_transformed'], kernel_type, 
                                              bandwidth1, bandwidth2, degree, points)
    eta_hat = eta_hat_d[:,0]

    scb_l = eta_hat -  critical * est_sd / np.sqrt(n)
    scb_u = eta_hat +  critical * est_sd / np.sqrt(n)
    width = np.mean(scb_u-scb_l)/2

    bs_result = {'scb_l':scb_l,'scb_u':scb_u,'est_sd':est_sd, 'c_value':critical, 'width': width}
    return bs_result




