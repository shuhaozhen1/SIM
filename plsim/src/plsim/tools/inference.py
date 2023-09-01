from est import optimize_plsim_h, local_polynomial_regression
import numpy as np

# Inference for non-parametric part \eta

def center_eta(data, kernel_type, degree, bandwidth):
    # Estimation
    
    beta_hat, theta_hat, bandwidth_opt = optimize_plsim_h(data, kernel_type, degree, bandwidth=bandwidth)
    else: 
        beta_hat, theta_hat, bandwidth_opt = optimize_plsim_h(data, kernel_type, degree, 
                                                            domain_check=True, lower=domain_l,upper=domain_u)
    # Plug-in for the centralization of the repsonse for non-parametric part \eta
    x = data['x']
    z = data['z']
    y = data['y']

    y_transformed = y - theta_hat.T @ z.T
    x_transformed = beta_hat.T @ x.T
    data_transformed = np.column_stack((x_transformed,y_transformed))
    y_hat = local_polynomial_regression(data_transformed, kernel_type = kernel_type, 
                                                                   bandwidth=bandwidth_opt, degree = degree, x0=x_transformed)

    y_centered = y - theta_hat.T @ z.T - y_hat[:,0]

    center_data = {'x_transformed':x_transformed, 'y_centered':y_centered,'y_transformed':y_transformed,
                    'h_opt': bandwidth_opt}

    return center_data

# Bootstrap for critical value and standard deviation

def bootstrap_inference(data, kernel_type, degree, b_time, x0, quantile, domain_l=None, domain_u=None):
    center_data = center_eta(data, kernel_type, degree, domain_l, domain_u)
    n = center_data['x_transformed'].shape[0]

    # Bootstrap
    bs_res = []
    x = center_data['x_transformed']
    y = center_data['y_centered']
    for i in range(b_time):
        y_new = y * np.random.randn(y.shape[0])
        data_new = np.column_stack((x,y_new))
        est_new = local_polynomial_regression(data=data_new, kernel_type=kernel_type, 
                                              bandwidth=center_data['h_opt'], degree=degree, x0= x0 )
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

    data_transformed = np.column_stack((center_data['x_transformed'],center_data['y_transformed']))
    eta_hat = local_polynomial_regression(data=data_transformed, kernel_type=kernel_type, 
                                              bandwidth=center_data['h_opt'], degree=degree, x0= x0 )
    eta_hat = eta_hat[:,0]
    
    scb_l = eta_hat -  critical * est_sd / np.sqrt(n)
    scb_u = eta_hat +  critical * est_sd / np.sqrt(n)
    width = np.mean(scb_u-scb_l)/2

    bs_result = {'scb_l':scb_l,'scb_u':scb_u,'est_sd':est_sd, 'c_value':critical, 'width': width}
    return bs_result




