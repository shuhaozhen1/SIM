import est2d
# from est2d import optimize_plsim_h_2d, local_polynomial_regression_2d, optimize_plsim_2d
import numpy as np

# Inference for non-parametric part \eta given bandwidth

def center_eta(data, kernel_type, degree, bandwidth1, bandwidth2):
    # Estimation
    beta_hat, theta_hat = est2d.optimize_plsim_h_2d(data, kernel_type, degree, bandwidth1, bandwidth2)

    mi = [len(subarray) for subarray in data['T']]
    
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
    eta_hat_d = est2d.local_polynomial_regression_2d(data_transformed, kernel_type, 
                                                     bandwidth1, bandwidth2, degree, xy0=tx, mi=mi)
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

def bootstrap_inference(data, bandwidth1, bandwidth2,  points,kernel_type = 'epa' , degree=1, b_time =2000 ,quantile=0.95):
    center_data = center_eta(data, kernel_type, degree, bandwidth1, bandwidth2)
    n = len(center_data['T'])
    lengths = [len(subarray) for subarray in center_data['T']]

    # Bootstrap
    bs_res = []

    t = np.concatenate(center_data['T'])
    u = np.concatenate(center_data['U'])
    e = np.concatenate(center_data['E'])

    eta_hat_d = est2d.local_polynomial_regression_2d(center_data['Data_transformed'], kernel_type, 
                                              bandwidth1, bandwidth2, degree, points, mi=lengths)
    eta_hat = eta_hat_d[:,0]

    for i in range(b_time):
    # Generate n standard normal variables
        std_norm_vars = np.random.standard_normal(n)

# Repeat each standard normal variable according to the lengths and flatten the list
        y_new = np.repeat(std_norm_vars, lengths) * e
        # y_new = np.multiply(np.concatenate([np.repeat(var, length) for var, length in zip(std_norm_vars, lengths)]),
        #                     e)
        
        data_new = np.column_stack((t,u,y_new))
        est_new = est2d.local_polynomial_regression_2d(data_new, kernel_type, 
                                              bandwidth1,bandwidth2, degree,points, mi=lengths)
        est_new_i = est_new[:,0]
        diff_i = np.sqrt(n)*  est_new_i - np.sqrt(n) * np.mean(std_norm_vars) * eta_hat

        bs_res.append(diff_i)
    
    bs_res = np.array(bs_res)

    # estimation of standard deviation
    est_var = np.mean(np.square(bs_res), axis=0)
    est_sd = np.sqrt(est_var)

    # obtain the critical value
    standardized_bs = bs_res / est_sd
    row_suprema = np.max(np.abs(standardized_bs), axis=1)
    critical = np.quantile(row_suprema, quantile)


    scb_l = eta_hat -  critical * est_sd / np.sqrt(n)
    scb_u = eta_hat +  critical * est_sd / np.sqrt(n)
    width = np.mean(scb_u-scb_l)/2

    bs_result = {'scb_l':scb_l,'scb_u':scb_u,'est_sd':est_sd, 'c_value':critical, 'width': width}
    return bs_result


# Inference for non-parametric part \eta (datadriven)

def center_eta_dd(data, kernel_type='epa', degree=1, k=1):
    # Estimation
    bandwidth1, bandwidth2, beta_hat, theta_hat = est2d.optimize_plsim_2d(data, kernel_type, degree, k=k)
    
    # Plug-in for the centralization of the repsonse for non-parametric part \eta
    t = np.concatenate(data['T'])
    x = np.concatenate(data['X'])
    z = np.concatenate(data['Z'])
    y = np.concatenate(data['Y'])

    mi = [len(subarray) for subarray in data['T']]

    # transformed model: y- \theta^T Z = eta(\beta^T X) + \epsilon
    y_transformed = y - np.dot(z, theta_hat)
    u_transformed = np.dot(x, beta_hat) 
    data_transformed = np.column_stack((t, u_transformed, y_transformed))
    tx = np.column_stack((t, u_transformed))

    # estimate eta
    eta_hat_d = est2d.local_polynomial_regression_2d(data_transformed, kernel_type, 
                                                     bandwidth1, bandwidth2, degree, xy0=tx, mi=mi)
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

    center_data = {'T':data['T'], 'U':u,'E':e, 'Data_transformed':data_transformed,'b1':bandwidth1,'b2':bandwidth2}

    return center_data

# Bootstrap for critical value and standard deviation

def bootstrap_inference_dd(data, points, kernel_type='epa', degree=1, b_time=2000,quantile=0.95,k=1):
    center_data = center_eta_dd(data, kernel_type, degree,k=k)
    n = len(data['T'])
    mi = [len(subarray) for subarray in data['T']]
    weights = np.repeat(mi,mi)

    bandwidth1 = center_data['b1']
    bandwidth2 = center_data['b2']

    bw1_forbs = 0.5 * bandwidth1
    bw2_forbs = 0.5 * bandwidth1

    if kernel_type == 'epa':
        kernel = est2d.epanechnikov_kernel_2d
    elif kernel_type == 'unif':
        kernel = est2d.uniform_kernel_2d
    elif kernel_type == 'triangle':
        kernel = est2d.triangle_kernel_2d
    else:
        raise ValueError('Unsupported kernel type')

    t = np.concatenate(center_data['T'])
    u = np.concatenate(center_data['U'])
    e = np.concatenate(center_data['E'])

    data_forsmooth = np.column_stack((t, u, e))

    center_i = []

    for i in range(n):
        x1 = data_forsmooth[:, 0]
        x2 = data_forsmooth[:, 1]

        ti = center_data['T'][i]
        ui = center_data['U'][i] 

        milen = len(ti)

        beta_hati = []

        for xi1, xi2 in points:
            W = np.diag(kernel(np.abs(x1-xi1)/bandwidth1, np.abs(x2-xi2)
                               /bandwidth2)/bandwidth1/bandwidth2/weights)
            Wi = np.diag(kernel(np.abs(ti-xi1)/bw1_forbs, np.abs(ui-xi2)
                               /bw2_forbs)/bw1_forbs/bw2_forbs/milen)
            X1_design = np.vander(x1 - xi1, degree + 1, increasing=True)
            ti_design = np.vander(ti - xi1, degree + 1, increasing=True)
            X2_design = np.vander(x2 - xi2, degree + 1, increasing=True)[:, 1:]
            ui_design = np.vander(ui - xi2, degree + 1, increasing=True)[:, 1:]
            X = np.column_stack((X1_design,X2_design))
            Xi = np.column_stack((ti_design,ui_design))
            yi = center_data['E'][i]
            beta_hat_i = np.linalg.inv(X.T @ W @ X/n) @ Xi.T @ Wi @ yi

            beta_hati.append(beta_hat_i)

        beta_hati = np.array(beta_hati)[:,0]
        
        center_i.append(beta_hati)

    fi = np.array(center_i)

    fi_mean = np.mean(fi, axis= 0)

    # estimation of standard deviation
    est_var = np.mean(np.square(fi -fi_mean), axis=0)
    est_sd = np.sqrt(est_var)

    bs_res = []

    for i in range(b_time):
    # Generate n standard normal variables
        std_norm_vars = np.random.standard_normal(n)

# Repeat each standard normal variable according to the lengths and flatten the list
        fi_new = ((fi - fi_mean) * std_norm_vars[:,np.newaxis]) / est_sd
        y_new = np.mean(fi_new, axis=0) * np.sqrt(n)
        sup_new = np.max(np.abs(y_new))

        bs_res.append(sup_new)
    
    bs_res = np.array(bs_res)

    critical = np.quantile(bs_res, quantile)

    eta_hat_d = est2d.local_polynomial_regression_2d(center_data['Data_transformed'], kernel_type, 
                                              bandwidth1, bandwidth2, degree, points, mi=mi)
    eta_hat = eta_hat_d[:,0]

    scb_l = eta_hat -  critical * est_sd / np.sqrt(n)
    scb_u = eta_hat +  critical * est_sd / np.sqrt(n)
    width = np.mean(scb_u-scb_l)/2

    bs_result = {'scb_l':scb_l,'scb_u':scb_u,'est_sd':est_sd, 'c_value':critical, 'width': width, 'est_eta':eta_hat}
    return bs_result



def bootstrap_inference_dd(data, points, kernel_type='epa', degree=1, b_time=2000,quantile=0.95,k=1):
    center_data = center_eta_dd(data, kernel_type, degree,k=k)
    n = len(data['T'])
    mi = [len(subarray) for subarray in data['T']]
    weights = np.repeat(mi,mi)

    bandwidth1 = center_data['b1']
    bandwidth2 = center_data['b2']

    bw1_forbs = 0.5 * bandwidth1
    bw2_forbs = 0.5 * bandwidth1

    if kernel_type == 'epa':
        kernel = est2d.epanechnikov_kernel_2d
    elif kernel_type == 'unif':
        kernel = est2d.uniform_kernel_2d
    elif kernel_type == 'triangle':
        kernel = est2d.triangle_kernel_2d
    else:
        raise ValueError('Unsupported kernel type')

    t = np.concatenate(center_data['T'])
    u = np.concatenate(center_data['U'])
    e = np.concatenate(center_data['E'])

    data_forsmooth = np.column_stack((t, u, e))

    center_i = []

    for i in range(n):
        x1 = data_forsmooth[:, 0]
        x2 = data_forsmooth[:, 1]

        ti = center_data['T'][i]
        ui = center_data['U'][i] 

        milen = len(ti)

        beta_hati = []

        for xi1, xi2 in points:
            W = np.diag(kernel(np.abs(x1-xi1)/bandwidth1, np.abs(x2-xi2)
                               /bandwidth2)/bandwidth1/bandwidth2/weights)
            Wi = np.diag(kernel(np.abs(ti-xi1)/bw1_forbs, np.abs(ui-xi2)
                               /bw2_forbs)/bw1_forbs/bw2_forbs/milen)
            X1_design = np.vander(x1 - xi1, degree + 1, increasing=True)
            ti_design = np.vander(ti - xi1, degree + 1, increasing=True)
            X2_design = np.vander(x2 - xi2, degree + 1, increasing=True)[:, 1:]
            ui_design = np.vander(ui - xi2, degree + 1, increasing=True)[:, 1:]
            X = np.column_stack((X1_design,X2_design))
            Xi = np.column_stack((ti_design,ui_design))
            yi = center_data['E'][i]
            beta_hat_i = np.linalg.inv(X.T @ W @ X/n) @ Xi.T @ Wi @ yi

            beta_hati.append(beta_hat_i)

        beta_hati = np.array(beta_hati)[:,0]
        
        center_i.append(beta_hati)

    fi = np.array(center_i)

    fi_mean = np.mean(fi, axis= 0)

    # estimation of standard deviation
    est_var = np.mean(np.square(fi -fi_mean), axis=0)
    est_sd = np.sqrt(est_var)

    bs_res = []

    for i in range(b_time):
    # Generate n standard normal variables
        std_norm_vars = np.random.standard_normal(n)

# Repeat each standard normal variable according to the lengths and flatten the list
        fi_new = ((fi - fi_mean) * std_norm_vars[:,np.newaxis]) / est_sd
        y_new = np.mean(fi_new, axis=0) * np.sqrt(n)
        sup_new = np.max(np.abs(y_new))

        bs_res.append(sup_new)
    
    bs_res = np.array(bs_res)

    critical = np.quantile(bs_res, quantile)

    eta_hat_d = est2d.local_polynomial_regression_2d(center_data['Data_transformed'], kernel_type, 
                                              bandwidth1, bandwidth2, degree, points, mi=mi)
    eta_hat = eta_hat_d[:,0]

    scb_l = eta_hat -  critical * est_sd / np.sqrt(n)
    scb_u = eta_hat +  critical * est_sd / np.sqrt(n)
    width = np.mean(scb_u-scb_l)/2

    bs_result = {'scb_l':scb_l,'scb_u':scb_u,'est_sd':est_sd, 'c_value':critical, 'width': width, 'est_eta':eta_hat}
    return bs_result


### pseudo part
def psuedo_center(data, kernel_type, degree, bandwidth1, bandwidth2, beta_hat, theta_hat):
    # Estimation

    mi = [len(subarray) for subarray in data['T']]
    
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
    eta_hat_d = est2d.local_polynomial_regression_2d(data_transformed, kernel_type, 
                                                     bandwidth1, bandwidth2, degree, xy0=tx, mi=mi)
    eta_hat = eta_hat_d[:,0]

    # residual
    e_est = y_transformed - eta_hat


    # Calculate the indices where the splits should happen
    indices = np.cumsum(mi)[:-1]

    # Use numpy.split to split the array into a list of the same structure
    e = np.split(e_est, indices)
    u = np.split(u_transformed, indices)

    center_data = {'T':data['T'], 'U':u,'E':e, 'Data_transformed':data_transformed}

    return center_data

def pseido_inference(data, bandwidth1, bandwidth2, beta_hat, theta_hat, 
                     points, kernel_type='epa', degree=1, b_time=2000,quantile=0.95):
    
    center_data = psuedo_center(data, kernel_type, degree, bandwidth1, bandwidth2, beta_hat, theta_hat)

    n = len(data['T'])
    mi = [len(subarray) for subarray in data['T']]
    weights = np.repeat(mi,mi)

    bw1_forbs = 0.71 * bandwidth1
    bw2_forbs = 0.71 * bandwidth2

    if kernel_type == 'epa':
        kernel = est2d.epanechnikov_kernel_2d
    elif kernel_type == 'unif':
        kernel = est2d.uniform_kernel_2d
    elif kernel_type == 'triangle':
        kernel = est2d.triangle_kernel_2d
    else:
        raise ValueError('Unsupported kernel type')

    t = np.concatenate(center_data['T'])
    u = np.concatenate(center_data['U'])
    e = np.concatenate(center_data['E'])

    data_forsmooth = np.column_stack((t, u, e))

    center_i = []

    for i in range(n):
        x1 = data_forsmooth[:, 0]
        x2 = data_forsmooth[:, 1]

        ti = center_data['T'][i]
        ui = center_data['U'][i] 

        milen = len(ti)

        beta_hati = []

        for xi1, xi2 in points:
            W = np.diag(kernel(np.abs(x1-xi1)/bandwidth1, np.abs(x2-xi2)
                               /bandwidth2)/bandwidth1/bandwidth2/weights)
            Wi = np.diag(kernel(np.abs(ti-xi1)/bw1_forbs, np.abs(ui-xi2)
                               /bw2_forbs)/bw1_forbs/bw2_forbs/milen)
            X1_design = np.vander(x1 - xi1, degree + 1, increasing=True)
            ti_design = np.vander(ti - xi1, degree + 1, increasing=True)
            X2_design = np.vander(x2 - xi2, degree + 1, increasing=True)[:, 1:]
            ui_design = np.vander(ui - xi2, degree + 1, increasing=True)[:, 1:]
            X = np.column_stack((X1_design,X2_design))
            Xi = np.column_stack((ti_design,ui_design))
            yi = center_data['E'][i]
            beta_hat_i = np.linalg.inv(X.T @ W @ X/n) @ Xi.T @ Wi @ yi

            beta_hati.append(beta_hat_i)

        beta_hati = np.array(beta_hati)[:,0]
        
        center_i.append(beta_hati)

    fi = np.array(center_i)

    fi_mean = np.mean(fi, axis= 0)

    # estimation of standard deviation
    est_var = np.mean(np.square(fi -fi_mean), axis=0)
    est_sd = np.sqrt(est_var)

    bs_res = []

    for i in range(b_time):
    # Generate n standard normal variables
        std_norm_vars = np.random.standard_normal(n)

# Repeat each standard normal variable according to the lengths and flatten the list
        fi_new = ((fi - fi_mean) * std_norm_vars[:,np.newaxis]) / est_sd
        y_new = np.mean(fi_new, axis=0) * np.sqrt(n)
        sup_new = np.max(np.abs(y_new))

        bs_res.append(sup_new)
    
    bs_res = np.array(bs_res)

    critical = np.quantile(bs_res, quantile)

    eta_hat_d = est2d.local_polynomial_regression_2d(center_data['Data_transformed'], kernel_type, 
                                              bandwidth1, bandwidth2, degree, points, mi=mi)
    eta_hat = eta_hat_d[:,0]

    scb_l = eta_hat -  critical * est_sd / np.sqrt(n)
    scb_u = eta_hat +  critical * est_sd / np.sqrt(n)
    width = np.mean(scb_u-scb_l)/2

    bs_result = {'scb_l':scb_l,'scb_u':scb_u,'est_sd':est_sd, 'c_value':critical, 'width': width, 'est_eta':eta_hat}
    return bs_result