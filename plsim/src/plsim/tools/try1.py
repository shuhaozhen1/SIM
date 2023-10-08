from datagenerate import simple_generate_data
from inference import bootstrap_inference, center_eta
from est2d import optimize_plsim_h_2d, local_polynomial_regression_2d

import numpy as np

# p = np.array([(xi,yi) for xi in np.linspace(0.1,0.9,11) for yi in np.linspace(min2,max2,11)])

time_points, X_samples, Z_samples, Y_samples =  simple_generate_data(100, 5, np.array([1]), np.array([2, 3]))

data = {'T':time_points, 'X':X_samples, 'Z': Z_samples, 'Y': Y_samples} 

# min2 = np.min(np.dot(np.concatenate(data['X']), np.array([0.8, 0.6])))
# max2 = np.max(np.dot(np.concatenate(data['X']), np.array([0.8, 0.6])))

# print(min2,max2)

# center_data = center_eta(data, 'epa', 1, 0.3, 0.3)

# n = len(center_data['T'])
# lengths = [len(subarray) for subarray in center_data['T']]

# # Bootstrap
# bs_res = []
# t = np.concatenate(center_data['T'])
# u = np.concatenate(center_data['U'])
# e = np.concatenate(center_data['E'])

# for i in range(10):
#     # Generate n standard normal variables
#     std_norm_vars = np.random.standard_normal(n)

# # Repeat each standard normal variable according to the lengths and flatten the list
#     y_new = np.multiply(np.concatenate([np.repeat(var, length) for var, length in zip(std_norm_vars, lengths)]),
#                             e)
        
#     data_new = np.column_stack((t,u,y_new))
#     est_new = local_polynomial_regression_2d(data_new, 'epa', 
#                                              0.3,0.3, 1,p)
#     est_new_i = np.sqrt(n) * est_new[:,0]
#     bs_res.append(est_new_i)
    
# bs_res = np.array(bs_res)

#     # estimation of standard deviation
# est_var = np.mean(np.square(bs_res), axis=0)
# est_sd = np.sqrt(est_var)

#     # obtain the critical value
# standardized_bs = bs_res / est_sd
# row_suprema = np.max(np.abs(standardized_bs), axis=1)
# critical = np.quantile(row_suprema, 0.95)

#     # eta_hat_d = local_polynomial_regression_2d(center_data['Data_transformed'], kernel_type, 
#     #                                           bandwidth1, bandwidth2, degree, points)
#     # eta_hat = eta_hat_d[:,0]

#     # scb_l = eta_hat -  critical * est_sd / np.sqrt(n)
#     # scb_u = eta_hat +  critical * est_sd / np.sqrt(n)
#     # width = np.mean(scb_u-scb_l)/2

#     # bs_result = {'scb_l':scb_l,'scb_u':scb_u,'est_sd':est_sd, 'c_value':critical, 'width': width}
#     # return bs_result




# # result = bootstrap_inference(data, 'epa', 1, 0.4, 0.4, 500, p)

# # print(result)