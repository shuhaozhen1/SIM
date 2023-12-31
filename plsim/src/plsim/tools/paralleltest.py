import datagenerate as dg
import inference as infer
import numpy as np
from multiprocessing import Pool

# def calculate_error_rate(i):
#     try:
#         time_points, X_samples, Z_samples, Y_samples =  dg.generate_data(200, 4, np.array([0.8,0.6]), np.array([2, 3]))

#         data = {'T':time_points, 'X':X_samples, 'Z': Z_samples, 'Y': Y_samples} 

#         grid = np.arange(0.2, 0.8, 0.1)
#         u_grid = 0.5 * np.sin(np.pi /2 * grid) + 0.5 * grid
#         points = np.column_stack((grid, u_grid))

#         true_eta = grid + u_grid ** 2

#         result = infer.bootstrap_inference_dd(data=data, points= points,k=5)

#         if np.all((result['scb_l'] <= true_eta) & (true_eta <= result['scb_u'])):
#             return 0
#         else:
#             return 1
#     except:
#         return None

# if __name__ == "__main__":
#     with Pool() as p:
#         error_rate = p.map(calculate_error_rate, range(8))
#     error_rate = [i for i in error_rate if i is not None]  # Remove None values
#     type1 = np.mean(error_rate)
#     print(type1)
#     print(error_rate)

n = 100
m = 50
beta = np.array([0.8,0.6])
theta = np.array([2, 3])
def calculate_error_rate(i):
    try:
        time_points, X_samples, Z_samples, Y_samples =  dg.generate_data(n, m, beta, theta)

        data = {'T':time_points, 'X':X_samples, 'Z': Z_samples, 'Y': Y_samples} 

        grid = np.arange(0.4, 0.8, 0.1)
        u_grid = 0.5 * np.sin(np.pi /2 * grid) + 0.5 * grid
        points = np.column_stack((grid, u_grid))

        true_eta = grid + u_grid ** 2

        std_norm_vars1 = 1.1 * np.random.standard_normal(n)

        std_norm_vars2 = 1.1 * np.random.standard_normal(n)

        beta_hat = beta + np.mean(std_norm_vars1)
        theta_hat = theta + np.mean(std_norm_vars2)

        beta_hat = beta 
        theta_hat = theta 


        result = infer.pseido_inference(data=data, bandwidth1= 0.11, bandwidth2= 0.11, 
                                        points= points, beta_hat=beta_hat, theta_hat=theta_hat, b_time=5000)

        if np.all((result['scb_l'] <= true_eta) & (true_eta <= result['scb_u'])):
            return 0
        else:
            return 1
    except:
        return None

if __name__ == "__main__":
    with Pool() as p:
        error_rate = p.map(calculate_error_rate, range(40))
    error_rate = [i for i in error_rate if i is not None]  # Remove None values
    type1 = np.mean(error_rate)
    print(type1)
    print(error_rate)
