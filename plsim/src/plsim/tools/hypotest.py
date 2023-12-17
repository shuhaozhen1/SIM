
import datagenerate as dg
import inference as infer

import numpy as np

error_rate = []
for i in range(20):
    time_points, X_samples, Z_samples, Y_samples =  dg.generate_data(100, 4, np.array([0.8,0.6]), np.array([2, 3]))

    data = {'T':time_points, 'X':X_samples, 'Z': Z_samples, 'Y': Y_samples} 

    grid = np.arange(0.2, 0.8, 0.1)
    u_grid = 0.5 * np.sin(np.pi /2 * grid) + 0.5 * grid
    points = np.column_stack((grid, u_grid))

    true_eta = grid + u_grid ** 2

    result = infer.bootstrap_inference_dd(data=data, points= points)

    if np.all((result['scb_l'] <= true_eta) & (true_eta <= result['scb_u'])):
        error_ratei = 0
    else:
        error_ratei = 1
    
    error_rate.append(error_ratei)

type1 = np.mean(np.array(error_rate))
print(type1)




