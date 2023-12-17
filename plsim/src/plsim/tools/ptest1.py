import datagenerate as dg
import inference as infer
import numpy as np
import time
from multiprocessing import Pool

start_time = time.perf_counter()
def calculate_error_rate(i):
    try:
        time_points, X_samples, Z_samples, Y_samples =  dg.generate_data(100, 15, np.array([0.8,0.6]), np.array([2, 3]))

        data = {'T':time_points, 'X':X_samples, 'Z': Z_samples, 'Y': Y_samples} 

        grid = np.arange(0.2, 0.8, 0.1)
        u_grid = 0.5 * np.sin(np.pi /2 * grid) + 0.5 * grid
        points = np.column_stack((grid, u_grid))

        true_eta = grid + u_grid ** 2

        result = infer.bootstrap_inference(data=data, points= points, bandwidth1= 0.3, bandwidth2= 0.3)

        if np.all((result['scb_l'] <= true_eta) & (true_eta <= result['scb_u'])):
            return 0
        else:
            return 1
    except:
        return None

if __name__ == "__main__":
    with Pool() as p:
        error_rate = p.map(calculate_error_rate, range(160))
    error_rate = [i for i in error_rate if i is not None]  # Remove None values
    type1 = np.mean(error_rate)
    print(type1)
    print(error_rate)

end_time = time.perf_counter()
run_time = end_time - start_time
print(run_time)
