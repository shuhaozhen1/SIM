from inference import bootstrap_inference

import numpy as np

# generate the data
n = 100
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




from multiprocessing import Pool

x0 = np.linspace(0.2, 2, 21)

from multiprocessing import Pool

def worker(i):
    try:
        result = bootstrap_inference(data, kernel_type, degree, b_time=2000, 
                              x0=x0, quantile=0.95)
        if (np.array(result['scb_l']) <= np.array(x0**2)).all() and (np.array(x0**2) <= np.array(result['scb_u'])).all():
            return 1
        else:
            return 0
    except Exception as e:
        return 0

if __name__ == '__main__':
    with Pool() as pool:
        results = pool.map(worker, range(2000))
        successful_tries = sum(results)
        if successful_tries > 0:
            ratio = successful_tries / sum(results)
        else:
            ratio = 0
    print(ratio)
