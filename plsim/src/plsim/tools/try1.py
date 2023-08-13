from inference import bootstrap_inference

import numpy as np

# generate the data
n = 200
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

x0 = np.linspace(0.2, 2 , 27)
result = bootstrap_inference(data, kernel_type, degree, b_time=2000, 
                              x0=x0, quantile=0.95)

import matplotlib.pyplot as plt


# Assuming bs_result is a dictionary containing the results
scb_l = result['scb_l']
scb_u = result['scb_u']

# Plot the band
plt.fill_between(x0, scb_l, scb_u, alpha=0.2)

# Plot the true function
# Assuming true_function is a function that takes x0 as input and returns the true values
true_values = np.array(x0**2)
plt.plot(x0, true_values)

# Show the plot
plt.show()
