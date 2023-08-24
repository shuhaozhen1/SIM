
from est import optimize_plsim_h, local_polynomial_regression

import numpy as np

# generate the data
n = 200

x12 = np.random.uniform(0,1,n)

# Example usage:
x= np.zeros((n,2))
x[:,0] = np.random.uniform(0,0.6,n)
x[:,1] =(x12 - 0.6*x[:,0])/0.8


# x[:,0] = np.random.uniform(0,1,size=n)
# x[:,1] = np.random.uniform(0,2,size=n)

z = np.random.uniform(-1,1,size=(n,2))
error = 0.1*np.random.randn(n)
y = (0.6*x[:,0] + 0.8*x[:,1])**2 + 0.5*z[:,0] + 0.8*z[:,1] + error
data = {'x': x, 'z': z, 'y': y}

# define the kernel type, bandwidth, and degree
kernel_type = 'epa'
degree = 1

x0 = np.linspace(0, 1, 200)

# result = optimize_plsim_h(data, kernel_type, degree, bandwidth=0.08)

theta_hat, beta_hat, bandwidth = optimize_plsim_h(data, kernel_type, degree, bandwidth=0.11)

x = data['x']
z = data['z']
y = data['y']

y_transformed = y - theta_hat.T @ z.T
x_transformed = beta_hat.T @ x.T
data_transformed = np.column_stack((x_transformed,y_transformed))

eta_hat = local_polynomial_regression(data=data_transformed, kernel_type=kernel_type, bandwidth=bandwidth, degree=degree, x0= x0)

import matplotlib.pyplot as plt

# # Assuming bs_result is a dictionary containing the results
# scb_l = result['scb_l']
# scb_u = result['scb_u']

# # Plot the band
# plt.fill_between(x0, scb_l, scb_u, alpha=0.2)

# # Plot the true function
# # Assuming true_function is a function that takes x0 as input and returns the true value
plt.plot(x0, eta_hat[:,0])
true_values = np.array(x0**2)
plt.plot(x0, true_values)

# # Show the plot
plt.show()
