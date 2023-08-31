
from est import local_polynomial_regression

import numpy as np

# Generate some sample data
n = 200
x1 = np.random.uniform(0,1,n)
x2 = np.random.uniform(0,1,n)
x = (x1 + x2) / 2

y = np.sin(2 * np.pi * x) + 0.1 * np.random.randn(n)
data = np.column_stack((x, y))

# Set the parameters
kernel_type = 'epa'
bandwidth = 0.1
degree = 1

x0 = np.linspace(0, 1, 100)

# Perform local polynomial regression
beta_hat = local_polynomial_regression(data, kernel_type, bandwidth, degree, x0)

# Plot the results
import matplotlib.pyplot as plt

plt.scatter(x, y)
plt.plot(x0, beta_hat[:, 0], 'r')
plt.show()

