import numpy as np
# Define kernel functions

def quartic_kernel(x):
    return np.where(np.abs(x) <= 1, 15/16 * (1 - x**2)**2, 0)

# quartic Kernel
def quartic_kernel_2d(x, y): 
    z = quartic_kernel(x) * quartic_kernel(y)
    return z

x = np.random.rand(10)
y = np.random.rand(10)


t_design = np.vander(x - 0.1, 2 + 1, increasing=True)
print(x)
print(t_design)