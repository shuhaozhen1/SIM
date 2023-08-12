import numpy as np
n = 200
x= np.zeros((n,2))
x[:,0] = np.random.uniform(0,1,size=n)
x[:,1] = np.random.uniform(0,2,size=n)
z = np.random.uniform(-1,1,size=(n,2))
error = 0.1*np.random.randn(n, 1)
y = (0.6*x[:,0] + 0.8*x[:,1])**2 + 0.5*z[:,0] + 0.8*z[:,1] + error
data = {'x': x, 'z': z, 'y': y}

print(data['y'].shape)