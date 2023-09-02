
import numpy as np

# assuming `t` and `x` are two 1D arrays of the same length
t = np.array([1, 2, 3])
x = np.array([4, 5, 6])

# stack the two arrays as columns into a 2D array
result = np.column_stack((t, x))

# print the resulting array


xy0 = np.array([(xi,yi) for xi in np.linspace(0,1,10) for yi in np.linspace(0,1,10)])

xy1 = np.column_stack((np.linspace(0,1,10),np.linspace(0,1,10)))

for x,y in xy1:
    print(x,y)