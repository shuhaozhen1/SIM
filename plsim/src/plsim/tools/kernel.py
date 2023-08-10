import numpy as np

def epanechnikov_kernel(x):
    return np.where(np.abs(x) <= 1, 3/4 * (1 - x**2), 0)

def uniform_kernel(x):
    return np.where(np.abs(x) <= 1, 1, 0)

def triangle_kernel(x):
    return np.where(np.abs(x) <= 1, 1 - np.abs(x), 0)

# Define a function that takes two arguments and returns their product
def multiply(x, y):
    return x * y

# Create an array of numbers
numbers = np.array([1, 2, 3, 4, 5])

# Define the second argument
y = 2

# Apply the multiply function to each element of the numbers array
# and provide the second argument y
result = multiply(numbers, y)
print(result) # [ 2  4  6  8 10]
