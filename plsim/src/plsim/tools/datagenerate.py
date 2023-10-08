import numpy as np


def generate_data(n, m, beta, theta):
    """
    Generates n sets of data for the partially linear single index model with functional data.

    Parameters:
    n (int): Number of realizations.
    m (int): Average number of time points for each realization.
    beta (ndarray): Array of shape (p,) containing the unknown parameters beta.
    theta (ndarray): Array of shape (q,) containing the unknown parameters theta.

    Returns:
    Tuple containing the following arrays:
        - time_points: List of length n containing arrays of shape (m_i,) containing the time points for each realization i.
        - X_samples: List of length n containing arrays of shape (m_i, p) containing the predictor variables X(T_{ij}) for each realization i.
        - Z_samples: List of length n containing arrays of shape (m_i, q) containing the predictor variables Z(T_{ij}) for each realization i.
        - Y_samples: List of length n containing arrays of shape (m_i,) containing the response variable Y(T_{ij}) for each realization i.
    """
    # Define functions for X and Z
    def X1(t):
        return t ** 2

    def X2(t):
        return t ** 3

    def Z1(t):
        return np.cos(2 * np.pi * t)

    def Z2(t):
        return np.sin(2 * np.pi * t)

    # Define function for mu
    def mu(x, y):
        return  x + y ** 2 

    # Define function for epsilon
    def epsilon(t):
        return t

    # Generate data for each realization
    time_points = []
    X_samples = []
    Z_samples = []
    Y_samples = []
    for i in range(n):
        # Generate m_i sets of time points (T_{ij}) uniformly on [0,1] for each realization i
        m_i = np.random.randint(m - 2, m + 3)
        time_points_i = np.sort(np.random.uniform(size=m_i))
        time_points.append(time_points_i)

        # Generate m_i sets of predictor variables X(T_{ij}) and Z(T_{ij}) using known functions
        X1_i = np.random.uniform(0,1,1) * X1(time_points_i)
        X2_i = np.random.uniform(0,1,1) * X2(time_points_i)
        Z1_i = np.random.normal(loc=1, scale=1, size=1) * Z1(time_points_i)
        Z2_i = np.random.normal(loc=1, scale=1, size=1) * Z2(time_points_i)
        X_i = np.column_stack((X1_i, X2_i))
        Z_i = np.column_stack((Z1_i, Z2_i))
        X_samples.append(X_i)
        Z_samples.append(Z_i)

        # Calculate U and mu
        U_i = np.dot(X_i, beta)
        mu_i = mu(time_points_i, U_i)

        # Calculate Y
        thetaZi = np.dot(Z_i, theta)
        e_i = np.random.normal(loc=0, scale=0.1, size=m_i) * epsilon(time_points_i)
        Y_i = mu_i + thetaZi + e_i
        Y_samples.append(Y_i)

    return time_points, X_samples, Z_samples, Y_samples


def simple_generate_data(n, m, beta, theta):
    """
    Generates n sets of data for the partially linear single index model with functional data.

    Parameters:
    n (int): Number of realizations.
    m (int): Average number of time points for each realization.
    beta (ndarray): Array of shape (p,) containing the unknown parameters beta.
    theta (ndarray): Array of shape (q,) containing the unknown parameters theta.

    Returns:
    Tuple containing the following arrays:
        - time_points: List of length n containing arrays of shape (m_i,) containing the time points for each realization i.
        - X_samples: List of length n containing arrays of shape (m_i, p) containing the predictor variables X(T_{ij}) for each realization i.
        - Z_samples: List of length n containing arrays of shape (m_i, q) containing the predictor variables Z(T_{ij}) for each realization i.
        - Y_samples: List of length n containing arrays of shape (m_i,) containing the response variable Y(T_{ij}) for each realization i.
    """
    # Define functions for X and Z
    def X1(t):
        return t ** 2

    def Z1(t):
        return np.cos(2 * np.pi * t)

    def Z2(t):
        return np.sin(2 * np.pi * t)

    # Define function for mu
    def mu(x, y):
        return  x + y ** 2 

    # Define function for epsilon
    def epsilon(t):
        return t

    # Generate data for each realization
    time_points = []
    X_samples = []
    Z_samples = []
    Y_samples = []
    for i in range(n):
        # Generate m_i sets of time points (T_{ij}) uniformly on [0,1] for each realization i
        m_i = np.random.randint(m - 2, m + 3)
        time_points_i = np.sort(np.random.uniform(size=m_i))
        time_points.append(time_points_i)

        # Generate m_i sets of predictor variables X(T_{ij}) and Z(T_{ij}) using known functions
        X_i = np.random.uniform(0,1,1) * X1(time_points_i)

        Z1_i = np.random.normal(loc=1, scale=1, size=1) * Z1(time_points_i)
        Z2_i = np.random.normal(loc=1, scale=1, size=1) * Z2(time_points_i)
        
        Z_i = np.column_stack((Z1_i, Z2_i))
        X_samples.append(X_i)
        Z_samples.append(Z_i)

        # Calculate U and mu
        U_i = np.dot(X_i, beta)
        mu_i = mu(time_points_i, U_i)

        # Calculate Y
        thetaZi = np.dot(Z_i, theta)
        e_i = np.random.normal(loc=0, scale=0.1, size=m_i) * epsilon(time_points_i)
        Y_i = mu_i + thetaZi + e_i
        Y_samples.append(Y_i)

    return time_points, X_samples, Z_samples, Y_samples