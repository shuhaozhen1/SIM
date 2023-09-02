import numpy as np

def generate_samples(n, m, mu, beta, theta):
    data = {'t': [], 'x': [], 'z': [], 'y': []}
    for i in range(n):
        m_i = int(np.random.normal(m, 1))
        t = np.random.uniform(0, 1, m_i)
        x = np.random.randn(m_i, beta.shape[0])
        z = np.random.randn(m_i)
        y = mu(t, x.dot(beta)) + theta * z + np.random.randn(m_i)
        data['t'].append(t)
        data['x'].append(x)
        data['z'].append(z)
        data['y'].append(y)
    return data
