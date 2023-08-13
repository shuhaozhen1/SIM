import numpy as np
x0 = np.linspace(0.2,2,11)
print(np.array(x0**2))
coverage = 0
x0 = np.linspace(0.2,2,11)
for i in range(100):
    result = bootstrap_inference(data, kernel_type, degree, b_time=2000, 
                          x0 = x0, quantile= 0.95, domain_l=0.05, domain_u=2.15)
    if (np.array(result['scb_l']) <= np.array(x0**2)).all() and (np.array(x0**2) <= np.array(result['scb_u'])).all()  :
        coverage += 1
ratio = coverage / 10

print(ratio)