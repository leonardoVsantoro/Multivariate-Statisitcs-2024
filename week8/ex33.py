import numpy as np
from scipy.stats import norm

def calculate_sample_correlation(X, Y):
    return np.sum((X - X.mean()) * (Y - Y.mean())) / np.sqrt(np.sum((X - X.mean())**2) * np.sum((Y - Y.mean())**2))

def fisher_z_transform(r):
    return 0.5 * np.log((1 + r) / (1 - r))

def inverse_fisher_z_transform(z):
    return np.tanh(z)



N = 100
X = np.random.normal(0, 1, N)
Y = np.random.normal(0, 1, N)
# Y = 3*X+2 + np.random.normal(0, 0.1, N)
r = calculate_sample_correlation(X, Y)
z = fisher_z_transform(r)


q025 = norm.ppf(0.025, loc=fisher_z_transform(r), scale=1 / np.sqrt(N - 3))
q975 = norm.ppf(0.975, loc=fisher_z_transform(r), scale=1 / np.sqrt(N - 3))

lower_bound = z - q025
upper_bound = z + q975
lower_bound_r = inverse_fisher_z_transform(lower_bound)
upper_bound_r = inverse_fisher_z_transform(upper_bound)
CI =  (lower_bound_r.round(3), upper_bound_r.round(3))

print(CI)