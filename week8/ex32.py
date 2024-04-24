import numpy as np
from scipy.stats import norm, wishart, chi2
import matplotlib.pyplot as plt


np.random.seed(42)  # For reproducibility

p = 5
n_x = 100
n_y = 150

mu_x = np.zeros(p)
Sigma = wishart.rvs(10, scale=np.eye(p))


def TestStatistic(X,Y,Sigma):
    return (n_x*n_y)/(n_x+n_y)*np.dot(X.mean(0) - Y.mean(0), np.linalg.inv(Sigma)@(X.mean(0) - Y.mean(0)))

thresh = chi2.ppf(0.95, p)

observed_values = []
NMC = 1000
for ITER in range(NMC):
    X = np.random.multivariate_normal(mu_x, Sigma, n_x)
    Y = np.random.multivariate_normal(mu_x, Sigma, n_y)
    observed_values.append(TestStatistic(X,Y,Sigma))

fig, ax = plt.subplots(figsize = (10,4))
ax.hist(observed_values, bins = 50, label = 'observed test values in Monte Carlo iterations')
ax.axvline(thresh, c='b', lw=3, label='test threshold')
ax.legend()
ax.set_title('Empirically observed level: {}'.format(sum([1 if val > thresh else 0 for val in observed_values])/NMC))
plt.show()