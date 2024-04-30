import numpy as np
from scipy.stats import norm, wishart, chi2
import matplotlib.pyplot as plt

p = 20

mu = np.zeros(p)
Sigma = wishart.rvs(p+2, scale=np.eye(p))

n=100
X = np.random.multivariate_normal(mu, Sigma, n)

empCov = np.cov(X.T)

def get_sorted_eigdecomp(cov):
    eigvals, eigvecs = np.linalg.eig(cov)
    ix_argsort = np.argsort(np.abs(eigvals))[::-1]
    eigvals, eigvecs = eigvals[ix_argsort], eigvecs.T[ix_argsort]
    return eigvals, eigvecs

eigvals, eigvecs = get_sorted_eigdecomp(empCov)
KL_coordinates = np.array([ eigvecs@x for x in X])
plt.plot(KL_coordinates.std(0)/KL_coordinates.std(0).sum())
# plt.plot(KL_coordinates.std(0).cumsum()/KL_coordinates.std(0).sum())

# KL decomp: the vairance of the components corresponds in fact precisely to the square root of the eigenvalues, oredered decreasingly
# PCA: the truuncated decomposition is optimal uniformly! So clearly the percentage of variance explained need be non-increasing