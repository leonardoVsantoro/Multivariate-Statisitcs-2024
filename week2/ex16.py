import numpy as np
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal
from scipy.stats import shapiro

rho = .5
cov = np.array([[1,rho], [rho, 1]])
mean = np.zeros(2)
N_samples = 1000
samples = multivariate_normal( mean, cov,N_samples)

# 3 different projections
v1 = np.array([1,0]) # x-axis
v2 = np.array([0,1]) # y-axis
v3 = np.array([1,1]); v3 = v3/np.linalg.norm(v3)
vs = [v1,v2,v3]



# scatterplot of samples
fig,ax = plt.subplots(figsize = (5,5))
ax.scatter(samples[:,0], samples[:,1]); 
fig.suptitle('Gaussian random samples')
ax.set_title('$\\rho$ = {}'.format(rho))

# 3 histograms of different projections
fig, axs = plt.subplots(figsize = (16,5), ncols = 3)
fig.suptitle('Histograms of projected data')
for i, (v, ax) in enumerate(zip(vs,axs)):
    projected_samples = np.array([ np.dot(x, v) for x in samples])
    ax.hist(projected_samples)
    ax.set_title('$v_{}$ = ({}, {})'.format(i+1, v[0].round(2), v[1].round(2)))
    ax.set_xlabel('Shapiro-Wilk test p-value: {:.2f}'.format(shapiro(projected_samples).pvalue))
    
plt.show()

