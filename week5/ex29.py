import numpy as np
from scipy.stats import f
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal
from scipy.linalg import sqrtm
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon


def boundary_sphere(r, n = 1000):
    theta = np.linspace(0, 2*np.pi, n)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack((x, y))

d = 2 
n = 100
mean = np.array([1,1])
cov = np.array([[1,+1],[+1, 2]])

X = multivariate_normal(mean, cov, n)
sample_mean = X.mean(0)
sample_cov = np.cov(X.T)


fig,ax = plt.subplots(figsize = (7,7))

alphas = [.01, 0.05, 0.1, .25]
cmap = mcolors.LinearSegmentedColormap.from_list("", ["white", "blue"])
shades = [mcolors.to_hex(cmap(i)) for i in np.linspace(.2, 1, len(alphas))]

for alpha, c in zip(alphas, shades):
    T2_alpha = d*(n-1)/(n-d)*f.ppf(1 - alpha, d, n-d)
    ball = boundary_sphere(T2_alpha)
    ellip = np.array([ sample_mean - n**(-.5)*sqrtm(sample_cov)@v for v in ball])
    # ax.plot(ellip[:,0], ellip[:,1], alpha=1, c = c, label = '{}% confidence ellipsoide'.format(int((1-alpha)*100)))
    polygon = Polygon(ellip, closed=True, edgecolor=None, facecolor=c, alpha=0.3 , label = '{}% confidence ellipsoide'.format(int((1-alpha)*100)))
    ax.add_patch(polygon)
ax.scatter(mean[0], mean[1], c = 'r', label = '$\mu$')
ax.scatter(sample_mean[0], sample_mean[1], c = 'g', marker = 'x', label = '$\overline{X}$')
ax.set_aspect('equal')
plt.legend(bbox_to_anchor=(1.1, 1.05))
plt.show()