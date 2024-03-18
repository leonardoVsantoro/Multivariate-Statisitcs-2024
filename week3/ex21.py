import numpy as np
from numpy.random import multivariate_normal
import matplotlib.pyplot as plt

# ---- 

d = 4
N_samples = 100

mean_X =  np.random.rand(d)
cov_X = np.array([[min(i+1,j+1) for i in range(d)] for j in range(d)])
Xs = multivariate_normal( mean_X, cov_X, N_samples)

A = np.random.rand(d,d)
b = np.random.rand(d)

eps = 1e-1

Ys = np.array([ A@X + b +  multivariate_normal(np.zeros(d), eps*np.eye(d)) for X in Xs])
Z = np.concatenate((Ys, Xs), axis = 1)

covariance_Z = np.cov(Z.T)
precision_Z = np.linalg.inv(covariance_Z )

# ----

fig, [axl, axr] = plt.subplots(figsize=(8, 4), ncols=2)

im1 = axl.imshow(covariance_Z)
axl.set_title('$\Sigma_Z$')
cbar1 = fig.colorbar(im1, ax=axl, fraction=0.046, pad=0.04)

im2 = axr.imshow(precision_Z, cmap='plasma')
axr.set_title('$\Sigma_Z^{-1}$')
cbar2 = fig.colorbar(im2, ax=axr, fraction=0.046, pad=0.04)

for ax in [axl,axr]:
    ax.set_xticks(np.arange(d+d)); ax.set_yticks(np.arange(d+d))
    ax.set_xticklabels(['$Y_{}$'.format(i+1) for i in range(d)] + ['$X_{}$'.format(i+1) for i in range(d)])
    ax.set_yticklabels(['$Y_{}$'.format(i+1) for i in range(d)] + ['$X_{}$'.format(i+1) for i in range(d)])
plt.tight_layout(); plt.show()

# ----
 
newXs =  multivariate_normal( mean_X, cov_X, N_samples)
newYs =  np.array([ A@X + b +  multivariate_normal(np.zeros(d), eps*np.eye(d)) for X in newXs])

BLPslope = covariance_Z[d:,:d].T@np.linalg.inv(covariance_Z[d:,d:])
BLPintercept = Ys.mean(0)  - covariance_Z[d:,:d].T@np.linalg.inv(covariance_Z[d:,d:])@Xs.mean(0)
BLP_YgivenX = lambda X : BLPintercept + BLPslope@X
predicted_newYs = np.array([BLP_YgivenX(X) for X in newXs])

print('Estimation on {} samples'.format(N_samples))
print('Validation on {} samples'.format(N_samples))
print('')
print('validated prediction error: (avrg squared norm):      {:.3f}'.format(np.mean([np.linalg.norm(py-y)**2 for py,y in zip(predicted_newYs, newYs)])))
print('slope difference (squared Frobenius norm):   {:.3f}'.format(np.linalg.norm( A -  BLPslope)**2))
print('intercept difference (squared norm):         {:.3f}'.format(np.linalg.norm( b -  BLPintercept)**2))

# ---- 

fig, axs = plt.subplots(figsize = (16,4), ncols = d); axs=axs.ravel()
mark = True
for X, pY, Y in zip(newXs, predicted_newYs, newYs):
    if mark:
        for ax, x, py, y in zip(axs, X, pY, Y):
            ax.scatter(x,y,marker ='o', color='b',alpha = .5, label ='Truth')
            ax.scatter(x,py, marker = 'x', color='r', label ='Best Linear Prediction')
        mark = False
    for ax, x, py, y in zip(axs, X, pY, Y):
        ax.scatter(x,y,marker ='o', color='b',alpha = .5)
        ax.scatter(x,py, marker = 'x', color='r')
for i, ax in enumerate(axs):
    ax.set_title('Dimension {}'.format(i+1))
    ax.legend()
plt.show()


