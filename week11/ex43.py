import numpy as np
import matplotlib.pyplot as plt

def pca(X, n_components = None):
    if n_components is None:
        n_components = X.shape[1]
    X_centered = X - X.mean(axis=0)
    C = np.cov(X.T)
    L, W = np.linalg.eig(C)
    idx = L.argsort()[::-1]   
    L = L[idx]; 
    W = W[:,idx]; W = W[:, :n_components]
    pcs = X_centered.dot(W); pc = W.T
    eigvals = L
    return pcs, pc, eigvals


d = 10 # Dimension
k = 3  # Number of orthogonal vectors

# Generate a random d x k matrix
vectors = np.random.randn(d, k).T
# Apply Gram-Schmidt process to obtain orthogonal vectors
orthogonal_vectors = np.linalg.qr(vectors.T)[0].T

eps = 1e-1 # noise parameter
sigma = 3 # scale parameter

n = 100 # Number of samples
Xs = [] # samples list
for _ in range(n):
    coordinates = np.random.normal(0,sigma,k)
    Xs.append(np.sum([c*v for c,v in zip(coordinates, orthogonal_vectors)], 0 ) + np.random.normal(0,eps, d))
Xs = np.array(Xs)
scores, components, eigvals = pca(Xs)

# variance explained
fig,[axl,axr] = plt.subplots(figsize = (10,3), ncols=2)
fig.suptitle('VARIANCE EXPLAINED')
variance_explained = eigvals/ eigvals.sum()

axl.plot(np.arange(1, d+1), variance_explained, '.-')
axl.set_ylabel('% relative variance explained')
axl.set_xlabel('number of components')
axl.axvline(k, c='r',linestyle  = '--')
axl.set_xticks(np.arange(1, d+1));

axr.plot(np.arange(1, d+1), variance_explained.cumsum(), '.-')
axr.set_ylabel('% total variance explained')
axr.set_xlabel('number of components')
axr.axvline(k, c='r',linestyle  = '--')
axr.set_xticks(np.arange(1, d+1)); 

plt.show()



# conditioning number
fig,ax = plt.subplots(figsize = (10,3))
fig.suptitle('CONDITIONING NUMBER')
truncatedcond_numbers = [eigvals[0]/ eigvals[j] for j in range(d)]
ax.plot(np.arange(1, d+1), truncatedcond_numbers, '.-')
ax.set_ylabel('conditioning number')
ax.set_xlabel('number of components')
ax.axvline(k, c='r',linestyle  = '--')
ax.set_xticks(np.arange(1, d+1))
plt.show()



# AIC

fig,[axl,axr] = plt.subplots(figsize = (10,3), ncols = 2)
fig.suptitle('AIC')

axl.plot(np.arange(1,d+1), eigvals,'.-')
axl.plot(np.arange(1,d+1), np.exp(-2/n)*np.ones(d), label = 'threshold')
axl.set_ylabel('value')
axl.legend()
axl.set_xlabel('number of components')
axl.axvline(k, c='r',linestyle  = '--')
axl.set_xticks(np.arange(1, d+1)); 

def AIC(k,eigvals):
    logL = -n/2*np.sum([np.log(lambda_i) for lambda_i in eigvals[k:] ]) + n*k/2
    return 2*k - 2*logL
aics = np.array([AIC(k,eigvals) for k in np.arange(1,d+1)])
axr.plot(np.arange(1,d+1), aics,'.-')
axr.set_ylabel('AIC')
axr.set_xlabel('number of components')
axr.axvline(k, c='r',linestyle  = '--')
axr.set_xticks(np.arange(1, d+1)); 

plt.show()




