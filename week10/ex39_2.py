import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
def pca(X, n_components):
    X_centered = X - X.mean(axis=0)
    C = np.cov(X.T)
    L, W = np.linalg.eig(C)
    idx = L.argsort()[::-1]   
    L = L[idx]; 
    W = W[:,idx]; W = W[:, :n_components]
    pcs = X_centered.dot(W); pc = W.T
    return pcs, pc

X, y = load_digits(return_X_y=True)
(n_samples, n_features), n_digits = X.shape, np.unique(y).size
print(f"# digits: {n_digits}; # samples: {n_samples}; # features {n_features}")

n_components = n_features
pcs, pc = pca(X, n_components)
fig,ax = plt.subplots(figsize = (6,2))
ax.set_title('% Explained Variance')
ax.set_xlabel('number of components')
ax.plot(100*pcs.std(0).cumsum()/pcs.std(0).sum()); plt.show()
fig,ax = plt.subplots(figsize = (18,2)); ax.set_title('Effect of 1st" principal component')
ax.plot(X.mean(0))
lam = pcs.std(0)[0]
ax.fill_between(np.arange(X.shape[1]), X.mean(0)- lam*pc[0], X.mean(0)+ lam*pc[0],alpha = .4)
plt.show()

fig,ax = plt.subplots(figsize = (18,2)); ax.set_title('Effect of 2nd principal component')
ax.plot(X.mean(0))
lam = pcs.std(0)[1]
ax.fill_between(np.arange(X.shape[1]), X.mean(0)- lam*pc[1], X.mean(0)+ lam*pc[1],alpha = .4)
plt.show()
fig,ax = plt.subplots(figsize = (6,6))
ax.set_title('First Two Principal Components')
for i in np.unique(y):

    ax.scatter(pcs[np.where(y==i),0],pcs[np.where(y==i),1],  alpha =.5, label = '{}'.format(i))
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
plt.legend(); plt.show()

from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import train_test_split

n_splits = 100
test_size = .1

for _ in range(n_splits):
    X_train, X_test, y_train, y_test = train_test_split(  X, y, test_size=test_size)
    clf = NearestCentroid(); clf.fit(X_train,y_train)
    validated_errors.append(( (np.sum(clf.predict(X_test)==y_test))/y_test.size).round(3))

fig,ax = plt.subplots(figsize = (6,2))
sns.boxplot(x = validated_errors)
fig.suptitle('Cross-validated error',y=1.1)
ax.set_title('{} random {}/{} train/test splits'.format(n_splits,1-test_size,test_size ))
plt.show()