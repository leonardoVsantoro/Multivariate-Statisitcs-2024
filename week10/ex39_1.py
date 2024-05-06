import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler

def pca(X, n_components):
    X_centered = X - X.mean(axis=0)
    C = np.cov(X.T)
    L, W = np.linalg.eig(C)
    idx = L.argsort()[::-1]   
    L = L[idx]; 
    W = W[:,idx]; W = W[:, :n_components]
    pcs = X_centered.dot(W); pc = W.T
    return pcs, pc

data = sklearn.datasets.load_breast_cancer()
# X = StandardScaler().fit_transform(data.data) 
X = data.data; X = X/X.std(0)
y = data.target
idx_0 = np.where(y==0); idx_1 = np.where(y==1)

n_components = 20
pcs, pc = pca(X, n_components)
fig,ax = plt.subplots(figsize = (18,2)); ax.set_title('Effect of 1st" principal component')
ax.plot(X.mean(0))
lam = pcs.std(0)[0]
ax.fill_between(np.arange(X.shape[1]), X.mean(0)- lam*pc[0], X.mean(0)+ lam*pc[0],alpha = .4)
ax.set_xticks(np.arange(data.feature_names.size))
ax.set_xticklabels(data.feature_names, rotation = 45 )
plt.show()

fig,ax = plt.subplots(figsize = (18,2)); ax.set_title('Effect of 2nd principal component')
ax.plot(X.mean(0))
lam = pcs.std(0)[1]
ax.fill_between(np.arange(X.shape[1]), X.mean(0)- lam*pc[1], X.mean(0)+ lam*pc[1],alpha = .4)
ax.set_xticks(np.arange(data.feature_names.size))
ax.set_xticklabels(data.feature_names, rotation = 45 )
plt.show()
fig,ax = plt.subplots(figsize = (6,2))
ax.set_title('% Explained Variance')
ax.set_xticks(np.arange(n_components)); ax.set_xticklabels(1+np.arange(n_components))
ax.set_xlabel('number of components')
ax.plot(100*pcs.std(0).cumsum()/pcs.std(0).sum()); plt.show()


from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import train_test_split
for _ in range(10):
    X_train, X_test, y_train, y_test = train_test_split(  X, y, test_size=0.1)
    clf = NearestCentroid(); clf.fit(X_train,y_train)
    print((1-(clf.predict(X_test)  - y_test)**2).mean().round(3))