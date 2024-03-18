import numpy as np
from numpy.random import multivariate_normal
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

ds = [5, 20, 50, 100, 500, 1000]

NMC = 100
Nsamples = 1000


norm_of_mean = {d : [] for d in ds}
for _ in tqdm(np.arange(NMC)):
    sample = multivariate_normal(np.zeros(ds[-1]), np.eye(ds[-1]), Nsamples)
    for d in ds:
        norm_of_mean[d].append( np.mean( [np.linalg.norm( value[:d]) for value in sample]) / d**.5 )

fig, ax = plt.subplots(figsize = (18,4))    
for d in ds:
    sns.kdeplot(data = norm_of_mean[d], fill=True, label = '{}'.format(d), ax=ax)
plt.legend(title='d')
# ax.set_yscale('log')
plt.show()