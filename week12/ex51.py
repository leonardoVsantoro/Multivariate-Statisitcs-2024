import numpy as np


rho = 0.9

sigma = 0.5
def noise():
    return np.random.normal(0,sigma)

start_point = np.random.normal(0, sigma/(1-rho))
chain = [start_point]


NSTEPS = 100

for _ in range(NSTEPS-1):
    chain.append(rho*chain[-1] + noise())

fig,ax = plt.subplots(figsize = (15,5))

NMC = 500
for iter in range(NMC):
    mcchain = [chain[-1]]
    for _ in range(5):
        mcchain.append(rho*mcchain[-1] + noise())
    ax.plot(np.arange(NSTEPS-1, NSTEPS+5), mcchain, '.-', c='r', alpha = .1)

ax.plot(np.arange(NSTEPS), chain, '.-')
ax.set_xlabel('time')
ax.set_ylabel('space')
