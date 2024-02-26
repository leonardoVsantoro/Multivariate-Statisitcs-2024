import numpy as np
import matplotlib.pyplot as plt

def power_iteration(A, num_iterations=1000, return_all = False):
    # Generate a random initial guess for the eigenvector
    n = A.shape[0]
    v = np.random.rand(n)
    eigvals = []; eigvecs = []
    # Perform power iteration
    for i in range(num_iterations):
        Av = np.dot(A, v)
        v_new = Av / np.linalg.norm(Av)  # Normalize the vector
        if return_all:
            eigvals.append( np.dot(v_new, np.dot(A, v_new)))
            eigvecs.append(v_new)
        if np.allclose(v, v_new):  # Check for convergence
            break
        v = v_new
    # Compute the corresponding eigenvalue
    eigenvalue = np.dot(v_new, np.dot(A, v_new))
    if return_all:
        return np.array(eigvecs), np.array(eigvals), i+1
    else:
        return v_new, eigenvalue, i+1
    


# ---------------------------------------------------- PART 1 ---------------------------------------------------- #
# verify rate of convergence of power iteration

# Generate a random matrix
d = 5  # Size of the matrix
A = np.random.rand(d, d)

# get the true eigendecomposition using numpy, and extract dominant ones
eigenvalues, eigenvectors = np.linalg.eig(A)
ixargsort = np.argsort(np.abs(eigenvalues))
dominant_eigenvalue = eigenvalues[ixargsort][-1]
dominant_eigenvector = eigenvectors[:, ixargsort][:, -1]
# get the ratio of the first two leading eigenvalues -- this determines the speed of convergence of the algorithm!
ratio = np.abs(eigenvalues[ixargsort][-2]/eigenvalues[ixargsort][-1])


# Run power iteration to find the dominant eigenvector and eigenvalue
num_iterations=1000
estimated_dominant_eigenvector, estimated_dominant_eigenvalue,stop_iter = power_iteration(A)
print('Stoped at {} iterations'.format(stop_iter))
print("Dominant Eigenvector:", estimated_dominant_eigenvector.round(3))
print("Dominant Eigenvalue:", estimated_dominant_eigenvalue.round(3))

seq_estimated_dominant_eigenvector, seq_estimated_dominant_eigenvalue, stop_iter = power_iteration(A, return_all = True)

fig, [axl,axr] = plt.subplots(figsize = (16,5), ncols=2)

eigvec_errors = [np.linalg.norm( v*np.sign(np.dot(dominant_eigenvector,v)) -dominant_eigenvector) for v in seq_estimated_dominant_eigenvector]
eigval_errors = [ np.linalg.norm(lam -dominant_eigenvalue) for lam in seq_estimated_dominant_eigenvalue]
axl.plot(np.arange(stop_iter),  eigvec_errors)
axr.plot(np.arange(stop_iter),  eigval_errors)

axl.plot(np.arange(stop_iter), [eigvec_errors[0]*ratio**k for k in np.arange(stop_iter)], alpha=.2, lw=10, c='k', label = 'theoretical speed of convergence: $O(k^{\lambda_2/\lambda_1})$')
axr.plot(np.arange(stop_iter), [eigval_errors[0]*ratio**k for k in np.arange(stop_iter)], alpha=.2, lw=10, c='k',  label = 'theoretical speed of convergence: $O(k^{\lambda_2/\lambda_1})$')

fig.suptitle('Power Iteration Method on {}$\\times${} random matrix'.format(d,d))
axl.legend()
axr.legend()
axl.set_title('Dominant eigenvector estimation error')
axr.set_title('Dominant eigenvalue estimation error')
axl.set_xlabel('iteration')
axr.set_xlabel('iteration')
axr.set_ylabel('error')
axl.set_ylabel('error')
axr.set_yscale('log')
axl.set_yscale('log')

plt.show()




# ---------------------------------------------------- PART 2 ---------------------------------------------------- #
# check behaviour with different dimensions

dimensions = [2,4,8,16,32,64,128,256]

fig, [axl, axr] = plt.subplots(figsize=(16, 5), ncols=2)
fig.suptitle('Power Iteration Method on Brownian Motion covariance - behaviour with dimension')

for d in dimensions:
    grid = np.linspace(0, 1, d)
    A = np.array([[min(s, t) for t in grid] for s in grid])

    seq_estimated_dominant_eigenvector, seq_estimated_dominant_eigenvalue, stop_iter = power_iteration(A, return_all=True)
    eigenvalues, eigenvectors = np.linalg.eig(A)
    ixargsort = np.argsort(np.abs(eigenvalues))
    dominant_eigenvalue = eigenvalues[ixargsort][-1]
    dominant_eigenvector = eigenvectors[:, ixargsort][:, -1]
    axl.plot(np.arange(stop_iter), [np.linalg.norm(v * np.sign(np.dot(dominant_eigenvector, v)) - dominant_eigenvector) for v in seq_estimated_dominant_eigenvector], label=str(d))
    axr.plot(np.arange(stop_iter), [np.linalg.norm(lam - dominant_eigenvalue) for lam in seq_estimated_dominant_eigenvalue], label=str(d))

axl.legend(title='dimension')
axr.legend(title='dimension')
axl.set_title('Dominant eigenvector estimation error')
axr.set_title('Dominant eigenvalue estimation error')
axl.set_xlabel('iteration')
axr.set_xlabel('iteration')
axr.set_ylabel('error')
axl.set_ylabel('error')
axr.set_yscale('log')
axl.set_yscale('log')

plt.show()
