generate_symmetric_matrix <- function(n) {
  A <- matrix(runif(n^2), nrow = n)
  A <- (A + t(A))/2  # Ensuring symmetry
  return(A)
}

# Power iteration algorithm
power_iteration <- function(A, num_iters) {
  n <- nrow(A)
  eigenvalues <- numeric(num_iters)
  eigenvectors <- matrix(0, nrow = n, ncol = num_iters)
  

  b <- runif(n)
  for (ii in 1:num_iters) {
    bnew <- A %*% b
    b <- bnew / sqrt(sum(bnew^2))
    eigval <-  t(b) %*% A %*% b
    eigenvalues[ii] <- eigval
    eigenvectors[,ii] <- b
  }
  return(list(eigenvalues = eigenvalues, eigenvectors = eigenvectors))
}

# Set parameters
n <- 5  # Size of the matrix
num_iters <- 10  # Number of iterations

# Generate a random symmetric matrix
set.seed(42)  # for reproducibility
A <- generate_symmetric_matrix(n)

generate_symmetric_matrix <- function(n) {
  A <- matrix(runif(n^2), nrow = n)
  A <- (A + t(A))/2  # Ensuring symmetry
  return(A)
}

# Run power iteration
result <- power_iteration(A, num_iters)

# Get true eigendecomposition
true_eigen <- eigen(A)
true_lead_eigval <- true_eigen$values[1]
true_lead_eigvec <- true_eigen$vectors[,1]
ratio <- true_eigen$values[2]/true_eigen$values[1]

eigvec_errs <-   numeric(num_iters)
eigval_errs <-   numeric(num_iters)
for (ii in 1:num_iters) {
  eigvec_errs[ii] <- min( sum((result$eigenvectors[,ii] - true_lead_eigvec)^2)^(.5), sum((-result$eigenvectors[,ii] - true_lead_eigvec)^2)^(.5))
  eigval_errs[ii] <- min( abs(result$eigenvalues[ii] - true_lead_eigval), abs(result$eigenvalues[ii] + true_lead_eigval) )**.5
}


par(mfrow = c(1, 2))
plot(1:num_iters, eigval_errs, log = "y", type = "l", xlab = "Iteration", ylab = "Error", main = "Dominant eigenvalue estimation error")
lines(1:num_iters, eigval_errs[1] * ratio ^ ((1:num_iters)-1), col = "red", lty = 2)
legend("topright", legend = c("Estimated", "Theoretical"), col = c("black", "red"), lty = c(1, 2))

plot(1:num_iters, eigvec_errs,log = "y", type = "l", xlab = "Iteration", ylab = "Error", main = "Dominant eigenvector estimation error")
lines(1:num_iters, eigvec_errs[1] * ratio ^  ((1:num_iters)-1), col = "red", lty = 2)
legend("topright", legend = c("Estimated", "Theoretical"), col = c("black", "red"), lty = c(1, 2))


