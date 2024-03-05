library(MASS)  # For generating multivariate normal samples
library(ggplot2)
library(gridExtra)
library(nortest)

# Parameters
rho <- 0.5
cov <- matrix(c(1, rho, rho, 1), nrow = 2)
mean <- c(0, 0)
N_samples <- 1000

# Generate samples
set.seed(123)  # for reproducibility
samples <- mvrnorm(n = N_samples, mu = mean, Sigma = cov)

# Define projection vectors
v1 <- c(1, 0)  # x-axis
v2 <- c(0, 1)  # y-axis
v3 <- c(1, 1) / sqrt(2)  # 45-degree line
vs <- list(v1, v2, v3)

# Scatterplot of samples
p1 <- ggplot(data = as.data.frame(samples), aes(x = samples[,1], y = samples[,2])) +
  geom_point() +
  ggtitle('Gaussian random samples') +
  theme(plot.title = element_text(hjust = 0.5))

# 3 histograms of different projections
histograms <- lapply(1:length(vs), function(i) {
  projected_samples <- as.vector(samples %*% vs[[i]])
  p <- ggplot(data = as.data.frame(projected_samples), aes(x = projected_samples)) +
    geom_histogram(binwidth = 0.5, fill = "skyblue", color = "black") +
    ggtitle(sprintf("v%d = (%.2f, %.2f)", i, vs[[i]][1], vs[[i]][2])) +
    xlab(sprintf("Shapiro-Wilk test p-value: %.2f", shapiro.test(projected_samples)$p.value)) +
    theme(plot.title = element_text(hjust = 0.5))
})

# Arrange plots
plots <- grid.arrange(p1, do.call(grid.arrange, histograms), ncol=2, widths = c(2, 1))

# Show plots
print(plots)
