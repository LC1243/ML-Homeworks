import numpy as np
from scipy.stats import multivariate_normal

# Given mean vector for y1 and y2
mean_y1y2 = np.array([0.44140, 0.41])  # Example mean values

# Given covariance matrix for y1 and y2
cov_y1y2 = np.array([[0.0491, -0.0211], [-0.0211, 0.0375]])  # Example covariance values

# Create a multivariate normal distribution for y1 and y2
y1y2_distribution = multivariate_normal(mean=mean_y1y2, cov=cov_y1y2)

# Point for which we want to calculate the probability
point = np.array([0.38,0.52 ])  # Example point

# Calculate the probability of the point under the assumed multivariate normal distribution
probability = y1y2_distribution.pdf(point)

print("Probability of point", point, "under the assumed distribution:", probability)
