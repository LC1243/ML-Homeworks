import numpy as np

#observations
X = np.array([[0.7, -0.3], [0.4, 0.5], [-0.2, 0.8], [-0.4, 0.3]])
#targets
Y = np.array([0.8, 0.6, 0.3, 0.3])

# Radial basis function
def radial_basis_function(x, c):
    return np.exp(-0.5 * np.linalg.norm(x - c) ** 2)

# Centers
C = np.array([[0, 0], [1, -1], [-1, 1]])

# Compute the design matrix
Phi = np.array([[radial_basis_function(x, c) for c in C] for x in X])
# Ridge regression closed-form solution with lambda (λ) = 0.1
lambda_val = 0.1
I = np.eye(4)  # 3 centers, so the identity matrix is 3x3

# Add a column of 1s for the bias term
Phi_with_bias = np.column_stack((np.ones(Phi.shape[0]), Phi))
print("Transformed Values: \n", Phi_with_bias)

x_time_x = np.dot(Phi_with_bias.T, Phi_with_bias)
print("X * X transpose: \n", x_time_x)

plus_lambda = x_time_x + lambda_val * I
print("X*X^T + lambda * I \n", np.round(plus_lambda, 5))

inverted_matrix = np.linalg.inv(plus_lambda)
print("Inverted matrix \n", np.round(inverted_matrix, 5))

#Inverted Matrix * X^T
dot = np.dot(inverted_matrix, Phi_with_bias.T)
print("Inverted Matrix * X^T: \n", np.round(dot, 5))

#Ridge regressionn coefficients 
print("Ridge regression coefficients: \n", np.round(np.dot(dot, Y), 5))
# Compute Ridge regression coefficients with bias
w_with_bias = np.dot(inverted_matrix, np.dot(Phi_with_bias.T, Y))

print("Ridge regression coefficients (with bias):\n", np.round(w_with_bias, 5))


#####################################Part B#####################################

# Predictions using the learned model
y_pred = np.dot(Phi_with_bias, w_with_bias)

print("Output predictions: \n", np.round(y_pred, 5))

# Compute RMSE (Root Mean Squared Error)
rmse = np.sqrt(np.mean((y_pred - Y) ** 2))

print("RMSE: \n", round(rmse,5) )