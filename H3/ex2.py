import numpy as np

# Given weights and biases
W1 = np.array([[1, 1, 1, 1], [1, 1, 2, 1], [1, 1, 1, 1]], dtype=np.float64)
b1 = np.array([1, 1, 1], dtype=np.float64)
W2 = np.array([[1, 4, 1], [1, 1, 1]], dtype=np.float64)
b2 = np.array([1, 1], dtype=np.float64)
W3 = np.array([[1, 1], [3, 1], [1, 1]], dtype=np.float64)
b3 = np.array([1, 1, 1], dtype=np.float64)

# Activation function
def activation(x):
    return np.tanh(0.5 * x - 2)

# Learning rate
eta = 0.1

# Training observations and targets
x1 = np.array([1, 1, 1, 1])
x2 = np.array([1, 0, 0, -1])
targets = []
target1 = np.array([0, 1, 0])  # Target for class B
target2 = np.array([1, 0, 0])  # Target for class A
targets.append(target1)
targets.append(target2)

observations = []
observations.append(x1)
observations.append(x2)
w1_lst = []
w2_lst = []
w3_lst = []
b1_lst = []
b2_lst = []
b3_lst = []


for i in range(2):
    # Forward pass
    print(np.dot(W1, observations[i]))
    print("Z[1] = ",np.dot(W1, observations[i]) + b1)
    z1 = activation(np.dot(W1, observations[i]) + b1)
    print("X[1] = ", np.round(z1, 5))
    print("\n")
    print(np.dot(W2, z1))
    print("Z[2] = ",np.dot(W2, z1) + b2)
    z2 = activation(np.dot(W2, z1) + b2)
    print("X[2] = ", np.round(z2, 5))
    print("\n")
    print(np.dot(W3, z2))
    print("Z[3] = ",np.dot(W3, z2) + b3)
    z3 = activation(np.dot(W3, z2) + b3)
    print("X[3] = ", np.round(z3, 5))

    delta3_part1 = np.subtract(z3, targets[i])
    print("Delta 3, parte 1 = \n", np.round(delta3_part1, 5))
    delta3_part2 = (0.5 - 0.5 * z3 ** 2)
    print("Delta 3, parte 2 = \n", np.round(delta3_part2, 5))
    delta3 = delta3_part1 * delta3_part2
    print("Delta 3 =\n", np.round(delta3, 5))

    delta2_parte1 = np.dot(W3.T, delta3)
    print("Delta 2, parte 1 = \n", np.round(delta2_parte1, 5))

    delta2_part2 = (0.5 - 0.5 * z2 ** 2)
    print("Delta 2, parte 2 = \n", np.round(delta2_part2, 5))

    delta2 = delta2_parte1 * delta2_part2
    print("Delta 2 =\n", delta2)

    delta1_parte1 = np.dot(W2.T, delta2)
    print("Delta 1, parte 1 = \n", np.round(delta1_parte1, 5))

    delta1_part2 = (0.5 - 0.5 * z1 ** 2)
    print("Delta 1, parte 2 = \n", np.round(delta1_part2, 5))

    delta1 = delta1_parte1 * delta1_part2
    print("Delta 1 =\n", delta1)

    delta_w1 = -0.1 * np.dot(delta1[:, np.newaxis], observations[i][np.newaxis, :])
    print("Delta W1: \n", delta_w1)
    w1_lst.append(delta_w1)

    delta_b1 = -0.1 * delta1
    print("Delta b1: \n", delta_b1)
    b1_lst.append(delta_b1)

    delta_w2 = -0.1 * np.dot(delta2[:, np.newaxis], z1[np.newaxis, :])
    print("Delta W2: \n", delta_w2)
    w2_lst.append(delta_w2)

    delta_b2 = -0.1 * delta2
    print("Delta b2: \n", delta_b2)
    b2_lst.append(delta_b2)

    delta_w3 = -0.1 * np.dot(delta3[:, np.newaxis], z2[np.newaxis, :])
    print("Delta W3: \n", delta_w3)
    w3_lst.append(delta_w3)

    delta_b3 = -0.1 * delta3
    print("Delta b3: \n", delta_b3)
    b3_lst.append(delta_b3)


# Gradient descent update
W1 += np.sum(w1_lst, axis=0, dtype=np.float64)
b1 += np.sum(b1_lst, axis=0, dtype=np.float64)
W2 += np.sum(w2_lst, axis=0, dtype=np.float64)
b2 += np.sum(b2_lst, axis=0, dtype=np.float64)
W3 += np.sum(w3_lst, axis=0, dtype=np.float64)
b3 += np.sum(b3_lst, axis=0, dtype=np.float64)

# Print updated weights and biases
print("Updated W1:\n", np.round(W1, 5))
print("Updated b1:\n", np.round(b1, 5))
print("Updated W2:\n", np.round(W2, 5))
print("Updated b2:\n", np.round(b2, 5))
print("Updated W3:\n", np.round(W3, 5))
print("Updated b3:\n", np.round(b3, 5))
