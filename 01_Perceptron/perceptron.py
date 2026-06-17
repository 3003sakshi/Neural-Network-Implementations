import numpy as np
import matplotlib.pyplot as plt
import time

# Step activation function
def step(x):
    return 1 if x >= 0 else 0

# Perceptron training function
def train_perceptron(X, y, title, lr=1.0, max_epochs=20):
    weights = np.zeros(X.shape[1])
    bias = 0
    errors_per_epoch = []
    converged = False

    for epoch in range(max_epochs):
        total_error = 0

        print(f"\nEpoch {epoch + 1}")
        print("x1 x2 | target | predicted | error | weights | bias")

        for i in range(len(X)):
            net = np.dot(X[i], weights) + bias
            pred = step(net)
            error = y[i] - pred

            # Update rule
            weights += lr * error * X[i]
            bias += lr * error

            total_error += abs(error)

            print(f"{X[i][0]}  {X[i][1]}   |   {y[i]}    |     {pred}     |   {error}   | {weights} | {bias}")

        errors_per_epoch.append(total_error)

        # Plot decision boundary
        plt.clf()
        for i in range(len(X)):
            if y[i] == step(np.dot(X[i], weights) + bias):
                plt.scatter(X[i][0], X[i][1], c='green', marker='o')
            else:
                plt.scatter(X[i][0], X[i][1], c='red', marker='x')

        x_vals = np.array([0, 1.5])
        if weights[1] != 0:
            y_vals = -(weights[0] * x_vals + bias) / weights[1]
            plt.plot(x_vals, y_vals)

        plt.title(f"{title} - Epoch {epoch + 1}")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.grid(True)
        plt.pause(2)

        if total_error == 0:
            converged = True
            break

    plt.close()

    print("\n----------------------------------")
    print(f"Training completed for {title}")
    if converged:
        print(f"Status: CONVERGED in {epoch + 1} epochs")
    else:
        print("Status: FAILED to converge (Not linearly separable)")
    print("----------------------------------")

    # Error vs Epoch plot
    plt.figure()
    plt.plot(errors_per_epoch, marker='o')
    plt.title(f"Error vs Epochs ({title})")
    plt.xlabel("Epoch")
    plt.ylabel("Total Error")
    plt.grid(True)
    plt.show()


# ================= DATASETS ================= #

# NAND gate
X_nand = np.array([[0,0],[0,1],[1,0],[1,1]])
y_nand = np.array([1,1,1,0])

# XOR gate
X_xor = np.array([[0,0],[0,1],[1,0],[1,1]])
y_xor = np.array([0,1,1,0])

# ================= RUN ================= #

train_perceptron(X_nand, y_nand, "NAND Gate")
train_perceptron(X_xor, y_xor, "XOR Gate")
