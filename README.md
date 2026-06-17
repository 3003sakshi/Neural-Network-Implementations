# Experiment 1 — Visualization of Perceptron Learning Algorithm


## Objective

To visualize the Perceptron Learning Algorithm using NumPy and Matplotlib and evaluate its performance on NAND and XOR truth table datasets.

---

## Description of the Model

A perceptron is a single-layer neural network used for binary classification. It computes a weighted sum of inputs and passes it through a step activation function to produce the output. During training, the perceptron updates its weights and bias using the perceptron learning rule based on the classification error. The model can correctly learn only linearly separable datasets.

---

## Python Implementation

```python
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
```

---

## Code Description

1. Required libraries (NumPy and Matplotlib) are imported.
2. A step activation function is defined to convert the net input into binary output.
3. The `train_perceptron` function initializes weights and bias to zero.
4. For each epoch, the algorithm calculates predicted output and error.
5. Weights and bias are updated using the perceptron learning rule.
6. Decision boundary and classification results are visualized using Matplotlib.
7. Error vs Epoch graph is plotted to observe the learning progress.

---

## Performance Evaluation

| Dataset | Result |
|---------|--------|
| **NAND Gate** | ✅ Converges — linearly separable; error reduces to zero after a few epochs |
| **XOR Gate** | ❌ Fails to converge — not linearly separable; error persists |

---

## Limitations & Scope of Improvement

- A single-layer perceptron **cannot** solve non-linearly separable problems (e.g., XOR).
- This limitation can be overcome using Multi-Layer Perceptrons (MLP) with hidden layers and nonlinear activation functions.
- Future work: implement MLP and compare performance with this single-layer model.

---

## Libraries Used

| Library | Purpose |
|---------|---------|
| `numpy` | Numerical computations |
| `matplotlib` | Visualization of decision boundary and error graphs |
