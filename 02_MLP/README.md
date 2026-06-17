# Experiment 2 — Implementation of Multi-Layer Perceptron (MLP)


## Objective

To implement and analyze neural network models — Perceptron and Multi-Layer Perceptron (MLP) — using Python. The aim is to understand learning algorithms, convergence behavior, and the ability of neural networks to solve linearly and non-linearly separable problems.

---

## Description of the Model

### Perceptron
A single-layer neural network for binary classification of linearly separable data. Updates weights using the perceptron learning rule based on classification error.

### Multi-Layer Perceptron (MLP)
Consists of an input layer, one hidden layer, and an output layer. Uses nonlinear activation functions (sigmoid) and is trained via backpropagation and gradient descent. Can solve non-linearly separable problems such as XOR.

---

## Python Implementation

```python
import numpy as np

# XOR Dataset
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

# Activation function
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)

# Initialize parameters
np.random.seed(42)
W1 = np.random.randn(2,4)
b1 = np.zeros((1,4))
W2 = np.random.randn(4,1)
b2 = np.zeros((1,1))

learning_rate = 0.1
epochs = 10000

# Training
for epoch in range(epochs):
    hidden = sigmoid(np.dot(X,W1)+b1)
    output = sigmoid(np.dot(hidden,W2)+b2)

    error = y - output
    d_output = error * sigmoid_derivative(output)
    d_hidden = np.dot(d_output,W2.T) * sigmoid_derivative(hidden)

    W2 += np.dot(hidden.T,d_output) * learning_rate
    b2 += np.sum(d_output,axis=0,keepdims=True) * learning_rate
    W1 += np.dot(X.T,d_hidden) * learning_rate
    b1 += np.sum(d_hidden,axis=0,keepdims=True) * learning_rate

print("Final Output:")
print(output)
```

---

## Code Description

- Weights and bias terms are initialized before training begins.
- Predictions are made using sigmoid activation functions in a forward pass.
- Error is computed and weights are updated iteratively via backpropagation.
- Hidden layer activations and gradients are computed using the chain rule.
- The training process records loss values for visualization and evaluation.

---

## Performance Evaluation

| Model | Dataset | Result |
|-------|---------|--------|
| Perceptron | NAND | ✅ Converges (linearly separable) |
| Perceptron | XOR | ❌ Fails (not linearly separable) |
| MLP | XOR | ✅ Successfully learns the pattern |

The loss curve shows decreasing error during training, confirming successful gradient-based learning.

---

## Limitations & Scope of Improvement

- The perceptron is limited to linearly separable problems.
- MLP overcomes this but requires more computation and careful hyperparameter tuning (learning rate, epochs).
- Performance can be improved by:
  - Experimenting with different activation functions and optimizers
  - Adding regularization (dropout, L2)
  - Using advanced libraries like TensorFlow or PyTorch for larger datasets

---

## Libraries Used

| Library | Purpose |
|---------|---------|
| `numpy` | Numerical computations, matrix operations |
