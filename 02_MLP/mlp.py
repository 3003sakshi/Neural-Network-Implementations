import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -------------------------------------------------
# 1. XOR DATA
# -------------------------------------------------
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

# -------------------------------------------------
# 2. ACTIVATION
# -------------------------------------------------
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)

# -------------------------------------------------
# 3. INITIALIZATION
# -------------------------------------------------
np.random.seed(42)

input_neurons = 2
hidden_neurons = 4
output_neurons = 1

W1 = np.random.randn(input_neurons, hidden_neurons)
b1 = np.zeros((1, hidden_neurons))

W2 = np.random.randn(hidden_neurons, output_neurons)
b2 = np.zeros((1, output_neurons))

learning_rate = 0.1
epochs = 10000

loss_history = []

# -------------------------------------------------
# 4. TRAINING (BACKPROP)
# -------------------------------------------------
for epoch in range(epochs):

    # Forward
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, W2) + b2
    predicted_output = sigmoid(final_input)

    # Loss (MSE)
    loss = np.mean((y - predicted_output)**2)
    loss_history.append(loss)

    # Backprop
    error = y - predicted_output
    d_output = error * sigmoid_derivative(predicted_output)

    error_hidden = np.dot(d_output, W2.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    # Update
    W2 += np.dot(hidden_output.T, d_output) * learning_rate
    b2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate

    W1 += np.dot(X.T, d_hidden) * learning_rate
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

print("Final Predictions:\n", predicted_output)

# -------------------------------------------------
# 5. LOSS vs EPOCH GRAPH
# -------------------------------------------------
plt.figure()
plt.plot(loss_history)
plt.title("Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.show()

# -------------------------------------------------
# 6. DECISION BOUNDARY
# -------------------------------------------------
x_min, x_max = -0.5, 1.5
y_min, y_max = -0.5, 1.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

grid = np.c_[xx.ravel(), yy.ravel()]

hidden_layer = sigmoid(np.dot(grid, W1) + b1)
output_layer = sigmoid(np.dot(hidden_layer, W2) + b2)

Z = output_layer.reshape(xx.shape)

plt.figure()
plt.contourf(xx, yy, Z, levels=50, cmap="coolwarm", alpha=0.6)

for i in range(len(X)):
    if y[i] == 0:
        plt.scatter(X[i,0], X[i,1])
    else:
        plt.scatter(X[i,0], X[i,1])

plt.title("MLP Decision Boundary for XOR")
plt.xlabel("Input 1")
plt.ylabel("Input 2")
plt.show()

# -------------------------------------------------
# 7. 3D LOSS SURFACE (2 Selected Weights)
# -------------------------------------------------
w_vals = np.linspace(-5,5,50)
b_vals = np.linspace(-5,5,50)

W_surf, B_surf = np.meshgrid(w_vals, b_vals)
Z_surf = np.zeros_like(W_surf)

# Use only one weight and one bias from output layer for visualization
for i in range(W_surf.shape[0]):
    for j in range(W_surf.shape[1]):

        temp_W2 = W2.copy()
        temp_b2 = b2.copy()

        temp_W2[0,0] = W_surf[i,j]
        temp_b2[0,0] = B_surf[i,j]

        hidden_layer = sigmoid(np.dot(X, W1) + b1)
        output_layer = sigmoid(np.dot(hidden_layer, temp_W2) + temp_b2)

        Z_surf[i,j] = np.mean((y - output_layer)**2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(W_surf, B_surf, Z_surf, cmap="coolwarm", alpha=0.8)

ax.set_xlabel("Selected W2[0]")
ax.set_ylabel("Selected b2")
ax.set_zlabel("Loss")
ax.set_title("3D Loss Surface")

plt.show()
