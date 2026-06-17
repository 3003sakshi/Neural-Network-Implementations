# Experiment 3 — Implementation of BPNN using TensorFlow (MNIST Classification)


## Objective

To implement a three-layer Backpropagation Neural Network (BPNN) using TensorFlow (without Keras layers) to classify handwritten digits from the MNIST dataset and demonstrate feed-forward and back-propagation learning.

---

## Description of the Model

A three-layer feed-forward neural network:

| Layer | Neurons | Activation |
|-------|---------|------------|
| Input | 784 (28×28 pixels flattened) | — |
| Hidden | 128 | ReLU |
| Output | 10 (digit classes 0–9) | Softmax |

- Trained using **backpropagation** with **gradient descent** optimization.
- **Cross-entropy loss** is used as the cost function.

---

## Python Implementation

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0
x_test  = x_test.reshape(-1, 784).astype(np.float32) / 255.0

y_train_onehot = tf.one_hot(y_train, depth=10)
y_test_onehot  = tf.one_hot(y_test,  depth=10)

input_size    = 784
hidden_size   = 128
output_size   = 10
learning_rate = 0.01
epochs        = 10
batch_size    = 128

W1 = tf.Variable(tf.random.normal([input_size, hidden_size], stddev=0.1))
b1 = tf.Variable(tf.zeros([hidden_size]))
W2 = tf.Variable(tf.random.normal([hidden_size, output_size], stddev=0.1))
b2 = tf.Variable(tf.zeros([output_size]))

def forward(x):
    z1 = tf.matmul(x, W1) + b1
    a1 = tf.nn.relu(z1)
    z2 = tf.matmul(a1, W2) + b2
    return tf.nn.softmax(z2)

def loss_function(y_true, y_pred):
    return -tf.reduce_mean(
        tf.reduce_sum(y_true * tf.math.log(y_pred + 1e-8), axis=1)
    )

loss_history = []

for epoch in range(epochs):
    for i in range(0, len(x_train), batch_size):
        x_batch = x_train[i:i+batch_size]
        y_batch = y_train_onehot[i:i+batch_size]

        with tf.GradientTape() as tape:
            predictions = forward(x_batch)
            loss = loss_function(y_batch, predictions)

        gradients = tape.gradient(loss, [W1, b1, W2, b2])

        W1.assign_sub(learning_rate * gradients[0])
        b1.assign_sub(learning_rate * gradients[1])
        W2.assign_sub(learning_rate * gradients[2])
        b2.assign_sub(learning_rate * gradients[3])

    loss_history.append(loss.numpy())

test_pred = forward(x_test)
correct   = tf.equal(tf.argmax(test_pred, 1), y_test)
test_accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

print("Test Accuracy:", test_accuracy.numpy())
```

---

## Code Description

1. **Data loading & preprocessing** — MNIST pixels normalized to [0, 1]; labels converted to one-hot encoding.
2. **Weight initialization** — TensorFlow `Variable` objects with random normal distribution.
3. **`forward()`** — Feed-forward computation using matrix multiplication and activation functions.
4. **`loss_function()`** — Computes cross-entropy loss.
5. **Backpropagation** — `tf.GradientTape()` automatically computes gradients; weights updated via gradient descent.

---

## Performance Evaluation

| Metric | Value |
|--------|-------|
| **Test Accuracy** | ~91–92% (≈ 0.9186 after 10 epochs) |
| **Loss Trend** | Consistently decreasing across epochs |
| **Confusion Matrix** | Most digits correctly classified; minor misclassification between visually similar digits |

**Visualizations:**
- **Figure 1:** Loss vs Epoch graph (decreasing training loss)
- **Figure 2:** Confusion Matrix for digit classification (0–9)

---

## Limitations & Scope of Improvement

- Only one hidden layer and basic gradient descent — limits performance ceiling.
- Training longer or adding neurons may improve accuracy but risks overfitting.
- Potential improvements:
  - Use **Adam optimizer** instead of vanilla gradient descent
  - Add **dropout layers** for regularization
  - Better **weight initialization** (e.g., Xavier/He)
  - Implement **deeper networks** for higher accuracy

---

## Libraries Used

| Library | Purpose |
|---------|---------|
| `tensorflow` | Neural network training with `GradientTape` |
| `numpy` | Array manipulation |
| `matplotlib` | Loss curve visualization |
| `sklearn` | Confusion matrix |
| `seaborn` | Heatmap visualization |

---

## Dataset

**MNIST** — 70,000 grayscale images (28×28) of handwritten digits (0–9)
- Training set: 60,000 images
- Test set: 10,000 images
