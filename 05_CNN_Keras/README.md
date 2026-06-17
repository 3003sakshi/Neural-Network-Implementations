# Experiment 5 — CNN using Keras for Fashion MNIST Classification


## Objective

To implement and train a Convolutional Neural Network (CNN) using the Keras library to classify images from the Fashion MNIST dataset and study the effect of different hyperparameters such as filter size, regularization, batch size, and optimization algorithms on model performance.

---

## Description of the Model

A CNN architecture for image recognition:

| Layer | Config | Purpose |
|-------|--------|---------|
| Conv2D | 32 filters, 3×3, ReLU | Spatial feature extraction |
| MaxPooling2D | 2×2 | Dimensionality reduction |
| Conv2D | 64 filters, 3×3, ReLU | Deeper feature extraction |
| MaxPooling2D | 2×2 | Dimensionality reduction |
| Flatten | — | Convert 2D features to 1D |
| Dense | 128, ReLU | Fully connected classification layer |
| Dense | 10, Softmax | Output (10 fashion categories) |

**Fashion MNIST Classes:** T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle Boot

---

## Python Implementation

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Normalize data
X_train = X_train / 255.0
X_test  = X_test  / 255.0

# Reshape for CNN input
X_train = X_train.reshape(-1, 28, 28, 1)
X_test  = X_test.reshape(-1,  28, 28, 1)

# Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=5, batch_size=64,
                    validation_data=(X_test, y_test))

test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_acc)
```

---

## Code Description

1. **Dataset loading** — Fashion MNIST loaded via TensorFlow.
2. **Normalization** — Pixel values scaled to [0, 1].
3. **CNN construction** — Conv layers detect image features; pooling layers reduce spatial dimensions.
4. **Flattening & Dense layers** — Extracted features passed through FC layers for classification.
5. **Compilation** — Adam optimizer + sparse categorical cross-entropy loss.
6. **Evaluation** — Accuracy and loss measured on the test set.

---

## Performance Evaluation

| Metric | Value |
|--------|-------|
| **Test Accuracy** | ~90% |
| **Epochs Trained** | 5 |
| **Batch Size** | 64 |

**Visualizations:**
- **Figure 1:** Training and Validation Loss vs Epochs
- **Figure 2:** Training and Validation Accuracy vs Epochs
- **Figure 3:** Confusion Matrix for classification across all 10 classes
- **Figure 4:** Sample predictions on test images

---

## Limitations & Scope of Improvement

- Low-resolution images (28×28) limit complexity of learnable patterns.
- Only 5 epochs trained; longer training could yield higher accuracy.
- Potential improvements:
  - Deeper CNN architectures (e.g., VGG-style, ResNet-style)
  - **Dropout** or **Batch Normalization** for regularization
  - **GPU acceleration** for faster training
  - **Data augmentation** to improve generalization

---

## Libraries Used

| Library | Purpose |
|---------|---------|
| `tensorflow` / `keras` | CNN model building and training |
| `numpy` | Array operations |
| `matplotlib` | Training curves and prediction visualization |

---

## Dataset

**Fashion MNIST** — 70,000 grayscale images (28×28) across 10 fashion categories
- Training set: 60,000 images
- Test set: 10,000 images
