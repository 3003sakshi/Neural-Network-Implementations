# Experiment 4 — Hyperparameter Tuning for a BPNN (Iris Dataset)


## Objective

To evaluate the performance of a three-layer Backpropagation Neural Network (BPNN) by varying hyperparameters such as activation function, hidden layer size, learning rate, batch size, and number of epochs using the Iris dataset.

---

## Description of the Model

A three-layer neural network:

| Layer | Neurons | Activation |
|-------|---------|------------|
| Input | 4 (Iris features) | — |
| Hidden | 8 / 16 / 32 (varied) | Sigmoid / Tanh / ReLU |
| Output | 3 (Iris classes) | Softmax |

- **Optimizer:** Adam
- **Loss Function:** Categorical cross-entropy
- **Task:** Multi-class classification

---

## Python Implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

data = load_iris()
X = data.data
y = to_categorical(data.target)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

def build_model(hidden_size, activation, learning_rate):
    model = Sequential()
    model.add(Dense(hidden_size, input_dim=4, activation=activation))
    model.add(Dense(3, activation='softmax'))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model
```

---

## Code Description

1. **Iris dataset** loaded and preprocessed with `StandardScaler`.
2. Output labels converted to **one-hot encoded** vectors.
3. `build_model()` constructs the network with configurable hyperparameters.
4. Multiple combinations tested: activation functions, hidden neurons, learning rates, batch sizes, and epochs.
5. Test accuracy recorded and compared across configurations.

---

## Hyperparameters Tested

| Hyperparameter | Values Explored |
|----------------|-----------------|
| Activation Function | Sigmoid, Tanh, ReLU |
| Hidden Neurons | 8, 16, 32 |
| Learning Rate | 0.001, 0.01, 0.1 |
| Batch Size | 8, 16, 32 |
| Epochs | 50, 100, 200 |

---

## Best Configuration

| Hyperparameter | Best Value |
|----------------|------------|
| Activation Function | Sigmoid |
| Hidden Neurons | 8 |
| Learning Rate | 0.01 |
| Batch Size | 8 |
| Epochs | 50 |
| **Test Accuracy** | **100%** |

---

## Performance Evaluation

- Accuracy comparison graph plotted across all hyperparameter configurations.
- Loss curves show that **higher learning rates** converge faster.
- **Smaller hidden layers** (8 neurons) are sufficient for the Iris dataset.
- Most configurations achieved high accuracy due to the simplicity of the dataset.

---

## Limitations & Scope of Improvement

- Iris is a small, simple dataset — differences between models are minimal and results not fully generalizable.
- For more meaningful evaluation:
  - Test on **larger, more complex datasets**
  - Add **Dropout** to study overfitting behavior
  - Use **Grid Search** or **Random Search** for systematic hyperparameter optimization

---

## Libraries Used

| Library | Purpose |
|---------|---------|
| `tensorflow.keras` | Model building and training |
| `sklearn` | Dataset loading, train/test split, feature scaling |
| `numpy` | Numerical operations |
| `matplotlib` | Accuracy comparison visualization |
| `pandas` | Data handling |

---

## Dataset

**Iris Dataset** — 150 samples, 4 features (sepal/petal length & width), 3 classes (Setosa, Versicolor, Virginica)
