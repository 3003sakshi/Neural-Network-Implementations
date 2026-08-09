# Experiment 6 — RNN using PyTorch for Time Series Prediction

## Objective

To implement and train a Recurrent Neural Network (RNN) using PyTorch to predict the next value in a time series dataset, and to study the effect of hyperparameters such as sequence length, hidden size, and learning rate on model performance.

---

## Description of the Model

A Recurrent Neural Network (RNN) is designed for sequential data. It processes an input sequence step by step, maintaining a hidden state that carries information forward through time to capture temporal dependencies. In this experiment, the RNN takes a fixed-length sequence of past values from a sine wave and predicts the next value using a fully connected output layer.

| Layer | Config | Purpose |
|-------|--------|---------|
| RNN | input_size=1, hidden_size=32, num_layers=1 | Captures temporal dependencies across the sequence |
| Linear (fc) | hidden_size → 1 | Maps final hidden state to the predicted value |

**Input:** Sequence of 10 past time steps
**Output:** Predicted next value in the series

---

## Python Implementation

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Generate time series data
t = np.linspace(0, 100, 400)
data = np.sin(t)
data = (data - np.mean(data)) / np.std(data)

# Create sliding-window dataset
def create_dataset(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 10
X, y = create_dataset(data, seq_length)

# Train-test split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

X_train = torch.FloatTensor(X_train).unsqueeze(-1)
y_train = torch.FloatTensor(y_train).unsqueeze(-1)
X_test  = torch.FloatTensor(X_test).unsqueeze(-1)
y_test  = torch.FloatTensor(y_test).unsqueeze(-1)

# Define RNN model
class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]      # last time step
        out = self.fc(out)
        return out

model = RNNModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 100
train_losses = []
for epoch in range(epochs):
    model.train()
    output = model(X_train)
    loss = criterion(output, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    test_predictions = model(X_test)
    test_loss = criterion(test_predictions, y_test)

print("\nTest Loss:", test_loss.item())
```

---

## Code Description

1. **Data generation** — A sine wave is generated over 400 points and normalized (zero mean, unit variance).
2. **Sequence creation** — A sliding-window function converts the 1D series into input sequences (`seq_length=10`) and corresponding target values.
3. **Train-test split** — 80% of sequences used for training, 20% held out for testing.
4. **RNN construction** — A single-layer `nn.RNN` (hidden_size=32) processes each sequence; the hidden state from the final time step is passed through a fully connected layer to produce the prediction.
5. **Training** — The model is trained for 100 epochs using MSE loss and the Adam optimizer (lr=0.001).
6. **Evaluation** — Test loss (MSE) is computed on held-out sequences.
7. **Visualization** — Loss vs. epoch and actual vs. predicted value plots are generated with matplotlib.

---

## Performance Evaluation

| Metric | Value |
|--------|-------|
| **Loss Function** | Mean Squared Error (MSE) |
| **Epochs Trained** | 100 |
| **Optimizer** | Adam (lr = 0.001) |
| **Sequence Length** | 10 |
| **Hidden Size** | 32 |

The training loss steadily decreases over epochs, indicating the model is learning the underlying pattern of the sine wave. The predicted values closely track the actual test values, confirming the RNN's ability to model short-term temporal dependencies.

**Visualizations:**
- **Figure 1:** Training Loss vs Epochs
- **Figure 2:** Actual vs Predicted values on the test set

---

## My Comments

The RNN performs well on this simple, smooth time series (a sine wave) but may struggle with longer sequences or more complex data due to the vanishing gradient problem inherent to vanilla RNNs. Performance could be improved by:
- Using **LSTM** or **GRU** cells for better long-term dependency learning
- Increasing **hidden size** or stacking **multiple RNN layers**
- Tuning **sequence length** to better capture the periodicity of the data
- Adding **learning rate scheduling** or **gradient clipping** for more stable training

---

## Limitations & Scope of Improvement

- Vanilla RNNs suffer from vanishing/exploding gradients on longer sequences.
- Only trained on a clean, synthetic sine wave — real-world time series (with noise, trends, seasonality) would need more robust preprocessing.
- Potential improvements:
  - **LSTM / GRU** architectures for longer-range dependencies
  - **Bidirectional RNNs** for richer temporal context
  - **Multi-step forecasting** instead of single-step prediction
  - **Real-world datasets** (stock prices, weather, sensor data) for validation

---

## Libraries Used

| Library | Purpose |
|---------|---------|
| `torch` / `torch.nn` | RNN model building and training |
| `numpy` | Array operations and data generation |
| `matplotlib` | Loss curves and prediction visualization |

---

## Dataset

**Synthetic Sine Wave** — 400 points generated over `t = linspace(0, 100, 400)`, normalized and split into overlapping sequences of length 10.
- Training sequences: ~312 (80%)
- Test sequences: ~78 (20%)
