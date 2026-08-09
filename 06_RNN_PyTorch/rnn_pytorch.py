# Import Libraries
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: Generate Time Series Data
# -------------------------------
t = np.linspace(0, 100, 400)
data = np.sin(t)

# Normalize data 
data = (data - np.mean(data)) / np.std(data)

# -------------------------------
# Step 2: Create Dataset
# -------------------------------
def create_dataset(data, seq_length):
    X, y = [], []
    
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    
    return np.array(X), np.array(y)

seq_length = 10
X, y = create_dataset(data, seq_length)

# Train-Test Split
train_size = int(len(X) * 0.8)

X_train = X[:train_size]
X_test = X[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]

# Convert to tensors
X_train = torch.FloatTensor(X_train).unsqueeze(-1)
y_train = torch.FloatTensor(y_train).unsqueeze(-1)

X_test = torch.FloatTensor(X_test).unsqueeze(-1)
y_test = torch.FloatTensor(y_test).unsqueeze(-1)

# -------------------------------
# Step 3: Define RNN Model
# -------------------------------
class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super(RNNModel, self).__init__()
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]  # last time step
        out = self.fc(out)
        return out

model = RNNModel()

# -------------------------------
# Step 4: Loss and Optimizer
# -------------------------------
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # improved LR

# -------------------------------
# Step 5: Training
# -------------------------------
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

# -------------------------------
# Step 6: Evaluation
# -------------------------------
model.eval()
with torch.no_grad():
    test_predictions = model(X_test)
    test_loss = criterion(test_predictions, y_test)

print("\nTest Loss:", test_loss.item())

# -------------------------------
# Step 7: Visualization
# -------------------------------
train_losses = np.array(train_losses)
test_predictions = test_predictions.numpy()
y_test_np = y_test.numpy()

# Epoch list
epochs_list = range(1, epochs + 1)

# Loss vs Epoch
plt.figure()
plt.plot(epochs_list, train_losses, marker='o')
plt.title("Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.show()

# Actual vs Predicted
plt.figure()
plt.plot(y_test_np, label="Actual")
plt.plot(test_predictions, label="Predicted")
plt.legend()
plt.title("RNN Time Series Prediction")
plt.show()