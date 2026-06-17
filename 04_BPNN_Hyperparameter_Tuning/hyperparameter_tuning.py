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

# --------------------------------------------------
# STEP 1: Load and Prepare Dataset
# --------------------------------------------------

data = load_iris()
X = data.data
y = to_categorical(data.target)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --------------------------------------------------
# STEP 2: Model Builder Function
# --------------------------------------------------

def build_model(hidden_size, activation, learning_rate):
    model = Sequential()
    model.add(Dense(hidden_size, input_dim=4, activation=activation))
    model.add(Dense(3, activation='softmax'))

    optimizer = Adam(learning_rate=learning_rate)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    return model

# --------------------------------------------------
# STEP 3: Hyperparameter Values
# --------------------------------------------------

activations = ['sigmoid', 'tanh', 'relu']
hidden_sizes = [8, 16, 32]
learning_rates = [0.001, 0.01]
batch_sizes = [8, 16]
epochs_list = [50, 100]

results = []

# --------------------------------------------------
# STEP 4: Hyperparameter Evaluation
# --------------------------------------------------

for activation in activations:
    for hidden in hidden_sizes:
        for lr in learning_rates:
            for batch in batch_sizes:
                for epochs in epochs_list:

                    model = build_model(hidden, activation, lr)

                    model.fit(
                        X_train, y_train,
                        epochs=epochs,
                        batch_size=batch,
                        verbose=0
                    )

                    loss, accuracy = model.evaluate(
                        X_test, y_test, verbose=0
                    )

                    results.append([
                        activation, hidden, lr, batch, epochs, accuracy
                    ])

# --------------------------------------------------
# STEP 5: Convert to DataFrame
# --------------------------------------------------

results_df = pd.DataFrame(results, columns=[
    "Activation", "Hidden", "LearningRate",
    "BatchSize", "Epochs", "Accuracy"
])

results_df["Accuracy"] = results_df["Accuracy"].astype(float)

# Sort by Accuracy
results_df = results_df.sort_values(by="Accuracy", ascending=False)

# --------------------------------------------------
# STEP 6: Print Clean Output
# --------------------------------------------------

print("\nTop 5 Configurations:")
print(results_df.head())

best = results_df.iloc[0]

print("\nBest Configuration:")
print("Activation Function :", best["Activation"])
print("Hidden Neurons      :", best["Hidden"])
print("Learning Rate       :", best["LearningRate"])
print("Batch Size          :", best["BatchSize"])
print("Epochs              :", best["Epochs"])
print("Test Accuracy       :", best["Accuracy"])

# --------------------------------------------------
# STEP 7: Accuracy Comparison Graph
# --------------------------------------------------

plt.figure()
plt.plot(results_df["Accuracy"].values)
plt.title("Hyperparameter Performance Comparison")
plt.xlabel("Experiment Index (Sorted)")
plt.ylabel("Accuracy")
plt.show()