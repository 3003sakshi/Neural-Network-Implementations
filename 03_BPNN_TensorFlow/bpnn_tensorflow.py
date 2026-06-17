import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# -------------------------------
# 1. Load Dataset
# -------------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# -------------------------------
# 2. Preprocessing
# -------------------------------
x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0
x_test = x_test.reshape(-1, 784).astype(np.float32) / 255.0

y_train_onehot = tf.one_hot(y_train, depth=10)
y_test_onehot = tf.one_hot(y_test, depth=10)

# -------------------------------
# 3. Define Parameters
# -------------------------------
input_size = 784
hidden_size = 128
output_size = 10
learning_rate = 0.01
epochs = 10
batch_size = 128

# -------------------------------
# 4. Initialize Weights
# -------------------------------
W1 = tf.Variable(tf.random.normal([input_size, hidden_size], stddev=0.1))
b1 = tf.Variable(tf.zeros([hidden_size]))

W2 = tf.Variable(tf.random.normal([hidden_size, output_size], stddev=0.1))
b2 = tf.Variable(tf.zeros([output_size]))

# -------------------------------
# 5. Forward Propagation
# -------------------------------
def forward(x):
    z1 = tf.matmul(x, W1) + b1
    a1 = tf.nn.relu(z1)
    z2 = tf.matmul(a1, W2) + b2
    return tf.nn.softmax(z2)

# -------------------------------
# 6. Loss Function
# -------------------------------
def loss_function(y_true, y_pred):
    return -tf.reduce_mean(
        tf.reduce_sum(y_true * tf.math.log(y_pred + 1e-8), axis=1)
    )

# -------------------------------
# 7. Training (Backpropagation)
# -------------------------------
loss_history = []
accuracy_history = []

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

    # Accuracy calculation
    train_pred = forward(x_train)
    correct = tf.equal(tf.argmax(train_pred, 1), y_train)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    loss_history.append(loss.numpy())
    accuracy_history.append(accuracy.numpy())

    print(f"Epoch {epoch+1}, Loss: {loss.numpy():.4f}, Accuracy: {accuracy.numpy():.4f}")

# -------------------------------
# 8. Test Accuracy
# -------------------------------
test_pred = forward(x_test)
test_correct = tf.equal(tf.argmax(test_pred, 1), y_test)
test_accuracy = tf.reduce_mean(tf.cast(test_correct, tf.float32))

print("\nTest Accuracy:", test_accuracy.numpy())

# -------------------------------
# 9. Loss Graph
# -------------------------------
plt.figure()
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epoch")
plt.show()

# -------------------------------
# 10. Confusion Matrix
# -------------------------------
y_pred_classes = tf.argmax(test_pred, 1).numpy()
cm = confusion_matrix(y_test, y_pred_classes)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()