import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Load Fashion MNIST dataset

fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Normalize the dataset
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape for CNN input
X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)

# Class names
class_names = ['T-shirt','Trouser','Pullover','Dress','Coat',
               'Sandal','Shirt','Sneaker','Bag','Ankle Boot']


# Function to build CNN model
def create_model(filter_size=3, reg=None, optimizer='adam'):

    model = models.Sequential()

    model.add(layers.Conv2D(32,(filter_size,filter_size),
              activation='relu',
              input_shape=(28,28,1),
              kernel_regularizer=reg))

    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Conv2D(64,(filter_size,filter_size),
              activation='relu',
              kernel_regularizer=reg))

    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Flatten())

    model.add(layers.Dense(128,activation='relu'))

    model.add(layers.Dense(10,activation='softmax'))

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# -----------------------------
# Experiment 1 : Filter Size
# -----------------------------

print("\nEffect of Filter Size")

filter_sizes = [3,5]

for f in filter_sizes:

    print("\nTraining with Filter Size:",f)

    model = create_model(filter_size=f)

    history = model.fit(X_train,y_train,
                        epochs=5,
                        batch_size=64,
                        validation_data=(X_test,y_test))

    test_loss,test_acc = model.evaluate(X_test,y_test)
    print("Test Accuracy:",test_acc)



# -----------------------------
# Experiment 2 : Regularization
# -----------------------------

print("\nEffect of Regularization")

model = create_model(reg=regularizers.l2(0.001))

history = model.fit(X_train,y_train,
                    epochs=5,
                    batch_size=64,
                    validation_data=(X_test,y_test))

test_loss,test_acc = model.evaluate(X_test,y_test)

print("Accuracy with Regularization:",test_acc)



# -----------------------------
# Experiment 3 : Batch Size
# -----------------------------

print("\nEffect of Batch Size")

batch_sizes = [32,128]

for b in batch_sizes:

    print("\nTraining with Batch Size:",b)

    model = create_model()

    history = model.fit(X_train,y_train,
                        epochs=5,
                        batch_size=b,
                        validation_data=(X_test,y_test))

    test_loss,test_acc = model.evaluate(X_test,y_test)
    print("Test Accuracy:",test_acc)



# -----------------------------
# Experiment 4 : Optimizer
# -----------------------------

print("\nEffect of Optimizer")

optimizers = ['adam','sgd']

for opt in optimizers:

    print("\nTraining with Optimizer:",opt)

    model = create_model(optimizer=opt)

    history = model.fit(X_train,y_train,
                        epochs=5,
                        batch_size=64,
                        validation_data=(X_test,y_test))

    test_loss,test_acc = model.evaluate(X_test,y_test)
    print("Test Accuracy:",test_acc)



# -----------------------------
# Confusion Matrix
# -----------------------------

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred,axis=1)

cm = confusion_matrix(y_test,y_pred_classes)

plt.figure(figsize=(8,6))
sns.heatmap(cm,annot=True,fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()



# -----------------------------
# Loss vs Epoch Graph
# -----------------------------

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title("Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend(['Training Loss','Validation Loss'])

plt.show()



# -----------------------------
# Accuracy vs Epoch Graph
# -----------------------------

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title("Accuracy vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend(['Training Accuracy','Validation Accuracy'])

plt.show()



# -----------------------------
# Sample Predictions
# -----------------------------

plt.figure(figsize=(10,10))

for i in range(9):

    plt.subplot(3,3,i+1)
    plt.imshow(X_test[i].reshape(28,28),cmap='gray')

    pred = np.argmax(model.predict(X_test[i].reshape(1,28,28,1)))

    plt.title("Pred: "+class_names[pred])

    plt.axis('off')

plt.show()