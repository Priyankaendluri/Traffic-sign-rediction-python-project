import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# Example: assume you have images loaded into X_train and labels into y_train
# For demonstration, let's create dummy data:
num_classes = 43  # GTSRB has 43 classes
input_shape = (32, 32, 3)  # traffic signs resized to 32x32 RGB

# Dummy data: 1000 samples
X_train = np.random.rand(1000, 32, 32, 3).astype(np.float32)
y_train = np.random.randint(0, num_classes, 1000)

X_test = np.random.rand(200, 32, 32, 3).astype(np.float32)
y_test = np.random.randint(0, num_classes, 200)

# Normalize pixel values
X_train /= 255.0
X_test /= 255.0

# Convert labels to categorical
y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)

# Build a simple CNN
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train_cat, epochs=10, batch_size=64, validation_split=0.2)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test_cat)
print(f"Test accuracy: {test_acc:.2f}")

# Predict example
sample = X_test[0:1]
prediction = model.predict(sample)
predicted_class = np.argmax(prediction)
print(f"Predicted traffic sign class: {predicted_class}")

