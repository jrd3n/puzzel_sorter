import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
import matplotlib.pyplot as plt

csv_path = 'object_data.csv'
df = pd.read_csv(csv_path)

image_dir = ''
image_filenames = df['Image File'].tolist()
labels = df[['Top Left X','Top Left Y','Bottom Right X','Bottom Right Y']].values

images = []
for filename in image_filenames:
    image_path = os.path.join(image_dir, filename)
    image = load_img(image_path, target_size=(640,480))
    image_array = img_to_array(image)
    images.append(image_array)

images = np.array(images)

X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.1, random_state=42)


model = tf.keras.Sequential([
    layers.Conv2D(32, (5, 5), activation='relu', input_shape=(640, 480, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4)  # Output layer with 4 units for bounding box regression
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])  # Adjust the optimizer and loss function as needed

history = model.fit(X_train, y_train, epochs=100, batch_size=128, validation_data=(X_val, y_val))

# Extract the loss values from the history object
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Plot the loss values
epochs = range(1, len(train_loss) + 1)
plt.plot(epochs, train_loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#loss = model.evaluate(X_val, y_val)

# Example preprocessing for a single image
image = load_img('images/captured_image_5.jpg', target_size=(640, 480))
image_array = img_to_array(image)
preprocessed_image = image_array / 255.0  # Normalize the pixel values if necessary
preprocessed_image = np.expand_dims(preprocessed_image, axis=0)  # Add batch dimension

predictions = model.predict(preprocessed_image)

print(predictions)