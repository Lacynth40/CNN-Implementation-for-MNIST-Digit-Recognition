from ucimlrepo import fetch_ucirepo   
# fetch dataset 
optical_recognition_of_handwritten_digits = fetch_ucirepo(id=80)   
# data (as pandas dataframes) 
X = optical_recognition_of_handwritten_digits.data.features 
y = optical_recognition_of_handwritten_digits.data.targets

# metadata 
print(optical_recognition_of_handwritten_digits.metadata)

# variable information 
print(optical_recognition_of_handwritten_digits.variables) 

#Build a CNN from scratch
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

X = X.values.reshape(-1,8,8,1)
y = y.values

#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Modify the model architecture by adjusting the pooling parameters
model = keras.Sequential(
    [
        keras.Input(shape=(8, 8, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(1, 1)),  # Adjust the pooling parameters
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(1, 1)),  # Adjust the pooling parameters
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ]
)

#Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Train the model
history = model.fit(X_train, y_train, epochs=10, validation_split=0.2)

#Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

#Plot the accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

#Plot the loss
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0, 1])
plt.legend(loc='lower right')  
plt.show()

#Make predictions
predictions = model.predict(X_test)
print(predictions)
print(np.argmax(predictions[0]))
print(y_test[0])

#Plot the first 10 test images, their predicted label, and the true label
fig = plt.figure(figsize=(10, 10))

for i in range(10):
    ax = fig.add_subplot(5, 5, i+1)
    ax.imshow(X_test[i].reshape(8, 8), cmap='gray')
    ax.set_title(f"Predicted: {np.argmax(predictions[i])}, True: {y_test[i]}")
    ax.axis('off')
plt.tight_layout()
plt.show()
