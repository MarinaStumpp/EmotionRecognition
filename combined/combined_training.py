import tensorflow as tf
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd


# load csv file
df = pd.read_csv('predictions.csv')


# data type
X =
y =


# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)


# model creation
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=16),

    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),

    tf.keras.layers.Dense(8)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# model training
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

model.save('combined-model.h5')

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Combined Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('combined-accuracy2.jpg')
plt.close()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Combined Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('combined-loss2.jpg')
plt.close()
