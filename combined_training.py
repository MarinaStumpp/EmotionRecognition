import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle


# load csv file
df = pd.read_csv('predictions.csv')
df = shuffle(df, random_state=42)


# data type
X = df.drop('label', axis=1)
y = df['label']


# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)


# model creation
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=[16]),

    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),

    tf.keras.layers.Dense(8, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# model training
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

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
