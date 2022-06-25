import tensorflow as tf
import pickle
import numpy as np
import matplotlib.pyplot as plt

# load data
pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)

# data type
X = np.array(X)
y = np.array(y)

# model creation
model = tf.keras.Sequential([

    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), input_shape=X.shape[1:]),

    tf.keras.layers.Conv2D(filters=32, kernel_size=2, activation='relu'),
    tf.keras.layers.Conv2D(filters=32, kernel_size=2, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(filters=32, kernel_size=2, activation='relu'),
    tf.keras.layers.Conv2D(filters=32, kernel_size=2, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(filters=32, kernel_size=2, activation='relu'),
    tf.keras.layers.Conv2D(filters=32, kernel_size=2, activation='relu'),
    tf.keras.layers.Conv2D(filters=32, kernel_size=2, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(filters=32, kernel_size=2, activation='relu'),
    tf.keras.layers.Conv2D(filters=32, kernel_size=2, activation='relu'),
    tf.keras.layers.Conv2D(filters=32, kernel_size=2, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(filters=32, kernel_size=2, activation='relu'),
    tf.keras.layers.Conv2D(filters=32, kernel_size=2, activation='relu'),
    tf.keras.layers.Conv2D(filters=32, kernel_size=2, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),

    tf.keras.layers.Dense(8)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# model training
history = model.fit(X, y, epochs=20, validation_split=0.2)

model.save('vgg16-model.h5')

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('VGG16 Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('vgg16-accuracy2.jpg')
plt.close()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('VGG16 Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('vgg16-loss2.jpg')
plt.close()
