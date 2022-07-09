import tensorflow as tf
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

# load data
pickle_in = open("../X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("../y.pickle", "rb")
y = pickle.load(pickle_in)

# data type
X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# model creation
model = tf.keras.Sequential([

    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), input_shape=X.shape[1:]),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Flatten(),

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
history = model.fit(X, y, epochs=30, validation_split=0.2)

model.save('model.h5')

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Own Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('own-model-accuracy.jpg')
plt.close()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Own Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('own-model-loss.jpg')
plt.close()


y_pred = model.predict(X_test).argmax(axis=1)

cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7], normalize='true')

display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"])
display.plot()
plt.savefig('own-confusion.jpg')
plt.close()

