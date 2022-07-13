import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import random

order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
random.shuffle(order)


# load csv file
df = pd.read_csv('predictions.csv')
df = shuffle(df)

# data type
X = df.drop('label', axis=1)
y = df['label']

'''
print(X.head())
X = X.iloc[:, order]
print(X.head())
'''

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
X_test, X_testtest, y_test, y_testtest = train_test_split(X_test, y_test, test_size=0.5)


'''
print(X_testtest.head())
X_testtest = X_testtest.iloc[:, [2, 3, 1, 0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]
print(X_testtest.head())
'''


# model creation
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=[16]),

    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),

    tf.keras.layers.Dense(8, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# model training
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))

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

y_pred = model.predict(X_testtest).argmax(axis=1)

cm = confusion_matrix(y_testtest, y_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7], normalize='true')

display = ConfusionMatrixDisplay(confusion_matrix=cm,
                                 display_labels=["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad",
                                                 "surprised"])
display.plot()
plt.savefig('combined-confusion.jpg')
plt.close()

model.evaluate(X_testtest, y_testtest)
